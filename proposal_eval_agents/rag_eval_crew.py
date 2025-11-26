from traceback import print_tb
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
import logging
from proposal_eval_agents.tools import SearchRelatedChunk, GetAgentsOutput
from openai import OpenAI
from dotenv import load_dotenv
# from .requirement_categorizer_agent import RequirementCategorizerAgent
# from .requirement_verifier_agent import RequirementVerifierAgent
import os
import json
from logger_config import get_logger
import re
from Unscored_Requirement.proposal_evaluator_agent import ProposalEvaluatorAgent

logger = get_logger(__file__)

class RAGEvaluationCrew:
    def __init__(self, language="eng", output_path = "outputs_proposal_eval/formatted_eval.json", logger_instance=None):
        # Configure logging
        self.logger = logger_instance if logger_instance is not None else get_logger(__name__) # Use passed logger or default
        # Removed basicConfig as it's handled by logger_config
        
        # Load environment
        load_dotenv()
        
        self.language = language
        self.model_name = "openrouter/qwen/qwen3-32b"
        self.search_tool = SearchRelatedChunk()
        # Initialize LLM
        self.llm = LLM(
            model=self.model_name,
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3
        )
        self.output_path = output_path

    def create_analyzer_agents(self, sections):
        """Create analyzer agents with ReAct framework for each RFP requirement section"""

        self.logger.info(f"Creating analyzer agents for {len(sections)} sections")
        analyzer_agents = []
        analyzer_tasks = []
        
        unscored_output_dir = "outputs_proposal_eval/rag_eval_crew_proposal_eval/"

        # Create the directory if it doesn't exist
        try:
            os.makedirs(unscored_output_dir, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Error creating directory {unscored_output_dir}: {e}") # Use self.logger
            
        dir = "requirements_analysis_at_Final_analysis"
        os.makedirs(dir, exist_ok=True)

        with open(f"{dir}/requirements_analysis_welcome_entry.json", "w") as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)

        if not sections:
            self.logger.error("No sections provided to create analyzer agents") # Use self.logger
            return [], []

        
        for i, section in enumerate(sections):
            try:
                requirement = section.get('requirement') or section.get('requirement_text', 'Unknown Requirement')
                context = section.get('justification') or section.get('context')
                weight_percentage = section.get('weight_percentage', None)
                is_scored = weight_percentage is not None
                
                # --- Handling for UNSCORED sections ----------------------------------------------The Debuuger Point--------------------------------------------")
                 
                if not is_scored:
                    self.logger.info(f"Section {i} is unscored. Storing its data to a file and skipping analysis.") # Use self.logger
                    status = section.get('status')
                    requirement_id = section.get('requirement_id')
                    # Define a unique filename for the unscored section
                    safe_filename = f"section_{i}_unscored_data.json"
                    output_path = os.path.join(unscored_output_dir, safe_filename)

                    # Prepare the data to be stored
                    unscored_data = {
                        "requirement_id": requirement_id,
                        "status": status,
                        "requirement": requirement,
                        "context": context
                    }

                    # Write the data to its own JSON file
                    try:
                        with open(output_path, 'w') as f:
                            json.dump(unscored_data, f, indent=4)
                    except Exception as e:
                        self.logger.error(f"Failed to write unscored section {i} to file {output_path}: {str(e)}") # Use self.logger
                    continue  # Skip creating agent for unscored sections
                else:
                    self.logger.info(f"Section {i} is scored. Creating agent and task.") # Use self.logger

                if not context:
                    self.logger.warning(f"Empty context for section {i}, skipping...") # Use self.logger
                    continue

                # Get relevant proposal chunks using SearchRelatedChunk with error handling
                try:
                    search_query = f"{requirement} {context}"[:500]  # Truncate to avoid very long queries
                    proposal_chunks = self.search_tool._run(search_query)

                    if not proposal_chunks:
                        self.logger.warning(f"No proposal chunks found for section {i}") # Use self.logger
                        proposal_chunks = ["No relevant content found"]
                except Exception as e:
                    self.logger.error(f"Error searching proposal chunks for section {i}: {str(e)}") # Use self.logger
                    proposal_chunks = ["Error retrieving relevant content"]

                # Create agent with error handling using index-based naming
                agent_type = "Scored"
                try:
                    agent = Agent(
                        role=f"Section {i} {agent_type} Proposal Evaluator",
                        goal=f"Evaluate proposal compliance against RFP {agent_type.lower()} requirement section {i} using provided proposal chunks",
                        backstory=f"An expert evaluator analyzing section {i} compliance by comparing RFP {agent_type.lower()} requirements with proposal content.",
                        llm=self.llm,
                        verbose=True
                    )
                except Exception as e:
                    self.logger.error(f"Error creating agent for section {i}: {str(e)}") # Use self.logger
                    continue

                # Use index-based filename
                safe_filename = f"section_{i}_{agent_type.lower()}_analysis.json"

                # Create different task descriptions based on whether it's scored or unscored
                if is_scored:
                    task_description = self._create_scored_task_description(i, requirement, context, weight_percentage, proposal_chunks)
                    expected_output = f"A detailed JSON analysis of proposal compliance for scored RFP requirement section {i} with scoring justification"
                    # Create task with error handling
                    try:
                        task = Task(
                            description=task_description,
                            expected_output=expected_output,
                            agent=agent,
                            context=[{
                                "name": f"rfp_{agent_type.lower()}_requirement_analysis",
                                "description": f"RFP {agent_type} Requirement Section {i}",
                                "role": "system",
                                "content": (
                                    f"You are an expert proposal evaluator analyzing compliance with RFP {agent_type.lower()} requirement section {i}\n\n"
                                    "## CRITICAL INSTRUCTIONS - FLEXIBLE, GROUNDED, CERTIFICATION-AWARE SCORING\n"
                                    "1. Determine if the RFP content is an approved-options list or individual requirements.\n"
                                    "2. For APPROVED OPTIONS lists: compliant if the proposal uses ANY approved option(s).\n"
                                    "3. For INDIVIDUAL REQUIREMENTS: assess each required element separately.\n"
                                    "4. FULL-CREDIT RULES:\n"
                                    "   - If the requirement is clearly satisfied by the proposal evidence, assign FULL or NEAR-FULL credit.\n"
                                    "   - Do NOT deduct points for unrelated or non-mandatory details that are not part of the requirement.\n"
                                    "   - If a requirement states a minimum (e.g., 'minimum 8 years') and the proposal shows equal or greater (e.g., 9 years), treat as FULLY MET.\n"
                                    "5A. QUANTIFIER INTERPRETATION (IMPORTANT):\n"
                                    "   - Identify quantifiers: 'all/each/every' = UNIVERSAL; 'presence of/at least/any' = EXISTENTIAL.\n"
                                    "   - If ambiguous, DEFAULT to EXISTENTIAL (evidence of qualified key expert(s) is sufficient).\n"
                                    "   - Example: 'Presence of consultants with >10 years' is FULLY MET if named key experts meet this (unless RFP explicitly says 'all consultants').\n"
                                    "5B. AUTHORIZED PARTNER RULE (VENDOR PHRASING LIKE 'FROM BIZZDESIGN'):\n"
                                    "   - Treat 'from <vendor>' as certification/authorized-partner/affiliation unless RFP explicitly requires vendor EMPLOYMENT with clear terms like 'employed by/on payroll/staffed by'.\n"
                                    "   - Authorized agent/partner status plus certified consultant(s) satisfies the requirement unless explicit employment is demanded.\n"
                                                 "5C. IMAGE-ONLY EVIDENCE ACCEPTANCE:\n"
             "   - If certificates are embedded as images but referenced (e.g., 'see page 61-63'), ACCEPT as valid evidence even if text is not OCR'd.\n"
             "5D. PROVEN SUCCESS INTERPRETATION:\n"
             "   - For 'proven success' requirements: Accept completed projects, certifications, authorized partnerships, or tool usage as evidence of success.\n"
             "   - Do NOT require specific metrics, case studies, or measurable outcomes unless the RFP explicitly demands them.\n"
             "   - Example: 'Proven success with Bizzdesign' is satisfied by Bizzdesign-certified consultants, authorized partnerships, or completed projects using Bizzdesign tools.\n"
                                    "6. EVIDENCE & GROUNDING:\n"
                                    "   - Base your analysis ONLY on the RFP and proposal content provided. Do not invent entities, ministries, or facts.\n"
                                    "   - Do NOT mention organizations not present in the inputs (e.g., do not mention 'Ministry of Media' unless it appears in the RFP/proposal content).\n"
                                    "   - Quote or reference specific snippets (and page numbers if available).\n"
                                    "   - If certificates are likely embedded as images and the text mentions or references them (with page numbers), treat that as valid evidence of certification.\n"
                                    "7. GAPS:\n"
                                    "   - Be explicit about actual requirement gaps only. Do not penalize for items outside the stated requirement.\n"
                                    "8. OUTPUT RULES:\n"
                                    "   - Do NOT include score explanations like 'Score of X reflects...'.\n"
                                    "   - Never include the word 'chunk' or chunk numbers in the output.\n\n"
                                    "## EVALUATION LOGIC\n"
                                    "1. Identify content type (options list vs individual requirements).\n"
                                    "2. For options lists: confirm approved options are used; flag only non-approved usage.\n"
                                    "3. For individual requirements: assess each required element for presence and adequacy.\n\n"
                                    f"## RFP REQUIREMENT\n{requirement}\n\n"
                                    f"## RFP CONTEXT\n{context}\n\n"
                                    "## PROPOSAL CONTENT CHUNKS\n"
                                    f"{chr(10).join(proposal_chunks)}"
                                ),
                                "expected_output": f"JSON array with smart analysis - single entry for options lists, separate entries for individual requirements",
                                "framework": "Options vs Requirements Analysis",
                                "evaluation_approach": (
                                    "1. Analyze RFP requirement structure (options list vs individual requirements).\n"
                                    "2. Apply certification policy and full-credit rules where applicable.\n"
                                    "3. Ground claims strictly in provided text and page-referenced evidence.\n"
                                    "4. Provide targeted recommendations based on actual compliance gaps only."
                                ),
                                "proposal_data_chunks": proposal_chunks
                            }],
                            output_file=f"outputs_proposal_eval/rag_eval_crew_proposal_eval/{safe_filename}"
                        )
                    except Exception as e:
                        self.logger.error(f"Error creating task for section {i}: {str(e)}") # Use self.logger
                        continue

                    analyzer_agents.append(agent)
                    analyzer_tasks.append(task)

            except Exception as e:
                self.logger.error(f"Error processing section {i}: {str(e)}") # Use self.logger
                continue

        if not analyzer_agents:
            self.logger.warning("No analyzer agents were created successfully") # Use self.logger

        return analyzer_agents, analyzer_tasks

    def _create_scored_task_description(self, i, requirement, context, weight_percentage, proposal_chunks):
        """Create task description for scored requirements"""
        return (
            f"# TASK: Evaluate Proposal Compliance for Scored RFP Requirement Section {i}\n\n"
            f"## RFP REQUIREMENT\n"
            f"```\n{requirement}\n```\n\n"
            f"## RFP CONTEXT/DETAILS\n"
            f"```\n{context}\n```\n\n"
            f"## WEIGHT PERCENTAGE\n"
            f"```\n{weight_percentage}\n```\n\n"
            "## INSTRUCTIONS\n"
            "1. Analyze the RFP requirement and identify exactly what is demanded.\n"
            "2. Determine if the requirement represents:\n"
            "   - A list of approved options (ANY approved option qualifies as compliant)\n"
            "   - Individual mandatory requirements (ALL must be addressed)\n"
            "   - Specific technical specifications that must be met\n"
            "3. Review the most relevant proposal content chunks for this requirement.\n"
            "4. Evaluate whether the proposal adequately addresses the RFP requirement.\n"
                         "5. Assign a score FAIRLY based on actual compliance (0-100). Use these principles:\n"
             "   - FULL CREDIT (100) when the requirement is clearly met by explicit evidence.\n"
             "   - If a CORE MANDATORY element specified by the RFP is missing (e.g., explicitly requires a vendor-employed consultant or a named certification with proof), assign 0.\n"
             "   - Otherwise use evidence-based thresholds: 75 (strong but minor gaps), 50 (partial), 25 (weak mention), 0 (no evidence).\n"
             "   - Do NOT grant credit for unrelated or non-mandatory details.\n"
             "   - Accept certification evidence (listed certificates, scanned images, or page references).\n"
             "   - Only assert vendor EMPLOYMENT if the RFP contains explicit phrases like \"employed by\", \"staffed by\", or \"on payroll of\" that vendor. Do NOT interpret ambiguous wording like \"from Bizzdesign\" as employment; treat such phrasing as certification/partnership unless explicit employment is stated. If explicit employment is required and not evidenced, assign 0.\n"
             "   - For \"proven success\" requirements: Accept evidence of completed projects, certifications, authorized partnerships, or tool usage as proof of success. Do NOT require specific metrics or case studies unless explicitly demanded by the RFP.\n"
            "6. Quantifier interpretation (IMPORTANT):\n"
            "   - UNIVERSAL: words like 'all', 'each', 'every' mean all items/people must comply.\n"
            "   - EXISTENTIAL: words like 'presence of', 'at least', 'any' mean evidence of qualified key expert(s) is sufficient.\n"
            "   - If ambiguous, default to EXISTENTIAL. Example: 'Presence of consultants with >10 years' is fully met if named key experts meet this (unless RFP explicitly requires 'all consultants').\n"
            "7. Authorized partner rule:\n"
            "   - For vendor phrasing like 'from Bizzdesign', accept certification/authorized-partner evidence unless explicit employment wording is present.\n"
            "   - Accept image-only certificate evidence when referenced by page numbers (treat OCR failures as acceptable).\n"
            "8. Document explicit gaps ONLY when they relate directly to the requirement.\n"
            "9. In 'proposal_compliance', describe SPECIFICALLY what the proposal states, with quotes or page references when available.\n"
            "10. NEVER include score calculations/explanations like 'Score of X reflects...'; state facts only.\n"
            "11. Do not include the word 'chunk' or chunk numbers in the output.\n\n"
            "## GROUNDING & ANTI-HALLUCINATION (MANDATORY)\n"
            "- Base analysis solely on the provided RFP/proposal text.\n"
            "- Do NOT invent ministries, organizations, or facts. Only mention entities that appear in the provided content.\n"
            "- If the input does not mention a specific ministry (e.g., 'Ministry of Media'), do not introduce it.\n"
            "- If certificates are referenced as images/pages, treat that as valid evidence of certification.\n\n"
            "## INTELLIGENT EVALUATION APPROACH\n"
            "1. If requirement says 'not less than 8 years' and the proposal shows 9+ years, consider FULLY MET.\n"
            "2. If the requirement asks for a specific certification, the presence of that certification (even as a scanned image or page reference) is enough to be COMPLIANT.\n"
            "3. For approved options lists: using any approved option = FULLY COMPLIANT; focus on flagging non-approved usage.\n"
            "4. For individual requirements: assess each one based on actual evidence.\n\n"
            "## PROPOSAL CONTENT CHUNKS\n"
            f"The following are the most relevant proposal content chunks for section {i}:\n"
            f"```\n{chr(10).join(proposal_chunks)}\n```\n\n"
            "## EVALUATION CRITERIA\n"
            "- For Options Lists: Is the proposal using approved options? Are non-approved options used?\n"
            "- For Individual Requirements: Is each requirement fully/partially/not addressed in the proposal?\n"
            "- For Technical Specifications: Does the proposal meet the specified technical requirements?\n\n"
            "## EXPECTED OUTPUT FORMAT\n"
            "Always output as a JSON array with the following format:\n\n"
            "```json\n"
            "[{\n"
            "  \"rfp_requirement\": \"Specific requirement from RFP\",\n"
            "  \"proposal_compliance\": \"Describe SPECIFICALLY what the proposal states about this requirement. Include actual content, technologies, approaches, or methods mentioned in the proposal that relate to the RFP requirement. Do NOT use generic terms like 'Met' or 'Partially Met' - provide concrete details from the proposal content.\",\n"
            "  \"technical_strengths\": \"List of strengths in the proposal\",\n"
            "  \"technical_concerns\": \"List of concerns or issues found\",\n"
            "  \"score\": <Appropriate score between 0 and 100>,\n"
            "  \"justification\": \"State facts only - NO score explanations. Example: 'Requirement asks for experience in submitting proposals by project managers. Proposal shows general project experience but does NOT explicitly mention proposal submission by project managers. Missing the core requirement of proposal submission experience.' OR 'Requirement asks for minimum 8 years experience. Proposal shows 9 years experience as Project Manager. Requirement is fully met.'\",\n"
            "  \"weight_percentage\": <number>\n"
            "}]\n"
        )

    # def _create_unscored_task_description(self, i, requirement, context, proposal_chunks):
    #     """Create task description for unscored requirements with a focus on identifying contradictions"""
    #     return (
    #         f"# TASK: Evaluate Proposal Compliance for Unscored RFP Requirement Section {i}\n\n"
    #         f"## RFP REQUIREMENT\n"
    #         f"```\n{requirement}\n```\n\n"
    #         f"## RFP CONTEXT/DETAILS\n"
    #         f"```\n{context}\n```\n\n"
    #         "## INSTRUCTIONS\n"
    #         "1. **ANALYZE** the RFP requirement above and identify what needs to be addressed.\n"
    #         "2. **DETERMINE** if the requirement represents:\n"
    #         "   - A list of approved options (where using ANY from the list is compliant)\n"
    #         "   - Individual mandatory requirements (where ALL must be addressed)\n"
    #         "   - Specific technical specifications that must be met\n"
    #         "3. **REVIEW** the provided proposal content chunks that are relevant to this requirement.\n"
    #         "4. **EVALUATE** whether the proposal adequately addresses the RFP requirement, with a specific focus on identifying contradictions (e.g., missing elements, non-compliance with specifics, or use of non-approved options).\n"
    #         "5. **DOCUMENT** your findings, and explicit contradictions between the RFP requirement and the proposal content.\n"
    #         "6. **CRITICAL**: In the 'proposal_compliance' field, describe SPECIFICALLY what the proposal states about this requirement. Include actual content, technologies, approaches, or methods mentioned. Do NOT use generic terms like 'Met' or 'Partially Met' - provide concrete details from the proposal content and explicitly note contradictions with the RFP requirement.\n"
    #         "7. **FILTER OUTPUT**: Only include requirements in the output where a contradiction is found (i.e., the proposal does not fully meet the RFP requirement, misses key components, or uses non-approved options). If no contradiction exists, exclude the requirement from the final list.\n\n"
    #         "8. Make sure to not include the chunk number or word `chunk` in the output.\n\n"
    #         "## EVALUATION APPROACH FOR SMART ANALYSIS\n"
    #         "1. **Identify Requirement Type**: Determine if RFP content is an options list, individual requirements, or technical specifications.\n"
    #         "2. **For Options Lists** (databases, frameworks, languages, etc.):\n"
    #         "   - Create ONE analysis entry for the entire category.\n"
    #         "   - proposal_compliance: Describe SPECIFICALLY what the proposal states about using approved options and highlight any use of non-approved options as a contradiction.\n"
    #         "   - Only recommend adding more options if the proposal has very limited choices or uses non-approved options.\n"
    #         "3. **For Individual Requirements**:\n"
    #         "   - Create separate entries for each distinct requirement.\n"
    #         "   - Assess each requirement individually, focusing on whether the proposal omits or deviates from required elements, explicitly documenting contradictions.\n"
    #         "4. **For Technical Specifications**:\n"
    #         "   - Verify if the proposal meets the specified technical requirements and identify any deviations or omissions as contradictions.\n\n"
    #         "## PROPOSAL CONTENT CHUNKS\n"
    #         f"The following are the most relevant proposal content chunks for section {i}:\n"
    #         f"```\n{chr(10).join(proposal_chunks)}\n```\n\n"
    #         "## EVALUATION CRITERIA\n"
    #         "Based on the requirement type:\n"
    #         "- **For Options Lists**: Does the proposal use approved options? Are non-approved options used, creating a contradiction?\n"
    #         "- **For Individual Requirements**: Is each requirement fully addressed? Identify contradictions where elements are missing, partially addressed, or misaligned with RFP specifics.\n"
    #         "- **For Technical Specifications**: Does the proposal meet the specified technical requirements? Highlight contradictions where technical specifications are not met or omitted.\n"
    #         "- **Critical Filter**: Only output requirements where a contradiction is identified (e.g., missing components, non-compliance, or deviations from RFP expectations).\n\n"
    #         "## EXPECTED OUTPUT FORMAT\n"
    #         "Always output as a JSON array with the following format, including only requirements with identified contradictions:\n\n"
    #         "```json\n"
    #         "[{\n"
    #         "  \"rfp_requirement\": \"Specific requirement from RFP\",\n"
    #         "  \"proposal_compliance\": \"Describe SPECIFICALLY what the proposal states about this requirement. Include actual content, technologies, approaches, or methods mentioned in the proposal that relate to the RFP requirement. Explicitly note contradictions, such as missing elements, non-compliance with specifics, or use of non-approved options.\",\n"
    #         "  \"technical_strengths\": \"List of strengths in the proposal related to this requirement\",\n"
    #         "  \"technical_concerns\": \"List of concerns or issues found, emphasizing contradictions with the RFP\",\n"
    #         "}]\n"
    #         '```'
    #     )

    def improvement_aggregator(self, scored_requirements, agents_output):
        """
        Combine every agent-analysis JSON into two buckets – 'scored' and 'unscored'.
        """

        if not agents_output:
            self.logger.error("No agent outputs provided") # Use self.logger
            return {}

        # output skeleton
        formatted = {"scored": [], "unscored": []}

        # Sort files by section number to ensure proper order (0, 1, 2, ...)
        def extract_section_number(filename):
            """Extract section number from filename like 'section_<num>_'"""
            match = re.search(r'section_(\d+)_', filename)
            if match:
                return int(match.group(1))
            return float('inf')  # Put files without section number at the end
        
        # Sort the agent outputs by section number
        sorted_outputs = sorted(agents_output.items(), key=lambda x: extract_section_number(x[0]))
        
        for fname, raw in sorted_outputs:
            try:
                # -------- 1. normalise / parse --------
                improvements = raw
                if isinstance(improvements, str):
                    # pull out fenced-JSON if present
                    if "```json" in improvements:
                        improvements = improvements.split("```json", 1)[1].split("```", 1)[0].strip()
                    improvements = json.loads(improvements)
                # at this point `improvements` must be a list (or list-like) of dicts
                if not isinstance(improvements, list):
                    improvements = [improvements]

                # -------- 2. route into scored / unscored bucket --------
                bucket = "scored" if "_scored_analysis" in fname else "unscored"
                formatted[bucket].extend(improvements)

            except Exception as e:
                self.logger.error(f"Error processing {fname}: {e}") # Use self.logger
                formatted.setdefault("errors", []).append(
                    {fname: f"Failed to parse – {e}"}
                )

        # Validate scored_requirements before processing
        validated_scored_requirements = []
        for i, req in enumerate(scored_requirements):
            if not isinstance(req, dict):
                self.logger.warning(f"Skipping requirement {i}: not a dictionary") # Use self.logger
                continue
            
            weight_percentage = req.get('weight_percentage')
            if weight_percentage is None:
                self.logger.warning(f"Skipping requirement {i}: missing 'weight_percentage' key") # Use self.logger
                continue
            
            # Check if weight_percentage is a number or string that can be converted to float
            is_valid = False
            weight_value = None
            if isinstance(weight_percentage, (int, float)):
                weight_value = float(weight_percentage)
                is_valid = weight_value > 0
            elif isinstance(weight_percentage, str):
                # Try to convert to float, with or without '%' symbol
                try:
                    # Remove '%' if present and convert to float
                    clean_value = weight_percentage.strip('%').strip()
                    weight_value = float(clean_value)
                    is_valid = weight_value > 0
                except ValueError:
                    pass
            
            if is_valid:
                validated_scored_requirements.append(req)
            else:
                self.logger.warning(f"Skipping requirement {i}: invalid 'weight_percentage' value '{weight_percentage}' - must be a number greater than 0 or string that can be converted to a number greater than 0") # Use self.logger
        
        # Update scored_requirements to use validated list
        scored_requirements = validated_scored_requirements

        # Mathematics based balanced score calculation
        weight_percentages = []
        for i in range(len(scored_requirements)):
            weight_value = scored_requirements[i]['weight_percentage']
            if isinstance(weight_value, str):   
                # Remove '%' if present and convert to float
                weight_percentages.append(float(weight_value.strip('%').strip()))
            else:
                weight_percentages.append(float(weight_value))


        total_percentage = sum(weight_percentages)
        
        if total_percentage == 100:
            scale_factor = 1
        elif total_percentage > 0:
            scale_factor = 100 / total_percentage
        else:
            scale_factor = 0
        
        weight_percentages = [round(wp * scale_factor,2) for wp in weight_percentages]

        score_outof_100 = [round(item['score'] * 0.01 * weight_percentages[i],2) for i, item in enumerate(formatted['scored'])]

        for i in range(len(formatted['scored'])):
            formatted['scored'][i]['assigned_score'] = score_outof_100[i]
            formatted['scored'][i]['requirement_score'] = weight_percentages[i]
            del formatted['scored'][i]['score']

        formatted['final_score_out_of_100'] = round(sum(score_outof_100),2)
        Total_assigned_score = sum(formatted['scored'][i]['assigned_score'] for i in range(len(formatted['scored'])))

        # -------- 3. persist to disk (optional) --------
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(formatted, f, indent=2, ensure_ascii=False)
            self.logger.info("Saved formatted improvements to file") # Use self.logger
        except Exception as e:
            self.logger.error(f"Error saving formatted improvements: {e}") # Use self.logger

        return formatted, self.output_path, Total_assigned_score

    def filter_unscored_requirements(self, unscored_requirements):
        """
        Filter unscored requirements using specialized agents
        """
        self.logger.info("Starting filtering of unscored requirements") # Use self.logger
        dirname = "unScore_filter_json"
        os.makedirs(dirname, exist_ok=True)

        # Save input requirements for debugging
        with open(f"{dirname}/core_unscored_requirements.json", "w") as f:
            json.dump(unscored_requirements, f, indent=2, ensure_ascii=False)

        # Create categorizer agent
        categorizer = RequirementCategorizerAgent(self.llm)
        categorizer_agent = categorizer.create_agent()
        self.logger.info("Created categorizer agent") # Use self.logger

        # Create categorization task
        categorization_task = Task(
            description=categorizer.create_categorization_prompt(unscored_requirements),
            agent=categorizer_agent,
            expected_output="""A JSON object containing categorized requirements in the following format:
            {
                "standalone_requirements": [
                    {
                        "requirement": "requirement text",
                        "context": "why this is standalone"
                    }
                ],
                "proposal_dependent_requirements": [
                    {
                        "requirement": "requirement text",
                        "context": "why this needs proposal verification"
                    }
                ],
                "basic_requirements_excluded": [
                    {
                        "requirement": "requirement text",
                        "context": "why this was identified as a basic requirement and excluded"
                    }
                ]
            }""",
            verbose=True,
        )
        
        # Run categorization
        self.logger.info("Running categorizer crew") # Use self.logger
        categorizer_crew = Crew(
            agents=[categorizer_agent],
            tasks=[categorization_task],                    
            process=Process.sequential,
            manager_llm=self.llm
        )
        self.logger.debug("Calling categorizer_crew.kickoff()...") # Use self.logger
        categorization_results = categorizer_crew.kickoff()
        self.logger.debug(f"categorizer_crew.kickoff() raw result type: {type(categorization_results)}") # Use self.logger
        self.logger.debug(f"categorizer_crew.kickoff() raw result (first 500 chars): {str(categorization_results)[:500]}") # Use self.logger
        if categorization_results is None or (isinstance(categorization_results, str) and not categorization_results.strip()):
            self.logger.error("Categorizer crew returned None or an empty string. This will likely cause a downstream error.") # Use self.logger
        self.logger.info(f"Categorization results: {categorization_results}") # Use self.logger

        # Parse categorization results
        if isinstance(categorization_results, str):
            json_str = categorization_results
        else:
            json_str = str(categorization_results)
            if "```json" in json_str:
                # Split on the first occurrence of ```json and take the part after it
                json_str = json_str.split("```json", 1)[-1]
                # Remove trailing ```
                if "```" in json_str:
                    json_str = json_str.split("```", 1)[0].strip()
            elif "```" in json_str:
                # Split on the first occurrence of ``` and take the part after it
                json_str = json_str.split("```", 1)[-1].strip()
            self.logger.info(f"Parsing JSON string: {json_str}") # Use self.logger
            # Ensure json_str is a string and not a list
            if isinstance(json_str, list):
                # Try to join list elements into a single string
                json_str = "\n".join(json_str)
            try:
                categorized = json.loads(json_str)
                self.logger.info("Successfully parsed categorization JSON") # Use self.logger
            except Exception as e:
                self.logger.error(f"Error parsing categorization JSON: {str(e)}. Attempting to extract JSON substring.") # Use self.logger
                # Try to extract JSON substring if possible
                import re
                match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if match:
                    try:
                        categorized = json.loads(match.group(0))
                        self.logger.info("Successfully parsed categorization JSON after extracting substring") # Use self.logger
                    except Exception as e2:
                        self.logger.error(f"Still failed to parse categorization JSON: {str(e2)}") # Use self.logger
                        categorized = {
                "standalone_requirements": [],
                "proposal_dependent_requirements": unscored_requirements,
                "basic_requirements_excluded": []
            }
        
        standalone_reqs = categorized.get('standalone_requirements', [])
        dependent_reqs = categorized.get('proposal_dependent_requirements', [])
        basic_reqs_excluded = categorized.get('basic_requirements_excluded', [])

        # Save categorized requirements
        with open(f"{dirname}/unscored_normal_requirements.json", "w") as f:
            json.dump(standalone_reqs, f, indent=2, ensure_ascii=False)
        with open(f"{dirname}/unscored_dependent_requirements.json", "w") as f:
            json.dump(dependent_reqs, f, indent=2, ensure_ascii=False)
        with open(f"{dirname}/unscored_basic_requirements_excluded.json", "w") as f:
            json.dump(basic_reqs_excluded, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Standalone requirements: {len(standalone_reqs)}, Proposal-dependent: {len(dependent_reqs)}, Basic excluded: {len(basic_reqs_excluded)}") # Use self.logger

        # Create verifier agent
        verifier = RequirementVerifierAgent(self.llm, self.search_tool)
        verifier_agent = verifier.create_agent()
        self.logger.info("Created verifier agent") # Use self.logger

        # Combine non-basic requirements for verification
        all_requirements_without_scored = standalone_reqs + dependent_reqs

        with open(f"{dirname}/all_requirements_without_scored.json", "w") as f:
            json.dump(all_requirements_without_scored, f, indent=2, ensure_ascii=False)


        verified_reqs = []

        # Verify requirements with retry logic
        for req in all_requirements_without_scored:
            self.logger.info(f"Verifying requirement: {req['requirement']}") # Use self.logger
            search_query = req['requirement']
            
            # Debug: Print search query
            self.logger.debug(f"\nSearching for requirement: {search_query}") # Use self.logger
            
            # Debug: Check embedding manager state
            self.logger.debug(f"Embedding count before search: {self.search_tool.embedding_manager.get_embedding_count()}") # Use self.logger
            
            proposal_chunks = self.search_tool._run(search_query)
            
            # Debug: Print chunk results
            self.logger.debug(f"Retrieved chunks: {len(proposal_chunks) if isinstance(proposal_chunks, list) else 0}") # Use self.logger
            if proposal_chunks and isinstance(proposal_chunks, list):
                self.logger.debug(f"First chunk preview: {proposal_chunks[0][:200]}...") # Use self.logger
            
            self.logger.info(f"Retrieved {len(proposal_chunks) if isinstance(proposal_chunks, list) else 1} proposal chunks") # Use self.logger
            
            dir="chunker_verification"  
            os.makedirs(dir, exist_ok=True)

            formated_req_chunks = {
                "requirement": req['requirement'],
                "chunks": proposal_chunks
            }
            with open(f"{dir}/chunker_verification_at_RAG_level.json", "w") as f:
                json.dump(formated_req_chunks, f, indent=2, ensure_ascii=False)

            max_retries = 3
            attempt = 0
            result = None
            last_raw_llm_output = ""

            while attempt < max_retries:
                self.logger.info(f"Attempt {attempt + 1}/{max_retries} for requirement: {req['requirement']}") # Use self.logger
                is_retry = (attempt > 0)
                
                verification_task = Task(
                    description=verifier.create_verification_prompt(req['requirement'], proposal_chunks, is_retry=is_retry),
                    agent=verifier_agent,
                    expected_output="""
                    {
                        "requirement": "requirement text",
                        "is_addressed": true/false,
                        "match_type": "FULL/PARTIAL/NONE",
                        "evidence": "relevant text from proposal or explanation",
                        "confidence": "HIGH/MEDIUM/LOW",
                        "is_basic_requirement": true/false,
                        "retrieved_chunks": "proposal chunks"
                    }
                    """,

                    verbose=True
                )

                verifier_crew = Crew(
                    agents=[verifier_agent],
                    tasks=[verification_task],
                    process=Process.sequential,
                    manager_llm=self.llm
                )
                self.logger.debug("Calling verifier_crew.kickoff()...") # Use self.logger
                verification_result = verifier_crew.kickoff()
                self.logger.debug(f"verifier_crew.kickoff() raw result type: {type(verification_result)}") # Use self.logger
                self.logger.debug(f"verifier_crew.kickoff() raw result (first 500 chars): {str(verification_result)[:500]}") # Use self.logger
                last_raw_llm_output = str(verification_result) # Always store the raw output

                # IMMEDIATE CHECK: If LLM returned None or empty, assign a default error and retry
                if verification_result is None or (isinstance(verification_result, str) and not verification_result.strip()):
                    self.logger.warning(f"LLM returned None or empty for {req['requirement']} (Attempt {attempt + 1}). Retrying...") # Use self.logger
                    # No need to set result = None here, as parsing attempt will handle it
                    # last_raw_llm_output is already set to str(verification_result)
                    attempt += 1
                    continue # Skip to next attempt

                # Save verification result for debugging for each attempt
                verifier_dir = "verifier"
                os.makedirs(verifier_dir, exist_ok=True)
                with open(f"{verifier_dir}/verification_result_{hash(req['requirement'])}_attempt{attempt+1}.json", "w") as f:
                    f.write(last_raw_llm_output)

                # Attempt to parse verification result
                try:
                    result_str = str(verification_result)
                    if "```json" in result_str:
                        result_str = result_str.split("```json", 1)[-1]
                        if "```" in result_str:
                            result_str = result_str.split("```", 1)[0].strip()
                    elif "```" in result_str: # Fallback for just triple backticks without 'json'
                        result_str = result_str.split("```")[1].strip()
                    
                    if not result_str: # If string is empty after stripping, raise error to trigger fallback
                        raise ValueError("Extracted JSON string is empty.")
                    
                    result = json.loads(result_str)
                    self.logger.info(f"Verification result for {req['requirement']}: {result}") # Use self.logger
                    break # Break out of retry loop if parsing is successful
                
                except (json.JSONDecodeError, ValueError) as parse_e:
                    self.logger.warning(f"JSON parsing failed for {req['requirement']} (Attempt {attempt + 1}): {parse_e}. Raw string prefix: \n{result_str[:500]}...") # Use self.logger
                    result = None # Ensure result is None if parsing fails
                except Exception as e:
                    self.logger.error(f"Unexpected error during parsing attempt {attempt + 1} for {req['requirement']}: {str(e)}") # Use self.logger
                    result = None
                
                attempt += 1
            
            # After all retries, if result is still None (parsing failed), create a fallback result
            if result is None:
                self.logger.error(f"Failed to get valid JSON after {max_retries} attempts for requirement: {req['requirement']}. Recording raw output.") # Use self.logger
                
                # Ensure last_raw_llm_output is not None
                if not last_raw_llm_output:
                    last_raw_llm_output = "LLM returned no content or an unhandled error."

                result = {
                    "requirement": req['requirement'],
                    "is_addressed": False,
                    "match_type": "NONE",
                    "evidence": f"LLM failed to return valid JSON after {max_retries} attempts. Last raw output: {last_raw_llm_output}",
                    "confidence": "LOW",
                    "is_basic_requirement": False,
                    "retrieved_chunks": "Error: LLM output unparseable after retries"
                }

            # Only include requirements in the final list if they are reportable
            # Include if addressed AND not basic, OR if NOT addressed AND not basic AND not an empty/unparseable JSON error
            # Changed to ensure 'evidence' key exists for safety
            if (result['is_addressed'] and not result.get('is_basic_requirement', False)) or \
               (not result['is_addressed'] and not result.get('is_basic_requirement', False) and not (isinstance(result.get('evidence'), str) and "LLM returned empty or unparseable JSON." in result.get('evidence'))):
                verified_reqs.append({
                    "requirement": req['requirement'],
                    "context": req['context'],
                    "verification_result": result
                })
            elif result.get('is_basic_requirement', False):
                self.logger.info(f"Excluding basic requirement: {req['requirement']}") # Use self.logger
            else:
                self.logger.info(f"Excluding fully addressed or non-reportable requirement: {req['requirement']} (Is Addressed: {result['is_addressed']}, Is Basic: {result.get('is_basic_requirement', False)})") # Use self.logger

        # Save final filtered requirements
        with open(f"{dirname}/unscored_requirements_filtered.json", "w", encoding="utf-8") as f:
            json.dump(verified_reqs, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Final filtered unscored requirements: {len(verified_reqs)}") # Use self.logger
        return verified_reqs

    def start_crew_process(self, rfp_requirements=None,rfp_path=None,proposal_path=None, retries: int = 3):
        """Run the complete evaluation process"""
        self.logger.info("Starting RAGEvaluationCrew process.") # Use self.logger
        
        # Handle new format with scored and unscored requirements
        scored_requirements = rfp_requirements.get('scored_requirements', [])
        # unscored_requirements = rfp_requirements.get('unscored_requirements', [])
        self.logger.debug("0"*100) # Use self.logger
        self.logger.debug(scored_requirements) # Use self.logger
        self.logger.debug("0"*100) # Use self.logger
        
        dirname = "unScore_tracker"
        os.makedirs(dirname, exist_ok=True)
        with open(f"{dirname}/scored_requirements.json", "w") as f:
            json.dump(scored_requirements, f, indent=2, ensure_ascii=False)
            
        # unscored_requirements = rfp_requirements.get('unscored_requirements', [])
        unscored_requirements_data = rfp_requirements.get('unscored_requirements', [])
        unscored_requirements = unscored_requirements_data[0]['requirements'] if unscored_requirements_data else []

        # Store unscored_requirements in JSON file
        unscored_requirements_dir = "outputs_proposal_eval"
        os.makedirs(unscored_requirements_dir, exist_ok=True)
        unscored_requirements_json_path = os.path.join(unscored_requirements_dir, "unscored_requirements.json")
        
        with open(unscored_requirements_json_path, "w", encoding="utf-8") as f:
            json.dump({"requirements": unscored_requirements}, f, indent=2, ensure_ascii=False)
        
        # self.rfp_requirements_output = unscored_requirements_json_path
        proposal_evaluator = ProposalEvaluatorAgent(
                requirements_json_path=unscored_requirements_json_path,
                proposal_path=proposal_path,
                output_path="outputs_proposal_eval_new/unscored_requirements",
                logger_instance=self.logger
            )
            
            # Run the evaluation
        self.logger.info("Running ProposalEvaluatorAgent for unscored requirements...") # Use self.logger
        unscored_requirements = proposal_evaluator.run_evaluation()
        self.logger.info("ProposalEvaluatorAgent for unscored requirements completed.") # Use self.logger

        unscored_requirements = unscored_requirements.get("all_evaluations", [])

        # dirname = "unScore_tracker"
        # os.makedirs(dirname, exist_ok=True)
        # with open(f"{dirname}/scored_requirements.json", "w") as f:
        #     json.dump(scored_requirements, f, indent=2, ensure_ascii=False)
        # with open(f"{dirname}/Row_unscored_requirements.json", "w") as f:
        #     json.dump(unscored_requirements, f, indent=2, ensure_ascii=False)

        # self.logger.info("going into filter unscored requirements") # Use self.logger
        # # Filter unscored requirements
        # filtered_unscored = self.filter_unscored_requirements(unscored_requirements)
        # self.logger.info("filtered_unscored:") # Use self.logger
        # unscored_requirements = filtered_unscored
        # self.logger.info("process end unscored_requirements:") # Use self.logger

        # with open(f"{dirname}/modified_unscored_requirements.json", "w") as f:
        #     json.dump(unscored_requirements, f, indent=2, ensure_ascii=False)

        all_sections = scored_requirements + unscored_requirements
        
        if not all_sections:
            self.logger.error("No requirements provided in the input") # Use self.logger
            return None, None, None # Return None if no sections to process

        with open(f"{dirname}/all_sections.json", "w") as f:
            json.dump(all_sections, f, indent=2, ensure_ascii=False)

        # Create analyzer agents and tasks
        analyzer_agents, analyzer_tasks = self.create_analyzer_agents(all_sections)
        if len(scored_requirements) > 0:
        # First run the analyzer agents to get all the analysis results
            analyzer_crew = Crew(
                agents=analyzer_agents,
                tasks=analyzer_tasks,                   
                process=Process.sequential,
                manager_llm=self.llm,
                model=self.model_name,
                verbose=True,
                full_output=True,
                output_log_file="log/analyzer_crew.log"
            )
            
            analyzer_results = None
            for attempt in range(retries):
                self.logger.info(f"Attempt {attempt + 1}/{retries} to run analyzer crew...") # Use self.logger
                try:
                    analyzer_results = analyzer_crew.kickoff()
                    self.logger.debug(f"analyzer_crew.kickoff() raw result type: {type(analyzer_results)}") # Use self.logger
                    self.logger.debug(f"analyzer_crew.kickoff() raw result (first 500 chars): {str(analyzer_results)[:500]}") # Use self.logger
                    
                    if analyzer_results is None or (isinstance(analyzer_results, str) and not analyzer_results.strip()):
                        self.logger.warning(f"Analyzer crew returned None or an empty string on attempt {attempt + 1}. Retrying...") # Use self.logger
                        continue # Continue to the next attempt

                    analyzer_results_str = str(analyzer_results)
                    json_to_save = analyzer_results_str
                    if "```json" in analyzer_results_str:
                        json_to_save = analyzer_results_str.split("```json", 1)[1].split("```", 1)[0].strip()
                    elif "```" in analyzer_results_str:
                        json_to_save = analyzer_results_str.split("```")[1].strip()

                    parsed_json = json.loads(json_to_save)
                    self.logger.info(f"Successfully parsed analyzer results as JSON on attempt {attempt + 1}.") # Use self.logger
                    break # Exit retry loop on success

                except json.JSONDecodeError as json_e:
                    self.logger.warning(f"Failed to parse extracted JSON for analyzer results on attempt {attempt + 1}: {json_e}. Retrying... Raw string prefix: {json_to_save[:500]}...") # Use self.logger
                except Exception as e:
                    self.logger.error(f"Error during analyzer crew execution on attempt {attempt + 1}: {e}. Retrying...") # Use self.logger
                analyzer_results = None # Reset analyzer_results to None for the next attempt


            if analyzer_results is None:
                self.logger.error(f"All {retries} attempts to run analyzer crew failed. Returning None.") # Use self.logger
                return None, None, None

            dir="analyzer_results"
            os.makedirs(dir, exist_ok=True)
            try: 
                # Convert analyzer_results to a string first
                analyzer_results_str = str(analyzer_results)
                
                # Apply JSON extraction logic from improvement_aggregator
                json_to_save = analyzer_results_str
                if "```json" in analyzer_results_str:
                    json_to_save = analyzer_results_str.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in analyzer_results_str: # Fallback for just triple backticks without 'json'
                    json_to_save = analyzer_results_str.split("```")[1].strip()

                # Attempt to parse to validate and re-serialize
                try:
                    parsed_json = json.loads(json_to_save)
                    with open(f"{dir}/analyzer_results.json", "a") as f:
                        json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                    self.logger.info("Successfully saved analyzer results as JSON.") # Use self.logger
                except json.JSONDecodeError as json_e:
                    self.logger.error(f"Failed to parse extracted JSON for saving: {json_e}. Saving as raw text. Raw string prefix: {json_to_save[:500]}...") # Use self.logger
                    with open(f"{dir}/analyzer_results_raw.txt", "a") as f:
                        f.write(f"Analyzer results (raw, failed JSON parse):\n{analyzer_results_str}\n\n")
                except Exception as e:
                    self.logger.error(f"Error during JSON processing for saving: {e}. Saving as raw text. Raw string prefix: {json_to_save[:500]}...") # Use self.logger
                    with open(f"{dir}/analyzer_results_raw.txt", "a") as f:
                        f.write(f"Analyzer results (raw, unexpected error):\n{analyzer_results_str}\n\n")
            except Exception as e:
                self.logger.error(f"Error handling analyzer results before saving: {e}") # Use self.logger
                with open(f"{dir}/analyzer_results_raw.txt", "a") as f:
                    f.write(f"Analyzer results (raw, top-level error):\n{str(analyzer_results)}\n\n")

            self.logger.info("✅ Analyzer agents completed their tasks") # Use self.logger
        
        # Get all agent outputs after analyzers have completed
        agents_output = GetAgentsOutput()._run()
        
        # Create the improvement formatter agent to process the analyzer outputs
        formatted_eval, output_path, Total_assigned_score = self.improvement_aggregator(scored_requirements, agents_output)

        self.logger.info("RAGEvaluationCrew process completed.") # Use self.logger
        return formatted_eval, output_path, Total_assigned_score

if __name__ == "__main__":
    # Example usage with new format
    rfp_requirements = {
        "scored_requirements": [
            {
                "requirement": "Bidder selection may be based on considerations beyond lowest price",
                "weight_percentage": "Not specified",
                "context": "The Ministry of Culture reserves the right to select bidders based on other considerations, not just the lowest price (Section 1.17)."
            },
            {
                "requirement": "Compliance with work scope",
                "weight_percentage": "20%",
                "context": "Evaluates whether the proposal addresses all specified work scope requirements including software implementation, testing, and phased approach."
            },
            {
                "requirement": "Technical architecture compliance",
                "weight_percentage": "25%",
                "context": "Assessment of technical architecture alignment with specified requirements including database platforms, operating systems, and security measures."
            }
        ],
        "unscored_requirements": [
            {
                "requirement": "Implement latest software versions",
                "context": "Must implement the latest software and applications provided by the company, including detailed design specifications."
            },
            {
                "requirement": "Bid must specify total estimated cost and allow cost reductions via independent letter until submission deadline",
                "context": "The bidder must specify the total estimated cost of the project implementation in the bid form, and any reduction in the total cost must be made through an independent letter until the date of submission of the bid documents."
            }
        ]
    }
    
    evaluator = RAGEvaluationCrew(
        language="eng"
    )
    
    # Use the new format
    results = evaluator.start_crew_process(rfp_requirements)
    print(results)