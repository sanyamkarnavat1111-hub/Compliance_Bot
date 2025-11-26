from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import logging
# from .tools import SearchRelatedChunk, GetAgentsOutput
from procurement_agentic_system.tools import SearchRelatedChunk, GetAgentsOutput
from posprocessing import extract_clean_json, extract_list_of_dicts
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import sys
from logger_config import get_logger
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = get_logger(__file__)

def run_analyzer_in_thread(agent, task, llm, model_name, index, llm_kwargs):
    """Run analyzer in thread and return result with index"""
    try:
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            manager_llm=llm,
            manager_llm_kwargs=llm_kwargs,
            model=model_name,
            verbose=True,
            full_output=True,
            output_log_file=f"log/analyzer_crew_log_{index}.log"
        )
        result = crew.kickoff()
        return index, result
    except Exception as e:
        return index, f"Error: {str(e)}"

class RFPEvaluationCrew:
    def __init__(self, ea_standard_text, additional_sections, max_concurrent_calls=10):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Load environment
        load_dotenv()
        
        self.ea_standard_text = ea_standard_text
        self.additional_sections = additional_sections
        self.model_name = "openrouter/qwen/qwen3-32b"
        self.search_tool = SearchRelatedChunk()
        self.max_concurrent_calls = max_concurrent_calls
        
        self.llm_kwargs = {
            "temperature": 0.1,
            "top_k": 1,
            "top_p": 1,
            "seed": 42
            }

        # Initialize LLM
        self.llm = LLM(
            model=self.model_name,
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
            temperature=self.llm_kwargs['temperature'],
            top_k=self.llm_kwargs['top_k'],
            top_p=self.llm_kwargs['top_p'],
            seed=self.llm_kwargs['seed']
        )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY'),
        )

    def create_extractor_crew(self):
        """Create and run the extractor crew with ReAct framework for comprehensive extraction"""
        extractor_agent = Agent(
            role="EA Section Extractor with ReAct Framework - Comprehensive Detail Mode",
            goal="Extract EVERY possible section, subsection, detail, and piece of information from the EA standard with maximum granularity.",
            backstory="An expert at exhaustive document analysis who never misses any detail, no matter how small or seemingly insignificant.",
            llm=self.llm,
            llm_kwargs=self.llm_kwargs,
            reasoning=True,
            verbose=True,
            max_reasoning_attempts=7,
            max_iterations=8
        )

        # ReAct structured context with actual content
        context = [
            {
                "description": "ReAct Framework Analysis for COMPREHENSIVE EA standard document extraction",
                "expected_output": "JSON array of ALL sections with maximum detail extraction and reasoning traces",
                "reasoning_framework": {
                    "thought_process": "Analyze EVERY aspect of the EA standard document with microscopic detail",
                    "action_plan": "Extract ALL content systematically with maximum granularity",
                    "observation": "Validate that NO detail has been missed",
                    "reflection": "Ensure COMPLETE coverage of all document elements"
                },
                "source_content": {
                    "ea_standard_document": self.ea_standard_text,
                    "additional_sections": self.additional_sections,
                    "extraction_instructions": [
                        "Parse EVERY line of the EA standard document",
                        "Extract ALL headers, subheaders, and micro-sections",
                        "Include EVERY table, row, column, and cell as separate entries",
                        "Capture ALL metadata, footnotes, and annotations",
                        "Extract ALL technical specifications and details",
                        "Include ALL examples, code snippets, and references",
                        "Maintain ALL original formatting and structure indicators"
                    ]
                }
            }
        ]

        extractor_task = Task(
            description=(
                f"Follow the ReAct framework to extract EVERY possible detail from the provided EA standard document and additional sections.\n\n"
                f"**EA STANDARD DOCUMENT TO ANALYZE:**\n{self.ea_standard_text}\n\n"
                f"**ADDITIONAL SECTIONS TO INCLUDE (COMPULSARILY):**\n{self.additional_sections}\n\n"

                "**COMPREHENSIVE EXTRACTION STRATEGY**\n"
                "1. **Maximum Granularity Approach**:\n"
                "   - Extract EVERY section, subsection, and micro-detail\n"
                "   - Create separate entries for each distinct piece of information\n"
                "   - Include ALL technical specifications, parameters, and values\n"
                "   - Capture ALL examples, code snippets, and illustrations\n\n"
                "2. **Multi-Level Content Analysis**:\n"
                "   - Level 1: Major sections and headers\n"
                "   - Level 2: Subsections and subheaders\n"
                "   - Level 3: Individual paragraphs and statements\n"
                "   - Level 4: Technical details, specifications, and parameters\n"
                "   - Level 5: Examples, code, references, and footnotes\n\n"
                "3. **Exhaustive Table Processing**:\n"
                "   - Extract table headers as separate sections\n"
                "   - Extract each table row as individual entries\n"
                "   - Extract each cell content as granular information\n"
                "   - Include table captions, footnotes, and metadata\n\n"
                "4. **Metadata and Structural Elements**:\n"
                "   - Extract ALL formatting indicators (bold, italic, underline)\n"
                "   - Include ALL delimiters, separators, and structural markers\n"
                "   - Capture ALL cross-references and links\n"
                "   - Include ALL version information and timestamps\n\n"

                "**THOUGHT**: Perform microscopic analysis of the EA standard document structure, ensuring active filtering of all financial content:\n"
                "- Scan EVERY character, word, and line for potential information, but flag and discard any financial or monetary terms.\n"
                "- Identify ALL possible section boundaries, no matter how subtle, but exclude sections with financial implications.\n"
                "- Look for headers, subheaders, bullet points, numbered lists, tables, code blocks, but ignore any that are financial in nature.\n"
                "- Identify technical specifications, parameters, values, and measurements, ensuring they are NOT financial.\n"
                "- Find examples, illustrations, diagrams, and references, but only if they are NON-FINANCIAL.\n"
                "- Locate metadata, version info, dates, and authorship details, ensuring they are NOT financial.\n"
                "- Consider semantic groupings and conceptual relationships, always excluding financial aspects.\n\n"

                "**ACTION**: Extract with maximum detail by, while strictly adhering to financial exclusion:\n"
                "1. **Document Structure Analysis (Non-Financial Focus)**:\n"
                "   - Scan for ALL headers (H1, H2, H3, etc.) and create sections, but exclude any that are financial.\n"
                "   - Extract ALL paragraph content as individual entries, but only if they are NON-FINANCIAL.\n"
                "   - Identify ALL list items (bulleted, numbered, nested) as separate sections, but only if they are NON-FINANCIAL.\n"
                "   - Extract ALL table structures completely, but process only NON-FINANCIAL tables and their contents.\n\n"

                "2. **Technical Content Extraction (Strictly Non-Financial)**:\n"
                "   - Extract ALL technical specifications and parameters, ensuring NO financial values are included.\n"
                "   - Include ALL code snippets, configurations, and examples, but only if they are NON-FINANCIAL.\n"
                "   - Capture ALL formulas, calculations, and measurements, but only if they are NON-FINANCIAL.\n"
                "   - Extract ALL standards, protocols, and references, ensuring NO financial implications.\n\n"

                "3. **Detailed Table Processing (Financial Exclusion)**:\n"
                "   - For each table: extract table title/caption as a section, but only if it is NON-FINANCIAL.\n"
                "   - Extract column headers as individual sections, but only if they are NON-FINANCIAL.\n"
                "   - Extract each row as a separate section with full row content, but only if it is NON-FINANCIAL and contains no financial terms.\n"
                "   - Extract each cell as granular information if it contains significant data, but only if it is NON-FINANCIAL.\n"
                "   - Include table footnotes and annotations, but only if they are NON-FINANCIAL.\n\n"

                "4. **Metadata and Auxiliary Information (Non-Financial)**:\n"
                "   - Extract ALL document properties (title, author, date, version), ensuring NO financial metadata.\n"
                "   - Include ALL footnotes, endnotes, and annotations, but only if they are NON-FINANCIAL.\n"
                "   - Capture ALL cross-references and hyperlinks, ensuring they do NOT point to financial content.\n"
                "   - Extract ALL formatting and structural indicators, ensuring they are NOT related to financial presentation.\n\n"

                "5. **Additional Sections Processing (Strictly Non-Financial)**:\n"
                "   - Include ALL additional sections exactly as provided, but ONLY if they are COMPLETELY NON-FINANCIAL.\n"
                "   - Extract sub-components of additional sections if they exist, but only if they are NON-FINANCIAL.\n"
                "   - Maintain complete fidelity to additional section content, as long as it is NON-FINANCIAL.\n\n"

                "6. **Contextual and Semantic Extraction (Financial Exclusion)**:\n"
                "   - Extract conceptual groupings and themes, but only for NON-FINANCIAL aspects.\n"
                "   - Include process descriptions and workflows, ensuring NO financial implications.\n"
                "   - Capture relationships and dependencies, ensuring NO financial connections.\n"
                "   - Extract business rules and constraints, but only for NON-FINANCIAL rules.\n\n"

                "**OBSERVATION**: Validate comprehensive extraction by ensuring STRICT EXCLUSION of all financial and monetary content:\n"
                "- EVERY line of the document has been analyzed, and financial content has been filtered out.\n"
                "- ALL headers, subheaders, and micro-sections are captured, but ONLY if they are NON-FINANCIAL.\n"
                "- EVERY table, row, and significant cell is extracted, but ONLY if it is NON-FINANCIAL.\n"
                "- ALL technical details and specifications are included, ensuring NO financial values are present.\n"
                "- EVERY example, code snippet, and reference is captured, but ONLY if it is NON-FINANCIAL.\n"
                "- ALL metadata and structural elements are preserved, but ONLY if they are NON-FINANCIAL.\n"
                "- NO financial content has been overlooked or included by mistake.\n"
                "- Additional sections are completely included, but ONLY if they are NON-FINANCIAL.\n\n"

                "**REFLECTION**: Perform final validation, ensuring absolute absence of financial information:\n"
                "- Have I extracted EVERY possible piece of information that is NON-FINANCIAL?\n"
                "- Are ALL technical specifications and parameters included, with NO financial values?\n"
                "- Have I captured ALL examples and code snippets that are NON-FINANCIAL?\n"
                "- Are ALL table elements extracted with maximum granularity, ensuring NO financial data?\n"
                "- Is ALL metadata and structural information preserved, with NO financial context?\n"
                "- Have I included EVERY detail from additional sections that is NON-FINANCIAL?\n"
                "- Would a domain expert find ANY financial information present that should have been excluded?\n"
                "- Is the extraction truly comprehensive and exhaustive, while being strictly NON-FINANCIAL?\n\n"

                "**OUTPUT FORMAT (STRICT JSON ONLY - NO FINANCIAL CONTENT)**:\n"
                "Return results in the following JSON structure — nothing else outside this structure. Ensure all 'name' and 'content' fields are completely devoid of any financial or monetary information:\n"
                "[\n"
                "  {\n"
                "    \"name\": \"Section Name (be specific and descriptive, NON-FINANCIAL)\",\n"
                "    \"content\": \"Complete actual content with ALL details (STRICTLY NO FINANCIAL CONTENT)\",\n"
                "    \"reasoning\": \"Detailed explanation of what was extracted and why (NON-FINANCIAL REASONING ONLY)\",\n"
                "    \"granularity_level\": \"Level of detail (Major/Sub/Detail/Technical/Micro)\",\n"
                "    \"content_type\": \"Type of content (Header/Paragraph/Table/Code/Metadata/etc.)\"\n"
                "  },\n"
                "  ...\n"
                "]\n\n"

                "**CRITICAL RULES FOR COMPREHENSIVE OUTPUT (STRICTLY NO FINANCIAL CONTENT)**\n"
                "1. EXTRACT EVERYTHING NON-FINANCIAL: No technical/qualitative detail is too small or insignificant. ALL financial content must be excluded.\n"
                "2. MAXIMUM GRANULARITY (NON-FINANCIAL): Break down complex non-financial content into granular pieces.\n"
                "3. PRESERVE ALL NON-FINANCIAL CONTENT: Include every word, number, and symbol from non-financial sections.\n"
                "4. DETAILED NAMING (NON-FINANCIAL): Use descriptive, specific section names that are free of financial terms.\n"
                "5. COMPLETE NON-FINANCIAL CONTENT: Content field must contain ALL relevant non-financial information.\n"
                "6. THOROUGH REASONING (NON-FINANCIAL): Explain what was extracted and its significance, without any financial context.\n"
                "7. CLASSIFICATION (NON-FINANCIAL): Include granularity level and content type, ensuring no financial classification.\n"
                "8. NO FINANCIAL OMISSIONS: Better to over-extract non-financial than under-extract, but ALL financial content must be omitted.\n"
                "9. FORMAT STRICTNESS (NON-FINANCIAL JSON): Output must be valid JSON, and strictly contain no financial data.\n"
                "10. EXHAUSTIVE NON-FINANCIAL COVERAGE: Ensure absolutely nothing non-financial is missed, and all financial content is excluded.\n"
            ),
            expected_output=(
                "A comprehensive JSON array containing EVERY possible section, subsection, detail, "
                "technical specification, table element, code snippet, metadata, and piece of information "
                "from the EA standard document, with complete reasoning traces following ReAct methodology."
            ),
            context=context,
            agent=extractor_agent
        )

        extractor_crew = Crew(
            agents=[extractor_agent],
            tasks=[extractor_task],
            process=Process.sequential,
            manager_llm=self.llm,
            manager_llm_kwargs=self.llm_kwargs,
            model=self.model_name,
            verbose=True
        )

        return extractor_crew

    def create_analyzer_agents(self, sections):
        """Create analyzer agents with ReAct framework for each EA standard section"""
        analyzer_agents = []
        analyzer_tasks = []

        if not sections:
            self.logger.error("No sections provided to create analyzer agents")
            return [], []

        for section in sections:
            try:
                name = section.get('name', 'Unknown Section')
                content = section.get('content', '')

                if not content:
                    self.logger.warning(f"Empty content for section {name}, skipping...")
                    continue

                # Get relevant RFP chunks using SearchRelatedChunk with error handling
                try:
                    search_query = f"{name} {content}"
                    rfp_chunks = self.search_tool._run(search_query)
                    if not rfp_chunks:
                        self.logger.warning(f"No RFP chunks found for section {name}")
                        rfp_chunks = ["No relevant content found"]
                except Exception as e:
                    self.logger.error(f"Error searching RFP chunks for section {name}: {str(e)}")
                    rfp_chunks = ["Error retrieving relevant content"]

                # Create agent with error handling
                try:
                    agent = Agent(
                        role=f"{name} RFP Analyzer",
                        goal=f"Analyze RFP compliance against EA Standard '{name}' using provided RFP chunks",
                        backstory=f"An expert evaluator analyzing '{name}' compliance by comparing EA standards with RFP content.",
                        llm=self.llm,
                        llm_kwargs=self.llm_kwargs,
                        reasoning=True,
                        verbose=True,
                        max_reasoning_attempts=5,
                        max_iterations=5
                    )
                except Exception as e:
                    self.logger.error(f"Error creating agent for section {name}: {str(e)}")
                    continue

                # Sanitize filename with error handling
                try:
                    safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name)
                    safe_name = safe_name.replace(' ', '_').lower()
                except Exception as e:
                    self.logger.error(f"Error sanitizing filename for section {name}: {str(e)}")
                    safe_name = f"section_{len(analyzer_agents)}"

                try:
                    task = Task(
                        description=(
                            f"# TASK: Analyze RFP Compliance for '{name}' Section (STRICTLY NO FINANCIAL CONTENT)\n\n"
                            f"## **EA STANDARD SECTION CONTENT (NON-FINANCIAL)**\n"
                            f"```\n{name}: {content}\n```\n\n"
                            "## CRITICAL INSTRUCTIONS - STRICT COMPLIANCE (NO FINANCIAL ANALYSIS)\n"
                            "1. **ANALYZE** ONLY the EA standard section above and identify the NON-FINANCIAL requirements.\n"
                            "2. **DETERMINE** if the content represents (only for NON-FINANCIAL aspects):\n"
                            "   - A list of approved options (where using ANY from the list is compliant)\n"
                            "   - Individual mandatory requirements (where ALL must be addressed)\n"
                            "3. **REVIEW** the provided RFP content chunks that are relevant to this section, and contain NO financial information.\n"
                            "4. **EVALUATE** compliance based on the requirement type, focusing SOLELY on technical and qualitative aspects, and STRICTLY EXCLUDING any financial or monetary considerations.\n"
                            "5. **DOCUMENT** your findings and recommendations, ensuring NO financial details are included.\n\n"

                            "## STRICT RECOMMENDATION RULES (NO FINANCIAL RECOMMENDATIONS)\n"
                            "**CRITICAL**: All recommendations MUST be based ONLY on the EA standard content provided above, and MUST NOT contain any financial, budgetary, cost, currency, amount, payment, or money-related terms or implications.\n"
                            "- DO NOT recommend anything that is not explicitly mentioned in the EA standard content, and ensure recommendations are NON-FINANCIAL.\n"
                            "- DO NOT add external knowledge, best practices, or industry standards, especially if they have financial implications.\n"
                            "- DO NOT suggest improvements beyond what the EA standard requires, and ensure they are NON-FINANCIAL.\n"
                            "- Recommendations should only address gaps between RFP and the specific EA standard content, and be STRICTLY NON-FINANCIAL.\n"
                            "- If the EA standard doesn't mention something, DO NOT recommend it, particularly if it relates to financial aspects.\n\n"

                            "## CRITICAL GAP ANALYSIS RULES (NO FINANCIAL GAPS)\n"
                            "**MANDATORY**: Only identify gaps for requirements that are EXPLICITLY stated in the EA standard content above, and are STRICTLY NON-FINANCIAL.\n"
                            "- If the EA standard doesn't specify exact technologies, DO NOT flag missing technologies, and definitely DO NOT flag financial omissions.\n"
                            "- Only flag gaps when the EA standard explicitly requires something that is missing from the RFP, and ensure the gap is NON-FINANCIAL.\n"
                            "- If the EA standard is general/vague, focus on whether the RFP addresses the general requirement, not specific implementations, and ensure NO financial gaps are inferred.\n"
                            "- DO NOT create gaps for industry best practices or common technologies unless explicitly required by the EA standard, and ensure NO financial gaps.\n\n"

                            "6. Make sure to not include the chunk number or word `chunk` in the output. STRICTLY EXCLUDE any discussion of project costs, budget, currency, specific amounts, pricing models, or any financial values from your analysis and recommendations.\n\n"

                            "## ADDITIONAL EVALUATION GUIDANCE (STRICTLY NON-FINANCIAL)\n"
                            "- When analyzing RFP coverage, check if each required role/component is described anywhere in the RFP, even if not in a list, ensuring NO financial information is included.\n"
                            "- If the required information is scattered or not in a structured list, note this is not a major gap, consider that information, and assign that point as `Met` in compliance_status (for NON-FINANCIAL aspects only).\n"
                            "- If the EA standard requirement is general/vague, focus on whether the RFP addresses the general requirement, not specific implementations, and ensure NO financial analysis.\n"
                            "- Only mark as `Not Met` if the EA standard explicitly requires something specific that is completely missing and is NON-FINANCIAL.\n\n"

                            "## EVALUATION APPROACH FOR SMART ANALYSIS (STRICTLY NON-FINANCIAL)\n"
                            "1. **Identify Content Type**: Determine if EA content is an options list or individual requirements (for NON-FINANCIAL aspects only).\n"
                            "2. **For Options Lists** (databases, frameworks, languages, etc. - NON-FINANCIAL):\n"
                            "   - Create ONE analysis entry for the entire category\n"
                            "   - ea_requirement: 'Use of approved [category] options (NON-FINANCIAL)'\n"
                            "   - Focus on identifying NON-APPROVED items being used in RFP (NON-FINANCIAL)\n"
                            "   - Status = Met if only approved options used, Partially Met if mix, Not Met if only non-approved\n"
                            "3. **For Individual Requirements**: (NON-FINANCIAL only)\n"
                            "   - Create separate entries for each distinct requirement\n"
                            "   - Assess each requirement individually\n\n"

                            "## **RFP CONTENT CHUNKS (STRICTLY NO FINANCIAL CONTENT)**\n"
                            f"The following are the most relevant **RFP content chunks** for '{name}':\n"
                            f"```\n{chr(10).join(rfp_chunks)}\n```\n\n"
                            "## EVALUATION CRITERIA (STRICTLY NON-FINANCIAL)\n"
                            "Based on the requirement type:\n"
                            "- **For Options Lists**: Is the RFP using approved options? Are non-approved options being used? (NON-FINANCIAL)\n"
                            "- **For Individual Requirements**: Is each requirement fully/partially/not addressed? (NON-FINANCIAL)\n"
                            "- **For Technical Specifications**: Does the proposal meet the specified technical requirements? Highlight contradictions where technical specifications are not met or omitted. (NON-FINANCIAL)\n\n"

                            "## EXPECTED OUTPUT FORMAT (STRICTLY NON-FINANCIAL ANALYSIS)\n"
                            "Always output as a JSON array with the original format, but adapt the analysis logic. Ensure ALL fields are completely devoid of any financial or monetary information:\n\n"

                            "For OPTIONS LISTS (like databases, frameworks, etc. - NON-FINANCIAL):\n"
                            "- Create ONE entry for overall compliance of the options category\n"
                            "- Focus on non-compliant usage rather than unused approved options\n"
                            "For INDIVIDUAL REQUIREMENTS (NON-FINANCIAL):\n"
                            "- Create separate entries for each distinct requirement\n\n"

                            "Required JSON format:\n"
                            "```json\n"
                            "[{\n"
                            "  \"ea_requirement\": \"Specific NON-FINANCIAL requirement from EA standard (short with good grammer and no extra words or redundant words)\",\n"
                            "  \"compliance_status\": \"Met/Partially Met/Not Met\",\n"
                            "  \"rfp_coverage\": \"Description of how RFP addresses this NON-FINANCIAL requirement (things that are included in RFP), ensuring NO financial or monetary details are mentioned.\",\n"
                            "  \"gap_analysis\": \"Specific NON-FINANCIAL gaps or issues found while comparing that thing of EA standard against RFP chunks (things that are not included in RFP but must be there as per reference from EA Standard). And if options are mentioned in EA standard, then for reference, mention the NON-FINANCIAL options that are not used in RFP). Ensure NO financial or monetary details are mentioned.\",\n"
                            "  \"recommendation\": {\n"
                            "    \"action\": \"Specific NON-FINANCIAL action to address the gap based ONLY on EA standard content, and containing NO financial or monetary terms.\",\n"
                            "    \"priority\": \"High/Medium/Low\",\n"
                            "    \"rationale\": \"Why this NON-FINANCIAL improvement is important based on EA standard requirements, and containing NO financial or monetary terms.\"\n"
                            "  }\n"
                            "}]\n"
                            '```'
                        ),
                        expected_output=f"A detailed JSON array analysis of RFP compliance for '{name}' section with smart handling of options vs individual requirements (STRICTLY NO FINANCIAL CONTENT)",
                        agent=agent,
                        context=[{
                            "name": "ea_standard_analysis",
                            "description": f"EA Standard Section: {name}",
                            "role": "system",
                            "content": (
                                f"You are an expert RFP evaluator analyzing compliance with EA standard: {name}\n\n"
                                "## CRITICAL INSTRUCTIONS - STRICT EA STANDARD COMPLIANCE (NO FINANCIAL ANALYSIS)\n"
                                "1. First determine if the EA standard content represents approved options or individual requirements (for NON-FINANCIAL aspects only).\n"
                                "2. APPROVED OPTIONS (comma/line separated lists): RFP is compliant if it uses ANY approved option(s) (for NON-FINANCIAL aspects only)\n"
                                "3. INDIVIDUAL REQUIREMENTS: Each requirement must be individually assessed (for NON-FINANCIAL aspects only)\n"
                                "4. Focus on compliance gaps, not missing options that aren't being used (for NON-FINANCIAL aspects only)\n\n"

                                "## STRICT RECOMMENDATION POLICY (NO FINANCIAL RECOMMENDATIONS)\n"
                                "**MANDATORY**: All recommendations MUST be based EXCLUSIVELY on the EA standard content provided, and MUST NOT contain any financial, budgetary, cost, currency, amount, payment, or money-related terms or implications.\n"
                                "- DO NOT recommend anything not explicitly mentioned in the EA standard, and ensure recommendations are NON-FINANCIAL.\n"
                                "- DO NOT add external knowledge, industry best practices, or additional requirements, especially if they have financial implications.\n"
                                "- DO NOT suggest improvements beyond what the EA standard specifically requires, and ensure they are NON-FINANCIAL.\n"
                                "- If the EA standard doesn't mention something, DO NOT recommend it, particularly if it relates to financial aspects.\n"
                                "- Recommendations should only address gaps between RFP and the specific EA standard content, and be STRICTLY NON-FINANCIAL.\n\n"

                                "## CRITICAL GAP ANALYSIS POLICY (NO FINANCIAL GAPS)\n"
                                "**MANDATORY**: Only identify gaps for requirements EXPLICITLY stated in the EA standard, and are STRICTLY NON-FINANCIAL.\n"
                                "- If EA standard doesn't specify exact technologies, DO NOT flag missing technologies, and definitely DO NOT flag financial omissions.\n"
                                "- Only flag gaps when EA standard explicitly requires something missing from RFP, and ensure the gap is NON-FINANCIAL.\n"
                                "- DO NOT create gaps for industry best practices unless explicitly required by EA standard, and ensure NO financial gaps.\n\n"

                                "## ADDITIONAL EVALUATION GUIDANCE (STRICTLY NON-FINANCIAL)\n"
                                "- When analyzing RFP coverage, check if each required role/component is described anywhere in the RFP, even if not in a list, ensuring NO financial information is included.\n"
                                "- If the required information is scattered or not in a structured list, note this is not a major gap, consider that information, and assign that point as `Met` in compliance_status (for NON-FINANCIAL aspects only).\n"
                                "- If the EA standard requirement is general/vague, focus on whether the RFP addresses the general requirement, not specific implementations, and ensure NO financial analysis.\n"
                                "- Only mark as `Not Met` if the EA standard explicitly requires something specific that is completely missing and is NON-FINANCIAL.\n\n"

                                "## EVALUATION LOGIC - CRITICAL (STRICTLY NON-FINANCIAL)\n"
                                "1. **DETERMINE CONTENT TYPE FIRST**:\n"
                                "   - Options List: Comma/line-separated items (databases, languages, frameworks) - NON-FINANCIAL ONLY.\n"
                                "   - Individual Requirements: Specific policies, procedures, or mandatory items - NON-FINANCIAL ONLY.\n\n"

                                "2. **FOR OPTIONS LISTS** - Create ONE JSON entry (NON-FINANCIAL ONLY):\n"
                                "   - ea_requirement: 'Use of approved [category] from EA standard (NON-FINANCIAL)'\n"
                                "   - compliance_status: Based on whether RFP uses approved options (NON-FINANCIAL)\n"
                                "   - Focus analysis on NON-APPROVED usage, not missing unused options (NON-FINANCIAL)\n"
                                "   - Recommend removing non-approved items as HIGH priority (NON-FINANCIAL)\n"
                                "   - Recommend adding more approved options only as MEDIUM/LOW priority AND only if EA standard suggests flexibility (NON-FINANCIAL)\n\n"

                                "3. **FOR INDIVIDUAL REQUIREMENTS** - Create separate entries for each (NON-FINANCIAL ONLY)\n\n"

                                f"## EA STANDARD CONTENT (STRICTLY NO FINANCIAL CONTENT)\n{content}\n\n"
                                "## RFP CONTENT CHUNKS (STRICTLY NO FINANCIAL CONTENT)\n"
                                f"{chr(10).join(rfp_chunks)}"
                            ),
                            "expected_output": f"JSON array with smart analysis - single entry for options lists, separate entries for individual requirements (STRICTLY NO FINANCIAL CONTENT)",
                            "framework": "Options vs Requirements Analysis",
                            "evaluation_approach": (
                                "1. Analyze EA standard content structure (options list vs individual requirements - NON-FINANCIAL ONLY)\n"
                                "2. For options lists: Check if RFP uses approved options and flag non-approved usage (NON-FINANCIAL)\n"
                                "3. For individual requirements: Assess each requirement separately (NON-FINANCIAL)\n"
                                "4. Provide targeted recommendations based on actual compliance gaps from EA standard only (NON-FINANCIAL)"
                            ),
                            "rfp_data_chunks": rfp_chunks
                        }],
                        output_file=f"outputs/{safe_name}_analysis.json"
                    )
                except Exception as e:
                    self.logger.error(f"Error creating task for section {name}: {str(e)}")
                    continue

                analyzer_agents.append(agent)
                analyzer_tasks.append(task)

            except Exception as e:
                self.logger.error(f"Error processing section: {str(e)}")
                continue

        if not analyzer_agents:
            self.logger.warning("No analyzer agents were created successfully")

        return analyzer_agents, analyzer_tasks

    def create_improvement_formatter_agent(self, agents_output):
        """Process each analysis file separately and combine the results"""
        
        if not agents_output:
            self.logger.error("No agent outputs provided")
            return {}

        formatted_improvements = {}
        
        try:
            # Process each analysis file separately
            for key, data in agents_output.items():
                try:
                    # improvements = self.extract_actions_only(data)
                    improvements = data
                    
                    # Parse the improvements from the agent's output
                    if isinstance(improvements, str):
                        try:
                            # Try to extract JSON from the string if needed
                            if '```json' in improvements:
                                improvements = improvements.split('```json')[1].split('```')[0].strip()
                            improvements_list = json.loads(improvements)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Error parsing JSON from string for {key}: {str(e)}")
                            improvements_list = [{"error": f"Failed to parse improvements: {str(e)}"}]
                    else:
                        improvements_list = improvements

                    # Get the category name from the key
                    category = key.replace('_analysis', '')
                        
                    # Add to the formatted improvements
                    formatted_improvements[category] = improvements_list
                    
                except Exception as e:
                    self.logger.error(f"Error processing {key} improvements: {str(e)}")
                    formatted_improvements[key] = [{"error": f"Failed to process improvements: {str(e)}"}]
            
            # Save the combined results with error handling
            try:
                os.makedirs("outputs", exist_ok=True)
                with open("outputs/formatted_improvements.json", "w", encoding='utf-8') as f:
                    json.dump(formatted_improvements, f, indent=2, ensure_ascii=False)
                self.logger.info("Successfully saved formatted improvements to file")
            except Exception as e:
                self.logger.error(f"Error saving formatted improvements to file: {str(e)}")
                
            return formatted_improvements
            
        except Exception as e:
            self.logger.error(f"Unexpected error in improvement formatter: {str(e)}")
            return {"error": f"Failed to process improvements: {str(e)}"}

    def final_sections_analyzer_and_updater(self, extracted_sections_json):
        """Process and deduplicate sections with error handling"""
        try:
            # Validate input
            if not extracted_sections_json:
                self.logger.error("Empty sections JSON provided")
                return []

            # Attempt API call with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model="qwen/qwen3-32b",
                        # model="qwen/qwen3-32b",
                        temperature=self.llm_kwargs["temperature"],
                        top_p=self.llm_kwargs["top_p"],
                        seed=self.llm_kwargs["seed"],
                        messages=[{
                            "role": "user",
                            "content": f"""Analyze the provided JSON array and intelligently deduplicate entries while preserving all unique and valuable information. Follow these comprehensive guidelines:

                        ## Primary Deduplication Rules:

                        ### 1. Content Overlap Analysis
                        - **Exact Duplicates**: Remove entries with identical content, titles, or descriptions
                        - **Near-Exact Matches**: Remove entries with >95% content similarity (raised threshold to preserve more nuanced differences)
                        - **Subset Relationships**: When one entry is completely contained within another, consider merging rather than removing to preserve context and source attribution
                        - **Superset Preservation**: Retain comprehensive versions but also preserve standalone entries if they contain unique details, formatting, or context not present in the superset

                        ### 2. Hierarchical Content Prioritization with Granular Preservation
                        - **Broader Categories**: Prioritize comprehensive sections but maintain granular entries when they provide:
                        - Specific implementation details not captured in broader sections
                        - Unique formatting or presentation that aids understanding
                        - Standalone reference value for specific topics
                        - Different levels of detail for different audiences
                        - **Contextual Completeness**: Retain both comprehensive overviews AND detailed specifications
                        - **Logical Groupings**: Create hybrid entries that combine related concepts while preserving individual component details
                        - **Multi-Level Structure**: Support both high-level summaries and detailed breakdowns within the same topic area

                        ### 3. Unique Value Preservation
                        - **Different Perspectives**: Keep entries with the same topic but different analytical approaches or contexts
                        - **Source Diversity**: Retain entries from different document sources/sections even if content overlaps (indicate source differences)
                        - **Temporal Variations**: Preserve entries representing different time periods or versions
                        - **Stakeholder Perspectives**: Keep entries that represent different viewpoints or requirements from various stakeholders

                        ### 4. Content Specificity Guidelines with Detail Preservation
                        - **Technical Details**: Always preserve specific technical specifications, metrics, or configurations, even if mentioned elsewhere
                        - **Implementation Details**: Retain unique implementation approaches, methodologies, or procedures with their specific context
                        - **Quantitative Data**: Always preserve numerical data, measurements, or specific values in their original context
                        - **Qualitative Insights**: Keep unique observations, recommendations, or analysis with their source reasoning
                        - **Formatting and Structure**: Preserve entries where specific formatting (lists, tables, hierarchical structure) adds value
                        - **Standalone Reference Value**: Maintain entries that serve as quick reference points even if information exists elsewhere

                        ## Advanced Deduplication Logic:

                        ### 5. Semantic Relationship Handling
                        - **Complementary Information**: Keep entries that complement each other rather than duplicate
                        - **Cross-References**: Preserve entries that reference or build upon other entries
                        - **Dependency Relationships**: Maintain entries that have logical dependencies between them

                        ### 6. Metadata and Context Preservation
                        - **Source Attribution**: Maintain entries with unique source information or provenance
                        - **Timestamps**: Preserve entries with different temporal contexts
                        - **Authority Levels**: Keep entries from different authority levels or approval stages
                        - **Version Control**: Retain entries representing different versions or iterations

                        ### 7. Quality and Completeness Metrics
                        - **Information Density**: Favor entries with higher information density and detail
                        - **Clarity and Structure**: Prioritize well-structured, clearly articulated entries
                        - **Actionability**: Prefer entries that provide actionable information or clear next steps
                        - **Comprehensiveness**: Retain entries that provide complete coverage of a topic

                        ## Specific Handling Instructions:

                        ### 8. Information Preservation Strategies
                        - **Merge Similar Entries**: Instead of removing, combine entries with 70-95% overlap into enriched versions that preserve all unique elements
                        - **Maintain Source Context**: When combining entries, preserve the original source context and reasoning from each component
                        - **Create Hybrid Sections**: Develop comprehensive sections that maintain granular subsections for detailed reference
                        - **Cross-Reference Preservation**: Maintain entries that serve as cross-references or provide alternative perspectives on the same topic
                        - **Audience-Specific Versions**: Keep both technical and summary versions when they serve different purposes

                        ### 9. Enhanced Deduplication Logic
                        - **Preserve Granular Details**: When consolidating, maintain specific details, bullet points, and structured information
                        - **Context-Aware Merging**: Combine entries while preserving the unique context and reasoning from each source
                        - **Incremental Information**: Keep entries that add incremental value even if they overlap with larger sections
                        - **Reference Utility**: Maintain entries that provide quick reference value for specific topics
                        - **Original Structure**: Preserve meaningful structural elements (lists, hierarchies, categorizations) that aid comprehension

                        ### 10. Final Validation Steps
                        - **Coverage Verification**: Ensure no important information is lost during deduplication
                        - **Logical Consistency**: Verify that remaining entries form a coherent and complete information set
                        - **Gap Analysis**: Check for any gaps created by removal of redundant entries
                        - **Relationship Mapping**: Ensure important relationships between entries are preserved

                        ## Output Requirements:
                        Return a cleaned JSON array that:
                        - **Eliminates only true redundancy** (>95% identical content) while preserving all unique information
                        - **Maintains granular detail** alongside comprehensive overviews
                        - **Preserves structured information** (lists, hierarchies, specific formatting) that aids understanding
                        - **Combines related content intelligently** through enriched entries that maintain all component details
                        - **Includes comprehensive reasoning** explaining deduplication decisions and what was preserved/combined
                        - **Ensures complete topic coverage** with both high-level and detailed perspectives
                        - **Maintains original context and source attribution** for all preserved information
                        - **Provides quick reference capability** through maintained granular entries
                        - **Supports multiple use cases** (quick reference, detailed analysis, implementation guidance)

                        ## Quality Assurance:
                        Before finalizing, verify that:
                        - **Zero information loss**: All unique facts, specifications, and details from the original are preserved
                        - **Enhanced accessibility**: Information is available at multiple levels of detail for different needs
                        - **Maintained context**: Original reasoning and source context is preserved for all content
                        - **Improved organization**: Related information is logically grouped while maintaining granular access
                        - **Complete coverage**: All original topics are covered comprehensively with appropriate detail levels
                        - **Preserved utility**: Both quick reference and detailed analysis capabilities are maintained
                        - **Structural integrity**: Important formatting, lists, and hierarchical information is preserved
                        - **Reasoning transparency**: Clear explanations of how content was combined, merged, or preserved

                        INPUT: {extracted_sections_json}

                **OUTPUT FORMAT (STRICT JSON ONLY)**:\n
                Return results in the following JSON structure — nothing else outside this structure:\n
                [
                  {{
                    "name": "Section Name",
                    "content": "Actual content",
                  }},
                  ...
                ]\n\n\n
                       """
                        }]
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        self.logger.error(f"Failed to process sections after {max_retries} attempts: {str(e)}")
                        return extracted_sections_json  # Return original if all retries fail
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    continue

            # Process the completion response
            try:
                processed_sections = extract_list_of_dicts(completion.choices[0].message.content)
                if not processed_sections:
                    self.logger.warning("No valid sections extracted from API response")
                    return extracted_sections_json
                return processed_sections
            except Exception as e:
                self.logger.error(f"Error extracting sections from API response: {str(e)}")
                return extracted_sections_json

        except Exception as e:
            self.logger.error(f"Unexpected error in sections analyzer: {str(e)}")
            return extracted_sections_json  # Return original sections as fallback

    def start_crew_process(self):
        """Run the complete evaluation process"""
        try:
            # Step 1: Extract sections from EA standard
            extractor_crew = self.create_extractor_crew()
            extracted_sections = extractor_crew.kickoff()

            extracted_sections_json = extract_clean_json(extracted_sections.raw)
            print(extracted_sections_json)
            print(f"Successfully parsed {len(extracted_sections_json)} sections")

            os.makedirs("tmp", exist_ok=True)
            with open("tmp/extracted_sections_json_before_posprocessing.json", "w") as f:
                json.dump(extracted_sections_json, f, indent=2, ensure_ascii=False)

            extracted_sections_json = self.final_sections_analyzer_and_updater(extracted_sections_json)
            print(extracted_sections_json)
            print(f"Successfully parsed {len(extracted_sections_json)} sections after posprocessing llm call.")

            os.makedirs("tmp", exist_ok=True)
            with open("tmp/extracted_sections_json.json", "w") as f:
                json.dump(extracted_sections_json, f, indent=2, ensure_ascii=False)

            # Create analyzer agents and tasks
            analyzer_agents, analyzer_tasks = self.create_analyzer_agents(extracted_sections_json)

            # With ThreadPoolExecutor for concurrent processing:
            results = {}
            max_workers = min(len(analyzer_agents), self.max_concurrent_calls)  # Limit concurrent workers to avoid resource issues
            
            for i, (agent, task) in enumerate(zip(analyzer_agents, analyzer_tasks)):
                try:
                    index, result = run_analyzer_in_thread(agent, task, self.llm, self.model_name, i, self.llm_kwargs)
                    results[index] = result
                    
                    if i % 5 == 0:
                        self.logger.info(f"Completed {i}/{len(analyzer_agents)} analyzer tasks")
                            
                except Exception as e:
                    index = i
                    self.logger.error(f"Error processing analyzer task {index}: {e}")
                    results[index] = f"Error: {str(e)}"

            # Ensure all results are present, even if tasks failed
            analyzer_results = []
            for i in range(len(analyzer_agents)):
                if i in results:
                    analyzer_results.append(results[i])
                else:
                    analyzer_results.append(f"Error: Task {i} failed to complete")
                    self.logger.error(f"Task {i} failed to complete")
            self.logger.info("✅ Analyzer agents completed their tasks")
            
            # Get all agent outputs after analyzers have completed
            agents_output = GetAgentsOutput()._run()
            
            # Create the improvement formatter agent to process the analyzer outputs
            formatted_improvements = self.create_improvement_formatter_agent(agents_output)
            
            print(formatted_improvements)
            saved_json_path, saved_arabic_json_path = "outputs/formatted_improvements.json", "outputs/arabic_translated_improvements.json"
            print("THE FORMATTED_IMPROVEMENTS IS SAVED AT THE: outputs/formatted_improvements.json")

            return formatted_improvements, saved_json_path, saved_arabic_json_path
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise

if __name__ == "__main__":
    evaluator = RFPEvaluationCrew(ea_standard_text="""
    Management Dashboard 12', '• System setup
• Methodology for opening/backup/security copies of statements
• Computer peripheral for EUC users
• Integration procedures for new users:

Distribution based on the following aspects:

1. Design and infrastructure of the solution – 15%
2. Patterns and loads – 15%
3. Capacity completeness – 10%
4. Scalability and performance – 10%
5. Report and data management – 10%
6. Solution design and user experience – 10%
7. Maintenance and setup – 10%
8. Training and knowledge transfer – 10%
9. Customizability and specialization capability – 5%
10. Plan for continuity and disaster recovery impact – 5%

--- Table 1 ---

• System setup
• Methodology for opening/backup/security copies of statements
• Peripheral computer usage for EUC users
• Integration procedures for new users:

Distribution based on the following aspects:

--- Table 2 ---

        1. Design and infrastructure of the solution – 15%
        2. Patterns and loads – 15%
        3. Capacity completeness – 10%
        4. Scalability and performance – 10%
        5. Report and data management – 10%
        6. Solution design and user experience – 10%
        7. Maintenance and setup – 10%
        8. Training and knowledge transfer – 10%
        9. Customizability and specialization capability – 5%
        10. Plan for continuity and disaster recovery impact – 5%
--------------TRANSLATED TEXT-----------------
Translated lines 301 to 350
Sure, here's the translated text with formatting and special characters preserved:
---
**The Primary Categories**
Database Platforms: Microsoft SQL Server 2022, Oracle, Postgres, MongoDB, CosmosDB
Compatible Operating Systems: Red Hat Enterprise Linux, Windows Server 2019
**Design Considerations**
The design should be aligned with Azure's minimal requirements and compliant with client-centric architecture. It should include zero-trust architecture with AES-256 encryption.
**Local and Cloud Hosting Management Requirements**
Technologies such as Kubernetes and Docker are required for managing containerized applications.
For content management systems, using open-source platforms like Drupal with blockchain verification would be preferred.
**Strategy and Recovery Considerations**
Disaster recovery (DR) strategy must align with Azure infrastructure compliance. The service provider should offer plans covering a minimum of 4 hours for Recovery Time Objective (RTO) and 30 minutes for Recovery Point Objective (RPO). Regular testing of disaster recovery plans is essential for effectively handling crises.
Redis is recommended for caching and temporary storage solutions considering all limitations and risks.
**Financial and Service Considerations**
A comprehensive cost analysis should include examining all demanded features to optimize resource allocation and cover the required specifications.
Recommended frameworks for mobile application development include React Native and Flutter.
--- Table 1 ---
---
**Primary Categories**
Database Platforms: Microsoft SQL Server 2022, Oracle, Postgres, MongoDB, CosmosDB
Operating Systems: Compatible with Linux (Red Hat Enterprise) and Windows Server 2019
**Design Alignment**
The design should fulfill Azure compliance and suit client-centric architecture needs, featuring zero-trust principles like AES-256 encryption.
**System Management**
Requirements for technologies, including Kubernetes and Docker, to organize applications are compulsory for system management.
For content management systems, it is suggested to use open-source like Drupal with the added security of blockchain verification.
**Strategy Planning**
It's crucial that disaster recovery (DR) strategies meet Azure compatibility, highlighting the necessity for a comprehensive approach in line with business objectives.
**Testing and Verification**
DR plans should ensure over 99.9% operational uptime (SLA) with exact time schedules for RTO and RPO criteria, tested regularly for effectiveness during potential disruptions.
**Supporting Technologies**
Technologies preferred for development include .Net, Java, Python, Node.js, and Go.
Additionally, multi-factor authentication, SSL certificates with 2048-bit encryption, SAN, and SAN demands must be met.
**Documentation and System Reviews**
From user acceptance testing plans to role-based outcome reviews, all must align with defined project deliverables to maintain consistent testing and final result acknowledgment.
Translation:
---
A maximum of 45-minute interruption, 99.9% operating standard, SLA Compliance level
Input
Service Level Agreement (SLA)
--- Table 1 ---
ArchiMate                       Language
Model
Comparison
Requirement and
Provision (99.9% SLA) for scalability and availability                  Requirements
Non-Functional
Technical
Preferred
Development
Framework
Documentation, Initial Model, User Acceptance Test Plan (UAT), Execution Test Plan, Delivery Plan, Main Solution
Results
Design of the solution with high performance at a low cost level, BRD
Project
Requirements
SSL Certificate
A maximum of 45-minute interruption, 99.9% operating standard                   Criteria
Compliance
Level
Service Level
(SLA)
● Requirements
- The solution must be layered with three layers.
- API gateways should be used for both external party management and internal communication management.
- A front-facing server is required in the DMZ for external users; another front-facing server is required for internal user communications.
- Communication should take place through a database connection broker in designing the third layer.
- Firewalls should be used to segregate communication channels with security enhancements through an IPS system to detect and prevent intrusions.
- The solution should be scalable and resilient for growth.
Micro Services / Serverless Architecture consideration is required.
Authentication through SSO (Single Sign-On), multi-factor authentication should be available for users.
High-level design documentation required.
The provider should present the HLD (High-Level Design) document based on the standard framework implemented.
1. Presentation:
- A document summary must be presented for quick referencing.
- Significant goals and purposes of the new solution should be comprehensively overviewed.
2. System Architecture:
- A description must be presented on how the system architecture functions on a broad spectrum.
- Clarity is required on the interaction between components and units.
- Clarity around microservices, server-client model, or any other model as necessary to envision the user architecture.
3. Additional:
- Possibility to add more required information or create new sections within the final document.
--- Table 2 ---
High-level design documentation required.
The provider should present the HLD (High-Level Design) document based on the implemented architectural standards.
---
This translation maintains the original formatting, organizational structure, and technical terminology from the source text, while converting it into English.
Sure, here's the translated text while maintaining the original formatting and special characters:
1. Introduction
- Provide a comprehensive view of the new solution and its main purpose, striving for documentation and reference.
- Present a general overview on how to verify it.
2. System Architecture
- Write documentation based on the general and pivotal structure of the system.
- Document how the components and events communicate with each other.
- Clarify the architecture model of the user, whether it's a client-server model, microservices, or any other paradigm.
- Include diagrams illustrating the architecture of the system with regards to databases, data storage, security considerations, communications, and connection protocols.
3. Components and Events
- Present a list of the major components and events within the system.
- Provide a detailed description of their responsibilities and functions for each component.
4. Data Flow
- Use diagrams or visual representations to illustrate how data moves systematically within the system.
- Detail how processing and storing of data are taken into account for end-user analytics.
5. Data Storage and Database Design
- Describe database mechanisms within the system.
- Provide a list of all entities involved alongside database queries or schemas used (if available).
- Define chosen database systems or structures for user data if applicable.
6. Security Considerations
- Provide details on the strategies and practices undertaken to secure user information and system integrity.
- Discuss authentication, authorization, and other security mechanisms available.
7. Communication Protocols
- Provide comprehensive information about the communication protocols and user interfaces within the system.
- Identify use cases for APIs, messaging lists, or other methods for exchanging data.
8. Integration Points
- Identify any external services or system integrations, specifying how they work with internal components.
- Ensure a seamless transfer of data and compatibility with external systems.
9. Scalability and Expansion
- Address system operational requirements concerning scalability and expansion capabilities.
- Provide insights into load balancing, temporary storage, and performance enhancements.
10. Error Handling and Logging
- Ensure efficient management of system exceptions and error handling.
- Offer descriptions of logging processes and system monitoring for error tracking and resolution.
11. Infrastructure and Deployment
- Provide strategies and plans for deployment, featuring environments like development, testing, and production.
- Define the infrastructure requirements essential for system operation.
12. Solution Management
--- Table 1 ---
- Provide a comprehensive illustration of the system's architecture covering databases, data storage, security considerations, communications, and connection protocols.
- Include a ranking of components and points of integration such as zoning and protocols.
- List important system components and events.
- Use diagrams or visual methodologies to accurately depict data flow within the system for end-user processes.
Management Dashboard 12', '• System setup
• Methodology for opening/backup/security copies of statements
• Computer peripheral for EUC users
• Integration procedures for new users:
Distribution based on the following aspects:
1. Design and infrastructure of the solution – 15%
2. Patterns and loads – 15%
3. Capacity completeness – 10%
4. Scalability and performance – 10%
5. Report and data management – 10%
6. Solution design and user experience – 10%
7. Maintenance and setup – 10%
8. Training and knowledge transfer – 10%
9. Customizability and specialization capability – 5%
10. Plan for continuity and disaster recovery impact – 5%
--- Table 1 ---
• System setup
• Methodology for opening/backup/security copies of statements
• Peripheral computer usage for EUC users
• Integration procedures for new users:
Distribution based on the following aspects:
--- Table 2 ---
1. Design and infrastructure of the solution – 15%
2. Patterns and loads – 15%
3. Capacity completeness – 10%
4. Scalability and performance – 10%
5. Report and data management – 10%
6. Solution design and user experience – 10%
7. Maintenance and setup – 10%
8. Training and knowledge transfer – 10%
9. Customizability and specialization capability – 5%
10. Plan for continuity and disaster recovery impact – 5%
    """, additional_sections="""Preferred Application Frameworks,Accepted Database,Accepted Operating System,Requirement of SSL Certificate,Non-Functional,Project Deliverables""")
    evaluator.start_crew_process()
