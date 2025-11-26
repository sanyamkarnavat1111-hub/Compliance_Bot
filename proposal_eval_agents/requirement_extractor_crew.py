from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from posprocessing import extract_clean_json
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
from logger_config import get_logger
import json
from Unscored_Requirement.rfp_requirement_extractor import RFPRequirementExtractorAgent

class RFPRequirementExtractor:
    def __init__(self, logger_instance=None):
        # Configure logging
        self.logger = logger_instance if logger_instance is not None else get_logger(__name__)
        
        # Load environment
        load_dotenv()
        self.model_name = "openrouter/qwen/qwen3-32b"

        # Initialize LLM
        self.llm = LLM(
            model=self.model_name,
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3
        )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY'),
        )

    def create_requirement_extractor_crew(self, rfp_chunk_sample):
        """Create and run the RFP requirement extractor crew with ReAct framework optimized for proposal verification"""
        extractor_agent = Agent(
            role="RFP Requirement Extractor for Proposal Verification",
            goal="Extract all verifiable requirements from RFP document chunks that can be cross-checked against proposals.",
            backstory="An expert at analyzing RFP documents and extracting requirements that can be systematically verified against proposal submissions for tender evaluation.",
            llm=self.llm,
            verbose=True
        )

        # ReAct structured context with verification-focused content
        context = [
            {
                "description": "ReAct Framework Analysis for RFP requirement extraction optimized for proposal verification",
                "expected_output": "JSON object with verifiable requirements categorized for cross-checking against proposals",
                "reasoning_framework": {
                    "thought_process": "Analyze RFP chunks and identify requirements that can be verified in proposals",
                    "action_plan": "Categorize requirements by verification method and importance",
                    "observation": "Validate requirement extraction completeness for verification purposes",
                    "reflection": "Ensure all verifiable requirement types are captured for proposal cross-checking"
                },
                "extraction_instructions": [
                    "Identify scored requirements with weights and criteria that can be verified in proposals",
                    "Extract unscored requirements including mandatory ones that can be cross-checked",
                    "Focus on requirements that have clear verification criteria or evidence",
                    "Maintain original context and relationships for accurate verification"
                ]
            }
        ]

        extractor_task = Task(
            description=(
                f"Follow the ReAct framework to extract verifiable requirements from the provided RFP document chunks for proposal cross-verification.\n\n"
                + "\n\n".join([f"**CHUNK {i+1}:**\n{chunk}" for i, chunk in enumerate(rfp_chunk_sample)]) 
                + f"\n\n"
                "**REQUIREMENT EXTRACTION GUIDELINES FOR PROPOSAL VERIFICATION**\n"
                "Extract and categorize requirements from ALL chunks in this batch that can be verified against proposals:\n\n"
                "1. **SCORED REQUIREMENTS** (with scoring weights or evaluation criteria):\n"
                "   - Extract ONLY from the official evaluation criteria table, identified by titles such as 'Technical Evaluation Criteria Table', 'Technical and Financial Evaluation Criteria Table', or 'evaluation criteria table'.\n"
                "   - The table MUST be located in the final portion of the RFP document. It might be between sections sometimes.\n"
                "   - DO NOT modify, reword, merge, split, or interpret any requirements or scores—capture each item EXACTLY as written in the table, including punctuation and spacing.\n"
                "   - Include their relative importance (weights/percentages or scores) EXACTLY as specified, copying the numerical values and units (e.g., '%', 'points') verbatim from the weight column.\n"
                "   - CRITICAL: For weight_percentage field, use the EXACT numerical weight value from the table (e.g., '20', '15', '10'), NOT sequential numbers or row positions.\n"
                "   - IMPORTANT: If a row contains 'M', 'Mandatory', 'Pass/Fail', or any non-numeric weight value, it should be ignore.\n"
                "   - ONLY include rows with actual numeric weight values (like 20%, 15%, 30) in scored_requirements.\n"
                "   - STRICTLY EXCLUDE all financial, pricing, cost, budget, or commercial evaluation items—even if they appear in the same table. Reject any row containing terms like 'price', 'cost', 'budget', 'commercial offer', or 'financial components'.\n"
                "   - ONLY extract technical evaluation criteria—such as experience, methodology, solution features, quality standards, delivery plans, etc.—based solely on the table's content.\n"
                "   - Focus EXCLUSIVELY on requirements explicitly listed in the evaluation table—ignore ALL narrative descriptions, footnotes, or content outside the table.\n"
                "   - These are the ONLY technical items that contribute to the final proposal score—extract NO other scored or financial criteria.\n"
                "   - MANDATORY: PROCESS the table row-by-row from TOP to BOTTOM, ensuring EVERY SINGLE ROW meeting the technical criteria is included, INCLUDING THE VERY LAST ROW, until the table completely ends.\n"
                "   - DOUBLE-CHECK: After processing, verify you have captured ALL technical rows from the table, paying special attention to the final row which is often missed.\n"
                "2. **UNSCORED REQUIREMENTS** (all other verifiable requirements):\n"
                "   - Mandatory requirements (must be met - deal-breakers, pass/fail criteria)\n"
                "   - Technical specifications that can be verified\n"
                "   - Service levels with measurable criteria\n"
                "   - Deliverables with clear specifications\n"
                "   - Timelines and milestones\n"
                "   - Compliance needs with verification methods\n"
                "   - Certifications and qualifications\n"
                "   - Experience requirements with verification criteria\n"
                "   - Team qualifications and credentials\n"
                "   - Any other requirements that can be cross-checked\n\n"
                "**VERIFICATION-FOCUSED GUIDELINES**\n"
                "- Extract requirements that have clear verification criteria or evidence\n"
                "- Focus on requirements that can be objectively assessed in proposals\n"
                "- Include specific metrics, standards, or criteria for verification\n"
                "- Preserve any specific evaluation or verification methods mentioned\n"
                "- Note dependencies between requirements where they exist\n"
                "- Include any deadline or timeline-related requirements\n"
                "- Highlight any critical or deal-breaker requirements\n"
                "- Focus only on requirements present in these chunks\n"
                "- Maintain context and relationships between requirements\n"
                "- Consolidate similar requirements across chunks to avoid duplication\n\n"
                "**THOUGHT**: First, analyze all RFP chunks in the batch and identify verifiable requirements.\n"
                "- Look for section headers, bullet points, numbered lists, and table content across all chunks\n"
                "- Identify evaluation criteria, scoring mechanisms, and mandatory language\n"
                "- Consider how requirements relate to each other within and across chunks\n"
                "- Note any references to other sections or chunks\n"
                "- Identify requirements that span multiple chunks in this batch\n"
                "- Focus on requirements that have clear verification criteria\n"
                "- For evaluation criteria tables, carefully examine each row to ensure complete extraction\n\n"
                "**ACTION**: Extract requirements systematically by:\n"
                "1. Scanning all chunks for scored requirements:\n"
                "   - Look for percentages, weights, scoring criteria\n"
                "   - Identify evaluation matrices or rubrics\n"
                "   - Note how requirements contribute to final score\n"
                "   - Include both weighted requirements and evaluation criteria\n"
                "   - Focus on verifiable aspects\n"
                "   - Process each technical requirement row, paying attention to:\n"
                "     * The exact requirement text from the first column\n"
                "     * The exact weight/percentage value from the weight column (not row numbers)\n"
                "     * Ensuring no technical rows are skipped, especially the last row\n"
                "2. Identifying unscored requirements across all chunks:\n"
                "   - Look for 'must', 'shall', 'required', 'mandatory' language\n"
                "   - Identify pass/fail criteria and deal-breakers\n"
                "   - Note minimum qualifications and prerequisites\n"
                "   - Technical specifications and standards\n"
                "   - Service level agreements and performance metrics\n"
                "   - Deliverables and timelines\n"
                "   - Compliance and certification requirements\n"
                "3. Documenting relationships and dependencies across chunks\n"
                "4. Consolidating similar requirements to avoid duplication\n"
                "5. Noting continuation indicators for requirements that span beyond this batch\n\n"
                "**OBSERVATION**: Review the extracted requirements to ensure:\n"
                "- Both categories are properly populated from all chunks\n"
                "- Requirements maintain their original context and relationships\n"
                "- No pricing or financial information is included\n"
                "- Dependencies and continuation indicators are noted\n"
                "- Critical requirements are highlighted\n"
                "- The categorization is accurate and complete\n"
                "- Similar requirements across chunks are consolidated\n"
                "- All requirements are verifiable against proposals\n"
                "- For scored requirements, confirm:\n"
                "  * All technical rows from the table are included\n"
                "  * Weight values match the actual table values\n"
                "  * The last row of the table is not missing\n"
                "  * No financial/pricing criteria are included\n\n"
                "**REFLECTION**: Validate the extraction by checking:\n"
                "- Have I captured all scored requirements with their weights and criteria from all chunks?\n"
                "- Have I identified all unscored requirements including mandatory ones from all chunks?\n"
                "- Have I extracted ALL technical rows from the evaluation criteria table?\n"
                "- Are the weight_percentage values the actual weights from the table (not sequential numbers)?\n"
                "- Did I specifically check and include the last row of the table?\n"
                "- Have I properly ignore, If a row contains 'M', 'Mandatory', 'Pass/Fail', or any non-numeric weight value.\n"
                "- Do ALL scored requirements have numeric weight values only?\n"
                "- Are the requirements properly categorized?\n"
                "- Have I maintained context and relationships across chunks?\n"
                "- Are there any requirements that span beyond this batch?\n"
                "- Have I consolidated similar requirements to avoid duplication?\n"
                "- Can all extracted requirements be verified against proposals?\n\n"
                "Return results in JSON format with the following structure:\n"
                "{\n"
                "  \"scored_requirements\": [\n"
                "    {\n"
                "      \"requirement\": \"EXACT text from requirement column\",\n"
                "      \"weight_percentage\": \"EXACT weight value from table (e.g., '20', '15', '10')\",\n"
                "      \"context\": \"Detailed description of the requirement\",\n"
                "    }\n"
                "  ],\n"
                "  \"unscored_requirements\": [\n"
                "    {\n"
                "      \"requirement\": \"Abstract of requirement\",\n"
                "      \"context\": \"Detailed description of the requirement\",\n"
                "    }\n"
                "  ],\n"
                "}\n\n"
                "CRITICAL RULES FOR OUTPUT:\n"
                "1. ACCURATE CATEGORIZATION: Properly categorize each requirement\n"
                "2. PRESERVE CONTEXT: Maintain original context and relationships\n"
                "3. EXCLUDE PRICING: Do not include any pricing or financial inf    ormation\n"
                "4. COMPLETE EXTRACTION: Capture all verifiable requirements present in all chunks, ESPECIALLY the last row of evaluation tables\n"
                "5. CONSOLIDATION: Consolidate similar requirements across chunks\n"
                "6. VERIFICATION FOCUS: Ensure all requirements can be verified against proposals\n"
                "7. VALID JSON: Ensure the output is valid JSON format\n"
                "8. EXACT WEIGHTS: Use actual weight values from the table, not sequential numbers\n"
                "9. FINAL ROW CHECK: Always verify the last row of the evaluation table is included"
            ),
            expected_output=(
                "A valid JSON object containing all verifiable requirements categorized into scored and unscored requirements, "
                "with verification criteria and evidence requirements for proposal cross-checking."
            ),
            context=context,
            agent=extractor_agent
        )

        extractor_crew = Crew(
            agents=[extractor_agent],
            tasks=[extractor_task],
            process=Process.sequential,
            manager_llm=self.llm,
            model=self.model_name,
            verbose=True
        )

        return extractor_crew

    def start_crew_process(self, rfp_chunks, k=5, output_path="outputs_proposal_eval/extracted_requirements.json",rfp_path=None):
        """Run the complete RFP requirement extraction process for proposal verification"""
        self.logger.info("Starting RFP requirement extraction process.")
        final_extracted_requirements = {'scored_requirements': [], 'unscored_requirements': []}
        # Step 1: Extract requirements from RFP chunk batch
        for i in range(0, len(rfp_chunks), k):
            self.logger.info(f"Processing RFP chunk batch {i//k + 1}/{len(rfp_chunks)//k + (1 if len(rfp_chunks)%k else 0)}")
            extractor_crew = self.create_requirement_extractor_crew(rfp_chunks[i:i+k])
            
            # Implement retry logic for crew kickoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Attempt {attempt + 1}/{max_retries} to kickoff RFP requirement extraction crew.")
                    extracted_requirements = extractor_crew.kickoff()
                    extracted_requirements_json = extract_clean_json(extracted_requirements.raw)
                    self.logger.info(f"Successfully extracted requirements in attempt {attempt + 1}.")
                    final_extracted_requirements["scored_requirements"].extend(extracted_requirements_json["scored_requirements"])
                    break # Break on success
                except Exception as e:
                    self.logger.error(f"Attempt {attempt + 1}/{max_retries} failed for RFP requirement extraction: {e}")
                    if attempt == max_retries - 1:
                        self.logger.error("All retries failed for RFP requirement extraction. Skipping this batch.")
                        # If all retries fail, append an empty list or a structured error to maintain pipeline flow
                        final_extracted_requirements["scored_requirements"].extend([]) # Or a structured error dict
            
        try:
            # Initialize the RFP requirement extractor agent
            requirement_extractor = RFPRequirementExtractorAgent(
                input_file_path=rfp_path,
                output_path=output_path,
                logger_instance=self.logger # Pass logger here
            )
            
            # Execute the extraction process
            self.logger.info("Running RFPRequirementExtractorAgent...")
            requirement_results = requirement_extractor.run_extraction()
            final_extracted_requirements["unscored_requirements"].append(requirement_results)
            self.logger.info("RFPRequirementExtractorAgent completed.")
            
        except Exception as e:
            error_message = f"An unexpected error occurred during RFP extraction: {e}"
            print(error_message)
            return False, error_message

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_extracted_requirements, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Final extracted requirements saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving final extracted requirements to {output_path}: {e}")

        self.logger.info("RFP requirement extraction process completed.")
        return final_extracted_requirements, output_path

if __name__ == "__main__":
    # Example usage for RFP requirement extraction with multiple chunks
    rfp_chunk_1 = """
    Evaluation Criteria and Scoring:
    
    1. Technical Approach (30%)
       - Solution architecture and design
       - Technology stack and implementation methodology
       - Scalability and performance considerations
    
    2. Experience and Qualifications (25%)
       - Relevant project experience (minimum 3 years)
       - Team qualifications and certifications
       - Past performance references
    """
    
    rfp_chunk_2 = """
    3. Cost Effectiveness (20%)
       - Total cost of ownership
       - Value for money proposition
    
    Mandatory Requirements:
    - Vendor must be ISO 27001 certified
    - Minimum annual revenue of $10M
    - Must provide 24/7 support coverage
    """
    
    rfp_chunk_3 = """
    Technical Specifications:
    - Cloud-native architecture required
    - Must support multi-tenancy
    - API-first design approach
    - Real-time data processing capabilities
    
    Service Level Requirements:
    - 99.9% uptime SLA
    - Maximum 4-hour response time for critical issues
    """
    
    rfp_chunk_4 = """
    Deliverables:
    - High-level design document
    - Implementation plan
    - User acceptance test plan
    - Training materials
    
    Timeline Requirements:
    - 30-day implementation timeline
    - Weekly progress reports
    """
    
    rfp_chunk_5 = """
    Compliance Requirements:
    - GDPR compliance
    - SOC 2 Type II certification
    - HIPAA compliance for healthcare data
    
    Additional Requirements:
    - Disaster recovery plan
    - Data backup procedures
    - Security audit reports
    """
    
    # Process a batch of 5 chunks
    batch_chunks = [rfp_chunk_1, rfp_chunk_2, rfp_chunk_3, rfp_chunk_4, rfp_chunk_5]
    
    extractor = RFPRequirementExtractor()
    results = extractor.start_crew_process(rfp_chunks=batch_chunks, k=5, output_path="outputs_proposal_eval/extracted_requirements.json")
    print(results)