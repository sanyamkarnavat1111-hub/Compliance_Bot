from crewai import Agent, Task, Crew
from crewai.llm import LLM
import os
import logging
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
from Unscored_Requirement.chunk_processing import ChunkProcessor
from logger_config import get_logger

class RFPRequirementExtractorAgent:
    """
    A CrewAI agent designed to extract all requirements from a Request for Proposal (RFP) document.
    Now enhanced with semantic chunking capabilities for better processing of large documents.
    """
    def __init__(self, language="eng", output_path="outputs_RFP_Requirements/formatted_requirements.json", 
                 chunk_size=30000, input_file_path="RFP_Input/687e6c22e19aaQKsCb1753115682.txt", logger_instance=None):
        """
        Initializes the agent with a language model and configuration.

        Args:
            language (str): The language for processing (default: "eng")
            output_path (str): Path for output files (default: "outputs_RFP_Requirements/formatted_requirements.json")
            chunk_size (int): Maximum tokens per chunk (default: 1024)
            input_file_path (str): Path to the input RFP text file
        """
        # Configure logging
        self.logger = logger_instance if logger_instance is not None else get_logger(__name__)
        
        # Load environment
        load_dotenv()
        
        self.language = language
        self.model_name = "openrouter/qwen/qwen3-32b"
        self.chunk_size = chunk_size
        self.input_file_path = input_file_path
        
        # Initialize chunk processor
        try:
            self.chunk_processor = ChunkProcessor(max_chunk_size=chunk_size)
            self.logger.info("ChunkProcessor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChunkProcessor: {str(e)}")
            self.chunk_processor = None
        
        # Initialize LLM
        self.llm = LLM(
            model=self.model_name,
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3
        )
        self.output_path = output_path
        self.chunks_output_dir = os.path.join(os.path.dirname(output_path), "chunks")

    def load_rfp_content(self) -> str:
        """
        Load RFP content from the input file.
        
        Returns:
            str: The content of the RFP document
        """
        try:
            with open(self.input_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.logger.info(f"Successfully loaded RFP content from {self.input_file_path}")
            return content
        except Exception as e:
            self.logger.error(f"Failed to load RFP content from {self.input_file_path}: {str(e)}")
            return ""

    def create_agent(self) -> Agent:
        """
        Creates the CrewAI agent with a specific role, goal, and backstory for RFP analysis.

        Returns:
            Agent: An instance of a CrewAI Agent.
        """
        return Agent(
            role="Expert RFP Analyst and Unscored Requirement Extractor",
            goal=(
                "To meticulously analyze a Request for Proposal (RFP) document chunk and extract all "
                "unscored requirements, while STRICTLY IGNORING any requirements related to scoring, evaluation criteria, "
                "financial requirements, financial offers, pricing, costs, budgets, or language specifications. The agent must identify every unscored statement that outlines a "
                "specific need, capability, constraint, or obligation that the proposal must address, but must "
                "exclude any requirements that pertain to how proposals are scored, evaluated, any financial "
                "criteria or obligations, financial terms, or language/translation requirements."
            ),
            backstory=(
                "Your expert to fetch all unscored requirements from a Request for Proposal (RFP) document. "
                "As a seasoned procurement specialist, I have honed my skills in deconstructing complex "
                "RFP documents. My core function is to identify and catalogue every unscored requirement, "
                "ensuring that the responding team has a complete and accurate checklist for full compliance "
                "with all non-scored obligations. I am trained to ignore any requirements related to scoring, "
                "evaluation points, financial requirements, financial offers, pricing, costs, budgets, or language specifications, focusing only on the unscored, "
                "substantive requirements that must be met."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        

    def create_extraction_task(self, rfp_chunk: str, chunk_number: int, total_chunks: int, agent: Agent) -> Task:
        """
        Creates a CrewAI task for the agent to perform the requirement extraction on a specific chunk.

        Args:
            rfp_chunk (str): A chunk of the RFP document content
            chunk_number (int): The current chunk number (1-indexed)
            total_chunks (int): Total number of chunks
            agent (Agent): The agent to assign this task to

        Returns:
            Task: An instance of a CrewAI Task with detailed instructions.
        """
        return Task(
            description=f"""
            # TASK: Extract All UNIQUE Unscored Requirements from RFP Content Chunk {chunk_number}/{total_chunks}
            
            **PRIORITY DIRECTIVE: ZERO TOLERANCE FOR REDUNDANCY - Each requirement must be genuinely unique and distinct**
            **CRITICAL: FINANCIAL REQUIREMENTS ARE STRICTLY FORBIDDEN - Never extract any requirement related to pricing, costs, financial offers, budgets, or financial terms**
            **CRITICAL: LANGUAGE REQUIREMENTS ARE STRICTLY FORBIDDEN - Never extract any requirement that specifies language of submission, translation, or language preferences**

            ## RFP CHUNK TO ANALYZE
            ```
            {rfp_chunk}
            ```

            ## WHAT IS AN UNSCORED REQUIREMENT?
            An Unscored Requirement is a mandatory condition, specification, or obligation that a vendor must meet for compliance. It is not assigned a score or weight in the evaluation process. Think of these as the "pass/fail" or "go/no-go" gates of the Request for Proposal (RFP). These requirements serve as mandatory hurdles that bidders must clear to be considered eligible for the contract.

            ## EXPERT INSTRUCTIONS
            As an expert RFP analyst with vast knowledge of procurement documents, your task is to extract ALL unscored requirements from the provided RFP chunk. Focus on requirements that must be met by bidders, but DO NOT include any requirements related to scoring, evaluation criteria, or financial requirements.

            ### REQUIREMENTS TO EXTRACT:
            - **Technical Requirements:** System capabilities, performance standards, technical specifications
            - **Functional Requirements:** Features, functionalities, operational needs
            - **Legal/Compliance Requirements:** Regulatory compliance, certifications, legal obligations
            - **Administrative Requirements:** Documentation, submission formats, deadlines, procedures
            - **Qualification Requirements:** Experience, certifications, personnel qualifications
            - **Service Level Requirements:** Availability, response times, support levels
            - **Security Requirements:** Data protection, access controls, security standards
            - **Implementation Requirements:** Project phases, deliverables, timelines

            ### REQUIREMENTS TO IGNORE:
            - **Scoring/Evaluation Criteria:** Any mention of points, weights, scoring percentages, evaluation matrices
            - **Financial Requirements:** ANY requirement related to financial matters, including but not limited to:
                * Pricing, cost breakdowns, financial evaluations, budget constraints
                * Financial offers, financial proposals, financial terms
                * Cost estimates, pricing structures, financial bids
                * Budget requirements, financial obligations, payment terms
                * Financial qualifications, financial capacity, financial standing
                * Cost analysis, financial analysis, financial statements
                * Financial guarantees, financial bonds, financial security
                * Financial performance, financial history, financial records
                * Financial compliance, financial regulations, financial standards
                * ANY requirement that mentions money, costs, pricing, budgets, financial terms, or financial obligations
                * ANY requirement that specifies financial conditions, financial criteria, or financial requirements
            - **Evaluation Process:** How proposals will be assessed or ranked
            - **Selection Criteria:** Factors used to choose winning proposal
            - **Submission Deadline Requirements:** ANY requirement that specifies deadlines, submission dates, or time constraints for offers, proposals, or documents. This includes but is not limited to:
                * "Deadline for submission of offers is [date/time]"
                * "Proposals must be submitted by [date/time]"
                * "The closing date for submissions is [date]"
                * "Bidders must submit their offers no later than [date/time]"
                * "Submission deadline: [date/time]"
                * "Offers received after [date/time] will not be considered"
                * "The final date for submission is [date]"
                * "Proposals must be received by [date/time]"
                * "Submission period ends on [date]"
                * "Deadline for receipt of offers: [date/time]"
                * ANY requirement that mentions submission deadlines, closing dates, or time constraints for document submission
                * ANY requirement that specifies when offers, proposals, or documents must be submitted
            - **Language Requirements:** ANY requirement that specifies language of submission, translation requirements, or language preferences. This includes but is not limited to:
              * "All offers must be presented in [language]"
              * "Proposals must be submitted in [language] and accompanied by [additional language]"
              * "The official language of the tender is [language]"
              * "Documents must be provided in [language] with [translation requirements]"
              * "Bidders must submit in [language] format"
              * "The bidder must ensure all offers are submitted in [language] with an additional [language] copy"
              * "All documents must be in [language] with [language] translation"
              * "Proposals must be in [language] with accompanying [language] version"
              * "Bidders must provide submissions in [language] and [language]"
              * "All offers are submitted in [language] with [language] copy"
              * ANY requirement that mentions language, translation, linguistic specifications, or language preferences
              * ANY requirement that specifies the language in which documents, offers, proposals, or submissions must be provided

            ### CRITICAL REDUNDANCY PREVENTION:
            1. **Strict No-Duplication Policy:** NEVER extract the same requirement multiple times, even if phrased differently or mentioned in various sections
            2. **Semantic Similarity Check:** Before extracting any requirement, verify it's not semantically equivalent to a previously extracted one
            3. **Consolidate Similar Requirements:** If multiple sentences express the same core obligation, extract ONLY the most complete and comprehensive version
            4. **Focus on Unique Obligations:** Each requirement must represent a genuinely distinct obligation, constraint, or specification
            5. **Context-Aware Extraction:** Consider the full context to avoid extracting fragments of larger requirements or restating the same requirement in different words
            6. **Cross-Reference Prevention:** Do not extract requirements that are merely cross-references or restatements of other requirements
            7. **Substance Over Repetition:** Prioritize substantive, actionable requirements over procedural repetitions

            ### EXTRACTION GUIDELINES:
            1. **Identify Keywords:** Look for "must," "shall," "will," "is required to," "should," "needs to," "has to"
            2. **Complete Requirements:** Extract full, meaningful requirements, not partial statements
            3. **Maintain Context:** Include sufficient source paragraph context for clarity
            4. **Cross-Chunk Awareness:** This is chunk {chunk_number} of {total_chunks} - extract partial requirements if they span boundaries
            5. **Precise Categorization:** Accurately categorize each requirement type
            6. **FINANCIAL EXCLUSION:** IMMEDIATELY REJECT any requirement that mentions pricing, costs, financial offers, budgets, financial terms, money, financial obligations, or any financial-related matters
            7. **LANGUAGE EXCLUSION:** IMMEDIATELY REJECT any requirement that mentions language, translation, or linguistic specifications
            8. **DEADLINE EXCLUSION:** IMMEDIATELY REJECT any requirement that mentions submission deadlines, closing dates, or time constraints for document submission

            ## EXPECTED OUTPUT FORMAT
            Return a single JSON object containing a list of all identified unscored requirements:

            {{
                "chunk_number": {chunk_number},
                "requirements": [
                    {{
                        "requirement_id": "REQ-{chunk_number:03d}-001",
                        "requirement_text": "The exact, complete text of the unscored requirement extracted from the document.",
                        "category": "Technical/Functional/Legal/Administrative/Qualification/Security/Implementation",
                        "source_paragraph": "The full paragraph from the RFP chunk where the requirement was found, providing sufficient context.",
                        "confidence": "High/Medium/Low"
                    }},
                    {{
                        "requirement_id": "REQ-{chunk_number:03d}-002",
                        "requirement_text": "The system must support at least 1000 concurrent users without performance degradation.",
                        "category": "Technical",
                        "source_paragraph": "Performance requirements specify that the system must support at least 1000 concurrent users without performance degradation and maintain 99.9% uptime.",
                        "confidence": "High"
                    }}
                ]
            }}

            **CRITICAL REQUIREMENTS FOR OUTPUT:**
            1. Your final output must be ONLY the JSON object, with no additional text, explanations, or commentary before or after it
            2. ABSOLUTELY NO REDUNDANT REQUIREMENTS: Each requirement must be unique and distinct
            3. QUALITY OVER QUANTITY: It's better to extract fewer, unique requirements than many redundant ones
            4. FINAL REDUNDANCY CHECK: Before finalizing your JSON output, review each requirement to ensure no duplicates or near-duplicates exist
            5. DISTINCT VALUE REQUIREMENT: Every extracted requirement must add distinct value and represent a separate obligation
            6. FINAL FINANCIAL CHECK: Before submitting, verify that NO requirement mentions pricing, costs, financial offers, budgets, financial terms, money, financial obligations, or any financial-related matters
            7. FINAL LANGUAGE CHECK: Before submitting, verify that NO requirement mentions language, translation, or linguistic specifications
            8. FINAL DEADLINE CHECK: Before submitting, verify that NO requirement mentions submission deadlines, closing dates, or time constraints for document submission
            """,
            agent=agent,
            expected_output=(
                "A JSON object containing 'chunk_number' and 'requirements' keys. The 'requirements' "
                "key holds a list of UNIQUE, NON-REDUNDANT unscored requirement objects. Each requirement must be "
                "semantically distinct and represent a separate obligation. Each object must contain "
                "'requirement_id', 'requirement_text', 'category', 'source_paragraph', and 'confidence'. "
                "CRITICAL: No duplicate or similar requirements allowed. EXCLUDE ALL scoring, evaluation, financial, "
                "language, and submission deadline requirements. Financial, language, and deadline requirements are STRICTLY FORBIDDEN "
                "and must never be extracted."
            )
        )

    def extract_json_from_result(self, result_text: str) -> dict:
        """
        Extract JSON from the crew result, handling various formats.
        
        Args:
            result_text (str): The raw result from the crew
            
        Returns:
            dict: Parsed JSON result
        """
        try:
            # If it's already a dict, return it
            if isinstance(result_text, dict):
                return result_text
            
            # Convert to string if needed
            result_str = str(result_text)
            
            # Try to find JSON in the string using regex
            json_pattern = r'\{.*\}'
            json_matches = re.findall(json_pattern, result_str, re.DOTALL)
            
            if json_matches:
                # Try each match until we find valid JSON
                for match in json_matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, try parsing the entire string
            return json.loads(result_str)
            
        except Exception as e:
            self.logger.error(f"Failed to extract JSON from result: {str(e)}")
            self.logger.error(f"Result text: {result_text}")
            return {"chunk_number": 0, "requirements": []}

    def save_chunk_result(self, chunk_number: int, chunk_result: dict):
        """
        Save individual chunk result to a separate file.
        
        Args:
            chunk_number (int): The chunk number
            chunk_result (dict): The chunk processing result
        """
        try:
            # Create chunks directory if it doesn't exist
            os.makedirs(self.chunks_output_dir, exist_ok=True)
            
            # Save chunk result
            chunk_file = os.path.join(self.chunks_output_dir, f"chunk_{chunk_number:03d}.json")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Chunk {chunk_number} result saved to {chunk_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save chunk {chunk_number} result: {str(e)}")

    def deduplicate_requirements(self, requirements: List[Dict]) -> List[Dict]:
        """
        Remove duplicate requirements based on text similarity.
        
        Args:
            requirements (List[Dict]): List of requirement dictionaries
            
        Returns:
            List[Dict]: Deduplicated requirements
        """
        if not requirements:
            return requirements
        
        deduplicated = []
        seen_texts = set()
        
        for req in requirements:
            req_text = req.get('requirement_text', '').strip().lower()
            
            # Simple deduplication based on exact text match
            if req_text and req_text not in seen_texts:
                seen_texts.add(req_text)
                deduplicated.append(req)
            elif req_text:
                self.logger.info(f"Duplicate requirement found and removed: {req.get('requirement_id', 'Unknown')}")
        
        self.logger.info(f"Deduplication: {len(requirements)} -> {len(deduplicated)} requirements")
        return deduplicated

    def process_chunks_and_extract_requirements(self, retries: int = 3) -> dict:
        """
        Process the RFP document by chunking it and extracting requirements from each chunk.
        
        Args:
            retries (int): Number of times to retry the agent call if JSON parsing fails or no requirements are extracted.
            
        Returns:
            dict: Consolidated requirements from all chunks
        """
        # Load RFP content
        rfp_content = self.load_rfp_content()
        if not rfp_content:
            self.logger.error("No RFP content loaded. Cannot proceed with requirement extraction.")
            return {"error": "No RFP content loaded"}
        
        # Create chunks if chunk processor is available
        if self.chunk_processor:
            chunks = self.chunk_processor.create_chunks(rfp_content)
            self.logger.info(f"Document split into {len(chunks)} chunks")
        else:
            # Fallback to processing entire document as single chunk
            chunks = [rfp_content]
            self.logger.warning("ChunkProcessor not available. Processing entire document as single chunk.")
        
        # Process each chunk
        all_requirements = []
        agent = self.create_agent()
        
        for i, chunk in enumerate(chunks, 1):
            self.logger.info(f"Processing chunk {i}/{len(chunks)}")
            
            chunk_requirements_extracted = False
            for attempt in range(retries):
                self.logger.info(f"Attempt {attempt + 1}/{retries} to extract requirements from chunk {i}")
                try:
                    # Create task for this chunk
                    task = self.create_extraction_task(chunk, i, len(chunks), agent)
                    
                    # Create crew and execute
                    crew = Crew(
                        agents=[agent],
                        tasks=[task],
                        verbose=True
                    )
                    
                    result = crew.kickoff()
                    self.logger.info(f"Raw result from chunk {i} (Attempt {attempt + 1}): {str(result)[:500]}...") # Log more of the raw result
                    
                    # Extract JSON from result
                    chunk_result = self.extract_json_from_result(result)
                    
                    # Save individual chunk result
                    self.save_chunk_result(i, chunk_result)
                    
                    # Add chunk results to overall results
                    if "requirements" in chunk_result and chunk_result["requirements"]:
                        all_requirements.extend(chunk_result["requirements"])
                        self.logger.info(f"Extracted {len(chunk_result['requirements'])} requirements from chunk {i} on attempt {attempt + 1}")
                        chunk_requirements_extracted = True
                        break # Exit retry loop on successful extraction
                    else:
                        self.logger.warning(f"No requirements found in chunk {i} on attempt {attempt + 1}. Retrying...")
                    
                except json.JSONDecodeError as json_e:
                    self.logger.warning(f"JSON parsing failed for chunk {i} (Attempt {attempt + 1}): {json_e}. Raw result: {str(result)[:500]}...")
                    # Log full raw result for failed JSON parsing attempts for better debugging
                    self.logger.debug(f"Full raw result for failed JSON parsing (Chunk {i}, Attempt {attempt + 1}): {result}")
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i} (Attempt {attempt + 1}): {str(e)}")
                    # Save error information if an exception occurs during a retry
                    error_result = {
                        "chunk_number": i,
                        "error": str(e),
                        "requirements": []
                    }
                    self.save_chunk_result(i, error_result)
            
            if not chunk_requirements_extracted:
                self.logger.error(f"All {retries} attempts failed to extract requirements from chunk {i}. Proceeding without requirements from this chunk.")
        
        # Deduplicate requirements
        self.logger.info(f"Before deduplication: {len(all_requirements)} requirements")
        deduplicated_requirements = self.deduplicate_requirements(all_requirements)
        
        # Consolidate results
        consolidated_result = {
            "total_chunks_processed": len(chunks),
            "total_requirements_extracted": len(deduplicated_requirements),
            "total_requirements_before_deduplication": len(all_requirements),
            "chunks_output_directory": self.chunks_output_dir,
            "requirements": deduplicated_requirements   
        }
        
        self.logger.info(f"Requirement extraction completed. Total requirements: {len(deduplicated_requirements)}")
        return consolidated_result

    def save_results(self, results: dict):
        """
        Save the extraction results to a JSON file.
        
        Args:
            results (dict): The consolidated requirements results
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {self.output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")

    def run_extraction(self) -> dict:
        """
        Main method to run the complete requirement extraction process.
        
        Returns:
            dict: The consolidated requirements results
        """
        self.logger.info("Starting RFP requirement extraction process")
        
        # Process chunks and extract requirements
        results = self.process_chunks_and_extract_requirements()
        
        # Save results
        if "error" not in results:
            self.save_results(results)
        
        return results

# Example of how to use the class (optional, for demonstration)
if __name__ == '__main__':
    
    # Instantiate the extractor class
    extractor = RFPRequirementExtractorAgent(logger_instance=get_logger(__name__))
    
    # Run the complete extraction process
    results = extractor.run_extraction()
    
    # Print summary
    if "error" not in results:
        print(f"\n--- EXTRACTION SUMMARY ---")
        print(f"Total chunks processed: {results['total_chunks_processed']}")
        print(f"Total requirements extracted: {results['total_requirements_extracted']}")
        print(f"Requirements before deduplication: {results.get('total_requirements_before_deduplication', 'N/A')}")
        print(f"Results saved to: {extractor.output_path}")
        print(f"Individual chunk results in: {results.get('chunks_output_directory', 'N/A')}")
    else:
        print(f"Error: {results['error']}")