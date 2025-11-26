import os
import json
import re
import logging
from typing import List, Dict
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.llm import LLM
from logger_config import get_logger

# Import chunking functionality
try:
    from chunk_processing import ChunkProcessor
    CHUNK_PROCESSOR_AVAILABLE = True
except ImportError:
    CHUNK_PROCESSOR_AVAILABLE = False
    print("Warning: ChunkProcessor not available. Large proposals may exceed context window.")

# --- Configuration and Setup ---

# Load environment variables from .env file
load_dotenv()


class ProposalEvaluatorAgent:
    """
    A CrewAI agent specialized in evaluating proposal compliance against RFP requirements.
    Enhanced with chunking to handle large texts. Provides comprehensive evaluation with all statuses:
    Addressed, Partially Addressed, Contradicted, Not Found, and Error cases.
    """
    def __init__(self, requirements_json_path: str, proposal_path: str, output_path: str, 
                 chunk_size: int = 30000, use_chunking: bool = True, logger_instance=None):
        """
        Initializes the evaluator with necessary file paths and LLM.

        Args:
            requirements_json_path (str): Path to the JSON file with RFP requirements.
            proposal_path (str): Path to the text file of the proposal.
            output_path (str): Path to save the evaluation report.
            chunk_size (int): Maximum tokens per chunk for proposal processing (default: 4000).
            use_chunking (bool): Whether to use chunking for large proposals (default: True).
        """
        # Configure logging
        self.logger = logger_instance if logger_instance is not None else get_logger(__name__)
        
        self.requirements_json_path = requirements_json_path
        self.proposal_path = proposal_path
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.use_chunking = use_chunking

        self.llm = LLM(
            model=os.getenv("OPENROUTER_MODEL", "openrouter/qwen/qwen3-32b"),
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1  # Low temperature for factual, deterministic analysis
        )
        
        # Initialize chunk processor if available and requested
        self.chunk_processor = None
        if self.use_chunking and CHUNK_PROCESSOR_AVAILABLE:
            try:
                self.chunk_processor = ChunkProcessor(max_chunk_size=chunk_size)
                self.logger.info("ChunkProcessor initialized for proposal processing")
            except Exception as e:
                self.logger.error(f"Failed to initialize ChunkProcessor: {str(e)}")
                self.logger.warning("Falling back to non-chunked processing")
        elif self.use_chunking and not CHUNK_PROCESSOR_AVAILABLE:
            self.logger.warning("Chunking requested but ChunkProcessor not available. Processing without chunking.")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Setup chunks output directory for detailed chunk evaluations
        self.chunks_output_dir = os.path.join(os.path.dirname(self.output_path), "proposal_chunks")
        if self.chunk_processor:
            # Clean the chunks directory before use
            self._clean_chunks_directory()
            os.makedirs(self.chunks_output_dir, exist_ok=True)
            
        self.logger.info("ProposalEvaluatorAgent initialized with chunking capabilities.")
        self.logger.info("Agent configured to report ALL requirement evaluation statuses.")

    def _clean_chunks_directory(self):
        """
        Clean the proposal_chunks directory by removing all existing files and subdirectories.
        This ensures a fresh start for each evaluation run.
        """
        if not self.chunks_output_dir:
            return
            
        try:
            if os.path.exists(self.chunks_output_dir):
                import shutil
                shutil.rmtree(self.chunks_output_dir)
                self.logger.info(f"Cleaned existing chunks directory: {self.chunks_output_dir}")
            else:
                self.logger.debug(f"Chunks directory does not exist yet: {self.chunks_output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to clean chunks directory {self.chunks_output_dir}: {str(e)}")

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the chunk processor if available, otherwise rough estimate.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        if self.chunk_processor:
            return self.chunk_processor.count_tokens(text)
        else:
            # Rough estimate: ~4 characters per token
            return len(text) // 4

    def _load_requirements(self) -> List[Dict]:
        """Loads the requirements from the JSON file."""
        try:
            with open(self.requirements_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle the nested structure: data["unscored_requirements"][0]["requirements"]
                requirements = []
                
                # Check for direct "requirements" key first
                if "requirements" in data:
                    requirements = data["requirements"]
                # Check for the nested structure with unscored_requirements
                elif "unscored_requirements" in data:
                    unscored_reqs = data["unscored_requirements"]
                    if isinstance(unscored_reqs, list) and len(unscored_reqs) > 0:
                        # Get the first item which contains the metadata and requirements
                        first_item = unscored_reqs[0]
                        if isinstance(first_item, dict) and "requirements" in first_item:
                            requirements = first_item["requirements"]
                            self.logger.info(f"Found {len(requirements)} requirements in nested unscored_requirements structure")
                        else:
                            self.logger.warning("Expected 'requirements' key in first unscored_requirements item")
                    else:
                        self.logger.warning("unscored_requirements is empty or not a list")
                
                if not requirements:
                    self.logger.warning(f"No 'requirements' found in {self.requirements_json_path}")
                    self.logger.info(f"Available keys: {list(data.keys())}")
                else:
                    self.logger.info(f"Successfully loaded {len(requirements)} requirements")
                    
                return requirements
        except FileNotFoundError:
            self.logger.error(f"Requirements file not found: {self.requirements_json_path}")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Could not decode JSON from {self.requirements_json_path}")
            return []

    def _load_proposal_text(self) -> str:
        """Loads the proposal content from the text file."""
        try:
            with open(self.proposal_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.logger.info(f"Successfully loaded proposal content from {self.proposal_path}")
            return content
        except FileNotFoundError:
            self.logger.error(f"Proposal file not found at: {self.proposal_path}")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to load proposal content: {str(e)}")
            return ""

    def _create_proposal_chunks(self, proposal_text: str) -> List[str]:
        """
        Create chunks from the proposal text.
        
        Args:
            proposal_text (str): The full proposal text
            
        Returns:
            List[str]: List of proposal chunks
        """
        if not proposal_text:
            return []
        
        # Check if chunking is needed
        token_count = self._count_tokens(proposal_text)
        self.logger.info(f"Proposal contains {token_count} tokens")
        
        if token_count <= self.chunk_size:
            self.logger.info("Proposal size is within token limit. No chunking required.")
            return [proposal_text]
        
        if not self.chunk_processor:
            self.logger.warning("Proposal exceeds token limit but no chunk processor available. Using simple word-based chunking.")
            return self._simple_chunk_text(proposal_text)
        
        self.logger.info(f"Proposal exceeds token limit of {self.chunk_size}. Creating semantic chunks...")
        chunks = self.chunk_processor.create_chunks(proposal_text)
        self.logger.info(f"Split proposal into {len(chunks)} chunks using semantic boundaries")
        
        # Log chunk statistics
        for i, chunk in enumerate(chunks):
            chunk_tokens = self._count_tokens(chunk)
            self.logger.info(f"Proposal chunk {i+1}: {chunk_tokens} tokens")
        
        return chunks

    def _simple_chunk_text(self, text: str) -> List[str]:
        """
        Simple fallback chunking method when ChunkProcessor is not available.
        Splits text by sentences and groups them to stay within token limits.
        
        Args:
            text (str): Text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        # Split by sentences (simple approach)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If adding this sentence would exceed the limit, start a new chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence + " "
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no chunks were created (very long single sentence), split by words
        if not chunks and text:
            words = text.split()
            current_chunk = ""
            current_tokens = 0
            
            for word in words:
                word_tokens = self._count_tokens(word + " ")
                
                if current_tokens + word_tokens > self.chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = word + " "
                    current_tokens = word_tokens
                else:
                    current_chunk += word + " "
                    current_tokens += word_tokens
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        self.logger.info(f"Simple chunking created {len(chunks)} chunks")
        return chunks if chunks else [text]  # Fallback to original text if all else fails

    def _save_chunk_evaluation(self, requirement_id: str, chunk_number: int, chunk_result: dict):
        """
        Save individual chunk evaluation result.
        
        Args:
            requirement_id (str): The requirement ID being evaluated
            chunk_number (int): The chunk number
            chunk_result (dict): The chunk evaluation result
        """
        try:
            if not self.chunks_output_dir:
                return
                
            # Create requirement-specific directory
            req_dir = os.path.join(self.chunks_output_dir, requirement_id)
            os.makedirs(req_dir, exist_ok=True)
            
            # Save chunk result
            chunk_file = os.path.join(req_dir, f"chunk_{chunk_number:03d}.json")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_result, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Chunk evaluation saved: {requirement_id}/chunk_{chunk_number:03d}.json")
            
        except Exception as e:
            self.logger.error(f"Failed to save chunk evaluation for {requirement_id}, chunk {chunk_number}: {str(e)}")

    def _create_agent_definition(self) -> Agent:
        """
        Creates the CrewAI agent specialized in CAUTIOUS and NUANCED compliance verification.
        """
        return Agent(
            role="Meticulous Compliance Auditor",
            goal=(
                "To verify with high accuracy whether a proposal fully addresses the core intent of an RFP requirement. "
                "Your primary function is to confirm alignment. You must only flag a requirement as 'Contradicted' if there is "
                "clear, direct, and undeniable evidence of a conflict. Prioritize avoiding false positives above all else. "
                "CRITICAL: Only use information explicitly stated in the proposal content - never make assumptions or reference external knowledge."
            ),
            backstory=(
                "You are a seasoned procurement auditor with a reputation for precision and meticulousness. You understand that "
                "incorrectly flagging a contradiction can disqualify a deserving bidder, which is a critical error. "
                "Your entire analytical process is built on the principle of 'verify alignment first.' You methodically break down "
                "each requirement to its core functional and non-functional intent. You then search the proposal for evidence of "
                "this intent being met. A contradiction is not a simple mismatch of keywords; it is a fundamental failure to meet "
                "the requirement's objective. You are immune to hyperbole and focus only on the substantive facts presented. "
                "If the proposal addresses the requirement's spirit and core function, you do not flag it, even if the wording differs. "
                "You strictly base all assessments and justifications ONLY on what is explicitly stated in the proposal content, "
                "never making assumptions or referencing external knowledge or industry standards not mentioned in the proposal."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            # Add a cache to avoid re-computing for the same inputs within a run
            # cache=True
        )

    def _create_chunk_evaluation_task(self, requirement: Dict, proposal_chunk: str,
                                    chunk_number: int, total_chunks: int, agent: Agent) -> Task:
        """
        Creates a task to check for compliance in a specific proposal chunk with a focus on avoiding false positives.
        """
        return Task(
            description=f"""
            # TASK: Analyze Proposal Chunk {chunk_number}/{total_chunks} for Compliance with EXTREME CAUTION

            **PRIMARY DIRECTIVE: Your goal is to find evidence of COMPLIANCE. Do NOT hunt for contradictions.** A contradiction is a last resort, used only for clear, direct conflicts.

            ---
            **1. The RFP Requirement to Verify:**
            *   **Requirement ID:** `{requirement.get('requirement_id')}`
            *   **Requirement Text:** `{requirement.get('requirement_text')}`
            *   **Category:** `{requirement.get('category')}`

            **2. The Proposal Chunk to Analyze:**
            ```            {proposal_chunk}
            ```
            ---

            **3. Your Analytical "Chain of Thought" (Follow these steps precisely):**

            *   **Step 1: Deconstruct the Requirement's Core Intent.** What is the fundamental goal of this requirement? Is it about a specific technology (e.g., must use 'Java'), a performance level (e.g., '99.9% uptime'), a security protocol (e.g., 'AES-256'), or a process (e.g., '24/7 support')?
            *   **Step 2: Scan the Proposal Chunk for Relevant Information.** Look for keywords and concepts related to the requirement's core intent.
            *   **Step 3: Evaluate the Findings against the Hierarchy of Compliance.** Based on what you found, assign ONE of the following statuses:

                *   `"Addressed"`: The proposal chunk contains specific information that directly and clearly meets the core intent of the requirement.
                *   `"Partially Addressed"`: The chunk addresses the topic but is missing key details or meets the requirement to a lesser degree. (e.g., Req: "Provide a detailed security plan"; Prop: "We have a robust security plan.")
                *   `"Contradicted"`: **(Use with EXTREME caution)** The chunk contains a statement in DIRECT and UNDENIABLE opposition to the requirement. This is not a minor difference, but a clear "no" or a conflicting specification.
                *   `"Not Found"`: The chunk does not contain any relevant information regarding this specific requirement. This is the default if you find no evidence.

            **4. Strict Rules for Determining a "Contradiction":**

            *   **A "Contradiction" IS:**
                *   An explicit refusal to comply (e.g., Req: "Must support SAML"; Prop: "We do not support SAML.").
                *   A specification that directly conflicts (e.g., Req: "System must be built in Python"; Prop: "Our solution is built entirely on .NET.").
                *   A stated capability that is verifiably less than a required minimum (e.g., Req: "Minimum 10 years experience"; Prop: "Our firm was founded 5 years ago.").

            *   **A "Contradiction" IS NOT:**
                *   Exceeding the requirement (e.g., Req: "24/7 support"; Prop: "We offer 24/7/365 support with a dedicated account manager."). -> **This is "Addressed".**
                *   Providing a compliant solution with a different name (e.g., Req: "Provide a disaster recovery plan"; Prop: "Our business continuity strategy ensures..."). -> **This is "Addressed".**
                *   The proposal simply not mentioning the requirement in this specific chunk. -> **This is "Not Found".**
                *   A minor, reasonable condition (e.g., Req: "99.9% uptime"; Prop: "We guarantee 99.9% uptime, excluding scheduled weekly maintenance."). -> **This is "Addressed".**

            **5. CRITICAL: Justification Rules - ONLY Use Proposal Content**
            
            **MANDATORY: Your justification MUST ONLY reference information that is explicitly stated in the proposal chunk above.**
            
            **FORBIDDEN in justifications:**
            - Any information not present in the proposal chunk
            - Assumptions about what the proposal might mean
            - Industry standards or best practices not mentioned in the proposal
            - Technical details not explicitly stated in the proposal
            - References to other sections or documents
            - General knowledge or external information
            
            **REQUIRED in justifications:**
            - Direct quotes from the proposal chunk when available
            - Specific statements from the proposal that support your assessment
            - Clear reference to what the proposal actually says (or doesn't say)
            
            **Examples of GOOD justifications:**
            - "The proposal states 'Our system is built on .NET framework' which conflicts with the Python requirement."
            - "The proposal mentions '24/7 support' which meets the requirement."
            - "The proposal does not contain any information about security protocols."
            
            **Examples of BAD justifications:**
            - "The proposal likely uses industry-standard security measures." (assumption)
            - "Most modern systems support this requirement." (external knowledge)
            - "The proposal should include detailed documentation." (opinion)

            **6. Required Output Format:**
            Your final output for this task MUST be a single, clean JSON object. Do not include any text or markdown before or after the JSON.

            ```json
            {{
                "requirement_text": "{requirement.get('requirement_text')}",
                "status": "Addressed" | "Partially Addressed" | "Contradicted" | "Not Found",
                "justification": "A brief, professional explanation based ONLY on the proposal content above. Quote relevant proposal text if it supports your decision.",
                "confidence_score": 0.0-1.0
            }}
            ```
            """,
            agent=agent,
            expected_output="A single JSON object with the keys 'requirement_text', 'status', 'justification', and 'confidence_score'."
        )

    def _create_comprehensive_analysis_agent(self) -> Agent:
        """
        Creates a specialized agent for analyzing all chunk results and providing definitive conclusions.
        """
        return Agent(
            role="Master Compliance Analyst",
            goal=(
                "To provide an accurate, unified assessment of whether the proposal contradicts the requirement, "
                "presenting findings as a seamless analysis of the entire proposal without revealing the underlying segmented review process. "
                "CRITICAL: Only use information explicitly stated in the chunk results - never make assumptions or reference external knowledge."
            ),
            backstory=(
                "You are an elite proposal analyst with extensive experience in evaluating compliance. "
                "You excel at synthesizing complex information and presenting clear, unified conclusions. "
                "Your assessments read as comprehensive reviews of entire proposals, never revealing that you analyzed "
                "the content in segments. You focus on what the proposal actually states or lacks, presenting your "
                "findings as direct observations about the proposal's content and alignment with requirements. "
                "You only conclude contradiction when there is clear, substantial evidence that the proposal "
                "fundamentally fails to meet the requirement's intent. "
                "You strictly base all assessments and justifications ONLY on what is explicitly stated in the chunk results, "
                "never making assumptions or referencing external knowledge or industry standards not mentioned in the evidence."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def _create_comprehensive_analysis_task(self, requirement: Dict, chunk_results: List[Dict], agent: Agent) -> Task:
        """
        Creates a task for a final, holistic analysis of all chunk evaluations for a single requirement.
        """
        return Task(
            description=f"""
            # TASK: Synthesize Chunk Analyses into a Final, Definitive Verdict

            You have been provided with analyses from different sections of a proposal. Your task is to synthesize these findings into a single, holistic, and final verdict. You must decide if the requirement was met based on the totality of the evidence.

            ---
            **1. The RFP Requirement Under Review:**
            ```json
            {json.dumps(requirement, indent=2)}
            ```

            **2. Evidence from All Proposal Sections:**
            Here is a list of findings from the sequential analysis of the proposal. Some sections may not have mentioned the requirement, while others may have addressed it.
            ```json
            {json.dumps(chunk_results, indent=2)}
            ```
            ---

            **3. Your Synthesis and Decision-Making Logic:**

            *   **Step 1: Look for a Definitive "Contradicted" Status.** If any chunk has a high-confidence "Contradicted" status, your final verdict is likely "Contradicted". Your justification should focus on that specific piece of evidence as the deciding factor.
            *   **Step 2: Look for a Definitive "Addressed" Status.** If there is no contradiction, check if one or more chunks have an "Addressed" status. If the requirement is clearly met anywhere in the proposal, the final verdict is "Addressed".
            *   **Step 3: Consider "Partially Addressed".** If there are no "Contradicted" or "Addressed" statuses, but there are "Partially Addressed" findings, then the final verdict is "Partially Addressed". It means the proposal acknowledges the topic but fails to provide sufficient detail.
            *   **Step 4: Default to "Not Found".** If all chunks reported "Not Found", it means the proposal completely failed to mention or address the requirement anywhere. The final verdict is "Not Found".

            **4. CRITICAL: Justification Rules - ONLY Use Evidence from Chunk Results**
            
            **MANDATORY: Your justification MUST ONLY reference information that is explicitly stated in the chunk results above.**
            
            **FORBIDDEN in justifications:**
            - Any information not present in the chunk results
            - Assumptions about what the proposal might contain
            - Industry standards or best practices not mentioned in the chunk results
            - Technical details not explicitly stated in the chunk results
            - General knowledge or external information
            - References to analysis processes or chunk numbers
            
            **REQUIRED in justifications:**
            - Direct quotes from the chunk results when available
            - Specific statements from the chunk results that support your assessment
            - Clear reference to what the proposal actually says (or doesn't say)
            
            **DO NOT use these phrases:**
            - "A direct contradiction was found in the proposal"
            - "The RFP requirement (REQ-XXX-XXX) explicitly states"
            - "in chunk 3," "one section says," or "the analysis shows"
            - "After thorough analysis," "Based on evaluation," etc.
            - "The evidence from the chunks shows..."
            - "Based on the chunk analysis..."

            **GOOD examples:**
            *   "The proposal states the system is built on .NET, but the requirement mandates Python."
            *   "The proposal confirms 24/7 support availability in the Support Model section."
            *   "The requirement is not addressed anywhere in the proposal."
            *   "The proposal specifies a maximum of 500 concurrent users, while the requirement calls for support of at least 1000 users."

            **BAD examples:**
            *   "The evidence from the chunks shows that chunk 5 contradicts the requirement."
            *   "A direct contradiction was found in the proposal."
            *   "The RFP requirement (REQ-007-002) explicitly states..."
            *   "The proposal likely uses industry-standard security measures." (assumption)
            *   "Most modern systems support this requirement." (external knowledge)

            **5. Required Output Format:**
            Provide a single, clean JSON object as your final answer.

            ```json
            {{
                "final_status": "Addressed" | "Partially Addressed" | "Contradicted" | "Not Found",
                "comprehensive_justification": "Your final, holistic justification based ONLY on the evidence from the chunk results above. Be concise and direct.",
                "key_evidence_quote": "The single most important quote from the proposal that supports your final verdict. Can be an empty string if not applicable."
            }}
            ```
            """,
            agent=agent,
            expected_output="A single JSON object with the final verdict ('final_status'), a holistic justification, and the key evidence quote."
        )

    def _aggregate_chunk_results(self, requirement_id: str, chunk_results: List[Dict]) -> Dict:
        """
        Aggregates chunk results. If any contradiction is found, it's final. Otherwise, it prioritizes 'Addressed' status.
        This now acts as a smart aggregator before potentially sending to a final analysis agent if needed for complex cases.
        """
        # Find the original requirement details
        requirement = next((req for req in self._loaded_requirements if req.get('requirement_id') == requirement_id), {})

        # 1. Immediately identify high-confidence contradictions. They trump everything.
        contradictions = [r for r in chunk_results if r.get('status') == 'Contradicted' and r.get('confidence_score', 0) > 0.8]
        if contradictions:
            # Take the strongest contradiction as the final word.
            strongest_contradiction = max(contradictions, key=lambda x: x.get('confidence_score', 0))
            justification = strongest_contradiction.get('justification', '')

            return {
                "requirement_id": requirement_id,
                "status": "Contradicted",
                "confidence_score": strongest_contradiction.get('confidence_score'),
                "justification": justification,
                "relevant_quote": self._extract_quote(strongest_contradiction.get('justification', '')),
                "requirement": requirement
            }

        # 2. If no contradictions, look for the best evidence that the requirement was addressed.
        addressed = [r for r in chunk_results if r.get('status') == 'Addressed']
        if addressed:
            best_evidence = max(addressed, key=lambda x: x.get('confidence_score', 0))
            justification = best_evidence.get('justification', '')

            return {
                "requirement_id": requirement_id,
                "status": "Addressed",
                "confidence_score": best_evidence.get('confidence_score'),
                "justification": justification,
                "relevant_quote": self._extract_quote(best_evidence.get('justification', '')),
                "requirement": requirement
            }

        # 3. If not addressed, check for partial fulfillment.
        partials = [r for r in chunk_results if r.get('status') == 'Partially Addressed']
        if partials:
            weakest_evidence = max(partials, key=lambda x: x.get('confidence_score', 0))
            justification = weakest_evidence.get('justification', '')
            return {
                "requirement_id": requirement_id,
                "status": "Partially Addressed",
                "confidence_score": weakest_evidence.get('confidence_score'),
                "justification": justification,
                "relevant_quote": self._extract_quote(weakest_evidence.get('justification', '')),
                "requirement": requirement
            }

        # 4. If no evidence was found anywhere, the requirement is considered not found.
        return {
            "requirement_id": requirement_id,
            "status": "Not Found",
            "confidence_score": 1.0, # High confidence that it wasn't found
            "justification": "After a thorough review of the proposal, no section was found to address this requirement.",
            "relevant_quote": "",
            "requirement": requirement
        }

    def _extract_quote(self, justification: str) -> str:
        """A helper to pull a quote from the justification text if one exists."""
        match = re.search(r'["\'](.*?)["\']', justification)
        if match:
            return match.group(1)
        return justification # Fallback to the whole justification if no clear quote

    def _evaluate_requirement_against_chunks(self, requirement: Dict, proposal_chunks: List[str], retries: int = 5) -> Dict:
        """
        Evaluate a single requirement against all proposal chunks and aggregate results.
        
        Args:
            requirement (Dict): The requirement to evaluate
            proposal_chunks (List[str]): List of proposal chunks
            retries (int): Number of times to retry the agent call if JSON parsing fails.
            
        Returns:
            Dict: Aggregated evaluation result
        """
        requirement_id = requirement.get('requirement_id', 'Unknown')
        self.logger.info(f"Evaluating requirement {requirement_id} against {len(proposal_chunks)} chunks")
        
        agent = self._create_agent_definition()
        chunk_results = []
        
        # Evaluate requirement against each chunk
        for chunk_num, chunk_text in enumerate(proposal_chunks, 1):
            self.logger.debug(f"Evaluating {requirement_id} against chunk {chunk_num}/{len(proposal_chunks)}")
            
            chunk_evaluation_successful = False
            for attempt in range(retries):
                try:
                    self.logger.info(f"Attempt {attempt + 1}/{retries} to evaluate chunk {chunk_num} for requirement {requirement_id}")
                    task = self._create_chunk_evaluation_task(
                        requirement, chunk_text, chunk_num, len(proposal_chunks), agent
                    )
                    crew = Crew(agents=[agent], tasks=[task], verbose=False)
                    crew_output = crew.kickoff()
                    
                    # Extract result
                    if hasattr(crew_output, 'raw'):
                        result_str = crew_output.raw
                    elif hasattr(crew_output, 'result'):
                        result_str = crew_output.result
                    elif hasattr(crew_output, 'text'):
                        result_str = crew_output.text
                    else:
                        result_str = str(crew_output)
                    
                    # Clean and parse JSON
                    cleaned_json_str = self._clean_json_response(result_str)
                    chunk_result = json.loads(cleaned_json_str)
                    
                    # If parsing is successful, break the retry loop
                    chunk_results.append(chunk_result)
                    self._save_chunk_evaluation(requirement_id, chunk_num, chunk_result)
                    self.logger.debug(f"Chunk {chunk_num} result: {chunk_result.get('status')} "
                              f"(confidence: {chunk_result.get('confidence_score', 0.0)})")
                    chunk_evaluation_successful = True
                    break
                    
                except json.JSONDecodeError as json_e:
                    self.logger.warning(f"JSON parsing failed for {requirement_id} against chunk {chunk_num} (Attempt {attempt + 1}): {json_e}. Raw string prefix: {cleaned_json_str[:500]}...")
                    # Do not append to chunk_results yet, just retry
                except Exception as e:
                    self.logger.error(f"Unexpected error during evaluation of {requirement_id} against chunk {chunk_num} (Attempt {attempt + 1}): {str(e)}")
                    # Do not append to chunk_results yet, just retry

            if not chunk_evaluation_successful:
                self.logger.error(f"All {retries} attempts failed for {requirement_id} against chunk {chunk_num}. Appending error result.")
                error_result = {
                    "requirement_id": requirement_id,
                    "chunk_number": chunk_num,
                    "total_chunks": len(proposal_chunks),
                    "status": "Error",
                    "confidence_score": 0.0,
                    "justification": f"All {retries} evaluation attempts failed: Agent did not return valid JSON or encountered an error.",
                    "relevant_quote": "Evaluation error",
                    "chunk_coverage": "Error in processing"
                }
                chunk_results.append(error_result)
                self._save_chunk_evaluation(requirement_id, chunk_num, error_result)
        
        # Aggregate results from all chunks
        aggregated_result = self._aggregate_chunk_results(requirement_id, chunk_results)
        self.logger.info(f"Aggregated result for {requirement_id}: {aggregated_result.get('status')} "
                   f"(confidence: {aggregated_result.get('confidence_score', 0.0)})")
        
        return aggregated_result

    def _clean_json_response(self, result_str: str) -> str:
        """
        Clean JSON response from markdown code blocks and other formatting.
        
        Args:
            result_str (str): Raw response string
            
        Returns:
            str: Cleaned JSON string
        """
        # Clean the JSON from markdown code blocks if present
        if '```' in result_str:
            # Extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', result_str, re.DOTALL)
            if json_match:
                result_str = json_match.group(1).strip()
            else:
                # Fallback: try to extract anything that looks like JSON
                lines = result_str.strip().split('\n')
                json_lines = []
                inside_json = False
                for line in lines:
                    if line.strip().startswith('```') and not inside_json:
                        inside_json = True
                        continue
                    elif line.strip() == '```' and inside_json:
                        break
                    elif inside_json:
                        json_lines.append(line)
                result_str = '\n'.join(json_lines).strip()
        
        return result_str.strip()

    def run_evaluation(self) -> Dict:
        """
        Runs the full evaluation process and generates a final JSON report
        containing ALL requirement evaluations with their respective statuses.
        
        Returns:
            Dict: The final evaluation report, or an error report if processing fails.
        """
        self.logger.info("--- Starting Comprehensive Proposal Evaluation ---")
        requirements = self._load_requirements()
        proposal_text = self._load_proposal_text()

        if not requirements or not proposal_text:
            self.logger.error("Missing requirements or proposal file. Aborting.")
            # Write an empty/error report
            error_report = {
                "summary": {"error": "Execution failed. Missing requirements or proposal file."},
                "all_evaluations": []
            }
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, indent=4)
            return error_report # Return error report instead of None

        self._loaded_requirements = requirements
        proposal_chunks = self._create_proposal_chunks(proposal_text)

        all_evaluations = []
        error_count = 0

        # Process each requirement through the analysis workflow
        for i, req in enumerate(requirements, 1):
            try:
                # This method uses our robust, multi-status analysis and aggregation
                # Pass the retries parameter to _evaluate_requirement_against_chunks
                result = self._evaluate_requirement_against_chunks(req, proposal_chunks, retries=5) # Set retries to 5
                all_evaluations.append(result)
            except Exception as e:
                self.logger.error(f"Critical error evaluating requirement {req.get('requirement_id')}: {str(e)}")
                error_count += 1
                all_evaluations.append({
                    "requirement_id": req.get('requirement_id'),
                    "status": "Error",
                    "justification": f"A system error occurred during evaluation: {e}",
                    "requirement": req
                })

        # --- Report Generation Logic ---

        # Include ALL evaluation results, not just contradicted ones
        contradicted_count = len([res for res in all_evaluations if res.get('status') == 'Contradicted'])

        # Build the final report structure with all evaluations
        final_report = {
            "summary": {
                "total_requirements_analyzed": len(requirements),
                "total_addressed": len([r for r in all_evaluations if r.get('status') == 'Addressed']),
                "total_contradicted": contradicted_count,
                "total_partially_addressed": len([r for r in all_evaluations if r.get('status') == 'Partially Addressed']),
                "total_not_found": len([r for r in all_evaluations if r.get('status') == 'Not Found']),
                "total_errors": error_count,
                "contradiction_percentage": round((contradicted_count / len(requirements) * 100), 2) if len(requirements) > 0 else 0
            },
            # Include ALL evaluations with all statuses
            "all_evaluations": [
                {
                    "requirement_id": res.get('requirement_id', 'Unknown'),
                    "requirement_text": res.get('requirement', {}).get('requirement_text', 'Unknown'),
                    "status": res.get('status', 'Unknown'),
                    "confidence_score": res.get('confidence_score', 0.0),
                    "justification": res.get('justification', 'No justification provided')
                }
                for res in all_evaluations  # Include ALL evaluation results
            ]
        }

        # Save the focused report to the output file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=4)

        self.logger.info(f"--- Evaluation Complete. Comprehensive report saved to: {self.output_path} ---")

        # --- Console Output for Immediate Feedback ---
        print("\n=== PROPOSAL EVALUATION RESULTS ===")
        print(f"Total Requirements Analyzed: {final_report['summary']['total_requirements_analyzed']}")
        print(f"Total Addressed: {final_report['summary']['total_addressed']}")
        print(f"Total Partially Addressed: {final_report['summary']['total_partially_addressed']}")
        print(f"Total Contradicted: {contradicted_count}")
        print(f"Total Not Found: {final_report['summary']['total_not_found']}")
        print(f"Total Errors: {error_count}")
        print(f"\nFinal report file '{self.output_path}' contains details for ALL requirements with their evaluation status.")

        if contradicted_count > 0:
            print(f"\n--- CONTRADICTED REQUIREMENTS ---")
            contradicted_evals = [res for res in final_report['all_evaluations'] if res.get('status') == 'Contradicted']
            for res in contradicted_evals:
                print(f"\n[!] {res['requirement_id']} - Contradicted (Confidence: {res['confidence_score']:.2f})")
                print(f"    Justification: {res['justification']}")
                print(f"    Evidence from Proposal: '{res.get('relevant_quote', 'N/A')}'")
        else:
            print("\n--- No direct contradictions were found. ---")
        return final_report

# --- Main Execution Block ---
if __name__ == '__main__':
    # Define file paths
    # REQUIREMENTS_JSON_FILE = "Proposal_Input/RFP_Reuirement.json"
    REQUIREMENTS_JSON_FILE = "outputs_proposal_eval/extracted_requirements.json"
    PROPOSAL_FILE = "proposal_pdf_extracted.txt"
    # PROPOSAL_FILE = "Unaligne_Proposal_Input/Proposal_input.txt"
    # PROPOSAL_FILE = "input_proposal_data/prop_2_text.txt"
    EVALUATION_OUTPUT_FILE = "outputs_evaluation/proposal_evaluation_contradicted.json"

    # Create a dummy proposal file if it doesn't exist
    os.makedirs("inputs", exist_ok=True)
    if not os.path.exists(PROPOSAL_FILE):
        print(f"Creating dummy proposal file at {PROPOSAL_FILE}")
        with open(PROPOSAL_FILE, "w", encoding="utf-8") as f:
            f.write(
                "Our state-of-the-art solution is built on the .NET framework, ensuring robust and scalable performance.\n"
                "We provide comprehensive support with a guaranteed response time of 2 hours for critical incidents.\n"
                "User authentication is handled via our proprietary login system.\n"
                "The system will be deployed on Microsoft Azure cloud infrastructure."
            )

    # Check if the requirements file exists before running the evaluator
    if not os.path.exists(REQUIREMENTS_JSON_FILE):
        print("\nERROR: Requirements file not found!")
        print(f"Please run `rfp_extractor_agent.py` first to generate '{REQUIREMENTS_JSON_FILE}'.")
    else:
        # Instantiate and run the evaluator with chunking capabilities
        evaluator = ProposalEvaluatorAgent(
            requirements_json_path=REQUIREMENTS_JSON_FILE,
            proposal_path=PROPOSAL_FILE,
            output_path=EVALUATION_OUTPUT_FILE,
            chunk_size=30000,  # Adjust chunk size as needed
            use_chunking=True  # Enable chunking for large proposals
        )
        final_report = evaluator.run_evaluation()