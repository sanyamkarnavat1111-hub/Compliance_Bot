import json
import numpy as np
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from collections import defaultdict

# Add CrewAI imports
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from embedding_generation import EmbeddingGeneration

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuplicatedGroupCategorizer:
    def __init__(self, max_concurrent_calls: int = 10):
        """
        Initialize the RAG processor for improvements analysis.
        
        Args:
            json_file_path: Path to the formatted improvements JSON file
            max_concurrent_calls: Maximum number of concurrent API calls
        """
        self.max_concurrent_calls = max_concurrent_calls
        
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.model_name = "openrouter/qwen/qwen3-32b"
    
        if not self.api_key:
            raise ValueError("API key is required. Set OPENROUTER_API_KEY environment variable.")
        
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
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGeneration()
        
        # Initialize tracking for excluded items
        self.excluded_compliant_items = []  # Track compliant items that were excluded
        
        # Thread-safe locks and tracking for concurrent processing
        self.processed_items = set()
        self.duplicate_groups = defaultdict(set)  # Maps representative ID to all duplicate IDs
        self.removed_items = set()
        
    def _load_json_data(self) -> Dict[str, Any]:
        """Load and parse the JSON file."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise
    
    def _flatten_data(self) -> List[Dict[str, Any]]:
        """
        Flatten all categories into a single list of items with IDs.
        Exclude items with compliance_status "Met" or "met" as they don't need improvements.
        
        Returns:
            List of dictionaries with added 'id' and 'category' fields
        """
        flattened = []
        item_id = 0
        
        for category, items in self.data.items():
            for item in items:
                # Skip items that are already compliant (Met/met)
                compliance_status = item.get('compliance_status', '').lower()
                if compliance_status == 'met':
                    logger.info(f"Skipping compliant item: {item.get('ea_requirement', 'N/A')[:50]}...")
                    self.excluded_compliant_items.append(item) # Track excluded items
                    continue
                
                # Add ID and category information
                item_with_metadata = item.copy()
                item_with_metadata['id'] = item_id
                item_with_metadata['category'] = category
                flattened.append(item_with_metadata)
                item_id += 1
        
        logger.info(f"Flattened {len(flattened)} non-compliant items from {len(self.data)} categories")
        return flattened
    
    def _extract_text_from_item(self, item: Dict[str, Any]) -> str:
        """
        Extract meaningful text from a JSON item for embedding.
        
        Args:
            item: Dictionary containing improvement item data
            
        Returns:
            Concatenated text from relevant fields
        """
        text_parts = []
        
        # Extract text from key fields
        if 'ea_requirement' in item:
            text_parts.append(f"Requirement: {item['ea_requirement']}")
        
        if 'rfp_coverage' in item:
            text_parts.append(f"Coverage: {item['rfp_coverage']}")
        
        if 'gap_analysis' in item:
            text_parts.append(f"Gap Analysis: {item['gap_analysis']}")
        
        return " ".join(text_parts)
    
    def _create_all_embeddings(self) -> torch.Tensor:
        """
        Create embeddings for all items in the dataset.
        
        Returns:
            Tensor of embeddings for all items
        """
        texts = []
        for item in self.flattened_items:
            text = self._extract_text_from_item(item)
            texts.append(text)
        
        # Create embeddings using Nomic model
        embeddings = self.embedding_generator.generate_embeddings(texts)
        return embeddings
    
    def _find_top_similar_items(self, target_id: int, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        Find top-k most similar items for a given target item (excluding itself).
        
        Args:
            target_id: ID of the target item
            top_k: Number of top similar items to return
            
        Returns:
            List of tuples (id, similarity_score, item_data)
        """
        # Create embeddings if not already created
        if self.item_embeddings is None:
            self.item_embeddings = self._create_all_embeddings()
        
        # Get target item embedding
        target_embedding = self.item_embeddings[target_id:target_id+1]
        
        # Calculate similarities with all other items
        similarities = cosine_similarity(target_embedding.numpy(), self.item_embeddings.numpy())[0]
        
        # Create list of (id, similarity) pairs, excluding the target item
        similarity_pairs = [(i, similarities[i]) for i in range(len(similarities)) if i != target_id]
        
        # Sort by similarity (descending) and get top-k
        similarity_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = similarity_pairs[:top_k]
        
        results = []
        for item_id, similarity_score in top_pairs:
            item_data = self.flattened_items[item_id]
            results.append((item_id, similarity_score, item_data))
        
        return results
    
    def _call_agent(self, target_item: Dict, similar_items: List[Tuple[int, float, Dict]]) -> Dict[str, Any]:
        """
        Call the AI agent to analyze duplicates using CrewAI, prioritizing keywords.
        
        Args:
            target_item: The target item to analyze
            similar_items: List of similar items with scores
            
        Returns:
            Agent's response as a dictionary
        """

        # Create the prompt internally
        prompt = f"""
You are an expert analyst comparing improvement recommendations to identify TRUE DUPLICATES ONLY.

TARGET ITEM (ID: {target_item['id']}, Category: {target_item['category']}):
- EA Requirement: {target_item.get('ea_requirement', 'N/A')}
- Compliance Status: {target_item.get('compliance_status', 'N/A')}
- RFP Coverage: {target_item.get('rfp_coverage', 'N/A')}
- Gap Analysis: {target_item.get('gap_analysis', 'N/A')}

SIMILAR ITEMS TO COMPARE:
"""
        
        for i, (item_id, score, item) in enumerate(similar_items, 1):
            prompt += f"\n--- ITEM {i} (ID: {item_id}, Similarity: {score:.3f}, Category: {item['category']}) ---\n"
            prompt += f"- EA Requirement: {item.get('ea_requirement', 'N/A')}\n"
            prompt += f"- Compliance Status: {item.get('compliance_status', 'N/A')}\n"
            prompt += f"- RFP Coverage: {item.get('rfp_coverage', 'N/A')}\n"
            prompt += f"- Gap Analysis: {item.get('gap_analysis', 'N/A')}\n"
        
        prompt += """

CRITICAL INSTRUCTIONS: 
Items are DUPLICATES if they address the SAME CORE REQUIREMENT. Your analysis MUST BEGIN by identifying matching technical keywords and phrases. Only after confirming keyword similarity should you then check for broader contextual overlap.

CRITERIA FOR DUPLICATES:
1. **KEYWORD MATCH**: Search for an exact or near-exact match of specific technical terms, protocols, standards, or product names (e.g., "OAuth2.0," "SSO," "multi-factor authentication," "GraphQL," "container orchestration"). This is the highest priority check.
2. **SAME CORE REQUIREMENT**: Must address the same fundamental technical requirement. This is verified *only after* a keyword match is found.
3. **SAME MISSING ELEMENT**: Must identify the same missing component or functionality in the RFP.
4. **OVERLAPPING SCOPE**: One item can be broader/narrower than the other, but they must overlap on the core requirement.

KEYWORD FOCUS PRINCIPLES:
- **PRIORITIZE KEYWORDS**: If a significant technical keyword or phrase is present in one item but missing from the other, they are likely NOT duplicates.
- **AVOID GENERIC TERMS**: Ignore generic terms like "security," "performance," or "integration" unless they are part of a specific technical phrase. Focus on concrete nouns and verbs.
- **SUBSET vs SUPERSET**: An item about "Authentication" is a superset of one about "OAuth 2.0." If both items explicitly mention "OAuth 2.0," they are duplicates. If only one does, they are not.

ANALYSIS PRINCIPLES:
1. **SAME CORE ELEMENT ≠ RELATED**: Items with the same specific technical element (keyword-based) are likely duplicates.
2. **BROADER vs NARROWER**: A broader requirement that includes a narrower one is a duplicate only if the narrower requirement's core keywords are also present in the broader one.
3. **DIFFERENT TECHNOLOGIES**: Items requiring different specific technologies or tools are NOT duplicates, even if their descriptions are conceptually similar. For example, a requirement for "SAML" is not a duplicate of one for "OpenID Connect."

DUPLICATE IDENTIFICATION RULES:
- Mark as duplicate if items address the same specific technical element, protocol, or tool requirement based on keyword analysis.
- Focus on the specific technical elements mentioned in the requirements.
- Different categories can contain duplicates if they address the same core technical requirement.
- Broader scope items that contain narrower requirements are duplicates of those narrower items.

DECISION FRAMEWORK:
- **KEYWORD FIRST**: The final decision for "is_duplicate_found" must be heavily influenced by keyword analysis.
- **BALANCED APPROACH**: Identify genuine duplicates while preserving distinct technical requirements.
- **CORE REQUIREMENT FOCUS**: If the core technical requirement is the same, consider as duplicate.
- **PRESERVE TECHNICAL DIVERSITY**: Keep items that address different technologies or architectural components.

IMPORTANT: The system will NOT remove the first item automatically. It will analyze all duplicates and select the most appropriate representative item from the duplicate group.

Provide your analysis in the following JSON format ONLY:

{
    "is_duplicate_found": true/false,
    "duplicate_item_ids": [list of item IDs that are duplicates],
    "reasoning": "Detailed explanation focusing on why items are considered duplicates or distinct, highlighting specific technical differences or similarities found through keyword analysis."
}

Respond with ONLY the JSON, no additional text.
"""
        
        # Create CrewAI agent for duplicate analysis
        duplicate_analyzer_agent = Agent(
            role="Duplicate Analysis Expert",
            goal="Analyze improvement recommendations to identify TRUE DUPLICATES ONLY with high precision and accuracy, prioritizing a keyword-first approach.",
            backstory="An expert analyst specializing in technical requirements and improvement recommendations with deep understanding of keyword-based and semantic similarity for duplicate detection. Must respond with ONLY valid JSON format, no additional text or explanations.",
            llm=self.llm,
            llm_kwargs=self.llm_kwargs,
            reasoning=True,
            verbose=True,
            max_reasoning_attempts=5,
            max_iterations=5
        )
        
        # Create CrewAI task for duplicate analysis
        duplicate_analysis_task = Task(
            description=prompt,
            expected_output="A JSON object with duplicate analysis results in the specified format",
            agent=duplicate_analyzer_agent,
            context=[{
                "name": "duplicate_analysis",
                "description": "Duplicate Analysis Task",
                "role": "system",
                "content": (
                    "You are an expert analyst comparing improvement recommendations to identify TRUE DUPLICATES ONLY.\n\n"
                    "SYSTEM REQUIREMENTS:\n"
                    "1. Respond with ONLY valid JSON format, no additional text or explanations\n"
                    "2. Use the exact format specified in the task description\n"
                    "3. Include detailed reasoning for duplicate identification decisions\n"
                    "4. Focus on technical requirements and implementation details, using a keyword-first approach\n"
                    "5. Apply the criteria and principles outlined in the task description"
                ),
                "expected_output": "JSON object with duplicate analysis results",
                "framework": "Duplicate Detection Analysis",
                "evaluation_approach": (
                    "1. Analyze each item pair for core requirement similarity, starting with keyword identification.\n"
                    "2. Identify technical overlap and scope relationships.\n"
                    "3. Determine if items address the same fundamental requirement based on keywords.\n"
                    "4. Provide detailed reasoning for duplicate decisions.\n"
                    "5. Ensure JSON output format compliance."
                )
            }]
        )
        
        # Create CrewAI crew for duplicate analysis
        duplicate_crew = Crew(
            agents=[duplicate_analyzer_agent],
            tasks=[duplicate_analysis_task],
            process=Process.sequential,
            manager_llm=self.llm,
            manager_llm_kwargs=self.llm_kwargs,
            model=self.model_name,
            verbose=True,
            full_output=True
        )
        
        # Execute the crew
        result = duplicate_crew.kickoff()
        
        # Extract content from the result
        if hasattr(result, 'raw'):
            content = result.raw
        elif hasattr(result, 'result'):
            content = result.result
        else:
            content = str(result)
        
        logger.info(f"Raw CrewAI response: {content}")
        
        # Find JSON in the response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        json_str = content[start_idx:end_idx]
        parsed_response = json.loads(json_str)
        logger.info(f"Parsed CrewAI response: {parsed_response}")
        return parsed_response
    
    def _process_item_safely(self, target_id: int) -> Dict[str, Any]:
        """
        Process a single item with thread-safe duplicate handling.
        
        Args:
            target_id: ID of the target item
            
        Returns:
            Dictionary containing processing results
        """
        # Check if item is already processed or removed
        if target_id in self.processed_items or target_id in self.removed_items:
            return {"skipped": True, "reason": "Already processed or removed"}
        
        # Mark as processed
        self.processed_items.add(target_id)
        
        if target_id >= len(self.flattened_items):
            raise ValueError(f"Target ID {target_id} is out of range")
        
        target_item = self.flattened_items[target_id]
        logger.info(f"Processing item {target_id} from category {target_item['category']}")
        
        # Find top 5 similar items
        similar_items = self._find_top_similar_items(target_id, top_k=5)
        
        logger.info(f"Found {len(similar_items)} similar items for item {target_id}")
        
        # Call agent to analyze duplicates
        agent_response = self._call_agent(target_item, similar_items)
        
        # Handle duplicates with thread-safe logic
        # Check if agent found duplicates (direct response, not nested)
        if "is_duplicate_found" in agent_response:
            if agent_response["is_duplicate_found"]:
                duplicate_ids = agent_response.get("duplicate_item_ids", [])
                logger.info(f"LLM found duplicates for item {target_id}: {duplicate_ids}")
                
                # Create a duplicate group with the target as representative
                all_ids = [target_id] + duplicate_ids
                representative_id = min(all_ids)  # Use smallest ID as representative
                
                # Check if any of these IDs are already in a group
                existing_group = None
                for rep_id, group_ids in self.duplicate_groups.items():
                    if any(id in group_ids for id in all_ids):
                        existing_group = rep_id
                        break
                
                if existing_group is not None:
                    # Merge with existing group
                    self.duplicate_groups[existing_group].update(all_ids)
                    # Mark duplicates for removal (keep representative)
                    for dup_id in all_ids:
                        if dup_id != existing_group:
                            self.removed_items.add(dup_id)
                    logger.info(f"Merged duplicates into existing group {existing_group}")
                else:
                    # Create new group
                    self.duplicate_groups[representative_id] = set(all_ids)
                    # Mark duplicates for removal (keep representative)
                    for dup_id in all_ids:
                        if dup_id != representative_id:
                            self.removed_items.add(dup_id)
                    logger.info(f"Created new duplicate group with representative {representative_id}")
            else:
                logger.info(f"LLM found no duplicates for item {target_id}")
        else:
            logger.warning(f"Unexpected agent response format for item {target_id}: {agent_response}")
        
        # Prepare results
        results = {
            "target_id": target_id,
            "target_item": target_item,
            "similar_items": similar_items,
            "agent_analysis": agent_response,
            "processed_at": str(np.datetime64('now')),
            "model_used": self.model_name,
            "embedding_model": "nomic-ai/nomic-embed-text-v1"
        }
        
        return results
    
    def process_item_for_duplicates(self, target_id: int) -> Dict[str, Any]:
        """
        Process a single item to find and analyze potential duplicates.
        
        Args:
            target_id: ID of the target item
            
        Returns:
            Dictionary containing processing results
        """
        return self._process_item_safely(target_id)
    
    def remove_duplicates_concurrent(self) -> List[Dict[str, Any]]:
        """
        Process all items concurrently and remove duplicates to get a final list of unique items.
        Excludes ALL items that are part of any duplicate group (including representatives).
        
        Returns:
            List of unique improvement items (excluding all items in duplicate groups)
        """
        logger.info(f"Starting sequential duplicate removal process for {len(self.flattened_items)} items")
        
        # Reset tracking
        self.processed_items.clear()
        self.removed_items.clear()
        self.duplicate_groups.clear()
        
        # Create list of all item IDs to process
        all_item_ids = list(range(len(self.flattened_items)))
        
        # Process items sequentially
        for item_id in all_item_ids:
            try:
                self._process_item_safely(item_id)
            except Exception as e:
                logger.error(f"Error processing item {item_id}: {e}")
        
        # Create set of all items that are part of duplicate groups (including representatives)
        all_duplicate_group_items = set()
        for rep_id, group_ids in self.duplicate_groups.items():
            all_duplicate_group_items.update(group_ids)
        
        # Create final list of unique items (exclude ALL items in duplicate groups)
        unique_items = []
        for item_id in range(len(self.flattened_items)):
            if item_id not in all_duplicate_group_items:
                unique_items.append(self.flattened_items[item_id])
        
        logger.info(f"Sequential duplicate removal completed.")
        logger.info(f"Kept {len(unique_items)} unique items out of {len(self.flattened_items)} total items")
        logger.info(f"Excluded {len(all_duplicate_group_items)} items that are part of duplicate groups")
        logger.info(f"Created {len(self.duplicate_groups)} duplicate groups")
        
        return unique_items
    
    def remove_duplicates(self) -> List[Dict[str, Any]]:
        """
        Legacy method - now calls the concurrent version.
        
        Returns:
            List of unique improvement items
        """
        return self.remove_duplicates_concurrent()
    
    def save_unique_items(self, unique_items: List[Dict[str, Any]], output_file: str):
        """
        Save the list of unique items to a JSON file.
        
        Args:
            unique_items: List of unique improvement items
            output_file: Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(unique_items, file, indent=2, ensure_ascii=False)
            logger.info(f"Unique items saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving unique items: {e}")
            raise
    
    def save_removed_items(self, output_file: str):
        """
        Save the list of removed duplicate items to a JSON file.
        Now includes ALL items from duplicate groups (including representatives).
        
        Args:
            output_file: Output file path
        """
        try:
            removed_items_list = []
            
            # Get all items that are part of duplicate groups (including representatives)
            all_duplicate_group_items = set()
            for rep_id, group_ids in self.duplicate_groups.items():
                all_duplicate_group_items.update(group_ids)
            
            for item_id in sorted(all_duplicate_group_items):
                item = self.flattened_items[item_id].copy()
                # Add metadata about why it was removed
                item['removal_reason'] = 'part_of_duplicate_group'
                item['original_id'] = item_id
                
                # Find which group it belongs to
                for rep_id, group_ids in self.duplicate_groups.items():
                    if item_id in group_ids:
                        item['duplicate_group_representative'] = rep_id
                        item['duplicate_group_members'] = sorted(list(group_ids))
                        item['is_representative'] = (item_id == rep_id)
                        # Add the representative item for easy comparison
                        item['representative_item'] = self.flattened_items[rep_id]
                        break
                
                removed_items_list.append(item)
            
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(removed_items_list, file, indent=2, ensure_ascii=False)
            logger.info(f"Removed items saved to {output_file}")
            logger.info(f"Saved {len(removed_items_list)} items from duplicate groups")
        except Exception as e:
            logger.error(f"Error saving removed items: {e}")
            raise
    
    def save_duplicate_groups(self, output_file: str):
        """
        Save the duplicate groups information to a JSON file.
        
        Args:
            output_file: Output file path
        """
        try:
            groups_info = {}
            for rep_id, group_ids in self.duplicate_groups.items():
                # Get all items in this group
                group_items = []
                for item_id in sorted(group_ids):
                    item = self.flattened_items[item_id].copy()
                    item['is_representative'] = (item_id == rep_id)
                    group_items.append(item)
                
                group_info = {
                    'representative_id': rep_id,
                    'representative_item': self.flattened_items[rep_id],
                    'group_members': sorted(list(group_ids)),
                    'group_size': len(group_ids),
                    'removed_members': sorted([id for id in group_ids if id != rep_id]),
                    'all_group_items': group_items  # Full content of all items in group
                }
                groups_info[f"group_{rep_id}"] = group_info
            
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(groups_info, file, indent=2, ensure_ascii=False)
            logger.info(f"Duplicate groups saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving duplicate groups: {e}")
            raise
    
    
    def generate_summary_report(self, original_count: int, unique_count: int) -> str:
        """
        Generate a summary report of the duplicate removal process.
        
        Args:
            original_count: Number of original items
            unique_count: Number of unique items after removal
            
        Returns:
            Formatted summary report
        """
        total_original_items = sum(len(items) for items in self.data.values())
        
        # Calculate items excluded from duplicate groups
        all_duplicate_group_items = set()
        for rep_id, group_ids in self.duplicate_groups.items():
            all_duplicate_group_items.update(group_ids)
        
        report = "=== IMPROVEMENTS DUPLICATE REMOVAL SUMMARY REPORT ===\n\n"
        
        report += f"Total Original Items: {total_original_items}\n"
        report += f"Compliant Items Excluded: {len(self.excluded_compliant_items)}\n"
        report += f"Non-Compliant Items Processed: {original_count}\n"
        report += f"Unique Items After Removal: {unique_count}\n"
        report += f"Items in Duplicate Groups (Excluded): {len(all_duplicate_group_items)}\n"
        report += f"Reduction Percentage: {(len(all_duplicate_group_items) / original_count * 100):.1f}%\n"
        report += f"Model Used: {self.model_name}\n"
        report += f"Embedding Model: nomic-ai/nomic-embed-text-v1\n"
        report += f"Concurrent API Calls: {self.max_concurrent_calls}\n"
        report += f"Analysis Framework: CrewAI with Reasoning Framework\n\n"
        
        report += f"Duplicate Groups Created: {len(self.duplicate_groups)}\n"
        for rep_id, group_ids in self.duplicate_groups.items():
            report += f"  Group {rep_id}: {sorted(group_ids)} (ALL excluded from unique items)\n"
        
        if self.excluded_compliant_items:
            report += f"\nCompliant Items Excluded (no improvements needed):\n"
            for i, item in enumerate(self.excluded_compliant_items[:10]):  # Show first 10
                report += f"  {i+1}. {item.get('ea_requirement', 'N/A')[:80]}...\n"
            if len(self.excluded_compliant_items) > 10:
                report += f"  ... and {len(self.excluded_compliant_items) - 10} more\n"
        
        return report

    def process_improvements_and_save_all_outputs(self, input_json_path='outputs/formatted_improvements.json', output_dir='outputs/duplicated_group_categorizer_crew'):
        """
        Process improvements JSON, remove duplicates, and save all outputs to the specified directory.
        Creates the output directory if it does not exist.

        Args:
            input_json_path: Path to the input improvements JSON file
            output_dir: Directory to save all output files (created if it doesn't exist)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        self.json_file_path = input_json_path

        # Load and flatten the JSON data
        self.data = self._load_json_data()
        self.flattened_items = self._flatten_data()
        self.item_embeddings = None

        # Process a single item for duplicates (optional, for demonstration)
        result = self.process_item_for_duplicates(0)
        # Optionally print or log result if needed

        # Remove all duplicates concurrently and get unique items
        unique_items = self.remove_duplicates_concurrent()

        # Prepare output file paths
        unique_items_path = os.path.join(output_dir, "unique_improvements.json")
        removed_items_path = os.path.join(output_dir, "removed_duplicates.json")
        duplicate_groups_path = os.path.join(output_dir, "duplicate_groups.json")
        summary_path = os.path.join(output_dir, "duplicate_removal_summary.txt")

        # Save all outputs
        self.save_unique_items(unique_items, unique_items_path)
        self.save_removed_items(removed_items_path)
        self.save_duplicate_groups(duplicate_groups_path)

        # Generate and save summary report
        summary = self.generate_summary_report(len(self.flattened_items), len(unique_items))
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        # Optionally return a summary dict
        return {
            "unique_items_path": unique_items_path,
            "removed_items_path": removed_items_path,
            "duplicate_groups_path": duplicate_groups_path,
            "summary_path": summary_path
        }

if __name__ == "__main__":
    duplicated_group_categorizer = DuplicatedGroupCategorizer()
    duplicated_group_categorizer.process_improvements_and_save_all_outputs(input_json_path='outputs/formatted_improvements.json', output_dir='outputs/duplicated_group_categorizer_crew')