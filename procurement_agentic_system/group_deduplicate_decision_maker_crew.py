import json
from typing import Dict, List, Any
from crewai.llm import LLM
from crewai import Agent, Crew, Process, Task
import os
from dotenv import load_dotenv
import sys
import argparse
from posprocessing import group_and_clean_json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

class GroupDeduplicateDecisionMaker:
    def __init__(self):
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

    def _call_duplicate_cluster_refiner(self, all_group_items: List[Dict]) -> Dict[str, Any]:
        """
        Call the AI agent to refine a group of items, identifying true duplicates for removal,
        and prioritizing the most comprehensive item using a generalized approach.
        
        Args:
            all_group_items: A list of items that have been pre-sorted as a potential group of duplicates.
            
        Returns:
            Agent's response as a dictionary containing a list of IDs to remove.
        """
        
        # Create the prompt internally
        # The prompt is now generalized, using abstract examples instead of specific ones from the data.
        prompt = f"""
You are an expert analyst with a meticulous eye for technical detail. Your task is to analyze a provided group of items and identify which ones are **true, redundant duplicates**. The group was pre-sorted by a general semantic similarity model, so many items within it are likely to be non-duplicates.

Your primary goal is to identify and preserve all unique technical requirements. Only items that are a genuine duplicate of another item in the same group should be flagged for removal.

ITEMS TO ANALYZE:
"""
        
        for i, item in enumerate(all_group_items, 1):
            prompt += f"\n--- (ITEM ID: {item['id']}, Category: {item['category']}) ---\n"
            prompt += f"- EA Requirement: {item.get('ea_requirement', 'N/A')}\n"
            prompt += f"- Compliance Status: {item.get('compliance_status', 'N/A')}\n"
            prompt += f"- RFP Coverage: {item.get('rfp_coverage', 'N/A')}\n"
            prompt += f"- Gap Analysis: {item.get('gap_analysis', 'N/A')}\n"
        
        prompt += """

ANALYSIS INSTRUCTIONS:
1.  **Group by Core Requirement**: Scan all items and mentally group them by their core technical requirement. Use a keyword-first approach. For example, all items mentioning "OAuth2.0" should be in one group. All items mentioning "container orchestration" should be in another. Items with unique requirements will form their own group of one.

2.  **Identify Duplicates**: Within each mental group, a duplicate is an item that is fully captured by a more comprehensive or representative item in the same group.

3.  **Decision Framework for Selecting a Representative (CRITICAL)**:
    When a group of duplicates is identified, you **MUST** select the item that contains the most comprehensive information to be the representative.
    * **PRIORITY 1: SCOPE**: Look for the item whose `ea_requirement` and `gap_analysis` fields contain the most technical keywords. A broader scope that encompasses narrower requirements is ideal.
    * **PRIORITY 2: DETAIL**: If two items have similar scope, choose the one with the longest or most descriptive `rfp_coverage` or `gap_analysis` text.
    
    *General Examples (Do NOT use the above items for these examples):*
    -   **Example A (Subset/Superset)**: An item with requirement "Implement TLS 1.3" is a subset of an item with requirement "Ensure secure communication protocols including TLS 1.3, TLS 1.2, and SSL." The latter is more comprehensive and should be the representative.
    -   **Example B (Multiple Concepts)**: An item discussing "Data Security" is a duplicate of one that discusses "AES-256 encryption" and "Access Control," if the "Data Security" item explicitly mentions those specific concepts. If not, they are distinct.

4.  **Select for Removal**: For each group of duplicates, identify the most comprehensive item as the **representative**. All other items in that group should be listed for removal. Items that are in a group of one (non-duplicates) must NOT be listed for removal.

5.  **Final Output**: Provide a JSON object with a single key `ids_to_remove` containing a list of all IDs that should be removed from the original group. Also provide a detailed reasoning.

Respond with ONLY the JSON object, no additional text.

{
    "ids_to_remove": [list of item IDs that is mentioned in each dictionary],
    "reasoning": "Detailed explanation of which items form duplicate groups and why the most comprehensive items were selected as representatives."
}
"""
        
        # The agent and task definitions are already general and do not need to be changed.
        # Create CrewAI agent for duplicate cluster refinement
        refiner_agent = Agent(
            role="Duplicate Cluster Refinement Expert",
            goal="Analyze a group of potentially similar items and accurately identify and list only the true duplicates to be removed, prioritizing the most comprehensive item in each group.",
            backstory="A meticulous technical analyst who specializes in data normalization. You excel at distinguishing between genuinely different technical requirements and redundant information, always prioritizing the preservation of the most comprehensive knowledge.",
            llm=self.llm,
            llm_kwargs=self.llm_kwargs,
            reasoning=True,
            verbose=True,
            max_reasoning_attempts=5,
            max_iterations=5
        )
        
        # Create CrewAI task for duplicate refinement
        refiner_task = Task(
            description=prompt,
            expected_output="A JSON object with a list of IDs to remove and a detailed reasoning.",
            agent=refiner_agent,
            context=[{
                "name": "refinement_analysis",
                "description": "Refining a group of similar items to identify true duplicates.",
                "role": "system",
                "content": (
                    "You are an expert analyst performing duplicate refinement. Your task is to apply a keyword-first, "
                    "group-and-filter methodology to identify redundant items.\n\n"
                    "SYSTEM REQUIREMENTS:\n"
                    "1. Respond with ONLY valid JSON format.\n"
                    "2. Use the exact format specified in the task description.\n"
                    "3. Provide detailed reasoning for your decisions.\n"
                    "4. Focus on the core technical requirements of each item, selecting the most comprehensive item as the representative."
                ),
                "expected_output": "JSON object with duplicate refinement results",
                "framework": "Duplicate Refinement Analysis",
                "evaluation_approach": (
                    "1. Mentally group items by core technical keywords.\n"
                    "2. Identify groups with more than one item.\n"
                    "3. Select the most comprehensive item from each duplicate group as the representative.\n"
                    "4. List the IDs of all other items in that group for removal.\n"
                    "5. Ensure JSON output compliance."
                )
            }]
        )
        
        # Create CrewAI crew
        refinement_crew = Crew(
            agents=[refiner_agent],
            tasks=[refiner_task],
            process=Process.sequential,
            manager_llm=self.llm,
            manager_llm_kwargs=self.llm_kwargs,
            model=self.model_name,
            verbose=True,
            full_output=True
        )
        
        # Execute the crew
        result = refinement_crew.kickoff()
        
        # Extract content and parse JSON
        if hasattr(result, 'raw'):
            content = result.raw
        elif hasattr(result, 'result'):
            content = result.result
        else:
            content = str(result)
        
        print(f"Raw CrewAI response: {content}")
        
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        json_str = content[start_idx:end_idx]
        parsed_response = json.loads(json_str)
        print(f"Parsed CrewAI response: {parsed_response}")
        return parsed_response
    
    def add_non_duplicate_items(self, duplicate_group_json_input, formated_improvements_json_input):
        # Load the JSON file
        with open(duplicate_group_json_input, "r") as f:
            groups_data = json.load(f)
        
        results = {}
        group_ids_to_remove = {}

        # Sequentially process groups for stability
        for group_key, group_info in groups_data.items():
            all_group_items = group_info.get("all_group_items", [])
            if not all_group_items:
                print(f"Warning: No 'all_group_items' found for {group_key}")
                continue
            print(f"Processing {group_key} with {len(all_group_items)} items...")
            try:
                result = self._call_duplicate_cluster_refiner(all_group_items)
                results[group_key] = result
                group_ids_to_remove[group_key] = result['ids_to_remove']
            except Exception as exc:
                print(f"{group_key} generated an exception: {exc}")

        # Append non duplicate items to the formatted improvements json file
        with open(formated_improvements_json_input, "r") as f:
            improvements_data = json.load(f)

        for group_key, group_ids in group_ids_to_remove.items():
            all_group_items = groups_data[group_key].get("all_group_items", [])
            ids_to_remove = set(group_ids)
            for item in all_group_items:
                if item["id"] not in ids_to_remove:
                    print(f"Appending non duplicate item: {item['id']}")
                    improvements_data.append(item)

        base_output_dir = os.path.join('outputs', 'group_deduplicate_decision_maker_crew')
        os.makedirs(base_output_dir, exist_ok=True)

        # Write updated improvements data to file
        formated_improvements_with_non_duplicate_items_path = os.path.join(base_output_dir, 'formated_improvements_with_non_duplicate_items.json')
        
        # Post processing the json file
        improvements_data =group_and_clean_json(improvements_data)
        
        with open(formated_improvements_with_non_duplicate_items_path, 'w') as f:
            json.dump(improvements_data, f, indent=2)

        # Save logs for traceability
        output_path = os.path.join(base_output_dir, 'results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print('The formatted improvements json file updated, appended non duplicate items')

        return improvements_data, formated_improvements_with_non_duplicate_items_path, f'{base_output_dir}/formated_improvements_with_non_duplicate_items_AR.json'

if __name__ == "__main__":
    deduplicater = GroupDeduplicateDecisionMaker()
    deduplicater.add_non_duplicate_items('outputs/duplicated_group_categorizer_crew/duplicate_groups.json', 'outputs/duplicated_group_categorizer_crew/unique_improvements.json')