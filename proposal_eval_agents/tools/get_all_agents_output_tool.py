from crewai.tools import BaseTool
import os
import json
import re
from typing import Dict, List, Any

class GetAgentsOutput(BaseTool):
    name: str = "GetAgentsOutput"
    description: str = "Retrieves the content of all agent output files from the outputs directory."

    def extract_valid_entries(self, input_str: str, filename: str):
        """
        Extracts a valid JSON array or a single JSON object from the input string.
        It prioritizes finding an array, but falls back to finding an object.
        """
        try:
            # MODIFICATION: First, try to find a JSON array (list of objects)
            json_match = re.search(r'\[\s*{.*?}\s*]', input_str, re.DOTALL)

            # MODIFICATION: If no array is found, try to find a single JSON object
            if not json_match:
                json_match = re.search(r'\{\s*".*?":.*?\s*\}', input_str, re.DOTALL)

            if not json_match:
                raise ValueError("No valid JSON array or object found in input.")
            
            json_str = json_match.group(0)

            # Try parsing the extracted JSON string
            data = json.loads(json_str)

            return data

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing input: {e} in {filename}")
            return None # Return None to indicate failure

    def _run(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Read all JSON files from the outputs directory and return their contents.
        Returns a dictionary with filenames (without extension) as keys and a list of file contents as values.
        """
        output_dir = "outputs_proposal_eval/rag_eval_crew_proposal_eval"
        results = {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            return {}

        for filename in os.listdir(output_dir):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    data = self.extract_valid_entries(content, filename)
                
                # If parsing failed or file was empty, skip it
                if data is None:
                    continue

                key = os.path.splitext(filename)[0]
                
                # MODIFICATION: Ensure the final output is always a list for consistency
                if isinstance(data, list):
                    results[key] = data
                else:
                    # If a single object was returned, wrap it in a list
                    results[key] = [data]
                
            except OSError as e:
                print(f"Error reading {filename}: {e}")
                continue

        return results

if __name__ == "__main__":
    tool = GetAgentsOutput()
    print(json.dumps(tool._run(), indent=2))