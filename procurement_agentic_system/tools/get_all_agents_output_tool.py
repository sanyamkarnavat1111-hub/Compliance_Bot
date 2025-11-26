from crewai.tools import BaseTool
import os
import json
import re
from typing import Dict, List, Any

class GetAgentsOutput(BaseTool):
    name: str = "GetAgentsOutput"
    description: str = "Retrieves the content of all agent output files from the outputs directory."

    def extract_valid_entries(self, input_str, filename):
        try:
            # Extract JSON array using a regex that captures the entire list of dicts
            json_match = re.search(r'\[\s*{.*?}\s*]', input_str, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON array found in input.")
            
            json_str = json_match.group(0)

            # Try parsing the JSON string
            data = json.loads(json_str)

            return data

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing input: {e} in {filename}")
            return []

    def _run(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Read all JSON files from the outputs directory and return their contents.
        Returns a dictionary with filenames (without extension) as keys and file contents as values.
        """
        output_dir = "outputs"
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
                    data = f.read()
                    data = self.extract_valid_entries(data, filename)
                
                # Use filename without extension as the key
                key = os.path.splitext(filename)[0]
                results[key] = data if isinstance(data, list) else [data]
                
            except (json.JSONDecodeError, OSError) as e:
                print(f"Error reading {filename}: {e}")
                continue

        return results

if __name__ == "__main__":
    tool = GetAgentsOutput()
    print(json.dumps(tool._run(), indent=2))