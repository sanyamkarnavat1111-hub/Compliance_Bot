from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Any, Optional, ClassVar
import json
import html
import logging

# Configure logging for traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JsonToHtmlTableInput(BaseModel):
    """Input schema for JsonToHtmlTable."""
    json_string: str = Field(
        ..., 
        description="JSON string containing elements as keys and lists of proposed improvements as values.",
        examples=[
            '''
            {
              "element 1": ["proposed improvement 1", "proposed improvement 2"],
              "element 2": ["proposed improvement 1", "proposed improvement 2"]
            }
            '''
        ]
    )

class JsonToHtmlTable(BaseTool):
    name: str = "JsonToHtmlTable"
    description: str = (
        "Converts a JSON string into an HTML table with two columns: 'Elements' and 'Proposed Improvements'. "
        "The JSON must be a dictionary where keys are elements and values are lists of proposed improvements."
    )
    args_schema: Type[BaseModel] = JsonToHtmlTableInput
    
    # HTML table template with RTL support
    HTML_TABLE_TEMPLATE: ClassVar[str] = """
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; direction: {direction};">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Category</th>
      <th>Improvements</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
"""

    def _run(self, json_string: str, direction: str = "ltr") -> str:
        """Convert a JSON string to a complete HTML table with optional RTL support."""
        try:
            # Parse JSON string
            json_data = json.loads(json_string)
            logger.info(f"Parsed JSON for HTML table conversion: {json_string[:100]}...")

            # Validate structure
            if not isinstance(json_data, dict):
                logger.error("Invalid JSON structure: Expected a dictionary.")
                return "<table><tr><td>Error</td><td>Invalid JSON structure: Expected a dictionary.</td></tr></table>"

            # Generate rows for each category and its improvements
            rows = []
            for category, improvements in json_data.items():
                # Escape category
                category_html = html.escape(category)
                
                # Process improvements
                if isinstance(improvements, list):
                    # Join improvements with <br> tags
                    improvements_html = "<br>".join([f"{i+1}. {html.escape(str(item))}" for i, item in enumerate(improvements)])
                else:
                    # Handle case where improvements is not a list
                    improvements_html = html.escape(str(improvements))
                
                # Create row
                row = f"<tr>\n      <td>{category_html}</td>\n      <td>{improvements_html}</td>\n    </tr>"
                rows.append(row)
            
            # Generate full table
            table_html = self.generate_full_table(rows, direction)
            logger.info(f"Generated HTML table: {table_html[:100]}...")
            return table_html

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return f"<table><tr><td>Error</td><td>Failed to parse JSON: {str(e)}</td></tr></table>"
        except Exception as e:
            logger.error(f"Error converting JSON to HTML table: {str(e)}")
            return f"<table><tr><td>Error</td><td>Error processing JSON: {str(e)}</td></tr></table>"
    
    def generate_full_table(self, rows: List[str], direction: str = "ltr") -> str:
        """Generate a complete HTML table using the specified template with optional RTL support."""
        return self.HTML_TABLE_TEMPLATE.format(rows="\n    ".join(rows), direction=direction)