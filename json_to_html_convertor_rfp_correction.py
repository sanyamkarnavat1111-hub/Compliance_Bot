import json
import html
import os
import re
import ast
import sys
import argparse

def parse_dict_string(dict_str):
    """
    Parse a dictionary string that may contain mixed quotes and Arabic text.
    """
    try:
        # First, try direct evaluation
        return eval(dict_str)
    except:
        pass
    
    try:
        # Try AST literal_eval for safety
        return ast.literal_eval(dict_str)
    except:
        pass
    
    try:
        # Manual parsing approach
        # Clean up the string for JSON parsing
        cleaned = dict_str.strip()
        
        # Replace single quotes around keys and simple values with double quotes
        # But be careful with Arabic text that might contain apostrophes
        
        # Pattern to match keys (assuming they're mostly englishlish or simple)
        key_pattern = r"'([^']*?)':"
        cleaned = re.sub(key_pattern, r'"\1":', cleaned)
        
        # Handle string values - this is tricky with Arabic text
        # We'll use a more conservative approach
        cleaned = manual_quote_replacement(cleaned)
        
        return json.loads(cleaned)
    except Exception as e:
        print(f"Dict parsing failed for: {dict_str[:100]}... Error: {e}")
        return {"error": f"Could not parse: {dict_str[:100]}..."}


def manual_quote_replacement(text):
    """
    Manually replace quotes in a more controlled way for Arabic text.
    """
    result = text
    
    # Handle nested dictionary values
    # Look for patterns like ': {' and replace quotes accordingly
    
    # First pass: handle simple string values
    # Pattern: ': 'value'' -> ': "value"'
    simple_value_pattern = r":\s*'([^']*?)'"
    
    def replace_simple_value(match):
        value = match.group(1)
        # If the value doesn't contain double quotes, we can safely replace
        if '"' not in value:
            return f': "{value}"'
        else:
            # Escape internal double quotes
            escaped_value = value.replace('"', '\\"')
            return f': "{escaped_value}"'
    
    result = re.sub(simple_value_pattern, replace_simple_value, result)
    
    return result

def json_to_html_table(json_data, language="english"):
    """
    Convert JSON compliance data to HTML table format with improved Arabic support.
    
    Args:
        json_data (dict or str): JSON data containing compliance analysis
        language (str): Language code ("english" or "arabic")
        
    Returns:
        str: HTML table representation
    """
    
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        if language == "arabic":
            data = extract_inner_json(json_data)
        else:
            data = json.loads(json_data)
    else:
        data = json_data

    if language == "english":
        language_based_header = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """
    else:
        language_based_header = """
        <!DOCTYPE html>
        <html lang="ar" dir="rtl">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """
    
    # Enhanced CSS for better Arabic support
    html_content = language_based_header + """
        <title>Compliance Analysis Report</title>
        <style>
            body {
                font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .category-header {
                background-color: #3498db;
                color: white;
                padding: 15px;
                margin: 20px 0 10px 0;
                border-radius: 5px;
                font-size: 18px;
                font-weight: bold;
                text-transform: uppercase;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                table-layout: fixed;
            }
            th {
                background-color: #34495e;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                word-wrap: break-word;
            }
            td {
                padding: 12px;
                border-bottom: 1px solid #ddd;
                vertical-align: top;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }
            tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            tr:hover {
                background-color: #e8f4f8;
            }
            .status-met {
                background-color: #d4edda;
                color: #155724;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            .status-not-met {
                background-color: #f8d7da;
                color: #721c24;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            .status-partially-met {
                background-color: #fff3cd;
                color: #856404;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            .priority-high, .priority-عالي, .priority-عالية {
                background-color: #dc3545;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }
            .priority-medium, .priority-متوسط, .priority-متوسطة {
                background-color: #fd7e14;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }
            .priority-low, .priority-منخفض, .priority-منخفضة {
                background-color: #28a745;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }
            .recommendation-cell {
                background-color: #f8f9fa;
                border-left: 4px solid #007bff;
                padding-left: 16px;
            }
            /* RTL specific styles */
            [dir="rtl"] th {
                text-align: right;
            }
            [dir="rtl"] .recommendation-cell {
                border-left: none;
                border-right: 4px solid #007bff;
                padding-left: 12px;
                padding-right: 16px;
            }
        </style>
    </head>
    """

    if language == "english":
        html_content += """
        <body>
            <div class="container">
                <h1>EA Standard Compliance Analysis Report</h1>
                <table>
                    <thead>
                        <tr>
                            <th style=\"width: 20%;\">EA Requirement</th>
                            <th style=\"width: 10%;\">Status</th>
                            <th style=\"width: 25%;\">RFP Coverage</th>
                            <th style=\"width: 20%;\">Gap Analysis</th>
                        </tr>
                    </thead>
                    <tbody>
        """
    else:
        html_content += """
        <body dir=\"rtl\">
            <div class="container">
                <h1>تقرير تحليل الامتثال لمعيار EA</h1>
                <table dir=\"rtl\">
                    <thead>
                        <tr>
                            <th style=\"width: 20%;\">متطلب EA</th>
                            <th style=\"width: 10%;\">الحالة</th>
                            <th style=\"width: 25%;\">تغطية RFP</th>
                            <th style=\"width: 20%;\">تحليل الفجوة</th>
                        </tr>
                    </thead>
                    <tbody>
        """
    
    # Collect and append all non-met requirements from all categories
    for category_name, requirements_list in data.items():
        if not requirements_list:
            continue
        
        # Parse each requirement
        parsed_requirements = []
        for req in requirements_list:
            if isinstance(req, str):
                try:
                    parsed_req = parse_dict_string(req)
                    if isinstance(parsed_req, dict):
                        parsed_requirements.append(parsed_req)
                except Exception as e:
                    print(f"Warning: Could not parse requirement: {req[:100]}... Error: {e}")
                    continue
            elif isinstance(req, dict):
                parsed_requirements.append(req)
        
        if not parsed_requirements:
            continue
        
        # Only add non-met requirements
        for req in parsed_requirements:
            if not isinstance(req, dict):
                continue
            status = req.get('compliance_status', 'Unknown')
            if status.lower() in ['met', 'fully met', 'full', 'fully', 'تم الامتثال بالكامل']:
                continue
            if status in ['Met', 'تم الامتثال']:
                status_class = 'status-met'
            elif status in ['Not Met', 'not met', 'not', 'غير متوافق']:
                status_class = 'status-not-met'
            elif status in ['Partially Met', 'partially met', 'partially', 'partial', 'متوافق جزئيًا']:
                status_class = 'status-partially-met'
            else:
                status_class = ''
            def format_value(value):
                if isinstance(value, list):
                    return ', '.join(str(item) for item in value)
                elif isinstance(value, dict):
                    return str(value)
                return str(value) if value is not None else ''
            html_content += f"""
                <tr>
                    <td>{html.escape(format_value(req.get('ea_requirement', '')))}</td>
                    <td><span class=\"{status_class}\">{html.escape(format_value(status))}</span></td>
                    <td>{html.escape(format_value(req.get('rfp_coverage', '')))}</td>
                    <td>{html.escape(format_value(req.get('gap_analysis', '')))}</td>
                </tr>
            """
    # Close table and HTML
    html_content += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
    """
    return html_content


def load_json_from_file(json_file_path):
    """
    Load JSON data from a file with error handling.
    
    Args:
        json_file_path (str): Path to JSON file
        
    Returns:
        dict: JSON data
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file {json_file_path}: {e}")
    except Exception as e:
        raise Exception(f"Error reading file {json_file_path}: {e}")


def save_html_file(json_input, output_filename='compliance_report.html', language="english"):
    """
    Convert JSON to HTML and save to file with improved Arabic support.
    
    Args:
        json_input (dict or str): JSON data or file path
        output_filename (str): Output HTML filename
        language (str): Language code ("english" or "arabic")
    """
    
    # Handle file input or direct JSON
    if isinstance(json_input, str) and os.path.exists(json_input):
        # It's a file path
        data = load_json_from_file(json_input)
    elif isinstance(json_input, str):
        # Parse the JSON string
        if language == "arabic":
            data = extract_inner_json(json_input)
        else:
            data = json.loads(json_input)
    else:
        data = json_input
    
    # Convert to HTML
    html_output = json_to_html_table(data, language=language)
    
    # Create folder if it doesn't exist
    folder_path = os.path.dirname(output_filename)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save to file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"HTML report saved to: {output_filename}")
    return html_output

def decode_nested_json(input_data):
    decoded = {}
    for key, val_list in input_data.items():
        decoded_vals = []
        for item in val_list:
            try:
                # First decode (string -> dict)
                decoded_item = json.loads(item)
                # In case item was doubly encoded, decode again
                if isinstance(decoded_item, str):
                    decoded_item = json.loads(decoded_item)
                decoded_vals.append(decoded_item)
            except json.JSONDecodeError as e:
                print(f"Error decoding item under '{key}': {e}")
                decoded_vals.append({"parse_error": str(e), "raw": item})
        decoded[key] = decoded_vals
    return decoded

def extract_inner_json(data_input):
    """
    Handles both raw JSON strings and dicts with nested JSON-encoded strings.
    """
    # Step 1: Load if input is a string
    if isinstance(data_input, str):
        try:
            data = json.loads(decode_nested_json(data_input))
        except json.JSONDecodeError as e:
            print("Error parsing JSON string:", e)
            raise
    elif isinstance(data_input, dict):
        data = data_input
    else:
        raise TypeError("Input must be a JSON string or Python dict")

    # Step 2: Parse nested JSON strings
    for key, value in data.items():
        new_list = []
        for item in value:
            if isinstance(item, str):
                try:
                    new_list.append(decode_nested_json(json.loads(item)))
                except json.JSONDecodeError:
                    # Try to clean and decode broken JSON string
                    try:
                        cleaned = item.replace('\\"', '"').replace('"{', '{').replace('}"', '}')
                        new_list.append(decode_nested_json(json.loads(cleaned)))
                    except Exception as e:
                        print(f"Failed to parse inner JSON for key '{key}': {e}")
                        new_list.append({"parse_error": str(e), "raw_content": item})
            else:
                new_list.append(item)
        data[key] = new_list

    return data


def main():
    """
    Main function to handle command line arguments and process JSON to HTML conversion.
    """
    parser = argparse.ArgumentParser(
        description='Convert JSON compliance data to HTML report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python python json_to_html_convertor.py input.json
  python python json_to_html_convertor.py input.json -o output.html
  python python json_to_html_convertor.py input.json -o output.html -l arabic
  python python json_to_html_convertor.py input.json --output reports/compliance.html --language english
        """
    )
    
    parser.add_argument('json_path', help='Path to JSON input file')
    parser.add_argument('-o', '--output', default='compliance_report.html', 
                       help='Output HTML filename (default: compliance_report.html)')
    parser.add_argument('-l', '--language', choices=['english', 'arabic'], default='english',
                       help='Report language (default: english)')
    
    args = parser.parse_args()
    
    try:
        # Validate input file
        if not os.path.exists(args.json_path):
            print(f"Error: JSON file '{args.json_path}' not found.")
            sys.exit(1)
        
        print(f"Processing JSON file: {args.json_path}")
        print(f"Output file: {args.output}")
        print(f"Language: {args.language}")
        
        # Process the conversion
        save_html_file(args.json_path, args.output, args.language)
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()