#!/usr/bin/env python3
"""
JSON to HTML Requirements Analysis Converter
Converts JSON requirements analysis data to formatted HTML report
"""

import json
import argparse
import os


def load_json_data(file_path):
    """Load and parse JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def validate_json_structure(data):
    """Validate that JSON has required structure."""
    if not isinstance(data, dict):
        return False, "JSON must be an object"
    
    if 'summary' not in data:
        return False, "Missing 'summary' field"
    
    if 'all_evaluations' not in data:
        return False, "Missing 'all_evaluations' field"
    
    required_summary_fields = [
        'total_requirements_analyzed', 'total_addressed', 
        'total_contradicted', 'total_partially_addressed', 
        'total_not_found', 'contradiction_percentage'
    ]
    
    for field in required_summary_fields:
        if field not in data['summary']:
            return False, f"Missing summary field: {field}"
    
    return True, "Valid"


def group_requirements_by_status(evaluations):
    """Group requirements by their status."""
    groups = {
        'Addressed': [],
        'Partially Addressed': [],
        'Contradicted': [],
        'Not Found': []
    }
    
    # Status mapping for both English and Arabic
    status_mapping = {
        # English statuses
        'Addressed': 'Addressed',
        'Partially Addressed': 'Partially Addressed',
        'Contradicted': 'Contradicted',
        'Not Found': 'Not Found',
        # Arabic statuses
        'تمت معالجته': 'Addressed',
        'تمت معالجتها': 'Addressed',
        'تمت معالجته جزئيًا': 'Partially Addressed',
        'يتعارض': 'Contradicted',
        'غير موجود': 'Not Found'
    }
    
    for req in evaluations:
        status = req.get('status', 'Unknown')
        mapped_status = status_mapping.get(status, status)
        
        if mapped_status in groups:
            groups[mapped_status].append(req)
        else:
            print(f"Warning: Unknown status '{status}' for requirement {req.get('requirement_id', 'Unknown')}")
    
    return groups


def generate_html_template():
    """Generate the HTML template with embedded CSS and JavaScript."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Requirements Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}

        .header .timestamp {{
            opacity: 0.8;
            font-size: 0.9em;
        }}



        .tabs {{
            display: flex;
            background: #ecf0f1;
            border-bottom: 1px solid #bdc3c7;
        }}

        .tab {{
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            background: #ecf0f1;
            border: none;
            font-size: 1.1em;
            font-weight: 500;
            transition: all 0.3s ease;
            position: relative;
        }}

        .tab:hover {{
            background: #d5dbdb;
        }}

        .tab.active {{
            background: white;
            color: #2c3e50;
        }}

        .tab.active::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: #3498db;
        }}

        .tab-content {{
            display: none;
            padding: 30px;
            animation: fadeIn 0.5s ease-in;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .table-container {{
            padding: 0;
            overflow-x: auto;
        }}

        .requirements-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        }}

        .requirements-table thead {{
            background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
            color: white;
        }}

        .requirements-table th {{
            padding: 20px 15px;
            text-align: left;
            font-weight: 600;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .requirements-table td {{
            padding: 20px 15px;
            border-bottom: 1px solid #e9ecef;
            vertical-align: top;
            line-height: 1.6;
        }}

        .requirements-table tbody tr:hover {{
            background: #f8f9fa;
            transition: background 0.3s ease;
        }}

        .requirements-table tbody tr:last-child td {{
            border-bottom: none;
        }}

        .requirement-text-cell {{
            color: #2c3e50;
            font-weight: 500;
            font-size: 1em;
            line-height: 1.5;
            width: 50%;
        }}

        .justification-cell {{
            color: #34495e;
            line-height: 1.6;
            text-align: justify;
            width: 50%;
        }}



        .requirements-table.status-addressed thead {{
            background: linear-gradient(135deg, #27ae60 0%, #2c3e50 100%);
        }}
        
        .requirements-table.status-partially-addressed thead {{
            background: linear-gradient(135deg, #f39c12 0%, #2c3e50 100%);
        }}
        
        .requirements-table.status-contradicted thead {{
            background: linear-gradient(135deg, #e74c3c 0%, #2c3e50 100%);
        }}
        
        .requirements-table.status-not-found thead {{
            background: linear-gradient(135deg, #95a5a6 0%, #2c3e50 100%);
        }}

        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #7f8c8d;
        }}

        .empty-state .icon {{
            font-size: 4em;
            margin-bottom: 20px;
            opacity: 0.5;
        }}

        @media (max-width: 768px) {{
            .tabs {{
                flex-direction: column;
            }}
            
            .container {{
                margin: 10px;
                border-radius: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .requirements-table th,
            .requirements-table td {{
                padding: 12px 8px;
                font-size: 0.9em;
            }}
            
            .requirement-text-cell {{
                width: 50%;
            }}
            
            .justification-cell {{
                width: 50%;
            }}
            
            .table-container {{
                margin: 0 -15px;
                padding: 0 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Unscored requirement</h1>
        </div>



        {tabs_html}

        {tab_contents_html}
    </div>

    <script>
        function showTab(tabName) {{
            // Remove active class from all tabs and contents
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            // Add active class to selected tab and content
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }}
    </script>
</body>
</html>"""


def generate_requirement_row(requirement):
    """Generate HTML for a single requirement table row."""
    requirement_text = requirement.get('requirement_text', 'No text provided')
    justification = requirement.get('justification', 'No justification provided')
    
    # Escape HTML characters
    requirement_text = requirement_text.replace('<', '&lt;').replace('>', '&gt;')
    justification = justification.replace('<', '&lt;').replace('>', '&gt;')
    
    return f"""
        <tr>
            <td class="requirement-text-cell">{requirement_text}</td>
            <td class="justification-cell">{justification}</td>
        </tr>"""


def generate_tab_content(requirements, status_class, status_name):
    """Generate content for a tab."""
    if not requirements:
        return f"""
        <div class="empty-state">
            <div class="icon">📋</div>
            <h3>No requirements found</h3>
            <p>There are no requirements with "{status_name}" status.</p>
        </div>"""
    
    rows = [generate_requirement_row(req) for req in requirements]
    
    return f"""
    <div class="table-container">
        <table class="requirements-table {status_class}">
            <thead>
                <tr>
                    <th class="requirement-text-header">Requirement Text</th>
                    <th class="justification-header">Justification</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>"""


def convert_json_to_html(json_data, show_addressed=True, show_partially_addressed=True, show_contradicted=True, show_not_found=True):
    """Convert JSON data to HTML report content and return as string."""
    
    # Validate JSON structure
    is_valid, message = validate_json_structure(json_data)
    if not is_valid:
        print(f"Error: {message}")
        return None
    
    # Extract summary data
    summary = json_data['summary']
    evaluations = json_data['all_evaluations']
    
    # Group requirements by status
    grouped = group_requirements_by_status(evaluations)
    
    # Generate tab contents
    status_mapping = {
        'Addressed': 'status-addressed',
        'Partially Addressed': 'status-partially-addressed',  
        'Contradicted': 'status-contradicted',
        'Not Found': 'status-not-found'
    }
    
    tab_contents = {}
    for status, requirements in grouped.items():
        status_class = status_mapping.get(status, 'status-default')
        tab_contents[status] = generate_tab_content(requirements, status_class, status)
    
    # Generate dynamic tabs and tab contents based on visibility settings
    tabs_html = generate_tabs_html(grouped, show_addressed, show_partially_addressed, show_contradicted, show_not_found)
    tab_contents_html = generate_tab_contents_html(grouped, tab_contents, show_addressed, show_partially_addressed, show_contradicted, show_not_found)
    
    # Get HTML template
    html_template = generate_html_template()
    
    # Fill template with data
    html_content = html_template.format(
        tabs_html=tabs_html,
        tab_contents_html=tab_contents_html
    )
    
    return html_content


def generate_tabs_html(grouped, show_addressed, show_partially_addressed, show_contradicted, show_not_found):
    """Generate HTML for tabs based on visibility settings."""
    tab_configs = [
        ('Addressed', 'addressed', 'Addressed', show_addressed),
        ('Partially Addressed', 'partially-addressed', 'Partially Addressed', show_partially_addressed),
        ('Contradicted', 'contradicted', 'Contradicted', show_contradicted),
        ('Not Found', 'not-found', 'Not Found', show_not_found)
    ]
    
    tabs = []
    first_tab = True
    
    for status, tab_id, display_name, should_show in tab_configs:
        if should_show:  # Only show tabs that are enabled
            count = len(grouped.get(status, []))
            active_class = "active" if first_tab else ""
            tabs.append(f'<button class="tab {active_class}" onclick="showTab(\'{tab_id}\')">{display_name} ({count})</button>')
            first_tab = False
    
    if not tabs:  # If no tabs are enabled, show a default message
        tabs.append('<button class="tab active" onclick="showTab(\'no-data\')">No Tabs Enabled</button>')
    
    return f'<div class="tabs">{"".join(tabs)}</div>'


def generate_tab_contents_html(grouped, tab_contents, show_addressed, show_partially_addressed, show_contradicted, show_not_found):
    """Generate HTML for tab contents based on visibility settings."""
    tab_configs = [
        ('Addressed', 'addressed', 'Addressed', show_addressed),
        ('Partially Addressed', 'partially-addressed', 'Partially Addressed', show_partially_addressed),
        ('Contradicted', 'contradicted', 'Contradicted', show_contradicted),
        ('Not Found', 'not-found', 'Not Found', show_not_found)
    ]
    
    contents = []
    first_content = True
    
    for status, tab_id, display_name, should_show in tab_configs:
        if should_show:  # Only show content for tabs that are enabled
            active_class = "active" if first_content else ""
            contents.append(f'<div id="{tab_id}" class="tab-content {active_class}">{tab_contents[status]}</div>')
            first_content = False
    
    if not contents:  # If no content is enabled, show a default message
        contents.append('''
        <div id="no-data" class="tab-content active">
            <div class="empty-state">
                <div class="icon">📋</div>
                <h3>No Tabs Enabled</h3>
                <p>No tabs are currently enabled for display.</p>
            </div>
        </div>''')
    
    return "".join(contents)


def save_html_to_file(html_content, output_file):
    """Save HTML content to file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(html_content)
        return True
    except Exception as e:
        print(f"Error writing HTML file: {e}")
        return False


def convert_json_to_html_report(input_file, output_file=None, verbose=False, show_addressed=True, show_partially_addressed=True, show_contradicted=True, show_not_found=True):
    """
    Convert JSON requirements analysis to HTML report.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str, optional): Output HTML file path. If None, auto-generates name.
        verbose (bool): Enable verbose output
        show_addressed (bool): Whether to show the Addressed tab
        show_partially_addressed (bool): Whether to show the Partially Addressed tab
        show_contradicted (bool): Whether to show the Contradicted tab
        show_not_found (bool): Whether to show the Not Found tab
    
    Returns:
        tuple: (success: bool, html_content: str or None)
    """
    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return False, None
    
    # Determine output file name
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_report.html"
    
    if verbose:
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print("Loading JSON data...")
    
    # Load JSON data
    json_data = load_json_data(input_file)
    if json_data is None:
        return False, None
    
    if verbose:
        print(f"Loaded {len(json_data.get('all_evaluations', []))} requirements")
        print("Converting to HTML...")
    
    # Convert to HTML with tab visibility settings
    html_content = convert_json_to_html(json_data, show_addressed, show_partially_addressed, show_contradicted, show_not_found)
    
    if html_content is not None:
        # Optionally save to file if output_file is provided
        if output_file:
            save_success = save_html_to_file(html_content, output_file)
            if save_success:
                print(f"✅ HTML report generated successfully: {output_file}")
            else:
                print("⚠️ HTML content generated but failed to save to file")
        
        print(f"📊 Summary: {json_data['summary']['total_requirements_analyzed']} requirements analyzed")
        return True, html_content
    else:
        print("❌ Failed to generate HTML report")
        return False, None


def main():
    """Main function for command line usage (backward compatibility)."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert JSON requirements analysis to HTML report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python json_to_html.py input.json
  python json_to_html.py input.json -o report.html
  python json_to_html.py requirements.json --output analysis_report.html
        """
    )
    
    parser.add_argument('input_file', 
                       help='Path to input JSON file')
    parser.add_argument('-o', '--output', 
                       help='Output HTML file path (default: input_file_report.html)')
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Call the main conversion function
    success, html_content = convert_json_to_html_report(args.input_file, args.output, args.verbose)
    return 0 if success else 1


# Driver function for use in other scripts
def generate_html_report(input_json_file, output_html_file=None, verbose=False, show_addressed=True, show_partially_addressed=True, show_contradicted=True, show_not_found=True):
    """
    Driver function to generate HTML report from JSON requirements analysis.
    
    Args:
        input_json_file (str): Path to the JSON file containing requirements analysis
        output_html_file (str, optional): Path for the output HTML file. 
                                        If None, only returns HTML content without saving.
        verbose (bool): Whether to print verbose output during conversion
        show_addressed (bool): Whether to show the Addressed tab
        show_partially_addressed (bool): Whether to show the Partially Addressed tab
        show_contradicted (bool): Whether to show the Contradicted tab
        show_not_found (bool): Whether to show the Not Found tab
    
    Returns:
        tuple: (success: bool, html_content: str or None, output_file_path: str or None)
    """
    try:
        success, html_content = convert_json_to_html_report(input_json_file, output_html_file, verbose, show_addressed, show_partially_addressed, show_contradicted, show_not_found)
        if success:
            # Determine the actual output file path if one was provided
            actual_output_file = None
            if output_html_file is not None:
                if output_html_file == "":  # Empty string means auto-generate
                    base_name = os.path.splitext(input_json_file)[0]
                    actual_output_file = f"{base_name}_report.html"
                else:
                    actual_output_file = output_html_file
            
            return True, html_content, actual_output_file
        else:
            return False, None, None
    except Exception as e:
        print(f"Error in generate_html_report: {e}")
        return False, None, None


def get_html_content_only(input_json_file, verbose=False, show_addressed=True, show_partially_addressed=True, show_contradicted=True, show_not_found=True):
    """
    Generate HTML content from JSON without saving to file.
    
    Args:
        input_json_file (str): Path to the JSON file containing requirements analysis
        verbose (bool): Whether to print verbose output during conversion
        show_addressed (bool): Whether to show the Addressed tab
        show_partially_addressed (bool): Whether to show the Partially Addressed tab
        show_contradicted (bool): Whether to show the Contradicted tab
        show_not_found (bool): Whether to show the Not Found tab
    
    Returns:
        str or None: HTML content string if successful, None if failed
    """
    success, html_content, _ = generate_html_report(input_json_file, None, verbose, show_addressed, show_partially_addressed, show_contradicted, show_not_found)
    return html_content if success else None


if __name__ == "__main__":
    # Define your input parameters here
    input_json_file = "outputs_proposal_eval/unscored_requirements_analysis.json"  # Change this to your JSON file path
    output_html_file = "outputs_proposal_eval/proposal_evaluation_contradicted_report.html"  # None for no file saving, or specify custom path like "my_report.html"
    verbose = True  # Set to True for detailed output, False for minimal output
    
    # Tab visibility settings - set to True to show, False to hide
    show_addressed = True
    show_partially_addressed = True
    show_contradicted = True
    show_not_found = True
    
    # Call the driver function with tab visibility settings
    success, html_content, output_file = generate_html_report(
        input_json_file, 
        output_html_file, 
        verbose,
        show_addressed,
        show_partially_addressed,
        show_contradicted,
        show_not_found
    )
    
    if success:
        if output_file:
            print(f"✅ Report successfully generated and saved at: {output_file}")
        else:
            print("✅ HTML content generated successfully (not saved to file)")
        
        # Now you can use html_content variable for further processing
        print(f"📏 HTML content length: {len(html_content)} characters")
        
        # Example: You could return or use html_content here
        # return html_content  # If this was inside a function
        
    else:
        print("❌ Failed to generate report")
        exit(1)