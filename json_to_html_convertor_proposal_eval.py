## New

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
    except Exception as e:
        print(f"Dict parsing failed for: {dict_str[:100]}... Error: {e}")
    
    try:
        # Try AST literal_eval for safety
        return ast.literal_eval(dict_str)
    except Exception as e:
        print(f"AST literal_eval failed for: {dict_str[:100]}... Error: {e}")
    
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
        title_text = "Compliance Analysis Report"
    else:
        language_based_header = """
        <!DOCTYPE html>
        <html lang="ar" dir="rtl">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """
        title_text = "تقرير تحليل الامتثال"
    
    # Enhanced CSS for better Arabic support
    html_content = language_based_header + f"""
        <title>{title_text}</title>
        <style>
            body {{
                font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }}
            .category-header {{
                background-color: #3498db;
                color: white;
                padding: 15px;
                margin: 20px 0 10px 0;
                border-radius: 5px;
                font-size: 18px;
                font-weight: bold;
                text-transform: uppercase;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                table-layout: fixed;
            }}
            th {{
                background-color: #34495e;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                word-wrap: break-word;
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #ddd;
                vertical-align: top;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            tr:hover {{
                background-color: #e8f4f8;
            }}
            .status-met {{
                background-color: #d4edda;
                color: #155724;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
            .status-not-met {{
                background-color: #f8d7da;
                color: #721c24;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
            .status-partially-met {{
                background-color: #fff3cd;
                color: #856404;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
            .priority-high, .priority-عالي, .priority-عالية {{
                background-color: #dc3545;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }}
            .priority-medium, .priority-متوسط, .priority-متوسطة {{
                background-color: #fd7e14;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }}
            .priority-low, .priority-منخفض, .priority-منخفضة {{
                background-color: #28a745;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
            }}
            .recommendation-cell {{
                background-color: #f8f9fa;
                border-left: 4px solid #007bff;
                padding-left: 16px;
            }}
            /* RTL specific styles */
            [dir="rtl"] th {{
                text-align: right;
            }}
            [dir="rtl"] .recommendation-cell {{
                border-left: none;
                border-right: 4px solid #007bff;
                padding-left: 12px;
                padding-right: 16px;
            }}
        </style>
    </head>
    """

    if language == "english":
        html_content += """
        <body>
            <div class="container">
                <h1>EA Standard Compliance Analysis Report</h1>
        """
    else:
        html_content += """
        <body dir="rtl">
            <div class="container">
                <h1>تقرير تحليل الامتثال لمعيار EA</h1>
        """
    
    # Process each category
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
        
        # Check if there are any non-met requirements in this category
        has_non_met_requirements = any(
            req.get('compliance_status', 'Unknown').lower() not in ['met', 'fully met', 'full', 'fully', 'تم الامتثال بالكامل']
            for req in parsed_requirements
        )
        
        # Skip this category if all requirements are met
        if not has_non_met_requirements:
            continue
            
        # Add category header
        category_display = category_name.replace('_', ' ').title()
        html_content += f'<div class="category-header">{html.escape(category_display)}</div>\n'
        
        if language == "english":
            # Start table for this category
            html_content += """
            <table>
                <thead>
                    <tr>
                        <th style="width: 20%;">EA Requirement</th>
                        <th style="width: 10%;">Status</th>
                        <th style="width: 25%;">RFP Coverage</th>
                        <th style="width: 20%;">Gap Analysis</th>
                    </tr>
                </thead>
                <tbody>
            """
        else:
            # Start table for this category - Arabic
            html_content += """
            <table dir="rtl">
                <thead>
                    <tr>
                        <th style="width: 20%;">متطلب EA</th>
                        <th style="width: 10%;">الحالة</th>
                        <th style="width: 25%;">تغطية RFP</th>
                        <th style="width: 20%;">تحليل الفجوة</th>
                    </tr>
                </thead>
                <tbody>
            """
            
        # Process each requirement in the category
        for req in parsed_requirements:
            if not isinstance(req, dict):
                continue
                
            # Format compliance status
            status = req.get('compliance_status', 'Unknown')

            # Skip if fully met
            if status.lower() in ['met', 'fully met', 'full', 'fully', 'تم الامتثال بالكامل']:
                continue

            # Determine status class
            if status in ['Met', 'تم الامتثال']:
                status_class = 'status-met'
            elif status in ['Not Met', 'not met', 'not', 'غير متوافق']:
                status_class = 'status-not-met'
            elif status in ['Partially Met', 'partially met', 'partially', 'partial', 'متوافق جزئيًا']:
                status_class = 'status-partially-met'
            else:
                # It should be nothing if not found, as this would create the problem
                status_class = ''
            
            # Helper function to handle different value types
            def format_value(value):
                if isinstance(value, list):
                    return ', '.join(str(item) for item in value)
                elif isinstance(value, dict):
                    return str(value)
                return str(value) if value is not None else ''
            
            # Add table row with formatted values
            html_content += f"""
                <tr>
                    <td>{html.escape(format_value(req.get('ea_requirement', '')))}</td>
                    <td><span class="{status_class}">{html.escape(format_value(status))}</span></td>
                    <td>{html.escape(format_value(req.get('rfp_coverage', '')))}</td>
                    <td>{html.escape(format_value(req.get('gap_analysis', '')))}</td>
                </tr>
            """
        
        # Close table
        html_content += """
            </tbody>
        </table>
        """
    
    # Close HTML
    html_content += """
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

def process_scored_requirements(scored_data, aggregated_scored_data, language="english"):
    """
    Process scored requirements data and return HTML section.
    
    Args:
        scored_data (list): List of scored requirement dictionaries
        aggregated_scored_data (list): List of scored requirement dictionaries from aggregated analysis
        language (str): Language code ("english" or "arabic")
        
    Returns:
        str: HTML section for scored requirements
    """
    if not scored_data:
        return ""
    
    if language == "english":
        html_content = '<div class="section-header">Scored Requirements</div>'
    else:
        html_content = '<div class="section-header">المتطلبات المقيّمة</div>'
    
    if language == "english":
        html_content += """
        <table>
            <thead>
                <tr>
                    <th style="width: 25%;">RFP Requirement</th>
                    <th style="width: 25%;">Proposal Compliance</th>
                    <th style="width: 20%;">Justification</th>
                    <th style="width: 10%;">Secured Score</th>
                    <th style="width: 10%;">Requirement Score</th>
                </tr>
            </thead>
            <tbody>
        """
    else:
        html_content += """
        <table dir="rtl">
            <thead>
                <tr>
                    <th style="width: 25%;">متطلب RFP</th>
                    <th style="width: 25%;">امتثال العرض</th>
                    <th style="width: 20%;">التبرير</th>
                    <th style="width: 10%;">الدرجة المخصصة</th>
                    <th style="width: 10%;">درجة المتطلب</th>
                </tr>
            </thead>
            <tbody>
        """
    
    # Helper function to format values
    def format_value(value):
        if isinstance(value, list):
            return ', '.join(str(item) for item in value)
        elif isinstance(value, dict):
            return str(value)
        return str(value) if value is not None else ''
    
    for req in scored_data:
        assigned_score = req.get('assigned_score', 0)
        requirement_score = req.get('requirement_score', 0)
        
        # Determine score class
        if assigned_score >= requirement_score * 0.8:
            score_class = 'score-high'
        elif assigned_score >= requirement_score * 0.6:
            score_class = 'score-medium'
        else:
            score_class = 'score-low'
        
        html_content += f"""
            <tr>
                <td>{html.escape(format_value(req.get('rfp_requirement', '')))}</td>
                <td>{html.escape(format_value(req.get('proposal_compliance', '')))}</td>
                <td class="justification-cell">{html.escape(format_value(req.get('justification', '')))}</td>
                <td><span class="{score_class}">{assigned_score}</span></td>
                <td>{requirement_score}</td>
            </tr>
        """
    
    html_content += """
        </tbody>
    </table>
    """
    
    # Add technical analysis section from aggregated data
    if aggregated_scored_data and isinstance(aggregated_scored_data, dict):
        html_content += '<div class="technical-analysis-section">'
        if language == "english":
            html_content += '<h3>Technical Analysis</h3>'
        else:
            html_content += '<h3>التحليل التقني</h3>'
        
        # Add technical strengths
        technical_strengths = aggregated_scored_data.get('technical_strengths', [])
        if technical_strengths:
            if language == "english":
                html_content += '<div class="analysis-item"><h4>Technical Strengths:</h4><ul>'
            else:
                html_content += '<div class="analysis-item"><h4>نقاط القوة التقنية:</h4><ul>'
            for strength in technical_strengths:
                html_content += f'<li>{html.escape(str(strength))}</li>'
            html_content += '</ul></div>'
        
        # Add technical concerns
        technical_concerns = aggregated_scored_data.get('technical_concerns', [])
        if technical_concerns:
            if language == "english":
                html_content += '<div class="analysis-item"><h4>Technical Concerns:</h4><ul>'
            else:
                html_content += '<div class="analysis-item"><h4>المخاوف التقنية:</h4><ul>'
            for concern in technical_concerns:
                html_content += f'<li>{html.escape(str(concern))}</li>'
            html_content += '</ul></div>'
        
        # Add technical risks
        technical_risks = aggregated_scored_data.get('technical_risks', [])
        if technical_risks:
            if language == "english":
                html_content += '<div class="analysis-item"><h4>Technical Risks:</h4><ul>'
            else:
                html_content += '<div class="analysis-item"><h4>المخاطر التقنية:</h4><ul>'
            for risk in technical_risks:
                html_content += f'<li>{html.escape(str(risk))}</li>'
            html_content += '</ul></div>'
        
        # Add technical summary
        technical_summary = aggregated_scored_data.get('technical_summary', '')
        if technical_summary:
            if language == "english":
                html_content += f'<div class="analysis-item"><h4>Technical Summary:</h4>'
            else:
                html_content += f'<div class="analysis-item"><h4>الملخص التقني:</h4>'
            html_content += f'<p>{html.escape(str(technical_summary))}</p></div>'

        html_content += '</div>'
    
    return html_content

import html
from collections import defaultdict

def process_unscored_requirements(unscored_data, aggregated_unscored_data=None, language="english"):
    """
    Process unscored requirements data and return HTML section with separate tables for each status.
    """
    
    if not unscored_data:
        return ""

    # Define order and display names for known statuses
    status_order = ["addressed", "partially addressed", "contradicted","تمت معالجته","يتعارض","تمت معالجته جزئيًا"]
    status_labels = {
        "english": {
            "addressed": "Addressed",
            "partially addressed": "Partially Addressed",
            "contradicted": "Contradicted"
        },
        "arabic": {
            "addressed": "تمت معالجته",
            "partially addressed": "تمت معالجته جزئيًا",
            "contradicted": "متناقض"
        }
    }

    # Group by status (ignore "Not Found" or "غير موجود")
    grouped = defaultdict(list)
    for req in unscored_data:
        status = req.get("status", "").strip().lower()
        if status in ["not found", "غير موجود"]:
            continue
        grouped[status].append(req)

    html_content = '<div class="section-header">'
    html_content += "Unscored Requirements" if language == "english" else "المتطلبات غير المقيّمة"
    html_content += "</div>"

    # Render each status section
    for status_key in status_order:
        if status_key not in grouped:
            continue

        display_label = status_labels[language].get(status_key, status_key.capitalize())
        html_content += f'<h3>{display_label}</h3>'

        if language == "english":
            html_content += """
            <table>
                <thead>
                    <tr>
                        <th style="width: 50%;">Requirement</th>
                        <th style="width: 50%;">Context / Explanation</th>
                    </tr>
                </thead>
                <tbody>
            """
        else:
            html_content += """
            <table dir="rtl">
                <thead>
                    <tr>
                        <th style="width: 50%;">المتطلب</th>
                        <th style="width: 50%;">السياق / الشرح</th>
                    </tr>
                </thead>
                <tbody>
            """

        for req in grouped[status_key]:
            requirement = html.escape(req.get('requirement', ''))
            context = html.escape(req.get('context', ''))
            html_content += f"""
                <tr>
                    <td>{requirement}</td>
                    <td>{context}</td>
                </tr>
            """
        html_content += """
                </tbody>
            </table>
            <br>
        """

    # Optional technical analysis block
    if aggregated_unscored_data and isinstance(aggregated_unscored_data, dict):
        html_content += '<div class="technical-analysis-section">'
        html_content += '<h3>Technical Analysis</h3>' if language == "english" else '<h3>التحليل التقني</h3>'

        def render_list_section(title_en, title_ar, items):
            if not items:
                return ""
            html_snippet = f'<div class="analysis-item"><h4>{title_en if language == "english" else title_ar}:</h4><ul>'
            for item in items:
                html_snippet += f'<li>{html.escape(str(item))}</li>'
            html_snippet += '</ul></div>'
            return html_snippet

        html_content += render_list_section("Technical Strengths", "نقاط القوة التقنية", aggregated_unscored_data.get("technical_strengths", []))
        html_content += render_list_section("Technical Concerns", "المخاوف التقنية", aggregated_unscored_data.get("technical_concerns", []))
        html_content += render_list_section("Technical Risks", "المخاطر التقنية", aggregated_unscored_data.get("technical_risks", []))

        summary = aggregated_unscored_data.get("technical_summary", "")
        if summary:
            html_content += f'<div class="analysis-item"><h4>{"Technical Summary" if language == "english" else "الملخص التقني"}:</h4><p>{html.escape(str(summary))}</p></div>'

        html_content += '</div>'

    return html_content


def process_final_score(final_score_data, aggregated_final_score_data, language="english"):
    """
    Process final score data and return HTML section.
    
    Args:
        final_score_data (dict or int): Final score and analysis data, or just the score number
        aggregated_final_score_data (dict): Final score and analysis data from aggregated analysis
        language (str): Language code ("english" or "arabic")
        
    Returns:
        str: HTML section for final score and analysis
    """
    if not final_score_data:
        return ""
    
    if language == "english":
        html_content = '<div class="section-header">Final Score & Analysis</div>'
    else:
        html_content = '<div class="section-header">الدرجة النهائية والتحليل</div>'
    html_content += '<div class="final-analysis">'
    
    # Handle different types of final_score_data
    if isinstance(final_score_data, (int, float)):
        # Simple numeric score
        final_score = final_score_data
        if language == "english":
            html_content += f"""
                <div class="score-display">
                    <h3>Final Score: {final_score}/100</h3>
                </div>
            """
        else:
            html_content += f"""
                <div class="score-display">
                    <h3>الدرجة النهائية: {final_score}/100</h3>
                </div>
            """
    elif isinstance(final_score_data, dict):
        # Complex structure with analysis
        final_score = final_score_data.get('final_score_out_of_100', 0)
        if language == "english":
            html_content += f"""
                <div class="score-display">
                    <h3>Final Score: {final_score}/100</h3>
                </div>
            """
        else:
            html_content += f"""
                <div class="score-display">
                    <h3>الدرجة النهائية: {final_score}/100</h3>
                </div>
            """
    
    # Add technical analysis section from aggregated data
    if aggregated_final_score_data and isinstance(aggregated_final_score_data, dict):
        if language == "english":
            html_content += '<div class="analysis-item"><h4>Conclusion:</h4>'
        else:
            html_content += '<div class="analysis-item"><h4>الخلاصة:</h4>'
        
        # Add technical overall assessment
        technical_overall_assessment = aggregated_final_score_data.get('overall_assessment', [])
        if technical_overall_assessment:
            if language == "english":
                html_content += '<h5>Overall Assessment:</h5><ul>'
            else:
                html_content += '<h5>التقييم العام:</h5><ul>'
            for assessment in technical_overall_assessment:
                html_content += f'<li>{html.escape(str(assessment))}</li>'
            html_content += '</ul>'
        
        # Add technical recommendations
        technical_recommendation = aggregated_final_score_data.get('recommendation', [])
        if technical_recommendation:
            if language == "english":
                html_content += '<h5>Recommendations:</h5><ul>'
            else:
                html_content += '<h5>التوصيات:</h5><ul>'
            for rec in technical_recommendation:
                html_content += f'<li>{html.escape(str(rec))}</li>'
            html_content += '</ul>'
        
        # Add technical next steps
        technical_next_steps = aggregated_final_score_data.get('next_steps', [])
        if technical_next_steps:
            if language == "english":
                html_content += '<h5>Next Steps:</h5><ul>'
            else:
                html_content += '<h5>الخطوات التالية:</h5><ul>'
            for step in technical_next_steps:
                html_content += f'<li>{html.escape(str(step))}</li>'
            html_content += '</ul>'
        
        html_content += '</div>'
    
    html_content += '</div>'
    
    return html_content

def aggregate_proposal_evaluation(scored_data, unscored_data, final_score_data, aggregated_scored_data, aggregated_unscored_data, aggregated_final_score_data, language="english"):
    """
    Central function that aggregates all processed data into final HTML report.
    
    Args:
        scored_data (list): Processed scored requirements data
        unscored_data (list): Processed unscored requirements data  
        final_score_data (dict): Processed final score and analysis data
        aggregated_scored_data (list): Processed scored requirements data from aggregated analysis
        aggregated_unscored_data (list): Processed unscored requirements data from aggregated analysis
        aggregated_final_score_data (dict): Processed final score and analysis data from aggregated analysis
        language (str): Language code ("english" or "arabic")
        
    Returns:
        str: Complete HTML report
    """
    
    if language == "english":
        language_based_header = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """
        title_text = "Proposal Evaluation Report"
    else:
        language_based_header = """
        <!DOCTYPE html>
        <html lang="ar" dir="rtl">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """
        title_text = "تقرير تقييم العرض"
    
    # Enhanced CSS for proposal evaluation
    html_content = language_based_header + f"""
        <title>{title_text}</title>
        <style>
            body {{
                font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 40px;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #3498db;
            }}
            .section-header {{
                background-color: #3498db;
                color: white;
                padding: 15px;
                margin: 20px 0 10px 0;
                border-radius: 5px;
                font-size: 18px;
                font-weight: bold;
                text-transform: uppercase;
            }}
            .analysis-section {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 20px;
                margin: 20px 0;
            }}
            .analysis-item {{
                margin-bottom: 15px;
                padding: 15px;
                background-color: white;
                border-left: 4px solid #007bff;
                border-radius: 3px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .analysis-item h4 {{
                margin: 0 0 15px 0;
                color: #495057;
                font-size: 16px;
                font-weight: 600;
            }}
            .analysis-item h5 {{
                margin: 15px 0 8px 0;
                color: #6c757d;
                font-size: 14px;
                font-weight: 600;
            }}
            .analysis-item ul {{
                margin: 0;
                padding-left: 20px;
            }}
            .analysis-item li {{
                margin-bottom: 8px;
                line-height: 1.5;
            }}
            .analysis-item p {{
                margin: 8px 0;
                line-height: 1.5;
            }}
            .score-display {{
                text-align: center;
                background-color: #e8f5e8;
                border: 2px solid #28a745;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }}
            .score-display h3 {{
                color: #155724;
                margin: 0;
                font-size: 24px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                table-layout: fixed;
            }}
            th {{
                background-color: #34495e;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                word-wrap: break-word;
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #ddd;
                vertical-align: top;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            tr:hover {{
                background-color: #e8f4f8;
            }}
            .score-high {{
                background-color: #d4edda;
                color: #155724;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
            .score-medium {{
                background-color: #fff3cd;
                color: #856404;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
            .score-low {{
                background-color: #f8d7da;
                color: #721c24;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
            .justification-cell {{
                background-color: #f8f9fa;
                border-left: 4px solid #007bff;
                padding-left: 16px;
            }}
            .final-analysis {{
                background-color: #e3f2fd;
                border: 2px solid #2196f3;
                border-radius: 8px;
                padding: 20px;
                margin: 30px 0;
            }}
            .final-analysis h3 {{
                color: #1976d2;
                margin-top: 0;
            }}
            /* RTL specific styles */
            [dir="rtl"] th {{
                text-align: right;
            }}
            [dir="rtl"] .justification-cell {{
                border-left: none;
                border-right: 4px solid #007bff;
                padding-left: 12px;
                padding-right: 16px;
            }}
            [dir="rtl"] .analysis-item {{
                border-left: none;
                border-right: 4px solid #007bff;
            }}
        </style>
    </head>
    """

    if language == "english":
        html_content += """
        <body>
            <div class="container">
                <h1>Proposal Evaluation Report</h1>
        """
    else:
        html_content += """
        <body dir="rtl">
            <div class="container">
                <h1>تقرير تقييم العرض</h1>
        """
    
    # Process each section
    scored_html = process_scored_requirements(scored_data, aggregated_scored_data, language)
    unscored_html = process_unscored_requirements(unscored_data, aggregated_unscored_data, language)
    final_score_html = process_final_score(final_score_data, aggregated_final_score_data, language)
    
    # Combine all sections
    html_content += scored_html
    html_content += unscored_html
    html_content += final_score_html
    
    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content

def save_proposal_evaluation_html(json_input, analysis_json_path_aggregated, output_filename='outputs_proposal_eval/proposal_eval.html', language="english"):
    """
    Convert proposal evaluation JSON to HTML and save to file.
    
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
    
    if isinstance(analysis_json_path_aggregated, str) and os.path.exists(analysis_json_path_aggregated):
        # It's a file path
        analysis_data = load_json_from_file(analysis_json_path_aggregated)
    elif isinstance(analysis_json_path_aggregated, str):
        # Parse the JSON string
        if language == "arabic":
            analysis_data = extract_inner_json(analysis_json_path_aggregated)
        else:
            analysis_data = json.loads(analysis_json_path_aggregated)
    else:
        analysis_data = analysis_json_path_aggregated
    
    # Extract the three main data components
    scored_data = data.get('scored', [])
    unscored_data = data.get('unscored', [])
    final_score_data = data.get('final_score_out_of_100', 0)

    # Extract the three main data components from the aggregated analysis
    aggregated_scored_data = analysis_data.get('scored_requirements_analysis', [])
    aggregated_unscored_data = analysis_data.get('unscored_requirements_analysis', [])
    aggregated_final_score_data = analysis_data.get('final_analysis', {})
    
    # Aggregate all data into final HTML
    html_output = aggregate_proposal_evaluation(scored_data, unscored_data, final_score_data, aggregated_scored_data, aggregated_unscored_data, aggregated_final_score_data, language)
    
    # Create folder if it doesn't exist
    folder_path = os.path.dirname(output_filename)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save to file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"Proposal evaluation HTML report saved to: {output_filename}")
    return html_output, output_filename

def main():
    """
    Main function to handle command line arguments and process JSON to HTML conversion.
    """
    parser = argparse.ArgumentParser(
        description='Convert JSON data to HTML report (compliance or proposal evaluation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compliance Analysis Report
  python json_to_html_convertor.py input.json --mode compliance
  python json_to_html_convertor.py input.json -o outputs_proposal_eval/proposal_eval.html -l english
  
  # Proposal Evaluation Report
  python json_to_html_convertor.py input.json --mode proposal
  python json_to_html_convertor.py input.json -o outputs_proposal_eval/proposal_eval.html -l english --mode proposal
        """
    )
    
    parser.add_argument('json_path', help='Path to JSON input per section (scored, unscored, final_score_out_of_100) file')
    parser.add_argument('analysis_json_path_aggregated', help='Path to JSON aggregated json summary file')
    parser.add_argument('-o', '--output', default='outputs_proposal_eval/proposal_eval.html', 
                       help='Output HTML filename (default: outputs_proposal_eval/proposal_eval.html)')
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
        
        save_proposal_evaluation_html(args.json_path, args.analysis_json_path_aggregated, args.output, args.language)
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()