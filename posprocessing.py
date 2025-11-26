import re
import json
import re
from pprint import pprint
from typing import Any, Dict, List, Tuple
import json
from collections import defaultdict

def extract_list_of_dicts(text: str):
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON list found in the text")
    
    json_substr = text[start:end + 1]
    try:
        data = json.loads(json_substr)
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        else:
            raise ValueError("Extracted content is not a list of dictionaries")
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON content extracted") from e

def extract_clean_json(raw_text: str):
    try:
        # Clean the text first
        cleaned_text = raw_text.strip()
        
        # Remove code block markers if any
        cleaned_text = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = cleaned_text.strip()
        
        # Handle bullet points and other special characters
        cleaned_text = cleaned_text.replace('•', '-')
        
        # Parse the JSON
        data = json.loads(cleaned_text)
        
        # If we got here, parsing was successful
        return data
        
    except json.JSONDecodeError as e:
        # If direct parsing fails, try to extract JSON array
        try:
            match = re.search(r'(\[\s*\{.*\}\s*\])', cleaned_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                # Clean up common JSON issues
                json_str = json_str.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
                json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
                return json.loads(json_str)
        except:
            pass
            
        # If we're here, all attempts failed
        raise ValueError(f"Failed to parse JSON. Error: {str(e)}. Text start: {cleaned_text[:100]}...")

def extract_last_json(text: str) -> dict:
    """
    Pull the last {...} block out of *text* and return it as a Python object.
    Raises ValueError if no complete JSON object is found.
    """
    # 1) Find the last opening brace
    start = text.rfind('{')
    if start == -1:
        raise ValueError("No opening '{' found — is there JSON here?")

    # 2) Walk forward, tracking brace depth until we match the final closing brace
    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1          # slice end is non-inclusive
                break

    if end is None:
        raise ValueError("Unbalanced braces — incomplete JSON object.")

    json_fragment = text[start:end]

    # 3) Parse it (this is where json.loads() previously failed for you)
    try:
        return json.loads(json_fragment)
    except json.JSONDecodeError as exc:
        # If the “JSON-looking” fragment isn’t actually valid JSON
        raise ValueError(f"Found braces, but not valid JSON: {exc}") from exc

def post_process_html_file(file_path):
    with open(file_path, 'r') as f:
        html_data = f.read()

    pattern = r'<table.*?>.*?<\/table>'
    match = re.search(pattern, html_data, re.DOTALL)
    if match:
        updated_html_data = match.group(0)
    else:
        print("No match found")
        return

    with open(file_path, 'w') as f:
        f.write(updated_html_data)

def _clean_text(text: str) -> Tuple[str, List[str]]:
    _ALLOWED_CHAR_RE = re.compile(
    r'[A-Za-z0-9'                 # Latin letters & digits
    r'\u0660-\u0669'              # Arabic-Indic digits
    r'\u0600-\u06FF'              # Arabic
    r'\u0750-\u077F'              # Arabic Supplement
    r'\u08A0-\u08FF'              # Arabic Extended-A
    r'\s'                         # whitespace
    r'\.,;:!?\-/\\()\[\]{}"\'…'   # basic punctuation
    r'%@#$^&*+=|~<>'              # additional keyboard symbols
    r'_'                          # underscore
    r']'                          # closing bracket
  )

    # Regex to *capture* any run of disallowed chars so we can report it
    _BAD_RUN_RE = re.compile(
        r'[^A-Za-z0-9'
        r'\u0660-\u0669'
        r'\u0600-\u06FF'
        r'\u0750-\u077F'
        r'\u08A0-\u08FF'
        r'\s\.,;:!?\-/\\()\[\]{}"\'…'
        r'%@#$^&*+=|~<>'
        r'_'
        r']+'
    )

    removed_segments = _BAD_RUN_RE.findall(text)

    cleaned_chars = []
    last_kept_is_letter = False

    for ch in text:
        if _ALLOWED_CHAR_RE.match(ch):
            cleaned_chars.append(ch)
            last_kept_is_letter = ch.isalpha()
        else:
            # avoid gluing words if foreign char was between letters
            if last_kept_is_letter:
                cleaned_chars.append(' ')
            last_kept_is_letter = False

    cleaned = re.sub(r'\s{2,}', ' ', ''.join(cleaned_chars)).strip()
    return cleaned, removed_segments

def scrub_non_ar_en(data: Any,
                    path: List[str] = None,
                    report: Dict[str, List[str]] = None) -> Any:
    """
    Returns cleaned data *and* collects a report mapping dotted JSON paths
    to lists of removed substrings.
    """
    if path is None:
        path = []
    if report is None:
        report = {}

    if isinstance(data, dict):
        return {
            k: scrub_non_ar_en(v, path + [k], report)
            for k, v in data.items()
        }
    elif isinstance(data, (list, tuple)):
        cleaned_seq = [
            scrub_non_ar_en(v, path + [str(i)], report)
            for i, v in enumerate(data)
        ]
        return type(data)(cleaned_seq)
    elif isinstance(data, str):
        cleaned, removed = _clean_text(data)
        if removed:
            report['.'.join(path)] = removed
        return cleaned
    else:
        return data  # ints, floats, None, etc.

def clean_and_report(data: Any) -> Tuple[Any, Dict[str, List[str]]]:
    report: Dict[str, List[str]] = {}
    cleaned = scrub_non_ar_en(data, report=report)
    print(f'cleaned: {cleaned}, report of removed characters: {report}')
    return cleaned, report

def group_and_clean_json(data):
    # Initialize a dictionary to hold the grouped data
    grouped_data = defaultdict(list)

    # Iterate through each dictionary in the input data
    for item in data:
        # Remove 'id' and 'category' from each dictionary
        category = item.pop('category', None)  # Get category to group by
        item.pop('id', None)  # Remove 'id'
        
        # Group the items by category
        grouped_data[category].append(item)

    # Convert the grouped data to a regular dictionary (removes defaultdict)
    grouped_data = dict(grouped_data)

    return grouped_data

# ------------------------ DEMO ------------------------
if __name__ == "__main__":
    # sample = {
    #     "ea_requirement": "تعريف شامل ل범ويّة المشروع ومتطلباته",
    #     "rfp_coverage": (
    #         "يحتوي طلب العرض على متطلبات مفصلة للعروض التقنية لكنه يفتقر إلى "
    #         "إجراءات واضحة لإدارة البُعد. ويشير إلى حق الوزارة في تعديل "
    #         "تواريخ التسليم لكنه لا يحدد إجراءات تغيير البُعد أو تقييمات التأثير."
    #     ),
    #     "gap_analysis": "لا يوجد إجراء رسمي لتغييرات البُعد أو تحليل التأثير"
    # }

    # sample = {
    #     "ea_requirement": "تعريف شامل 尼尔ل범ويّة المشروع 닐ومتطلباته",
    #     "rfp_coverage": (
    #         "يحتوي طلب العرض على متطلبات مفصلة للعروض التقنية لكنه يفتقر إلى "
    #         "إجراءات واضحة لإدارة البُعد. ويشير إلى حق الوزارة في تعديل "
    #         "تواريخ التسليم لكنه لا يحدد إجراءات تغيير البُعد أو تقييمات التأثير."
    #     ),
    #     "gap_analysis": "لا يوجد إجراء رسمي لتغييرات البُعد أو تحليل التأثير"
    # }
    # clean_sample = clean_and_report(sample)
    # pprint(clean_sample[0])

    raw_text = '''
    ...everything before...
    {"some":"json"} some stray text
    {"key": 1}   /* another piece you don't want */
    """{"final": ["good", "json"], "ok": true}"""  # plus comments after
    ...and a note in Arabic...
    '''

    data = extract_last_json(raw_text)
    print(data)