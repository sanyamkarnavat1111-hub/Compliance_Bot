import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import logging
from typing import Any, Dict, List
from logger_config import get_logger
from posprocessing import clean_and_report, extract_last_json

from dotenv import load_dotenv
from openai import OpenAI
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed
logger = get_logger(__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 2. The Main Translator Class with CrewAI Integration ---

class ArabicJsonTranslator:
    """
    Processes nested JSON structures by translating string values to Arabic
    using a CrewAI-powered workflow for translation and review.
    """
    def __init__(self, model="openrouter/qwen/qwen3-32b"):
        try:
            # IMPORTANT: Create a .env file in the same directory as this script
            # with the line: OPENROUTER_API_KEY="your_api_key_here"
            load_dotenv()
            self.api_key = os.getenv('OPENROUTER_API_KEY')
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not found. Please create a .env file.")

            self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
            self.model = model
            self.max_retries = 3
            self.delay_between_retries = 5  # seconds

            self.llm_kwargs = {"temperature": 0.2, "top_k": 1, "top_p": 1, "seed": 42}
            self.crew_llm = LLM(
                model=self.model, api_key=self.api_key, base_url="https://openrouter.ai/api/v1",
                temperature=self.llm_kwargs['temperature'], top_k=self.llm_kwargs['top_k'],
                top_p=self.llm_kwargs['top_p'], seed=self.llm_kwargs['seed']
            )
        except Exception as e:
            logger.error(f"Error initializing ArabicJsonTranslator: {e}")
            raise

    def map_compliance_status(self, status: str) -> str:
        """Maps English compliance status to Arabic."""
        lower_status = str(status).lower()
        if lower_status in ['partially met', 'partially', 'partial', 'needs_review']:
            return 'متوافق جزئيًا'
        elif lower_status in ['not met', 'not fully met', 'not fully', 'not full', 'not']:
            return 'غير متوافق'
        return status

    def translate_and_review_with_crewai(self, english_json_string: str) -> str:
        """Translate and review JSON string to Arabic using a CrewAI crew."""
        retries = 0
        while retries < self.max_retries:
            try:
                translator_agent = Agent(
                    role="Professional Arabic Translator",
                    goal=f"Translate the given English JSON string into Arabic, ensuring technical accuracy and adherence to all rules. The input JSON is: {english_json_string}",
                    backstory="You are an expert in translating complex technical documents into Arabic. You are meticulous about preserving technical terms and JSON structure.",
                    llm=self.crew_llm, reasoning=True, verbose=False, allow_delegation=False, max_iter=10
                )
                translator_task = Task(
                    description=f"""
                    Translate the following JSON string from English to Arabic.
                    
                    **IMPORTANT RULES FOR TRANSLATION**:
                    1.  **JSON Structure**: Keep all JSON keys in their original language (do not translate keys). Only translate the VALUES associated with the keys. Maintain exact JSON structure and formatting.
                    2.  **Character Set**: Use standard Arabic characters only. Leave numerals in English (not Arabic numerals).
                    3.  **Forbidden Content**: DO NOT USE any non-Arabic language (e.g., Chinese, Japanese, Korean, Russian) in values. If a value contains foreign characters (like 범, 漢, 例, etc.), REMOVE them entirely.
                    4.  **Technical Terms**: DO NOT transliterate or phonetically spell technical terms, programming languages, acronyms, product names, or standards. **KEEP these as they appear in Latin characters** (e.g., MySQL, Redis, JavaScript, ISO, API, Microservices, Cybersecurity, RFP, EA, Kubernetes, Docker, HTTP, FTP, SSL, TLS).
                    5.  **Consistency**: For important technical/business/compliance terms or acronyms that appear in the text, always use the **same Arabic translation every time** if translated. If the term is best left in English, keep it in English—do not translate.
                    6.  **Ambiguity**: If the meaning of a technical or business term is unclear or ambiguous, prefer leaving it in English (Latin characters) to prevent incorrect translation.
                    7.  **Quality**: Ensure proper Arabic grammar and word order for compound terms. Eliminate any fragmented, unclear, or corrupted text segments.
                    8.  **Output**: Return ONLY the properly formatted JSON with translated Arabic values. Do not add any comments, explanations, or extra text outside the JSON.

                    **EXAMPLE INPUT AND OUTPUT**:
                    Input: {{"key": "English text using MySQL and Redis", "status": "completed"}}
                    Output: {{"key": "نص عربي يستخدم MySQL و Redis", "status": "مكتمل"}}

                    Input: {{"key": ["monitoring_and_management", "high-level_design_document_requirements", "accepted_database_systems"]}}
                    Output: {{"key": ["المراقبة والإدارة" ,"متطلبات المستندات الخاصة بالتصميم على مستوى عالٍ" ,"أنظمة قواعد البيانات المقبولة"]}}

                    Input: {{"key": "Technical requirements in the RFP must follow EA standards.", "status": "completed"}}
                    Output: {{"key": "يجب أن تتبع المتطلبات الفنية في RFP معايير EA.", "status": "مكتمل"}}
                    
                    Input: {{"key": "Content management system with advanced security", "status": "active"}}
                    Output: {{"key": "نظام إدارة المحتوى مع أمان متقدم", "status": "نشط"}}

                    JSON TO TRANSLATE: {english_json_string}
                    """,
                    expected_output="A JSON string with values translated to Arabic, following all rules.",
                    agent=translator_agent
                )
                reviewer_agent = Agent(
                    role="Arabic Translation Quality Reviewer",
                    goal="Review an Arabic translation of a JSON string against its original English version and correct any errors.",
                    backstory="You are a detail-oriented linguistic expert specializing in technical content. You ensure precision and adherence to guidelines.",
                    llm=self.crew_llm,
                    verbose=False, allow_delegation=False, max_iter=10
                )
                reviewer_task = Task(
                    description=f"""
                    You are provided with an original English JSON string and its Arabic translation.
                    Your task is to thoroughly review the Arabic translation based on the original English text and the following rules.
                    
                    **REVIEW GUIDELINES**:
                    1.  **Accuracy**: Does the Arabic translation accurately convey the meaning of the English original?
                    2.  **Rule Adherence (CRITICAL)**:
                        * Are all JSON keys untouched (not translated)?
                        * Are only VALUES translated?
                        * Is the JSON structure perfectly maintained?
                        * Are only standard Arabic characters used (no foreign characters like 범, 漢, 例, etc. – if found, REMOVE them)?
                        * Are numerals left in English form?
                        * Are **technical terms, programming languages, acronyms, product names, and standards (e.g., MySQL, Redis, JavaScript, ISO, API, Microservices, Cybersecurity, RFP, EA, Kubernetes, Docker, HTTP, FTP, SSL, TLS)** left in their original **English (Latin characters)** form? This is a frequent mistake and must be corrected if found.
                        * Is the same Arabic translation used consistently for recurring concepts?
                        * Are there any fragmented, unclear, or corrupted text segments?
                        * Is proper Arabic grammar and word order maintained?
                    
                    **Input to Review**:
                    Original English JSON: {english_json_string}
                    Arabic Translation to Review: {{initial_arabic_json_output}} # This will be replaced by context

                    If any corrections are needed, provide the corrected, full Arabic JSON. If the translation is perfect, return the original Arabic translation as is.
                    
                    **EXAMPLE OF COMMON MISTAKE AND CORRECTION (addressing the "Mostly Accurate" issues)**:
                    Original English: {{"key": "Technical terms - 'Sailing'"}}
                    Incorrect Arabic Translation (example): {{"key": "مصطلحات فنية - 'الإبحار'"}} (Incorrectly translated "Sailing" which is a technical term here)
                    Correct Arabic Output: {{"key": "مصطلحات فنية - 'Sailing'"}} (Keeping 'Sailing' in English as a technical term)
                    
                    Original English: {{"key": "Using Kubernetes, Docker, and FTP for deployment."}}
                    Incorrect Arabic Translation (example): {{"key": "استخدام كوبرنيتيس، دوكر، وبروتوكول نقل الملفات للنشر."}} (Incorrectly transliterated Kubernetes, Docker, and translated FTP)
                    Correct Arabic Output: {{"key": "استخدام Kubernetes و Docker و FTP للنشر."}} (Keeping Kubernetes, Docker, and FTP in English)

                    Return ONLY the corrected or verified Arabic JSON. Do not include any explanations or extra text outside the JSON.
                    """,
                    expected_output="A final, accurate Arabic JSON string.",
                    agent=reviewer_agent, context=[translator_task]
                )
                translation_crew = Crew(
                    agents=[translator_agent, reviewer_agent], tasks=[translator_task, reviewer_task],
                    process=Process.sequential, manager_llm=self.crew_llm, verbose=False, full_output=True
                )
                crew_result = translation_crew.kickoff()
                final_arabic_json_raw = crew_result.result if hasattr(crew_result, 'result') else str(crew_result)
                
                # Extract clean JSON from the output
                clean_json = json.dumps(extract_last_json(final_arabic_json_raw))
                logger.info("CrewAI Translation Successful.")
                return clean_json
            except Exception as e:
                logger.error(f"Error in CrewAI translation attempt {retries + 1}: {e}")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.delay_between_retries)
        logger.error(f"Failed to translate after {self.max_retries} attempts.")
        return english_json_string

    def process_item(self, item: Any) -> Any:
        """Processes a single data item using the CrewAI workflow."""
        if isinstance(item, dict):
            if str(item.get('compliance_status')).lower() == 'met':
                logger.info(f"Skipping item due to 'compliance_status': 'Met'.")
                return item

            try:
                processed = item.copy()
                original_compliance_status = processed.pop('compliance_status', None)
                original_recommendation = processed.pop('recommendation', None)

                processed_content = {}
                if processed:
                    translated_json_string = self.translate_and_review_with_crewai(json.dumps(processed))
                    processed_content = json.loads(translated_json_string)

                if original_compliance_status is not None:
                    processed_content['compliance_status'] = self.map_compliance_status(original_compliance_status)
                if original_recommendation is not None:
                    # Recommendation also needs translation
                    rec_trans_str = self.translate_and_review_with_crewai(json.dumps({"recommendation": original_recommendation}))
                    processed_content['recommendation'] = json.loads(rec_trans_str).get('recommendation', original_recommendation)
                
                return processed_content
            except Exception as e:
                logger.error(f"Error processing item: {e}. Returning original item.")
                return item
        return item

    def _process_list_of_dicts(self, data_list: List[Dict]) -> List[Dict]:
        """Helper method to process a list of dictionaries sequentially (more stable)."""
        processed_list = []
        for idx, item in enumerate(data_list):
            try:
                processed_list.append(self.process_item(item))
            except Exception as e:
                logger.error(f"Error processing item in list (index {idx}): {e}")
                processed_list.append(item)
        return processed_list

    def process_nested_structure(self, data: Any) -> Any:
        """Recursively processes a nested data structure sequentially."""
        if isinstance(data, dict):
            if str(data.get('compliance_status')).lower() == 'met':
                return data

            special_keys = {'compliance_status', 'recommendation'}
            current_special_values = {k: data[k] for k in special_keys if k in data}
            remaining_data = {k: v for k, v in data.items() if k not in special_keys}

            simple_values = {}
            complex_values = {}
            for key, value in remaining_data.items():
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    complex_values[key] = value
                elif isinstance(value, dict):
                    complex_values[key] = value
                else:
                    simple_values[key] = value

            processed_results = {}
            if simple_values:
                try:
                    processed_results["simple_values"] = self.process_item(simple_values)
                except Exception as e:
                    logger.error(f"Error processing simple_values: {e}")
                    processed_results["simple_values"] = simple_values
            for key, value in complex_values.items():
                try:
                    if isinstance(value, list):
                        processed_results[key] = self._process_list_of_dicts(value)
                    else:  # dict
                        processed_results[key] = self.process_nested_structure(value)
                except Exception as e:
                    logger.error(f"Error processing key '{key}': {e}")
                    processed_results[key] = value

            final_result = processed_results.pop("simple_values", {})
            final_result.update(processed_results)

            if 'compliance_status' in current_special_values:
                final_result['compliance_status'] = self.map_compliance_status(current_special_values['compliance_status'])
            if 'recommendation' in current_special_values:
                rec_trans_str = self.translate_and_review_with_crewai(json.dumps({"recommendation": current_special_values['recommendation']}))
                final_result['recommendation'] = json.loads(rec_trans_str).get('recommendation')

            return final_result
        elif isinstance(data, list):
            return self._process_list_of_dicts(data)
        return self.process_item(data)

    def process_json_file_for_rfp_eval(self, input_file: str, output_file: str):
        """Processes a nested JSON file for translation."""
        try:
            logger.info(f"Starting to process file: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            processed_data = self.process_nested_structure(data)
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully processed and saved to: {output_file}")
            return processed_data
        except Exception as e:
            logger.error(f"Error processing JSON file: {e}", exc_info=True)
            return None

# --- 3. Main Execution Block ---

def main():
    """Main function to run the translation process."""
    input_json = "outputs/group_deduplicate_decision_maker_crew/formated_improvements_with_non_duplicate_items.json"
    output_json = "outputs/group_deduplicate_decision_maker_crew/arabic_formated_improvements_with_non_duplicate_items.json"
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    try:
        start_time = time.time()
        logger.info(f"Starting translation process at {time.ctime(start_time)}")
        
        translator = ArabicJsonTranslator()
        result = translator.process_json_file_for_rfp_eval(input_json, output_json)
        
        if result is not None:
            duration = time.time() - start_time
            logger.info(f"Translation completed successfully in {duration:.2f} seconds")
            logger.info(f"Output saved to: {os.path.abspath(output_json)}")
            return 0
        else:
            logger.error("Translation process failed")
            return 1
    except KeyboardInterrupt:
        logger.info("\nTranslation process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
