import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Union, List, Optional
import logging
from posprocessing import clean_and_report, extract_last_json
from logger_config import get_logger



class ArabicJsonTranslator:
    # qwen/qwen3-32b
    # meta-llama/llama-3.1-8b-instruct
    # qwen/qwen2.5-vl-72b-instruct
    def __init__(self, model="qwen/qwen3-32b", logger_instance=None):
        try:
            load_dotenv()
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not found")
            self.logger = logger_instance if logger_instance is not None else get_logger(__name__)
            
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            self.model = model
            self.max_retries = 3
            self.delay_between_retries = 5  # seconds
        except Exception as e:
            print(f"Error initializing ArabicJsonTranslator: {e}")
            raise

    def translate_text(self, text: str) -> str:
        """Translate text to Arabic using the OpenAI API with retry logic."""
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"""You are an expert Arabic translator specializing in high-stakes technical, legal, and business documents, with a focus on Enterprise Architecture (EA) and Request for Proposals (RFP). Your translations must be precise, consistent, and adhere to professional standards.

                            **CRITICAL GUIDELINE: SCOPE OF WORK**
                            Your function is to translate the JSON values provided in the `{text}` input. You must work exclusively with the source text given.
                            - DO NOT add information that is missing from the source (e.g., if "Project Deliverables" are not in the input, do not invent them).
                            - DO NOT cross-reference external documents or standards (e.g., an "EA standard file").
                            - DO NOT identify or report gaps, duplicates, or omissions in the source content.
                            Your sole responsibility is to accurately translate the provided text.

                            **MANDATORY GLOSSARY & TERMINOLOGY RULES:**
                            You MUST use the following translations for these specific terms, without exception, to ensure consistency.

                            | English Term          | Approved Arabic/Technical Term | Notes                                                  |
                            |-----------------------|--------------------------------|--------------------------------------------------------|
                            | Blockchain            | Blockchain                     | Keep in English. Do not translate.                     |
                            | Cybersecurity         | الأمن السيبراني                | Use this exact phrase. Note the final "ي".             |
                            | Open Source           | مفتوح المصدر                   | Use this exact grammatical structure.                  |
                            | Microservices         | Microservices                  | Keep in English. Do not abbreviate or transliterate.   |
                            | Single Sign-On        | SSO                            | Use the English acronym "SSO".                         |
                            | Addressed             | تمت معالجته                    | Use this exact phrase for the 'status' value.          |
                            | Partially Addressed   | تمت معالجته جزئيًا              | Use this exact phrase for the 'status' value.          |
                            | Contradicted          | يتعارض                         | Use this exact phrase for the 'status' value.          |
                            | Not Found             | غير موجود                      | Use this exact phrase for the 'status' value.          |

                            **GENERAL TRANSLATION RULES:**
                            1.  **JSON Structure**: Absolutely maintain the original JSON structure. Translate **only the values**, never the keys.
                            2.  **Technical Names & Acronyms**: For all other technical terms, product names, programming languages, standards, or acronyms not in the glossary (e.g., MySQL, Redis, JavaScript, ISO, API, RFP, EA), **KEEP them in their original English (Latin) characters.** Do not attempt to translate or transliterate them.
                            3.  **Grammar & Syntax**:
                                *   Translations must be grammatically flawless in Arabic. Pay close attention to noun-adjective agreement. For instance, "Critical Services" is "الخدمات الحرجة", not "مخدمات الحرجة".
                                *   Ensure correct word order. For example, "Open Source Content" correctly translates to "محتوى مفتوح المصدر".
                            4.  **Character Sets**:
                                *   Use only standard Arabic characters for translated text.
                                *   All numerals must be kept in their English form (e.g., 1, 2, 3), not converted to Arabic numerals (e.g., ١, ٢, ٣).
                                *   If any value contains foreign characters from other languages (e.g., 범, 漢, 例), remove them completely.
                            5.  **Clarity**:
                                *   Avoid ambiguity. If a technical term's meaning is unclear, it is safer to keep it in English.
                                *   If you encounter garbled text or nonsensical artifacts (e.g., "ي ك في"), remove them to ensure the output is clean and professional.
                            6.  **No Extra Content**: Return **only** the translated JSON object. Do not add comments, explanations, apologies, or any text outside of the JSON structure.
                            7.  **Transliteration & Spelling**:
                                *   Do NOT transliterate or phonetically spell technical terms, programming languages, acronyms, product names, or standards. **KEEP these as they appear in Latin characters** (e.g., MySQL, Redis, JavaScript, ISO, API).
                                *   For important technical/business/compliance terms or acronyms that appear in the text, always use the **same Arabic translation every time**. If the term is best left in English, keep it in English—do not translate.
                                *   If you encounter multiple Arabic terms for a single English concept, always choose **one** and use it consistently throughout the output.
                                *   If a value contains foreign characters (like 범, 漢, 例, etc.), REMOVE them entirely.
                                *   Do not translate numerals to Arabic numerals; leave them in their original English form.
                                *   If the meaning of a technical or business term is unclear or ambiguous, prefer leaving it in English (Latin characters) to prevent incorrect translation.

                            ---
                            **SPECIAL INSTRUCTION FOR DISASTER RECOVERY PLAN PHRASES:**
                            If you encounter English phrases similar to:
                            - "The RFP mentions deliverables such as 'Project Plan / Project Schedule' and other documents, which can be interpreted to include the Disaster Recovery Plan."
                            or any sentence that implies the RFP lists documents or deliverables that could be interpreted as including a Disaster Recovery Plan, you must translate it to clear, professional Arabic that accurately reflects the nuance that the Disaster Recovery Plan is not explicitly mentioned, but could be inferred from the general mention of plans or documents. For example:
                            - "يذكر طلب العروض (RFP) مخرجات مثل 'خطة المشروع / جدول المشروع' وغيرها من الوثائق، والتي يمكن تفسيرها لتضمين خطة استعادة الكوارث."
                            Ensure the translation is faithful to the original meaning and does not add or omit any information.

                            ---
                            **TEXT TO TRANSLATE:** {text}

                            Return ONLY the properly formatted JSON with accurately translated Arabic values.
                            """
                        },
                    ],
                    temperature=0.2,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0.5
                )
                # response = self.client.chat.completions.create(
                #     model=self.model,
                #     messages=[
                #         {
                #             "role": "system",
                #             "content": f"""You are a professional Arabic translator specializing in technical, legal, and business content. You must ensure all terminology is consistent and accurate.

                #                 GENERAL TRANSLATION RULES:
                #                 1. Keep all JSON keys in their original language (do not translate keys).
                #                 2. Only translate the VALUES associated with the keys.
                #                 3. Maintain exact JSON structure and formatting.
                #                 4. Use standard Arabic characters only. Leave numerals in English (not Arabic numerals).
                #                 5. DO NOT USE any non-Arabic language (Chinese, Japanese, Korean, Russian, etc.) in values.
                #                 6. Do not add any comments, explanations, or extra text—return only the translated JSON.
                #                 7. Do NOT transliterate or phonetically spell technical terms, programming languages, acronyms, product names, or standards. **KEEP these as they appear in Latin characters** (e.g., MySQL, Redis, JavaScript, ISO, API).
                #                 8. For important technical/business/compliance terms or acronyms that appear in the text, always use the **same Arabic translation every time**. If the term is best left in English, keep it in English—do not translate.
                #                 9. If you encounter multiple Arabic terms for a single English concept, always choose **one** and use it consistently throughout the output.
                #                 10. If a value contains foreign characters (like 범, 漢, 例, etc.), REMOVE them entirely.
                #                 11. Do not translate numerals to Arabic numerals; leave them in their original English form.
                #                 12. If the meaning of a technical or business term is unclear or ambiguous, prefer leaving it in English (Latin characters) to prevent incorrect translation.


                #                 EXAMPLE FORMAT 1:
                #                 Input: {{"key": "English text using MySQL and Redis", "status": "completed"}}
                #                 Output: {{"key": "نص عربي يستخدم MySQL و Redis", "status": "مكتمل"}}

                #                 EXAMPLE FORMAT 2:
                #                 Input: {{"key": ["monitoring_and_management", "high-level_design_document_requirements", "accepted_database_systems"]}}
                #                 Output: {{"key": ["المراقبة والإدارة" ,"متطلبات المستندات الخاصة بالتصميم على مستوى عالٍ" ,"أنظمة قواعد البيانات المقبولة"]}}

                #                 EXAMPLE FORMAT 3:
                #                 Input: {{"key": "Technical requirements in the RFP must follow EA standards.", "status": "completed"}}
                #                 Output: {{"key": "يجب أن تتبع المتطلبات الفنية في RFP معايير EA.", "status": "مكتمل"}}

                #                 TEXT TO TRANSLATE: {text}

                #                 Return ONLY the properly formatted JSON with translated Arabic values. If a value contains foreign characters (like 범, 漢, 例, etc.), REMOVE them entirely."""
                #         },
                #     ],
                #     temperature=0.2,
                #     top_p=1,
                #     frequency_penalty=0,
                #     presence_penalty=0.5
                # )
                  
                translated_text = response.choices[0].message.content.strip()
                print(translated_text)
                return translated_text
                
            except Exception as e:
                print(f"Error in translation attempt {retries + 1}: {e}")
                last_error = e
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.delay_between_retries)
                    
        print(f"Failed to translate after {self.max_retries} attempts. Last error: {last_error}")
        return text  # Return original text as fallback

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item and translate its fields."""
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            try:
                # Create a copy of the item to avoid modifying the original
                processed = item.copy()
                
                try:
                    processed = self.translate_text(json.dumps(item))
                    if isinstance(processed, str):
                        try:
                            processed = json.loads(processed)
                        except json.JSONDecodeError as e:
                            try:
                                processed = extract_last_json(processed)
                            except Exception as e:
                                print(f"Error parsing translated JSON: {e}")
                                retries += 1
                                if retries < max_retries:
                                    time.sleep(self.delay_between_retries)
                                    continue
                                else:
                                    return item
                except Exception as e:
                    print(f"Error in translation process: {e}")
                    retries += 1
                    if retries < max_retries:
                        time.sleep(self.delay_between_retries)
                        continue
                    else:
                        return item
                
                return processed
                
            except Exception as e:
                print(f"Error processing item: {e}")
                retries += 1
                if retries < max_retries:
                    time.sleep(self.delay_between_retries)
                    continue
                else:
                    return item
        
        # If all retries failed, return original item
        return item

    def process_json_file_for_rfp_completeness_check(self, input_file: str, output_file: str):
        """Process a JSON file, translate its content, and save to a new file with progress tracking."""
        try:
            self.logger.info(f"Starting to process file: {input_file}")
            
            # Read the input JSON file
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading input file: {e}")
                return None
            
            processed_data = {}
            
            # Process category keys with retry logic
            max_retries = 3
            retries = 0
            arabic_keys = None
            
            while retries < max_retries and arabic_keys is None:
                try:
                    category_keys = list(data.keys())
                    print(category_keys)
                    
                    # Check if we need batch processing (more than 10 items)
                    if len(category_keys) > 10:
                        self.logger.info(f"Processing {len(category_keys)} category keys in batches of 10")
                        arabic_keys = []
                        
                        # Process in batches of 10
                        for i in range(0, len(category_keys), 10):
                            batch = category_keys[i:i+10]
                            self.logger.info(f"Processing batch {i//10 + 1}: {batch}")
                            
                            # Convert batch to a JSON-compatible string
                            keys_str = {"keys": batch}
                            batch_result = self.process_item(keys_str)
                            batch_arabic_keys = batch_result['keys']
                            
                            # Append batch results to main list
                            arabic_keys.extend(batch_arabic_keys)
                    else:
                        # Process all keys at once if 10 or fewer
                        keys_str = {"keys": category_keys}
                        arabic_keys = self.process_item(keys_str)
                        arabic_keys = arabic_keys['keys']
                    
                    print(arabic_keys)
                except Exception as e:
                    retries += 1
                    self.logger.error(f"Error processing category keys (attempt {retries}/{max_retries}): {e}")
                    if retries < max_retries:
                        time.sleep(self.delay_between_retries)
                    else:
                        self.logger.error("Failed to process category keys after all retries. Using original keys.")
                        arabic_keys = list(data.keys())

            # Process each category in the data
            for k, (category, items) in enumerate(data.items()):
                try:
                    self.logger.info(f"Processing category: {category} {arabic_keys[k]} {k}")
                    
                    if not isinstance(items, list):
                        processed_data[category] = items
                        continue
                    
                    processed_items = []
                    total_items = len(items)
                    
                    for i, item in enumerate(items, 1):
                        try:
                            # Store original compliance status for later use
                            original_compliance_status = item.get('compliance_status', '').lower()
                            
                            # Skip items that are fully met
                            if original_compliance_status in ['met', 'fully met', 'full', 'fully']:
                                continue
                            
                            self.logger.info(f"Processing item {i} of {total_items} in category '{category}'")

                            # Create a copy of the item for processing
                            item_for_translation = item.copy()
                            
                            # Remove compliance_status and recommendation from the input sent to LLM
                            item_for_translation.pop('compliance_status', None)
                            item_for_translation.pop('recommendation', None)
                            
                            # Process the item without compliance_status
                            processed_item = self.process_item(item_for_translation)
                            
                            # Add back the appropriate compliance status based on original value
                            if original_compliance_status in ['partially met', 'partially', 'partial']:
                                processed_item['compliance_status'] = 'متوافق جزئيًا'
                            elif original_compliance_status in ['not met', 'not fully met', 'not fully', 'not full', 'not']:
                                processed_item['compliance_status'] = 'غير متوافق'
                            else:
                                # For any other status, keep the original (or could be translated if needed)
                                processed_item['compliance_status'] = item.get('compliance_status', '')

                            processed_item_verified, report = clean_and_report(processed_item)
                            
                            processed_items.append(processed_item_verified)
                            
                        except Exception as item_error:
                            self.logger.error(f"Error processing item {i}: {item_error}")
                            continue
        
                    if len(processed_items) > 0:
                        processed_data[arabic_keys[k]] = processed_items
                except Exception as category_error:
                    self.logger.error(f"Error processing category {category}: {category_error}")
                    continue
            
            # Save the processed data to the output file
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2, default=str)
                
                self.logger.info(f"Successfully processed and saved to: {output_file}")
                return processed_data
            except Exception as save_error:
                self.logger.error(f"Errors saving output file: {save_error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing JSON file: {str(e)}", exc_info=True)
            return None
        
    def process_nested_structure(self, data: Any) -> Any:
        """
        Recursively process nested data structure:
        1. If value is a list of dicts, process each dict individually
        2. If value is a dict, recursively process it
        3. If value is a simple type, process it with other simple values
        """
        if isinstance(data, dict):
            # Separate complex values (lists and dicts) from simple values
            simple_values = {}
            complex_values = {}
            
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    # List of dictionaries - will be processed individually
                    complex_values[key] = value
                elif isinstance(value, dict):
                    # Nested dictionary - will be processed recursively
                    complex_values[key] = value
                else:
                    # Simple value - will be processed with other simple values
                    simple_values[key] = value
            
            # Process simple values together if there are any
            if simple_values:
                try:
                    self.logger.info(f"Processing simple values: {list(simple_values.keys())}")
                    processed_simple = self.process_item(simple_values)
                    if isinstance(processed_simple, str):
                        processed_simple = json.loads(processed_simple)
                except Exception as e:
                    self.logger.error(f"Error processing simple values: {e}")
                    processed_simple = simple_values
            else:
                processed_simple = {}
            
            # Process complex values
            processed_complex = {}
            for key, value in complex_values.items():
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    # Process each dict in the list individually
                    self.logger.info(f"Processing list of {len(value)} dicts for key: {key}")
                    processed_list = []
                    for i, item in enumerate(value, 1):
                        try:
                            self.logger.info(f"Processing dict {i} of {len(value)} in key '{key}'")
                            processed_item = self.process_item(item)
                            if isinstance(processed_item, str):
                                processed_item = json.loads(processed_item)
                            processed_list.append(processed_item)
                        except Exception as e:
                            self.logger.error(f"Error processing dict {i} in key '{key}': {e}")
                            processed_list.append(item)
                    processed_complex[key] = processed_list
                elif isinstance(value, dict):
                    # Recursively process nested dictionary
                    self.logger.info(f"Recursively processing nested dict for key: {key}")
                    processed_complex[key] = self.process_nested_structure(value)
            
            # Combine processed simple and complex values
            result = {**processed_simple, **processed_complex}
            verified_result, report = clean_and_report(result)
            return verified_result
            
        elif isinstance(data, list):
            # If it's a list of dictionaries, process each one
            if all(isinstance(item, dict) for item in data):
                self.logger.info(f"Processing list of {len(data)} dicts")
                processed_list = []
                for i, item in enumerate(data, 1):
                    try:
                        self.logger.info(f"Processing dict {i} of {len(data)}")
                        processed_item = self.process_item(item)
                        if isinstance(processed_item, str):
                            processed_item = json.loads(processed_item)
                        processed_list.append(processed_item)
                    except Exception as e:
                        self.logger.error(f"Error processing dict {i}: {e}")
                        processed_list.append(item)
                return processed_list
            else:
                # If it's a list of simple values, process them together
                try:
                    self.logger.info(f"Processing list of {len(data)} simple values")
                    list_dict = {"values": data}
                    processed = self.process_item(list_dict)
                    if isinstance(processed, str):
                        processed = json.loads(processed)
                    return processed.get("values", data)
                except Exception as e:
                    self.logger.error(f"Error processing list of simple values: {e}")
                    return data
        else:
            # Simple value - return as is (will be processed with other simple values)
            verified_result, report = clean_and_report(data)
            return verified_result

    def process_json_file_for_proposal_eval(self, input_file: str, output_file: str):
        """Process a JSON file with nested structure, translate its content, and save to a new file."""
        try:
            self.logger.info(f"Starting to process file: {input_file}")
            
            # Read the input JSON file
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading input file: {e}")
                return None
            
            # Process the entire data structure recursively
            self.logger.info("Processing nested data structure")
            processed_data = self.process_nested_structure(data)
            
            # Save the processed data to the output file
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2, default=str)
                
                self.logger.info(f"Successfully processed and saved to: {output_file}")
                return processed_data
            except Exception as save_error:
                self.logger.error(f"Error saving output file: {save_error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing JSON file: {str(e)}", exc_info=True)
            return None

def main():
    try:
        # Initialize logger for the main function
        main_logger = get_logger(__name__)

        input_json = "/home/ubuntu/Tendor_POC/custom_proposal/tendor_poc/proposal_test/english_json_test.json"
        output_json = "outputs_proposal_eval/arabic_translated_test.json"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        
        try:
            # Process the JSON file
            start_time = time.time()
            main_logger.info(f"Starting translation process at {time.ctime(start_time)}")
            
            # Initialize the translator, passing the main_logger instance
            translator = ArabicJsonTranslator(logger_instance=main_logger)
            # result = translator.process_json_file_for_rfp_completeness_check(input_json, output_json)
            result = translator.process_json_file_for_proposal_eval(input_json, output_json)
            
            if result is not None:
                end_time = time.time()
                duration = end_time - start_time
                main_logger.info(f"Translation completed successfully in {duration:.2f} seconds")
                main_logger.info(f"Output saved to: {os.path.abspath(output_json)}")
                return 0
            else:
                main_logger.error("Translation process failed")
                return 1
                
        except KeyboardInterrupt:
            main_logger.info("\nTranslation process interrupted by user")
            return 1
        except Exception as e:
            main_logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
            return 1
    except Exception as e:
        main_logger.error(f"Critical error in main: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        # This logger is for the __name__ == "__main__" block's direct exceptions
        # It's better to get the logger here too if main() itself raises an unhandled exception
        logger = get_logger(__name__) # Ensure logger is defined here if needed
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)