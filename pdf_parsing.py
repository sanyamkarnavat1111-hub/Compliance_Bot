import pdfplumber
import fitz  # PyMuPDF
from tqdm import tqdm
from openai import OpenAI
from logger import custom_logger as llog # Revert to llog import
import base64
import os
import shutil
from dotenv import load_dotenv
import tempfile
from PIL import Image
import io
import langdetect
from deep_translator import GoogleTranslator
import re
from typing import Tuple, Optional # Added Optional
from config import ARABIC_PERCENT 


class PDFParser():
    def __init__(self):
        """
        Initialize the PDF parser with OpenAI integration for image-based text extraction and translation.
        """
        self.translator = GoogleTranslator(source='auto', target='en')
        
        # Statistics for validation
        self.stats = {
            "total_pages": 0,
            "pages_with_text": 0,
            "pages_translated": 0,
            "total_text_length": 0,
            "pages_using_image_fallback": 0
        }

    def __encode_image(self, image_path, log_save_file_name: str):
        """Encode an image to base64 for API calls"""
        llog("PDFParser", f"Encoding image: {image_path}", log_save_file_name)
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def __render_page_to_image(self, page, page_num, log_save_file_name: str):
        """
        Render a PDF page to an image.
        
        Args:
            page: PyMuPDF page object
            page_num (int): Current page number (0-based)
            log_save_file_name (str): Name of the log file

        Returns:
            Optional[str]: Path to the temporary image file, or None if rendering fails.
        """
        llog("PDFParser", f"Rendering page {page_num + 1} to image", log_save_file_name)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_img_path = tmp.name
        
        zoom = 4  # Higher zoom for better text clarity
        mat = fitz.Matrix(zoom, zoom)
        
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(temp_img_path)
            
            if os.path.exists(temp_img_path) and os.path.getsize(temp_img_path) > 1000:
                return temp_img_path
            else:
                llog("PDFParser", f"Warning: Rendered image is too small: {os.path.getsize(temp_img_path)} bytes", log_save_file_name)
                return None
        except Exception as e:
            llog("PDFParser", f"Error rendering page to image: {str(e)}", log_save_file_name)
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            return None

    def __generate_image_text(self, image_path, model, log_save_file_name: str):
        """Extract text from an image using the specified model"""
        llog("PDFParser", f"Generating text from image: {image_path} using {model}", log_save_file_name)
        base64_image = self.__encode_image(image_path, log_save_file_name)

        prompt = """You are a precise document transcriber. Your task is to convert this image into text exactly as it appears, maintaining all formatting and content. Please:

        1. Transcribe ALL visible text exactly as written, including:
        - Headers and footers
        - Main body text
        - Table contents
        - Bullet points and numbered lists
        - Any visible numbers, references, or citations
        
        2. Preserve the formatting using markdown:
        - Use tables for tabular data
        - Maintain bullet points and numbering
        - Preserve paragraph breaks
        - Keep text alignment (left/center/right)
        
        3. For any graphics or images:
        - Describe their location and content
        - Include any text or labels within them
        
        4. If any part is unclear or partially visible:
        - Transcribe what you can see
        - Note any uncertainties with [unclear: description]
        
        5. Look carefully for any faint text, watermarks, or security patterns.
        
        Output the content exactly as it appears, maintaining the original structure and format in markdown."""

        
        llog("PDFParser", f"seleted model: {model}", log_save_file_name)
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY") 
        )
        self.model_name = "gpt-4o"
        
        try:
            if model == "openai":
                llog("PDFParser", "Sending request to OpenAI API...", log_save_file_name)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    max_tokens=3000,
                    timeout=1200
                )
                content = response.choices[0].message.content
                llog("PDFParser", "Text extracted successfully with OpenAI", log_save_file_name)

            elif model == "opensource":
                llog("PDFParser", "Sending request to Qwen 2.5 VL 32B...", log_save_file_name)
                response = self.openrouter_client.chat.completions.create(
                    model="qwen/qwen2.5-vl-32b-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    max_tokens=3000,
                    temperature=0.2,
                    timeout=120
                )
                content = response.choices[0].message.content
                
                regex = r'```markdown\n|```'
                def clean_response(text):
                    return re.sub(regex, '', text)
                
                content = clean_response(content)
                
                llog("PDFParser", f"Content: {content}", log_save_file_name)
                llog("PDFParser", "Text extracted successfully with Qwen", log_save_file_name)

            else:
                llog("PDFParser", f"Unsupported model: {model}", log_save_file_name)
                return f"[Unsupported model for text extraction: {model}]"

            lines = content.split('\n')
            result_lines = [line for line in lines if not line.strip().startswith('```')]
            return '\n'.join(result_lines)

        except Exception as e:
            llog("PDFParser", f"Error extracting text from image: {str(e)}", log_save_file_name)
            return f"[Error extracting text: {str(e)}]"

    def __translate_to_english(self, text, page_num, log_save_file_name: str):
        """Detect language and translate Arabic to English if needed"""
        llog("PDFParser", f"Detecting language for page {page_num + 1}", log_save_file_name)
        try:
            if not text or text.startswith("[Error") or "no text" in text.lower():
                return text
            
            lang = langdetect.detect(text)
            llog("PDFParser", f"Detected language inbuild: {lang}", log_save_file_name)
            
            if lang == 'ar':
                llog("PDFParser", f"Translating Arabic to English for page {page_num + 1}", log_save_file_name)
                self.stats["pages_translated"] += 1
                chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
                translated_chunks = []
                for chunk in chunks:
                    translated = self.translator.translate(chunk)
                    translated_chunks.append(translated if translated else chunk)
                return "\n".join(translated_chunks)
            return text
        except Exception as e:
            llog("PDFParser", f"Error in language detection/translation: {str(e)}", log_save_file_name)
            return text

    def get_extraction_stats(self):
        """Get statistics about the extraction process"""
        if self.stats["total_pages"] == 0:
            return {
                "total_pages": 0,
                "pages_with_text": 0,
                "pages_translated": 0,
                "text_coverage_percent": 0,
                "avg_text_per_page": 0,
                "pages_using_image_fallback": 0,
                "total_text_length": 0
            }
            
        return {
            "total_pages": self.stats["total_pages"],
            "pages_with_text": self.stats["pages_with_text"],
            "pages_translated": self.stats["pages_translated"],
            "text_coverage_percent": (self.stats["pages_with_text"] / self.stats["total_pages"]) * 100,
            "avg_text_per_page": self.stats["total_text_length"] / self.stats["total_pages"],
            "pages_using_image_fallback": self.stats["pages_using_image_fallback"],
            "total_text_length": self.stats["total_text_length"]
        }

    def parse_pdf_to_text(self, input_pdf, output_txt_path, model="openai", log_save_file_name: str = "custom_logs") -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Processes a PDF document by rendering each page as an image, extracting text, and translating to English.
        
        Args:
            input_pdf: Can be either a file path (str) or a PyMuPDF document object
            output_txt_path (str): Path to the output text file
            model (str): Model to use for text extraction ('openai' or 'opensource')
            log_save_file_name (str): Name of the log file for this process.
        
        Returns:
            Tuple[Optional[str], Optional[str], Optional[str]]: (error_msg_for_developer, error_msg_for_user, detected_language)
        """
        if model not in ["openai", "opensource"]:
            error_msg_dev = f"Invalid model: {model}. Supported models are 'openai' or 'opensource'"
            error_msg_user = f"Invalid model {model} specified. Please use 'openai' or 'opensource'."
            llog("PDFParser", error_msg_dev, log_save_file_name)
            return error_msg_dev, error_msg_user, None
            
        llog("PDFParser", f"Starting PDF processing with {model} model", log_save_file_name)
        output = []
        
        self.stats = {
            "total_pages": 0,
            "pages_with_text": 0,
            "pages_translated": 0,
            "total_text_length": 0,
            "pages_using_image_fallback": 0
        }
        
        try:
            if isinstance(input_pdf, str):
                llog("PDFParser", f"Opening PDF from path: {input_pdf}", log_save_file_name)
                pdf_document = fitz.open(input_pdf)
            else:
                llog("PDFParser", "Using provided PDF document", log_save_file_name)
                temp_path = "temp_pdf_file.pdf"
                input_pdf.save(temp_path)
                pdf_document = fitz.open(temp_path)
                
            if not pdf_document:
                error_msg_dev = "Failed to open PDF document"
                error_msg_user = "Failed to open the document. Please ensure it's a valid PDF."
                llog("PDFParser", error_msg_dev, log_save_file_name)
                return error_msg_dev, error_msg_user, None
                
            llog("PDFParser", "PDF opened successfully", log_save_file_name)
            
            total_pages = len(pdf_document)
            self.stats["total_pages"] = total_pages
            llog("PDFParser", f"Total pages in document: {total_pages}", log_save_file_name)
            
            for page_num in tqdm(range(total_pages), desc="Processing pages"):
                try:
                    llog("PDFParser", f"Processing page {page_num + 1}", log_save_file_name)
                    output.append(f"\n\n--- Page {page_num + 1} ---\n\n")
                    page = pdf_document[page_num]
                    
                    self.stats["pages_using_image_fallback"] += 1
                    page_image_path = self.__render_page_to_image(page, page_num, log_save_file_name)
                    
                    if not page_image_path:
                        output.append("[Failed to render page as image]")
                        continue
                        
                    try:
                        page_content = self.__generate_image_text(page_image_path, model, log_save_file_name)
                 
                        dir_name = "language_detection"
                        if not os.path.exists(dir_name):
                            os.makedirs(dir_name)

                        page_file_path = os.path.join(dir_name, "page_content.txt")
                        with open(page_file_path, "a", encoding="utf-8") as f:
                            f.write(f"\n\n--- Page {page_num + 1} ---\n\n")
                            f.write(page_content)           
                        
                        page_content = self.__translate_to_english(page_content, page_num, log_save_file_name)
                        
                        if page_content and not page_content.startswith("[Error") and "no text" not in page_content.lower():
                            self.stats["pages_with_text"] += 1
                            self.stats["total_text_length"] += len(page_content)
                        
                        output.append(page_content)
                        
                    finally:
                        if os.path.exists(page_image_path):
                            os.remove(page_image_path)
                            
                except Exception as page_error:
                    llog("PDFParser", f"Error processing page {page_num + 1}: {str(page_error)}", log_save_file_name)
                    output.append(f"[Error processing page: {str(page_error)}]")
                    continue
            
            if not os.path.exists(page_file_path):
                error_msg_dev = "No page content file was created. Cannot perform language detection."
                error_msg_user = "Failed to extract any readable content from the document for language detection."
                llog("PDFParser", error_msg_dev, log_save_file_name)
                return error_msg_dev, error_msg_user, None

            with open(page_file_path, "r", encoding="utf-8") as f:
                loaded_text = f.read()

            def detect_document_language(text: str, log_save_file_name: str) -> str:
                try:
                    if not text.strip():
                        error_msg = "The document appears to be empty"
                        llog("PDFParser", error_msg, log_save_file_name)
                        return "error: " + error_msg
                    
                    cleaned_text = re.sub(r'[^a-zA-Z\u0600-\u06FF]', '', text)
                    
                    arabic_chars = sum(1 for c in cleaned_text if '\u0600' <= c <= '\u06FF' and c.isalpha())
                    english_chars = sum(1 for c in cleaned_text if c.isalpha() and ord(c) < 128)
                    
                    llog("PDFParser", f"Arabic chars: {arabic_chars}, English chars: {english_chars}", log_save_file_name)
                    
                    total_chars = arabic_chars + english_chars
                    
                    arabic_percentage = (arabic_chars / total_chars) * 100 if total_chars > 0 else 0
                    llog("PDFParser", f"Arabic character percentage: {arabic_percentage:.2f}%", log_save_file_name)
                    
                    if arabic_percentage >= ARABIC_PERCENT:
                        llog("PDFParser", "Classifying document as Arabic due to \u226530% Arabic characters", log_save_file_name)
                        return 'ar'
                    else:
                        llog("PDFParser", "Classifying document as English due to <30% Arabic characters", log_save_file_name)
                        return 'en'
                    
                except Exception as e:
                    error_msg = f"Error in language detection: {str(e)}"
                    llog("PDFParser", error_msg, log_save_file_name)
                    return "error: " + error_msg
            
            lang_result = detect_document_language(loaded_text, log_save_file_name)
            llog("PDFParser", f"raw response from function: {lang_result}", log_save_file_name)
            print(f"Detected language: {lang_result}")
            
            with open(page_file_path, "w", encoding="utf-8") as f:
                f.write("")
        
            stats = self.get_extraction_stats()
            llog("PDFParser", f"Extraction stats: {stats}", log_save_file_name)

            try:
                llog("PDFParser", f"Writing output to: {output_txt_path}", log_save_file_name)
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(output))
                llog("PDFParser", "Output file written successfully", log_save_file_name)
                
            except Exception as write_error:
                error_msg_dev = f"Error writing output file: {str(write_error)}"
                llog("PDFParser", error_msg_dev, log_save_file_name)
                return error_msg_dev, "Failed to save the processed text", None

            if not isinstance(input_pdf, str) and os.path.exists("temp_pdf_file.pdf"):
                os.remove("temp_pdf_file.pdf")

            if lang_result.startswith("error:"):
                dev_error = lang_result
                user_error = "An error occurred during language detection."
                return dev_error, user_error, None
            else:
                return None, None, lang_result

        except Exception as pdf_error:
            error_msg_dev = f"Error processing PDF file: {str(pdf_error)}"
            error_msg_user = "An error occurred while processing the PDF file"
            llog("PDFParser", error_msg_dev, log_save_file_name)
            return error_msg_dev, error_msg_user, None

if __name__ == "__main__":
    input_pdf_path = "/root/SOW_for_NAFIS_PLATFROM_Developmen_1.pdf"
    output_txt_path = "/root/RFP/SOW_for_NAFIS_PLATFROM_Developmen_1.txt"
    document_parser = PDFParser()
    error_msg_for_developer, error_msg_for_user, language = document_parser.parse_pdf_to_text(
        input_pdf_path, output_txt_path, model="opensource", log_save_file_name="test_pdf_parsing_log"
    )
    if error_msg_for_developer:
        print(f"Developer Error: {error_msg_for_developer}")
    if error_msg_for_user:
        print(f"User Error: {error_msg_for_user}")
    llog("PDFParser", "Example completed!", "test_pdf_parsing_log")
    if language:
        print(f"Detected language: {language}")