import os
import requests
from urllib.parse import urlparse, parse_qs, urlencode
import json
import fitz  # PyMuPDF
import requests
import io
import datetime
import uuid
from typing import List, Dict, Optional, Tuple, Any
from pdf_parsing import PDFParser  # Updated to use unified PDF parser
# from pdf_parsing_neel import PDFParser  # Updated to use unified PDF parser
from pdf_standard_parser import PDFStandardParser
from logger import custom_logger as llog # Corrected llog import
from ea_standard_eval import evaluation
from openai import OpenAI
import os
import hashlib  # Added for PDF content hashing
from dotenv import load_dotenv
from docx2pdf import convert
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
# from model_provider import ModelProvider  # Disabled - CUDA not available
import subprocess
import re 
import concurrent.futures
from pipeline import Pipeline
from convert_docx_to_pdf import convert_docx_to_pdf
from encryption_utils import EncryptionUtil # Import EncryptionUtil
from bs4 import BeautifulSoup # Import BeautifulSoup for HTML parsing
class RFPCompleteness:
    def __init__(self, model_provider=None):
        """
        Initialize the RFP Completeness checker.
        
        Args:
            model_provider: Optional shared model provider instance
        """
        self.detected_language = None
        load_dotenv()

        # Use a fixed log file name for general initialization logs
        llog("MAIN", "Initializing MAIN ")
        llog("RFP_Completeness", f"Initializing RFPCompleteness")
        self.pipeline = Pipeline()
        self.model_provider = None
        app_key = os.getenv("APP_KEY")
        print(f"ED API KEY IS : {app_key}")
        if not app_key:
            llog("RFPCompleteness", "APP_KEY environment variable not set. Encryption/Decryption will not be available.")
            self.encryptor = None
        else:
            try:
                self.encryptor = EncryptionUtil(app_key)
                llog("RFPCompleteness", "EncryptionUtil initialized successfully.")
            except ValueError as e:
                llog("RFPCompleteness", f"Error initializing EncryptionUtil: {e}. Encryption/Decryption will not be available.")
                self.encryptor = None

    def _decrypt_json_string(self, encrypted_json_str: str, log_id: str) -> Optional[str]:
        """
        Decrypts a JSON-wrapped encrypted string (Fernet token).
        Expects a JSON string like: {"iv":"...", "value":"...", "mac":"...", "tag":""}
        """
        llog("RFPCompleteness", f"_decrypt_json_string received: {encrypted_json_str}", log_id)
        if not self.encryptor or not encrypted_json_str:
            llog("RFPCompleteness", "_decrypt_json_string: No encryptor or empty string. Returning original.", log_id)
            return encrypted_json_str # Return as is if no encryptor or empty string

        try:
            # FIRST ATTEMPT: Try to decrypt directly as a Fernet token
            llog("RFPCompleteness", "_decrypt_json_string: Attempting direct decryption as Fernet token.", log_id)
            decrypted_string = self.encryptor.decrypt_text(encrypted_json_str)
            llog("RFPCompleteness", f"_decrypt_json_string: Direct decryption successful. Decrypted content (first 50 chars): {decrypted_string[:50]}...", log_id)
            return decrypted_string
        except Exception as direct_decrypt_e:
            llog("RFPCompleteness", f"_decrypt_json_string: Direct decryption failed: {direct_decrypt_e}. Attempting JSON-wrapped decryption fallback.", log_id)
            # If direct decryption fails, try to parse as JSON-wrapped Fernet token
            try:
                # Attempt to parse the string as JSON
                data = json.loads(encrypted_json_str)
                llog("RFPCompleteness", f"_decrypt_json_string: Successfully parsed JSON. Data keys: {data.keys()}", log_id)
                
                # Assuming the 'value' field contains the actual Fernet token
                encrypted_value = data.get("value")
                if encrypted_value:
                    llog("RFPCompleteness", f"_decrypt_json_string: Extracted 'value' field. Attempting decryption of: {encrypted_value}", log_id)
                    # The 'value' is expected to be base64-url-safe encoded
                    decrypted_string = self.encryptor.decrypt_text(encrypted_value)
                    llog("RFPCompleteness", f"_decrypt_json_string: Decrypted JSON-wrapped string successfully. Decrypted content (first 50 chars): {decrypted_string[:50]}...", log_id)
                    return decrypted_string
                else:
                    llog("RFPCompleteness", f"_decrypt_json_string: No 'value' field found in encrypted JSON string. Returning original.", log_id)
                    return encrypted_json_str # Return original if no value field
            except json.JSONDecodeError as jde:
                llog("RFPCompleteness", f"_decrypt_json_string: Input is not a valid JSON string (after direct decrypt failed). JSONDecodeError: {jde}. Returning original. Input: {encrypted_json_str}", log_id)
                return encrypted_json_str # Not JSON, return original
            except Exception as e:
                llog("RFPCompleteness", f"_decrypt_json_string: Error during JSON-wrapped decryption process: {e}. Returning original. Input: {encrypted_json_str}", log_id)
                return encrypted_json_str # Return original on decryption error

    def is_complete(self,
                    id: str = "1", 
                    model: str = "openai", 
                    # rfp_url: str = "/home/khantil/Music/01 RFP.pdf",
                    rfp_url: str = "https://compliancebotai.blob.core.windows.net/compliancebotdev/rfp/document/67b5be9c749273GORz1739964060.pdf",
                    ea_standard_eval_url: str = "https://compliancebotai.blob.core.windows.net/compliancebotdev/ministry/document/6747f7dd5c51cgK33W1732769757.docx",
                    log_save_file_name: str = "rfp_job_processing_log",
                    output_tokens: str = "1", 
                    industry_standards: str = "",
                    ministry_compliances: str = "",
                    output_language: str = "english") -> Tuple[str, Optional[Dict], Optional[str], Optional[str]]:
        error_msg_for_user = None  # Initialize to None
        technical_error_msg = None # Initialize to None
        try:
            self.log_save_file_name = log_save_file_name # This will now be "rfp_job_processing_log"

            # Decryption of industry_standards and ministry_compliances is now handled upstream in database.py
            # using the Laravel decryption method. This fallback logic is no longer needed.
            # if self.encryptor:
            #     initial_industry_standards = industry_standards # Store original for comparison
            #     initial_ministry_compliances = ministry_compliances # Store original for comparison

            #     industry_standards = self._decrypt_json_string(industry_standards, self.log_save_file_name)
            #     if industry_standards != initial_industry_standards:
            #         llog("RFPCompleteness", f"Successfully decrypted industry_standards (fallback) (first 50 chars): {industry_standards[:50]}...", self.log_save_file_name)
            #     else:
            #         llog("RFPCompleteness", f"industry_standards not decrypted (fallback). Original value: {industry_standards}", self.log_save_file_name)

            #     ministry_compliances = self._decrypt_json_string(ministry_compliances, self.log_save_file_name)
            #     if ministry_compliances != initial_ministry_compliances:
            #         llog("RFPCompleteness", f"Successfully decrypted ministry_compliances (fallback) (first 50 chars): {ministry_compliances[:50]}...", self.log_save_file_name)
            #     else:
            #         llog("RFPCompleteness", f"ministry_compliances not decrypted (fallback). Original value: {ministry_compliances}", self.log_save_file_name)

            # Validate and process topics
            llog("RFPCompleteness", f"Calling _validate_topics with industry_standards: {industry_standards} and ministry_compliances: {ministry_compliances}", self.log_save_file_name)
            topics = self._validate_topics(industry_standards, ministry_compliances)
            llog("RFPCompleteness", f"Topics after validation: {topics}", self.log_save_file_name)
            
            # Set output tokens
            self.output_tokens = output_tokens
            self.model = model
            print("-"*100)
            print(model)

            # --- Start: Refactored File Download and Processing Logic ---
            llog("RFPCompleteness", f"Starting file processing for RFP URL: {rfp_url}", self.log_save_file_name)
            rfp_document = self._process_file(rfp_url, "rfp_file")
            if isinstance(rfp_document, tuple):
                llog("RFPCompleteness", f"RFP file processing failed: {rfp_document[2]}", self.log_save_file_name)
                return (id,) + rfp_document

            llog("RFPCompleteness", f"Starting file processing for EA Standard URL: {ea_standard_eval_url}", self.log_save_file_name)
            ea_standard_document = self._process_file(ea_standard_eval_url, "ea_standard_eval_file")
            if isinstance(ea_standard_document, tuple):
                llog("RFPCompleteness", f"EA Standard file processing failed: {ea_standard_document[2]}", self.log_save_file_name)
                return (id,) + ea_standard_document

            # Save processed documents to temporary files for pipeline
            temp_dir = "input_data"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate unique filenames for the PDF outputs
            rfp_file_path = os.path.join(temp_dir, f"rfp_document_{id}.pdf")
            ea_standard_file_path = os.path.join(temp_dir, f"ea_standard_document_{id}.pdf")

            try:
                rfp_document.save(rfp_file_path)
                llog("RFPCompleteness", f"Saved RFP document to {rfp_file_path}", self.log_save_file_name)
            except Exception as e:
                error_msg_for_user = "Failed to save processed RFP document."
                technical_error_msg = f"Error saving RFP document {rfp_file_path}: {e}"
                llog("RFPCompleteness", technical_error_msg, self.log_save_file_name)
                return id, None, error_msg_for_user, technical_error_msg

            try:
                ea_standard_document.save(ea_standard_file_path)
                llog("RFPCompleteness", f"Saved EA Standard document to {ea_standard_file_path}", self.log_save_file_name)
            except Exception as e:
                error_msg_for_user = "Failed to save processed EA Standard document."
                technical_error_msg = f"Error saving EA Standard document {ea_standard_file_path}: {e}"
                llog("RFPCompleteness", technical_error_msg, self.log_save_file_name)
                return id, None, error_msg_for_user, technical_error_msg
            # --- End: Refactored File Download and Processing Logic ---

            # Create the entity.txt file and write the topics to it
            entity_file_path = "sample_data/01/entity.txt"
            try:
                os.makedirs(os.path.dirname(entity_file_path), exist_ok=True)
                with open(entity_file_path, "w", encoding="utf-8") as entity_file:
                    if isinstance(topics, (list, tuple)):
                        entity_file.write(", ".join(str(topic) for topic in topics))
                    else:
                        entity_file.write(str(topics))
                llog("RFPCompleteness", f"Topics being written to {entity_file_path}: {topics}", self.log_save_file_name) # Added log for topics
                llog("RFPCompleteness", f"Successfully wrote topics to {entity_file_path}", self.log_save_file_name)
            except Exception as e:
                llog("RFPCompleteness", f"Failed to write topics to {entity_file_path}: {e}", self.log_save_file_name)
                
                # Process files using the pipeline

            llog("RFPCompleteness", "Custom ready to call pipeline", self.log_save_file_name)
            path_final_html = self.pipeline.process_pdfs(
                rfp_pdf_path=[rfp_file_path],
                ea_standard_pdf_path=[ea_standard_file_path],
                entity_path="sample_data/01/entity.txt",
                tmp_output_path="tmp/tmp_output",
                output_language=output_language
            )

            # short cut for testing
            # path_final_html = "final_outputss/report.html"

            # path_final_html = "final_outputs/report.html"
            llog("RFPCompleteness", f"Result from pipeline: {path_final_html}", self.log_save_file_name)
            
            # Read the HTML report
            html_report = None
            if not path_final_html or not os.path.exists(path_final_html):
                llog("RFPCompleteness", f"Report path invalid or missing: {path_final_html}", self.log_save_file_name)
                return id, None, "Report generation failed.", f"Invalid report path: {path_final_html}"
            with open(path_final_html, "r", encoding="utf-8") as f:
                html_report = f.read()
            
            llog("RFPCompleteness", f"HTML report read from {path_final_html}", self.log_save_file_name)
            print("*0"*100)
            print(f"Final HTML report: {html_report}")
    
            llog("RFPCompleteness", f"Result from pipeline: {html_report}", self.log_save_file_name)
            
            # Encrypt the HTML report if encryptor is available
            if self.encryptor:
                try:
                    encrypted_html_report = self.encryptor.encrypt_text(html_report)
                    llog("RFPCompleteness", "HTML report encrypted successfully.", self.log_save_file_name)
                    html_report = encrypted_html_report  # Use the encrypted report
                except Exception as e:
                    llog("RFPCompleteness", f"Error encrypting HTML report: {e}. Proceeding with unencrypted report.", self.log_save_file_name)

            formatted_json = {"result": {"ea_standard_eval": {"report": html_report}}}
            llog("RFPCompleteness", f"Formatted results: {html_report}", self.log_save_file_name)
           
            # now it's not usefull because there we perfomr Encryption and Decryption in the download_files function
            # report_html = formatted_json['result']['ea_standard_eval']['report']
            # llog("RFPCompleteness", f"Before regax & report_html: {report_html}", self.log_save_file_name)


            #while use groq [just for groq not for openai]
            # import re
            # regex_pattern = r'<table\b[^>]*>[\s\S]*?</table>'
            # report_html = re.findall(regex_pattern, report_html, re.DOTALL)
            # print(report_html[0])
            # llog("RFPCompleteness", f"After regax Final Report after Groq to HTML table fetch: {report_html[0]}", self.log_save_file_name)
            # formatted_json = {"results": {"ea_standard_eval": {"report": report_html[0]}}}
            
            
            # report_html_str = "".join(report_html)  # Combine all tables into one string
            # print(report_html_str)
            # llog("RFPCompleteness", f"Final Report after Groq to HTML table fetch: {report_html_str}", self.log_save_file_name)

            # Define directory and file path
            folder_name = "FINAL_RFP_COMPLETENESS"
            file_name = "ea_standard_eval_report.html"
            file_path = os.path.join(folder_name, file_name)

            # Create directory if it doesn't exist
            os.makedirs(folder_name, exist_ok=True)

            # Write the HTML content to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_report)

            print(f"Report saved to {file_path}")
            
            # llog("RFPCompleteness", f"Formatted JSON response: {formatted_json}", self.log_save_file_name)
            
            return id, formatted_json, error_msg_for_user, technical_error_msg
            
        except Exception as e:
            error_msg_for_user = "An unexpected error occurred while processing the PDF."
            technical_error_msg = f"Unexpected error in is_complete: {str(e)}"
            llog("RFPCompleteness", f"Error in is_complete: {technical_error_msg}", self.log_save_file_name)
            return id, None, error_msg_for_user, technical_error_msg
            
    def _process_file(self, url: str, file_type: str) -> fitz.Document | Tuple[None, str, str]:
        temp_html_path = None
        try:
            llog("RFPCompleteness", f"Attempting to download {file_type} from URL: {url}", self.log_save_file_name)
            response = requests.get(url)
            llog("RFPCompleteness", f"File download response: {response.status_code}", self.log_save_file_name)
        except Exception as e:
            error_msg_for_user = f"We encountered an issue while downloading the {file_type}. Please check if the file is accessible and try again."
            technical_error_msg = f"Failed to download file. Error : {str(e)}"
            llog("RFPCompleteness", f"File download error: {technical_error_msg}", self.log_save_file_name)
            return None, error_msg_for_user, technical_error_msg
            
        if response.status_code != 200:
            error_msg_for_user = f"We couldn\'t access the {file_type}. Please check if the file is accessible and try again."
            technical_error_msg = f"Failed to download file. Status code: {response.status_code}"
            llog("RFPCompleteness", f"File download error: {technical_error_msg}", self.log_save_file_name)
            return None, error_msg_for_user, technical_error_msg
            
        try:
            content_type = response.headers.get('Content-Type', '').lower()
            llog("RFPCompleteness", f"File content type: {content_type}", self.log_save_file_name)
            
            # --- Start: HTML/PHP Content Workaround ---
            if 'text/html' in content_type or 'application/x-httpd-php' in content_type:
                llog("RFPCompleteness", f"Detected HTML/PHP content for {file_type}. Attempting to find document link within.", self.log_save_file_name)
                
                # Save the HTML content to a temporary file for parsing
                temp_html_fd, temp_html_path = tempfile.mkstemp(suffix='.html')
                os.close(temp_html_fd)
                with open(temp_html_path, 'wb') as f:
                    f.write(response.content)
                
                with open(temp_html_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Heuristics to find a document link:
                # 1. Look for <a> tags with href ending in .pdf or .docx
                # 2. Prioritize links that contain 'download' or similar keywords
                
                found_link = None
                
                # Broad search for PDF/DOCX links
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if '.pdf' in href.lower() or '.docx' in href.lower():
                        # Basic URL join to handle relative paths
                        parsed_base_url = urlparse(url)
                        potential_doc_url = parsed_base_url.scheme + "://" + parsed_base_url.netloc + href if href.startswith('/') else href
                        
                        # Heuristic: prioritize links with 'download' or 'file' in them
                        if 'download' in href.lower() or 'file' in href.lower():
                            found_link = potential_doc_url
                            llog("RFPCompleteness", f"Found high-priority document link in HTML: {found_link}", self.log_save_file_name)
                            break # Take the first high-priority link
                        elif not found_link: # Store the first general doc link if no high-priority found yet
                            found_link = potential_doc_url
                            llog("RFPCompleteness", f"Found potential document link in HTML: {found_link}", self.log_save_file_name)

                if found_link:
                    llog("RFPCompleteness", f"Attempting secondary download from extracted link: {found_link}", self.log_save_file_name)
                    # Recursively call _process_file with the new URL
                    # This will handle the actual document download and conversion
                    os.unlink(temp_html_path) # Clean up temporary HTML file before recursive call
                    return self._process_file(found_link, file_type)
                else:
                    os.unlink(temp_html_path) # Clean up temporary HTML file
                    error_msg_for_user = f"The {file_type} URL returned an HTML page, but no direct document download link (PDF/DOCX) could be found within it. Please ensure the provided URL directly points to a document or contains a clearly accessible download link."
                    technical_error_msg = f"HTML parsing failed to find a document link from {url}"
                    llog("RFPCompleteness", f"HTML parsing error: {technical_error_msg}", self.log_save_file_name)
                    return None, error_msg_for_user, technical_error_msg
            # --- End: HTML/PHP Content Workaround ---
            
            # Handle PDF files
            elif 'application/pdf' in content_type:
                pdf_stream = io.BytesIO(response.content)
                pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
                
            # Handle Word documents (.docx, .doc)
            elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type or 'application/msword' in content_type:
                # Save the docx file temporarily
                temp_docx_fd, temp_docx_path = tempfile.mkstemp(suffix='.docx')
                os.close(temp_docx_fd)
                
                with open(temp_docx_path, 'wb') as f:
                    f.write(response.content)
                    
                # Convert to PDF
                temp_pdf_fd, temp_pdf_path = tempfile.mkstemp(suffix='.pdf')
                os.close(temp_pdf_fd)
                
                try:
                    convert(temp_docx_path, temp_pdf_path)
                except Exception as e:

                    print("Error converting docx to pdf :- ", str(e))
                    # Fallback to LibreOffice if docx2pdf fails
                    llog("RFPCompleteness", f"docx2pdf conversion failed, trying LibreOffice: {str(e)}", self.log_save_file_name)
                    try:
                        subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', os.path.dirname(temp_pdf_path), temp_docx_path], check=True)
                        # Rename the output file to match the expected path
                        libreoffice_output = os.path.join(os.path.dirname(temp_pdf_path), os.path.basename(temp_docx_path).replace('.docx', '.pdf'))
                        os.rename(libreoffice_output, temp_pdf_path)
                    except Exception as e:
                        llog("RFPCompleteness", f"LibreOffice conversion failed: {str(e)}", self.log_save_file_name)
                        raise
                
                # Read the PDF file
                with open(temp_pdf_path, 'rb') as pdf_file:
                    pdf_data = pdf_file.read()
                
                # Clean up temporary files
                os.unlink(temp_docx_path)
                os.unlink(temp_pdf_path)
                
                pdf_stream = io.BytesIO(pdf_data)
                pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
            else:
                error_msg_for_user = f"The {file_type} format is not supported. Please upload a PDF or Word document."
                technical_error_msg = f"Unsupported file format: {content_type}"
                llog("RFPCompleteness", f"File format error: {technical_error_msg}", self.log_save_file_name)
                return None, error_msg_for_user, technical_error_msg
                
            # Verify the PDF has content
            if pdf_document.page_count == 0:
                error_msg_for_user = f"The {file_type} appears to be empty. Please check the file and try again."
                technical_error_msg = "PDF has no pages"
                llog("RFPCompleteness", f"File content error: {technical_error_msg}", self.log_save_file_name)
                return None, error_msg_for_user, technical_error_msg
                
            # Check if the PDF has extractable text
            has_content = False
            for page in pdf_document:
                if page.get_text().strip() or len(page.get_images()) > 0:
                    has_content = True
                    break
                    
            if not has_content:
                return None, "The file appears to be blank or contains no readable content.", "PDF has pages but no extractable text or image content."
                
            llog("RFPCompleteness", "File processed successfully", self.log_save_file_name)
            return pdf_document
        except Exception as e:
            llog("RFPCompleteness", f"File processing error: {str(e)}", self.log_save_file_name)
            return None, "Error processing the file. Please ensure it\'s a valid and accessible PDF or Word document.", f"Error processing file: {str(e)}"
        finally:
            if temp_html_path and os.path.exists(temp_html_path):
                os.unlink(temp_html_path)

    def _validate_topics(self, industry_standards: str, ministry_compliances: str) -> List[str]:
        # Handle None values before combining
        combined_topics = ""
        
        if industry_standards is not None:
            combined_topics += industry_standards
            
        combined_topics += ","
        
        if ministry_compliances is not None:
            combined_topics += ministry_compliances
        
        topics = [topic.strip() for topic in combined_topics.split(',') if topic.strip()]
        llog("RFPCompleteness", f"Validating topics: {topics}", self.log_save_file_name)
        
        # Return the topics as is, even if empty
        return topics  
    
    # def detect_document_language(self, text: str) -> str | Tuple[str, str]:
    #     """Detect language from processed text"""
    #     try:
    #         if not text.strip():
    #             error_msg = "The document appears to be empty"
    #             llog("Main", error_msg, self.log_save_file_name)
    #             return "The document appears to be empty.", error_msg
            
    #         # Preprocess text: remove numbers, punctuation, spaces, and special characters
    #         # Keep only Arabic and English alphabetic characters
    #         cleaned_text = re.sub(r'[^a-zA-Z\u0600-\u06FF]', '', text)
            
    #         # Count Arabic alphabetic characters (exclude diacritics like َ, ُ)
    #         arabic_chars = sum(1 for c in cleaned_text if '\u0600' <= c <= '\u06FF' and c.isalpha())
    #         # Count English alphabetic characters (a-z, A-Z)
    #         english_chars = sum(1 for c in cleaned_text if c.isalpha() and ord(c) < 128)
            
    #         llog("Main", f"Arabic chars: {arabic_chars}, English chars: {english_chars}", self.log_save_file_name)
            
    #         #total char's
    #         total_chars = arabic_chars + english_chars
            
    #          # Calculate Arabic character percentage
    #         arabic_percentage = (arabic_chars / total_chars) * 100 if total_chars > 0 else 0
    #         llog("Main", f"Arabic character percentage: {arabic_percentage:.2f}%", self.log_save_file_name)
            
    #         # Classify as Arabic if Arabic characters are at least 30% of total
    #         if arabic_percentage >= ARABIC_PERCENT:
    #             llog("Main", "Classifying document as Arabic due to ≥30% Arabic characters", self.log_save_file_name)
    #             return 'ar'
    #         else:
    #             llog("Main", "Classifying document as English due to <30% Arabic characters", self.log_save_file_name)
    #             return 'en'

    #     except Exception as e:
    #         error_msg = f"Error in language detection: {str(e)}"
    #         llog("Main", error_msg, self.log_save_file_name)
    #         return "Error detecting language.", error_msg
    
    def _process_openai(self, id: str, rfp_url: str, ea_standard_eval_url: str, topics: List[str], token=None) -> Tuple[str, Optional[Dict], Optional[str], Optional[str]]:
        error_msg_for_user = None
        technical_error_msg = None
        language = None
        llog("RFPCompleteness", "Starting processing", self.log_save_file_name)

        # Create txt_cache folder
        cache_dir = "txt_cache"
        os.makedirs(cache_dir, exist_ok=True)
        llog("RFPCompleteness", f"Text cache directory ensured: {cache_dir}", self.log_save_file_name)

        # Process RFP file first
        rfp_result = self._process_file(rfp_url, "rfp_file")
        # rfp_result = self._process_file("/home/khantil/Music/01 RFP.pdf", "rfp_file")
        if isinstance(rfp_result, tuple):
            llog("RFPCompleteness", "RFP file processing failed", self.log_save_file_name)
            return (id,) + rfp_result

        pdf_document = rfp_result
        
        # Process ea_standard_eval file
        
        ea_standard_eval_file = self._process_file(ea_standard_eval_url, "ea_standard_eval_file")
        # ea_standard_eval_file = self._process_file("/home/khantil/Music/4.2./4.2.1/02 EA Standards.pdf", "ea_standard_eval_file")
        if isinstance(ea_standard_eval_file, tuple):
            llog("RFPCompleteness", "EA standard eval file processing failed", self.log_save_file_name)
            return (id,) + ea_standard_eval_file

        # Process PDFs using PDFParser
        llog("RFPCompleteness", "Processing PDFs using PDFParser", self.log_save_file_name)
        output_txt = "translated.txt"
        ea_standard_txt = "ea_standard.txt"
        pdf_parser = PDFParser(self.log_save_file_name)
        Standard_eval_parser = PDFStandardParser(self.log_save_file_name,self.model)
        llog("RFPCompleteness", "PDFParser initialized", self.log_save_file_name)
        # Generate hash for RFP PDF content
        pdf_content = b""
        for page in pdf_document:
            pdf_content += page.get_text("text").encode('utf-8')
        rfp_hash = hashlib.sha256(pdf_content).hexdigest()
        rfp_cache_file = os.path.join(cache_dir, f"rfp_{rfp_hash}.txt")
        llog("RFPCompleteness", f"RFP PDF hash: {rfp_hash}, cache file: {rfp_cache_file}", self.log_save_file_name)

        # Generate hash for EA standard PDF content
        ea_content = b""
        for page in ea_standard_eval_file:
            ea_content += page.get_text("text").encode('utf-8')
        ea_hash = hashlib.sha256(ea_content).hexdigest()
        ea_cache_file = os.path.join(cache_dir, f"ea_{ea_hash}.txt")
        llog("RFPCompleteness", f"EA standard PDF hash: {ea_hash}, cache file: {ea_cache_file}", self.log_save_file_name)

        # Check if RFP cached .txt exists
        # if os.path.exists(rfp_cache_file):
        #     llog("RFPCompleteness", f"Found cached RFP text file: {rfp_cache_file}", self.log_save_file_name)
        #     output_txt = rfp_cache_file
            
        #     language = self.detect_document_language(output_txt)
        #     llog("RFPCompleteness", f"Detected language with cache file: {language}", self.log_save_file_name)
        # else:
        llog("RFPCompleteness", f"No cache found for RFP, extracting text", self.log_save_file_name)
        llog("RFPCompleteness", f"model passed to EA is : {self.model}", self.log_save_file_name)
        error_msg_for_developer, error_msg_for_user, language = pdf_parser.parse_pdf_to_text(pdf_document, output_txt, self.model, self.log_save_file_name)
        
        if error_msg_for_developer:
            llog("RFPCompleteness", f"RFP PDF processing error: {error_msg_for_developer}", self.log_save_file_name)
            return id, None, error_msg_for_user, error_msg_for_developer
        # Save to cache
        with open(output_txt, 'r', encoding='utf-8') as src, open(rfp_cache_file, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        llog("RFPCompleteness", f"Saved RFP text to cache: {rfp_cache_file}", self.log_save_file_name)
            
        # Check if EA standard cached .txt exists
        # if os.path.exists(ea_cache_file):
        #     llog("RFPCompleteness", f"Found cached EA standard text file: {ea_cache_file}", self.log_save_file_name)
        #     ea_standard_txt = ea_cache_file
        # else:
        llog("RFPCompleteness", f"No cache found for EA standard, extracting text", self.log_save_file_name)
        llog("RFPCompleteness", f"model passed to EA is : {self.model}", self.log_save_file_name)
        error_msg_for_developer, error_msg_for_user = Standard_eval_parser.parse_pdf_to_text(ea_standard_eval_file, ea_standard_txt, self.model, self.log_save_file_name)
        if error_msg_for_developer:
            llog("RFPCompleteness", f"EA standard PDF processing error: {error_msg_for_developer}", self.log_save_file_name)
            return id, None, error_msg_for_user, error_msg_for_developer
        # Save to cache
        with open(ea_standard_txt, 'r', encoding='utf-8') as src, open(ea_cache_file, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        llog("RFPCompleteness", f"Saved EA standard text to cache: {ea_cache_file}", self.log_save_file_name)

        llog("RFPCompleteness", "PDF processing completed", self.log_save_file_name)
        llog('RFPCompleteness', "Starting document analysis", self.log_save_file_name)

        # Read the translated text and detect language
        try:
            # with open(output_txt, "r", encoding='utf-8') as file:
            #     text = file.read()
                
            # Detect document language
            document_language = language
            # document_language = "ar"
            llog("RFPCompleteness", f"Detected language: {document_language}", self.log_save_file_name)
            if isinstance(document_language, tuple):
                llog("RFPCompleteness", "Document language detection failed", self.log_save_file_name)
                return (id,) + document_language
            
            # Update the detected language
            self.detected_language = document_language
            llog("RFPCompleteness", f"Document language detected: {self.detected_language}", self.log_save_file_name)
            
        except FileNotFoundError:
            error_msg = "Translation file not found"
            llog("RFPCompleteness", error_msg, self.log_save_file_name)
            return id, None, error_msg, "translated.txt file not found"
        except Exception as e:
            error_msg = f"Error reading translation file: {str(e)}"
            llog("RFPCompleteness", error_msg, self.log_save_file_name)
            return id, None, "Error processing translation", error_msg

        # Process topic-specific evaluations if any topics are provided
        topic_results = None
        llog("RFPCompleteness", f"Processing with topics", self.log_save_file_name)
            
        # Process EA standard evaluation
        # old logic
        
        # try:
        #     if self.model_provider and self.model_provider.model_type == 'opensource':
        #         # For opensource model, use the shared model provider
        #         llog("RFPCompleteness", "Using shared model provider for evaluation", self.log_save_file_name)
        #         eval_result = evaluation(output_txt, ea_standard_txt, topics, self.detected_language, self.output_tokens, self.log_save_file_name, self.model_provider,token=token)
        #     else:
        #         # For OpenAI model, don't pass model_provider
        #         llog("RFPCompleteness", "Using OpenAI for evaluation", self.log_save_file_name)
        #         eval_result = evaluation(output_txt, ea_standard_txt, topics, self.detected_language, self.output_tokens, self.log_save_file_name)
                
        # agent pipeline code logic 
        print(f"Selected Model: {self.model}")   
        llog("RFPCompleteness", f"Selected Model: {self.model}", self.log_save_file_name)
        try:
            if self.model == 'opensource':
                # For opensource model, use the shared model provider
                llog("RFPCompleteness", "Using openSource model provider for evaluation", self.log_save_file_name)
                eval_result = evaluation(output_txt, ea_standard_txt, topics, self.detected_language, self.log_save_file_name, model_provider="opensource")
            else:
                # For OpenAI model, don't pass model_provider
                llog("RFPCompleteness", "Using OpenAI for evaluation", self.log_save_file_name)
                eval_result = evaluation(output_txt, ea_standard_txt, topics, self.detected_language, self.log_save_file_name, model_provider="openai")
                
       
            # Combine the EA evaluation with topic results if any
            results = {**eval_result}
            if topic_results:
                results.update(topic_results)

            llog("RFPCompleteness", f"fresh results: {results}", self.log_save_file_name)
            formatted_json = {"result": results}
            llog("RFPCompleteness", f"Formatted results: {results}", self.log_save_file_name)
            report_html = formatted_json['result']['ea_standard_eval']['report']
            llog("RFPCompleteness", f"Before regax & report_html: {report_html}", self.log_save_file_name)


            #while use groq [just for groq not for openai]
            # import re
            # regex_pattern = r'<table\b[^>]*>[\s\S]*?</table>'
            # report_html = re.findall(regex_pattern, report_html, re.DOTALL)
            # print(report_html[0])
            # llog("RFPCompleteness", f"After regax Final Report after Groq to HTML table fetch: {report_html[0]}", self.log_save_file_name)
            # formatted_json = {"results": {"ea_standard_eval": {"report": report_html[0]}}}
            
            
            # report_html_str = "".join(report_html)  # Combine all tables into one string
            # print(report_html_str)
            # llog("RFPCompleteness", f"Final Report after Groq to HTML table fetch: {report_html_str}", self.log_save_file_name)

            # Define directory and file path
            folder_name = "FINAL_RFP_COMPLETENESS"
            file_name = "ea_standard_eval_report.html"
            file_path = os.path.join(folder_name, file_name)

            # Create directory if it doesn't exist
            os.makedirs(folder_name, exist_ok=True)

            # Write the HTML content to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_html[0])

            print(f"Report saved to {file_path}")
            
            # llog("RFPCompleteness", f"Formatted JSON response: {formatted_json}", self.log_save_file_name)
            
            return id, formatted_json, error_msg_for_user, technical_error_msg
            
        except Exception as e:
            error_msg = f"Error in evaluation: {str(e)}"
            llog("RFPCompleteness", error_msg, self.log_save_file_name)
            return id, None, "An error occurred during evaluation.", error_msg

if __name__ == "__main__":
    rfpcomplete = RFPCompleteness()
    rfpcomplete.is_complete()


#  - Verify if security and data protection requirements specific to the entity are properly addressed
#    - Check if performance metrics and SLAs align with entity operational expectations
#    - Evaluate if project timeline recommendations match entity procurement cycles

# 4. **Mandatory Points Coverage** - Do the improvements identify if the following specific points are included in the RFP?
#    - Data Governance requirements
#    - Sustainability Considerations
#    - If these points are missing from the RFP, verify they are included in the HTML improvement recommendations