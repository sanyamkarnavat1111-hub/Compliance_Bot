from dotenv import load_dotenv
import fitz
# from logger import custom_logger as llog # Old import
from logger_config import get_logger, setup_process_logger # New imports
from openai import OpenAI
from urllib.parse import urlparse, parse_qs, urlencode
import io
import requests
import os
import tempfile
import datetime
import uuid
import subprocess
from docx2pdf import convert
from typing import Tuple, Dict, Optional, Any
from rfp_proposal_eval import AgentCluster
from langchain_core.caches import BaseCache
from pydantic import BaseModel
# from logger import custom_logger as llog # Re-commented out this line
# from model_provider import ModelProvider  # Disabled - CUDA not available
from pipeline_proposal_eval import ProposalEvalPipeline
import re
# from config import ARABIC_PERCENT
import sys
from encryption_utils import EncryptionUtil # Import EncryptionUtil

# Add root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_parsing import PDFParser
import tiktoken
import json
from convert_docx_to_pdf import convert_docx_to_pdf

# Global logger for this module (for general messages not tied to a specific process ID)
global_module_logger = get_logger(__file__)

class ProposalEvaluation:
    def __init__(self, model_provider=None):
        """
        Initialize the Proposal Evaluation module
        
        Args:
            model_provider: Optional shared model provider instance
        """
        # ModelProvider disabled - CUDA not available
        # self.model_provider = model_provider if model_provider is not None else ModelProvider(log_save_file_name=f"rfp_proposal_eval_{datetime.datetime.now().strftime('%m%d%H%M%S')}.txt")
        load_dotenv()
        self.model_provider = None
        self.agent = AgentCluster(model_provider)
        self.detected_language = None
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rfp_texts_dir = "rfp_texts"  # Directory to store processed RFP text files
        os.makedirs(self.rfp_texts_dir, exist_ok=True)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.pdf_parsing = PDFParser() # Updated: Pass global_module_logger to PDFParser
        self.proposal = ProposalEvalPipeline() # Updated: Pass global_module_logger to ProposalEvalPipeline

        # Initialize EncryptionUtil
        app_key = os.getenv("APP_KEY")
        print(f"ED API KEY IS : {app_key}")
        if not app_key:
            global_module_logger.error("APP_KEY environment variable not set. Encryption/Decryption will not be available.")
            self.encryptor = None
        else:
            try:
                self.encryptor = EncryptionUtil(app_key)
                global_module_logger.info("EncryptionUtil initialized successfully.")
            except ValueError as e:
                global_module_logger.error(f"Error initializing EncryptionUtil: {e}. Encryption/Decryption will not be available.")
                self.encryptor = None
        
    def start_evaluation(self,
                        #  rfp_id: str = "ad16304d-f198-4430-80d8-1dd3779f1e63", 
                        #  rfp_id: str = "11144", 
                         rfp_id: str = "oksfb", 
                        #  rfp_id: str = "67ebae961d48efC9UH1743498902",
                         model: str = "openai", 
                         rfp_url: str = "https://compliancebotai.blob.core.windows.net/compliancebotdev/rfp/document/67ebae961d48efC9UH1743498902.pdf", 
                         proposal_url: str = "https://compliancebotai.blob.core.windows.net/compliancebotdev/proposal/document/67ebc6cac48a84Uib01743505098.pdf", 
                         log_save_file_name: str = "log.txt",
                         output_language: str = "english"
                         ) -> Tuple[str, dict, str, str]:

        # proposal_url = "https://compliancebotai.blob.core.windows.net/compliancebotdev/proposal/document/68949c258519airyhC1754569765.pdf"
        # rfp_url = "https://compliancebotai.blob.core.windows.net/compliancebotdev/rfp/document/68827415e6cc15gFSy1753379861.pdf"
        
        # Extract proposal_id from proposal_url for unique logging
        
        try:
            parsed_url = urlparse(proposal_url)
            path_segments = parsed_url.path.split('/')
            # The ID is typically the filename before the extension
            proposal_filename = path_segments[-1] if path_segments else 'no_filename'
            proposal_id = os.path.splitext(proposal_filename)[0]
            if not proposal_id: # Fallback if no valid ID extracted
                proposal_id = rfp_id # Fallback to rfp_id if proposal_id can't be extracted
        except Exception as e:
            global_module_logger.error(f"Error extracting proposal_id from URL {proposal_url}: {e}. Using rfp_id as fallback.")
            proposal_id = rfp_id # Fallback to rfp_id if extraction fails
        
        # Setup the process-specific logger using the extracted proposal_id
        process_logger = setup_process_logger(proposal_id)
        process_logger.info(f"Starting evaluation process for Proposal ID: {proposal_id} (RFP ID: {rfp_id}) and model: {model}")
        process_logger.info(f"output_language is : {output_language}")
        print("\n============ STARTING EVALUATION PROCESS =================")
        try:
            self.log_save_file_name = log_save_file_name # Removed: No longer needed as logging is handled by process_logger/global_module_logger
            model_type = model.lower()
            rfp_language = None  # Initialize rfp_language at the beginning of try block
            
            # Re-initialize pdf_parsing and proposal with the process_logger
            self.pdf_parsing = PDFParser() 
            self.proposal = ProposalEvalPipeline(logger_instance=process_logger)
            
            def download_files(rfp_url: str, proposal_url: str, save_dir: str = "input_data") -> dict:
                """
                Downloads the RFP and proposal files from the given URLs 
                and saves them in the specified local directory.

                Args:
                    rfp_url (str): URL to the RFP document.
                    proposal_url (str): URL to the proposal document.
                    save_dir (str): Directory to save downloaded files (default: 'input_data').

                Returns:
                    dict: Paths to the downloaded files with keys 'rfp_path' and 'proposal_path'.
                """
                os.makedirs(save_dir, exist_ok=True)

                def download_file(url: str, url_type: str) -> str:
                    # Parse the URL to get scheme, netloc, path, query, fragment
                    parsed_url = urlparse(url)
                    query_params = parse_qs(parsed_url.query)

                    # Determine which secret key parameter to look for
                    secret_key_param = None
                    if url_type == "rfp":
                        secret_key_param = 'rfp-file-access-secret-key'
                    elif url_type == "proposal":
                        secret_key_param = 'proposal-file-access-secret-key'
                    else: # Default or for other types like ea-standard-file or hld-file
                        secret_key_param = 'secret-key'

                    if secret_key_param in query_params and self.encryptor:
                        encrypted_secret_key = query_params[secret_key_param][0]
                        # No decryption needed for file access keys
                        # The secret keys themselves are not encrypted Fernet tokens, but plain text tokens
                        query_params[secret_key_param] = [encrypted_secret_key] # Use original (encrypted) key
                        process_logger.info(f"Using original secret key for {url_type} URL as no decryption is needed.")
                    
                    # Reconstruct the URL with potentially decrypted secret key
                    updated_query = urlencode(query_params, doseq=True)
                    final_url = parsed_url._replace(query=updated_query).geturl()
                    process_logger.info(f"Final URL for {url_type} download: {final_url}")
                   
                    # Determine filename with original extension
                    original_path_basename = os.path.basename(urlparse(final_url).path)
                    filename = original_path_basename
                    process_logger.info(f"Initial filename from URL path: {filename}")

                    if not filename or '.' not in filename:
                        process_logger.info(f"Filename '{filename}' lacks an extension. Attempting to infer from headers for {url_type}.")
                        # Try to get filename from Content-Disposition header
                        with requests.get(final_url, stream=True) as head_response:
                            head_response.raise_for_status()
                            content_disposition = head_response.headers.get('Content-Disposition')
                            content_type = head_response.headers.get('Content-Type', '').lower()
                            process_logger.info(f"Headers for {url_type}: Content-Disposition: {content_disposition}, Content-Type: {content_type}")

                            if content_disposition:
                                fname_match = re.findall(r'filename="(.+)"', content_disposition)
                                if fname_match:
                                    filename = fname_match[0]
                                    process_logger.info(f"Filename from Content-Disposition: {filename}")
                            
                            if not filename or '.' not in filename:
                                process_logger.info(f"Filename still lacks an extension. Attempting to infer from Content-Type: {content_type}.")
                                if 'pdf' in content_type: # Check for common types
                                    filename = original_path_basename + '.pdf' if '.' not in original_path_basename else original_path_basename
                                    process_logger.info(f"Inferred PDF extension: {filename}")
                                elif 'word' in content_type or 'document' in content_type: # Check for word docs
                                    filename = original_path_basename + '.docx' if '.' not in original_path_basename else original_path_basename
                                    process_logger.info(f"Inferred DOCX extension: {filename}")
                                elif 'image' in content_type:
                                    if 'jpeg' in content_type or 'jpg' in content_type:
                                        filename = original_path_basename + '.jpeg' if '.' not in original_path_basename else original_path_basename
                                    elif 'png' in content_type:
                                        filename = original_path_basename + '.png' if '.' not in original_path_basename else original_path_basename
                                    process_logger.info(f"Inferred image extension: {filename}")

                    if not filename or '.' not in filename: # Final check
                        process_logger.warning(f"Could not infer a file extension for {url_type}. Using filename: {filename} as is. This may cause issues.")

                    if not filename:
                        raise ValueError(f"URL does not contain a valid filename or content type: {final_url}")
                    file_path = os.path.join(save_dir, filename)

                    response = requests.get(final_url, stream=True)
                    response.raise_for_status()  # Raise error for bad responses

                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    return file_path

                rfp_path = download_file(rfp_url, "rfp")
                proposal_path = download_file(proposal_url, "proposal")

                return {
                    "rfp_path": rfp_path,
                    "proposal_path": proposal_path
                }
                
            # If using opensource model, ensure it's properly set in the model provider
            if model_type == 'opensource': # Added check for self.model_provider
                process_logger.info("Using open-source model.")
                rfp_proposal_file_paths = download_files(rfp_url, proposal_url)
                process_logger.info(f"Downloaded RFP and proposal files to {rfp_proposal_file_paths}")
                rfp_url_path = rfp_proposal_file_paths["rfp_path"]
                proposal_url_path = rfp_proposal_file_paths["proposal_path"]
            
                process_logger.info(f"Converting RFP and proposal files to PDF: {rfp_url_path}, {proposal_url_path}")
                rfp_file_path = convert_docx_to_pdf(rfp_url_path)
                proposal_file_path = convert_docx_to_pdf(proposal_url_path)
                
                process_logger.info(f"Converted RFP and proposal files to PDF: {rfp_file_path}, {proposal_file_path}")
                print(f"DEBUG: Calling process_pdfs with rfp_path={[rfp_file_path]}, proposal_path={[proposal_file_path]}")

                # Orignal code to process the proposal
                proposal_path, Total_assigned_score = self.proposal.process_pdfs(
                    rfp_pdf_path=[rfp_file_path], 
                    proposal_pdf_path=[proposal_file_path],
                    tmp_output_path="tmp_proposal_eval/tmp_output",
                    final_report_language=output_language
                )

                # shortcut code to process the proposal
                # proposal_path, Total_assigned_score = "/home/ubuntu/Tendor_POC/kaushik_final_rfp/tendor_poc/proposal_test/final_outputss/report.html", 94

                print(f"DEBUG: Returned from process_pdfs: proposal_path={proposal_path}, Total_assigned_score={Total_assigned_score}")
                
                Proposal_html = None
                if proposal_path:
                    with open(proposal_path, 'r', encoding='utf-8')as f:
                        Proposal_html = f.read()
                else:
                    error_msg = "Proposal evaluation did not return a valid HTML path."
                    process_logger.error(error_msg)
                    return rfp_id, None, "Proposal evaluation failed to generate HTML report.", error_msg

                # Encrypt the HTML report if encryptor is available
                if self.encryptor:
                    try:
                        encrypted_proposal_html = self.encryptor.encrypt_text(Proposal_html)
                        process_logger.info("Proposal HTML report encrypted successfully.")
                        Proposal_html = encrypted_proposal_html  # Use the encrypted report
                    except Exception as e:
                        process_logger.error(f"Error encrypting Proposal HTML report: {e}. Proceeding with unencrypted report.")
                
                process_logger.info(f"Saved evaluation result to {proposal_path}")
                print(f"Evaluation result saved to {proposal_path}")
                
                formated_propasal = {
                    "results": Proposal_html,
                    "score": Total_assigned_score  # temporary score, can be updated later
                }
                # Encrypt the score if encryptor is available
                if self.encryptor:
                    try:
                        encrypted_score = self.encryptor.encrypt_text(str(Total_assigned_score)) # Encrypt as string
                        process_logger.info("Total assigned score encrypted successfully.")
                        formated_propasal["score"] = encrypted_score  # Use the encrypted score
                    except Exception as e:
                        process_logger.error(f"Error encrypting total assigned score: {e}. Proceeding with unencrypted score.")

                process_logger.info(f"Formatted proposal: {formated_propasal}")
                return rfp_id, formated_propasal, None, None

                
            elif model_type == 'openai':
                
                rfp_text_path = os.path.join(self.rfp_texts_dir, f"rfp_{rfp_id}.txt")
                proposal_text_path = os.path.join(self.rfp_texts_dir, f"proposal_{rfp_id}.txt")
                
                # Step 1: Check if we already have processed RFP text
                rfp_content = ""
                if os.path.exists(rfp_text_path):
                    process_logger.info(f"Using existing processed RFP text from {rfp_text_path}")
                    rfp_content = self._read_text_file(rfp_text_path)
                else:
                    # Step 2: Process RFP file
                    process_logger.info("Processing RFP file")
                    rfp_result = self._process_file(rfp_url, "RFP")
                    if isinstance(rfp_result, tuple):
                        error_msg_for_user, technical_error_msg = rfp_result[1], rfp_result[2]
                        process_logger.error(f"RFP file processing failed: {technical_error_msg}")
                        return rfp_id, None, error_msg_for_user, technical_error_msg
                        

                    print("Step 2a: RFP file downloaded AND converting into txt..")
                    # Parse PDF to text first
                    pdf_document = rfp_result
                    
                    error_msg_for_developer, error_msg_for_user, rfp_language = self.pdf_parsing.parse_pdf_to_text(pdf_document, rfp_text_path, model_type) # Updated: Removed log_save_file_name
                    if error_msg_for_developer:
                        print(f"ERROR: Failed to parse RFP PDF: {error_msg_for_developer}")
                        process_logger.error(f"Error parsing PDF to text: {error_msg_for_developer}")
                        return rfp_id, None, error_msg_for_user, error_msg_for_developer
                        
                    print("Step 2b: RFP file storing")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    unique_id = str(uuid.uuid4())

                    unique_rfp_text_path = os.path.join(self.rfp_texts_dir, f"rfp_pdf_to_text_{timestamp}_{unique_id}.txt")
                    # Copy parsed RFP text to unique file
                    with open(rfp_text_path, 'r', encoding='utf-8') as src:
                        with open(unique_rfp_text_path, 'w', encoding='utf-8') as dst:
                            dst.write(src.read())
                    process_logger.info(f"Saved RFP text to unique file: {unique_rfp_text_path}")
                    rfp_content = self._read_text_file(unique_rfp_text_path)
                    
                # Validate content is not empty
                if not rfp_content.strip():
                    error_msg = "The processed RFP text appears to be empty"
                    process_logger.error(error_msg)
                    return rfp_id, None, "The RFP document appears to be empty or could not be processed correctly.", error_msg
                
                # Use the language detected by PDFParser, default to "english" if None
                self.detected_language = rfp_language if rfp_language is not None else "english" # Ensure self.detected_language is never None
                process_logger.info(f"Detected language for RFP: {self.detected_language}")
                
             # Step 3: Check if we already have processed proposal text
                proposal_content = ""
                if os.path.exists(proposal_text_path):
                    process_logger.info(f"Using existing processed proposal text from {proposal_text_path}")
                    proposal_content = self._read_text_file(proposal_text_path)
                else:
                    # Step 3a: Process proposal file
                    process_logger.info("Processing proposal file")
                    proposal_result = self._process_file(proposal_url, "Proposal")
                    if isinstance(proposal_result, tuple):
                        error_msg_for_user, technical_error_msg = proposal_result[1], proposal_result[2]
                        process_logger.error(f"Proposal file processing failed: {technical_error_msg}")
                        return rfp_id, None, error_msg_for_user, technical_error_msg

                    print("Step 3b: Proposal file downloaded and converting into txt...")
                    # Parse PDF to text
                    pdf_document = proposal_result
                    
                    proposal_language = None
                    error_msg_for_developer, error_msg_for_user, proposal_language = self.pdf_parsing.parse_pdf_to_text(pdf_document, proposal_text_path, model_type) # Updated: Removed log_save_file_name
                    if error_msg_for_developer:
                        print(f"ERROR: Failed to parse proposal PDF: {error_msg_for_developer}")
                        process_logger.error(f"Error parsing proposal PDF to text: {error_msg_for_developer}")
                        return rfp_id, None, error_msg_for_user, error_msg_for_developer

                    print("Step 3c: Proposal file storing")
                    # Create a unique file for the parsed proposal text
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    unique_id = str(uuid.uuid4())
                    unique_proposal_text_path = os.path.join(self.rfp_texts_dir, f"proposal_pdf_to_text_{timestamp}_{unique_id}.txt")

                    # Copy parsed proposal text to unique file
                    with open(proposal_text_path, 'r', encoding='utf-8') as src:
                        with open(unique_proposal_text_path, 'w', encoding='utf-8') as dst:
                            dst.write(src.read())
                    process_logger.info(f"Saved proposal text to unique file: {unique_proposal_text_path}")
                    proposal_content = self._read_text_file(unique_proposal_text_path)

            
                # Validate content is not empty
                if not proposal_content.strip():
                    error_msg = "The processed proposal text appears to be empty"
                    process_logger.error(error_msg)
                    return rfp_id, None, "The proposal document appears to be empty or could not be processed correctly.", error_msg
                    
                # Step 4: Process the evaluation using the agent
                print("Step 4: Starting proposal evaluation")
                process_logger.info("Starting proposal evaluation with agent")
                
                # Count tokens for logging purposes
                rfp_tokens = self.count_tokens(rfp_content)
                proposal_tokens = self.count_tokens(proposal_content)
                process_logger.info(f"RFP tokens: {rfp_tokens}, Proposal tokens: {proposal_tokens}")
                
                # Process the evaluation using the agent
                try:
                    evaluation_result = self.agent.process_proposal(
                        rfp_content=rfp_content,
                        proposal_content=proposal_content,
                        language=self.detected_language,
                        model_type=model_type,
                        # log_save_file_name=self.log_save_file_name # Removed: Logging handled by process_logger/global_module_logger
                    )
                    
                    # Debug: Log what was returned
                    process_logger.info(f"Type of evaluation_result: {type(evaluation_result)}")
                    process_logger.info(f"evaluation_result content: {evaluation_result}")
                    
                    # Check if evaluation_result has error
                    if isinstance(evaluation_result, dict) and 'error' in evaluation_result:
                        error_msg = f"Agent returned error: {evaluation_result['error']}"
                        process_logger.error(error_msg)
                        return rfp_id, None, "An error occurred during the proposal evaluation.", error_msg
                    
                    # Check if evaluation_result has expected structure
                    if not isinstance(evaluation_result, dict) or 'results' not in evaluation_result:
                        error_msg = f"Invalid evaluation result format. Expected dict with 'results' key, got: {type(evaluation_result)}"
                        process_logger.error(error_msg)
                        return rfp_id, None, "The evaluation did not return expected format.", error_msg
                    
                    # Encrypt the HTML report from evaluation_result if encryptor is available
                    final_html_report = evaluation_result['results']
                    if self.encryptor:
                        try:
                            encrypted_final_html_report = self.encryptor.encrypt_text(final_html_report)
                            process_logger.info("Final evaluation HTML report encrypted successfully.")
                            final_html_report = encrypted_final_html_report
                            evaluation_result['results'] = final_html_report # Update the result dictionary
                        except Exception as e:
                            process_logger.error(f"Error encrypting final evaluation HTML report: {e}. Proceeding with unencrypted report.")

                    # print(f"THIS IS ============={evaluation_result}")
                    # i want to store the result of evaluation_result in .html in folder name of FINAL_RESULT
                    # Create a directory for final results if it doesn't exist
                    final_result_dir = "FINAL_RESULT_PROPOSAL"
                    os.makedirs(final_result_dir, exist_ok=True)
                    # Create a unique filename for the evaluation result
                    evaluation_result_filename = os.path.join(final_result_dir, f"evaluation_result_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html")
                    # Save the evaluation result to an HTML file
                    with open(evaluation_result_filename, 'w', encoding='utf-8') as f:
                        f.write(final_html_report) # Write the potentially encrypted HTML
                    process_logger.info(f"Saved evaluation result to {evaluation_result_filename}")
                    print(f"Evaluation result saved to {evaluation_result_filename}")
                    # Log the evaluation result
                    # process_logger.info(f"Evaluation result: {evaluation_result}") # Replaced llog
                    # Print the evaluation result
                    # print(f"Evaluation result: {evaluation_result}")
                    # Print the evaluation result
                    print("Step 4: Proposal evaluation completed successfully")
                    process_logger.info("Proposal evaluation completed successfully")
                    
                    # Return the evaluation result
                    return rfp_id, evaluation_result, None, None
                    
                except Exception as e:
                    error_msg = f"Error in proposal evaluation: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    process_logger.error(error_msg)
                    return rfp_id, None, "An error occurred during the proposal evaluation. Please try again.", error_msg
 
                
        except Exception as e:
            error_msg = f"Unexpected error in start_evaluation: {str(e)}"
            print(f"ERROR: {error_msg}")
            process_logger.error(error_msg)
            return rfp_id, None, "An unexpected error occurred. Please try again.", error_msg
            
    def _process_file(self, url: str, file_type: str = "Document") -> fitz.Document | Tuple[None, str, str]:
        if not url or not isinstance(url, str) or len(url.strip()) == 0:
            error_msg = f"Invalid or empty URL provided for {file_type}"
            print(f"ERROR in _process_file: {error_msg}")
            global_module_logger.error(error_msg) # Replaced llog
            return None, f"Invalid or empty URL provided for {file_type}. Please check the file path.", error_msg
            
        global_module_logger.info(f"Processing {file_type} file from URL: {url}") # Replaced llog
        print(f"Processing {file_type} file from URL: {url}")
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                error_msg = f"Failed to download {file_type} file. Status code: {response.status_code}"
                global_module_logger.error(error_msg) # Replaced llog
                return None, f"We couldn't access the {file_type}. Please check if the file is accessible and try again.", error_msg
                
            content_type = response.headers.get('Content-Type', '').lower()
            global_module_logger.info(f"{file_type} file content type: {content_type}") # Replaced llog
            
            # Handle PDF files
            if 'application/pdf' in content_type:
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
                    # Try docx2pdf first
                    convert(temp_docx_path, temp_pdf_path)
                except Exception as e:
                    # Fallback to LibreOffice if docx2pdf fails
                    global_module_logger.warning(f"docx2pdf conversion failed, trying LibreOffice: {str(e)}") # Replaced llog
                    try:
                        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', os.path.dirname(temp_pdf_path), temp_docx_path], check=True)
                        # Rename the output file to match the expected path
                        libreoffice_output = os.path.join(os.path.dirname(temp_pdf_path), os.path.basename(temp_docx_path).replace('.docx', '.pdf'))
                        os.rename(libreoffice_output, temp_pdf_path)
                    except Exception as e:
                        global_module_logger.error(f"LibreOffice conversion failed: {str(e)}") # Replaced llog
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
                error_msg = f"Unsupported file format for {file_type}: {content_type}"
                global_module_logger.error(error_msg) # Replaced llog
                return None, f"The {file_type} format is not supported. Please upload a PDF or Word document.", error_msg
                
            # Verify the PDF has content
            if pdf_document.page_count == 0:
                error_msg = f"The {file_type} appears to be empty (no pages)"
                global_module_logger.error(error_msg) # Replaced llog
                return None, f"The {file_type} appears to be empty. Please check the file and try again.", error_msg
                
            # Check if the PDF has extractable text
            has_content = False
            for page in pdf_document:
                if page.get_text().strip() or len(page.get_images()) > 0:
                    has_content = True
                    break
                    
            if not has_content:
                error_msg = f"The {file_type} appears to have no extractable content"
                global_module_logger.error(error_msg) # Replaced llog
                return None, f"The {file_type} appears to be blank or contains no readable content.", error_msg
                
            global_module_logger.info(f"{file_type} file processed successfully") # Replaced llog
            return pdf_document
            
        except Exception as e:
            error_msg = f"Error processing {file_type} file: {str(e)}"
            global_module_logger.error(error_msg) # Replaced llog
            return None, f"Error processing the {file_type}. Please ensure it's a valid and accessible PDF or Word document.", error_msg
            
    def count_tokens(self, text):
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))
        
    def _validate_extraction_quality(self, stats, doc_type="Document"):
        """Log extraction statistics but always return True since validation is handled in PDFParser"""
        global_module_logger.info(f"Logging {doc_type} extraction quality: {stats}") # Replaced llog
        
        # Calculate text coverage based on pages with sufficient text
        text_coverage = (stats["pages_with_sufficient_text"] / stats["total_pages"]) * 100 if stats["total_pages"] > 0 else 0
        global_module_logger.info(f"{doc_type} text coverage: {text_coverage:.1f}% (pages with sufficient text)") # Replaced llog
        
        # Also log how many pages were processed using image fallback
        image_fallback_percentage = (stats["pages_using_image_fallback"] / stats["total_pages"]) * 100 if stats["total_pages"] > 0 else 0
        global_module_logger.info(f"{doc_type} pages using image fallback: {image_fallback_percentage:.1f}%") # Replaced llog
        
        # Always return True since we handle minimal text at the page level in PDFParser
        return True
        
    def _read_text_file(self, file_path: str) -> str:
        global_module_logger.info(f"Reading text file: {file_path}") # Replaced llog
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Removed the _detect_text_language method as its logic is now handled by pdf_parsing.py

if __name__ == "__main__":
    proposal_evaluation = ProposalEvaluation()
    proposal_evaluation.start_evaluation()