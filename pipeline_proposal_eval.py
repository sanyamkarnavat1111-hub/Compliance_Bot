#!/usr/bin/env python
import sys
import warnings
import os
import shutil
import time
from pathlib import Path
from logger_config import get_logger
import json
import numpy as np
import time
# from config import Language_Threshold
# import torch  # Disabled - CUDA not available


# Removed the module-level logger as it will be passed via __init__
# logger = get_logger(__file__)

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_parsing_vision import PDFParser
from chunk_processing_proposal import ChunkProcessor
from embedding_generation import EmbeddingGeneration
from embedding_management import EmbeddingManager
from is_arabic_detection import ArabicEnglishDetector
from pdf_auto_translation_experiment import PDFVisionProcessor
from proposal_eval_agents.requirement_extractor_crew import RFPRequirementExtractor
from proposal_eval_agents.rag_eval_crew import RAGEvaluationCrew
from proposal_eval_agents.analysis_crew import AnalysisCrew
from json_to_html_convertor_proposal_eval import save_proposal_evaluation_html
from process_dicts_to_arabic import ArabicJsonTranslator
from posprocessing import clean_and_report

class ProposalEvalPipeline:
    def __init__(self, logger_instance=None):
        self.logger = logger_instance if logger_instance is not None else get_logger(__name__) # Use passed logger or default
        try:
            self.max_chunk_size = 2048
            self.embedding_dim = 768
            self.top_k_chunks = 5
            self.pdf_parser = PDFParser()
            self.pdf_processor = PDFVisionProcessor()
            self.chunk_processor = ChunkProcessor()
            self.embedding_generation = EmbeddingGeneration()
            self.embedding_manager = EmbeddingManager()
            self.arabic_english_detector = ArabicEnglishDetector()
            self.rfp_requirement_extractor = RFPRequirementExtractor()
            self.rag_eval_crew = RAGEvaluationCrew()
            self.analysis_crew = AnalysisCrew()
            self.translator = ArabicJsonTranslator()
        except Exception as e:
            self.logger.info(f"Error initializing Pipeline: {e}") # Use self.logger
            raise
    
    def make_copy_of_json(self, formatted_eval_path: str, new_key: str = "_eng"):
        src = Path(formatted_eval_path)

        # Create new path: same directory, stem + "_eng" + original suffix
        dest = src.with_name(f"{src.stem}{new_key}{src.suffix}")

        # Make sure the directory exists (usually already true)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy(src, dest)

        return str(dest)
    
    def strip_unscored_keys(self, json_path: str) -> None:
        """
        Remove 'technical_strengths' and 'technical_concerns' keys
        from each dict inside the 'unscored' list of the JSON file,
        and overwrite the original file with the updated content.

        Parameters
        ----------
        json_path : str
            Path to the JSON file to be modified in-place.
        """
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"No JSON file found at: {json_path}")

        # Load JSON
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Modify 'unscored' section
        for item in data.get("unscored", []):
            item.pop("technical_strengths", None)
            item.pop("technical_concerns", None)

        # Overwrite the file
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Updated JSON saved (overwritten): {path.resolve()}")

    def process_pdfs(self, rfp_pdf_path: list, proposal_pdf_path: list, tmp_output_path: str, final_report_language: int):
        print("DEBUG: Inside process_pdfs method - Entry point.")
        """
        Process multiple PDF files through the pipeline.
        
        Args:
            rfp_pdf_path: List of paths to the RFP PDF files to process
            proposal_pdf_path: Path to the proposal PDF file
            tmp_output_path: Path to save the processed output
        """
        self.logger.info("Entering process_pdfs method.") # Use self.logger
        try:
            # Start the timer
            start_time = time.time()
            print("Enter iN evaluation part")
            # Cleaning up output folder
            try:
                shutil.rmtree("outputs_proposal_eval/", ignore_errors=True)
                shutil.rmtree("proposal_eval_final_outputs/", ignore_errors=True)
                shutil.rmtree("tmp_proposal_eval/", ignore_errors=True)
                shutil.rmtree("outputs_proposal_eval_new/proposal_chunks/", ignore_errors=True)
                shutil.rmtree("outputs_proposal_eval/rag_eval_crew_proposal_eval/", ignore_errors=True)
                self.logger.info("Cleaned up previous output directories.") # Use self.logger
            except Exception as e:
                self.logger.info(f"Warning: Failed to clean up directories: {e}") # Use self.logger
            
            # Create the tmp folder and the tmp_output.txt file
            try:
                os.makedirs('outputs_proposal_eval', exist_ok=True)
                os.makedirs('proposal_eval_final_outputs', exist_ok=True)
                os.makedirs('tmp_proposal_eval', exist_ok=True)
                self.logger.info("Created temporary and output directories.") # Use self.logger
            except Exception as e:
                self.logger.info(f"Error creating temporary files: {e}") # Use self.logger
            print("proposal_pdf_path", proposal_pdf_path)
            self.logger.info('--------------------------PROPOSAL TEXT PROCESS STARTED--------------------------------------') # Use self.logger
            
            proposal_text = "" # Initialize proposal_text
            try:
                proposal_texts = []
                for pdf in proposal_pdf_path:
                    if pdf is None:
                        self.logger.warning(f"Skipping NoneType PDF path in proposal_pdf_path.") # Use self.logger
                        continue
                    # pdf_path = Path(pdf
                    # Convert PDF path to TXT path using string manipulation
                    proposal_txt_path = pdf.replace('.pdf', '.txt')
                    # proposal_txt_path = "input_data/687f779b35c14YH9RR1753184155.txt"
                   
                    # if proposal_txt_path.exists():
                    if os.path.exists(proposal_txt_path):
                        with open(proposal_txt_path, 'r', encoding='utf-8') as f:
                        # with open(proposal_txt_path, 'r') as f:
                            text = f.read()
                        print(f"Loaded existing extracted text from {proposal_txt_path}")
                        self.logger.info(f"Loaded existing proposal text: {proposal_txt_path}") # Use self.logger
                    else:
                        self.logger.info(f"Processing new proposal PDF: {pdf}") # Use self.logger
                        page_texts = self.pdf_processor.process_and_save([str(pdf)], str(proposal_txt_path))
                        text = "\n\n ".join(page_texts if isinstance(page_texts, list) else [str(page_texts)])
                        with open(proposal_txt_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        print(f"Extracted and saved to {proposal_txt_path}")
                        self.logger.info(f"Extracted and saved new proposal text: {proposal_txt_path}") # Use self.logger
                    proposal_texts.append(text)
                proposal_text = "\n\n ".join(proposal_texts)
                
                # Convert text to string if it's a list
                if isinstance(proposal_text, list):
                    proposal_text = "\n\n ".join(proposal_text)
                else:
                    proposal_text = str(proposal_text)
                    
                # Save proposal_text to proposal_extracted.txt                
                proposal_extracted_path = "proposal_pdf_extracted.txt"
                if not proposal_text.strip():
                    self.logger.warning("No proposal text extracted. Skipping saving to file.") # Use self.logger
                    proposal_extracted_path = "" # Ensure path is empty if no content
                else:
                    with open(proposal_extracted_path, "w", encoding="utf-8") as f:
                        f.write(proposal_text)
                    print(f"Proposal text extracted and saved to: {proposal_extracted_path}")
                    self.logger.info(f"Proposal text saved to {proposal_extracted_path}") # Use self.logger
                
                # Clean and report the proposal text
                self.logger.info("Cleaning and reporting proposal text.") # Use self.logger
                proposal_text, proposal_report = clean_and_report(proposal_text)
                self.logger.info(proposal_report) # Use self.logger
                
                # Ensure proposal_text is a string after cleaning
                if isinstance(proposal_text, list):
                    proposal_text = "\n\n ".join(proposal_text)
                else:
                    proposal_text = str(proposal_text)
                self.logger.info("Proposal text cleaned.") # Use self.logger
                
                # proposal_language = self.arabic_english_detector.detect_language("\n\n ".join(self.pdf_parser.parse_pdfs_to_text(input_pdfs = proposal_pdf_path, output_txt_path = tmp_output_path+"_proposal_pdf_extraction.txt")))
            except Exception as e:
                self.logger.error(f"Error processing PROPOSAL text: {e}") # Use self.logger
                proposal_text = "" # Ensure proposal_text is defined as empty string on error

            self.logger.info('--------------------------PROPOSAL TEXT PROCESS COMPLETED--------------------------------------') # Use self.logger
            self.logger.info(f"Proposal text length: {len(proposal_text)}") # Use self.logger
            self.logger.info('--------------------------RFP TEXT PROCESS STARTED--------------------------------------') # Use self.logger

            rfp_text = "" # Initialize rfp_text
            rfp_extracted_path = "" # Initialize rfp_extracted_path
            try:
                rfp_texts = []
                for pdf in rfp_pdf_path:
                    if pdf is None:
                        self.logger.warning(f"Skipping NoneType PDF path in rfp_pdf_path.") # Use self.logger
                        continue
                    # pdf_path = Path(pdf)
                    # rfp_txt_path = pdf_path.with_suffix('.txt')
                    rfp_txt_path = pdf.replace('.pdf', '.txt')
                    if os.path.exists(rfp_txt_path):
                        with open(rfp_txt_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        print(f"Loaded existing extracted text from {rfp_txt_path}")
                        self.logger.info(f"Loaded existing RFP text: {rfp_txt_path}") # Use self.logger
                    else:
                        self.logger.info(f"Processing new RFP PDF: {pdf}") # Use self.logger
                        page_texts = self.pdf_processor.process_and_save([str(pdf)], str(rfp_txt_path))
                        text = "\n\n ".join(page_texts if isinstance(page_texts, list) else [str(page_texts)])
                        with open(rfp_txt_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        print(f"Extracted and saved to {rfp_txt_path}")
                        self.logger.info(f"Extracted and saved new RFP text: {rfp_txt_path}") # Use self.logger
                    rfp_texts.append(text)
                rfp_text = "\n\n ".join(rfp_texts)
                
                # Convert text to string if it's a list
                if isinstance(rfp_text, list):
                    rfp_text = "\n\n ".join(rfp_text)
                else:
                    rfp_text = str(rfp_text)
                
                # Save rfp_text to rfp_pdf_extracted.txt                
                rfp_extracted_path = "rfp_pdf_extracted.txt"
                if not rfp_text.strip():
                    self.logger.warning("No RFP text extracted. Skipping saving to file.") # Use self.logger
                    rfp_extracted_path = "" # Ensure path is empty if no content
                else:
                    with open(rfp_extracted_path, "w", encoding="utf-8") as f:
                        f.write(rfp_text)
                    print(f"RFP text extracted and saved to: {rfp_extracted_path}")
                    self.logger.info(f"RFP text saved to {rfp_extracted_path}") # Use self.logger
                
                self.logger.info("Cleaning and reporting RFP text.") # Use self.logger
                rfp_text, rfp_report = clean_and_report(rfp_text)
                self.logger.info(rfp_report) # Use self.logger
                
                # Ensure rfp_text is a string after cleaning
                rfp_text = str(rfp_text)
                
                # New method to detect the language of the RFP text
                rfp_language = final_report_language if final_report_language in ("english", "arabic") else "english"

                # Old method to detect the language of the RFP text
                # rfp_language = self.arabic_english_detector.detect_language(rfp_text)
                # rfp_language = self.arabic_english_detector.detect_language("\n\n ".join(self.pdf_parser.parse_pdfs_to_text(input_pdfs = rfp_pdf_path, output_txt_path = tmp_output_path+"_rfp_pdf_extraction.txt")))
                # rfp_language = "arabic"
                # rfp_language = "english"
            except Exception as e:
                self.logger.error(f"Error processing RFP text: {e}") # Use self.logger
                rfp_text = "" # Ensure rfp_text is defined as empty string on error
            
            self.logger.info(f"RFP text length: {len(rfp_text)}") # Use self.logger

            # Storing the proposal chunks to the vector database
            self.logger.info("Starting proposal chunks processing.") # Use self.logger
            try:
                # Need to make sure this storage and the phase 1 storage location are different
                # Ensure proposal_text is a string before processing
                if not proposal_text.strip():
                    self.logger.warning("Skipping proposal chunking: proposal_text is empty.") # Use self.logger
                    proposal_chunks = []
                else:
                    if isinstance(proposal_text, list):
                        proposal_text_str = "\n\n ".join(proposal_text)
                    else:
                        proposal_text_str = str(proposal_text)
                    
                    proposal_chunks = self.chunk_processor.create_chunks(proposal_text_str)

                dir="pipiline_proposal_chunks"
                dir_embeddings= "pipiline_proposal_embeddings"
                os.makedirs(dir, exist_ok=True)
                if proposal_chunks:
                    with open(f"{dir}/proposal_chunks.json", "w", encoding="utf-8") as f:
                        json.dump(proposal_chunks, f, indent=2, ensure_ascii=False)
                    print(f"Proposal chunks saved to {dir}/proposal_chunks.json")
                    self.logger.info(f"Proposal chunks saved: {dir}/proposal_chunks.json") # Use self.logger
                else:
                    self.logger.warning("No proposal chunks to save.") # Use self.logger

                self.logger.info("Generating proposal embeddings.") # Use self.logger
                proposal_doc_embeddings = None # Initialize to None
                if proposal_chunks:
                    proposal_doc_embeddings = self.embedding_generation.generate_embeddings(proposal_chunks)
                    # Assuming proposal_doc_embeddings is a tensor (or list of tensors)
                    # Convert tensors to regular Python lists (torch disabled)
                    embeddings = None
                    #if isinstance(proposal_doc_embeddings, torch.Tensor):
                    # Note: torch.Tensor checks disabled since CUDA not available
                    if hasattr(proposal_doc_embeddings, 'tolist'):
                        proposal_doc_embeddings = proposal_doc_embeddings.tolist()
                    elif isinstance(proposal_doc_embeddings, list) and hasattr(proposal_doc_embeddings[0], 'tolist'):
                        proposal_doc_embeddings = [e.tolist() for e in proposal_doc_embeddings]
                    # else: proposal_doc_embeddings is already a list
                    os.makedirs(dir_embeddings, exist_ok=True)
                    with open(f"{dir_embeddings}/proposal_embeddings.json", "w") as f:
                        json.dump(proposal_doc_embeddings, f, indent=2, ensure_ascii=False)
                    print(f"Proposal embeddings saved to {dir}/proposal_embeddings.json")
                    self.logger.info(f"Proposal embeddings saved: {dir_embeddings}/proposal_embeddings.json") # Use self.logger
                else:
                    self.logger.warning("No proposal embeddings to generate or save.") # Use self.logger
                
            except Exception as e:
                self.logger.error(f"Error processing proposal chunks and embeddings: {e}") # Use self.logger

            self.logger.info("Adding embeddings to manager.") # Use self.logger
            try:
                # Here, The auto old data delete functionality is being added, to remove the old data from the embedding manager
                if proposal_chunks and proposal_doc_embeddings and len(proposal_doc_embeddings) > 0:
                    # Ensure embeddings are numpy array before adding to manager
                    if not isinstance(proposal_doc_embeddings, np.ndarray):
                        proposal_doc_embeddings = np.array(proposal_doc_embeddings)

                    self.embedding_manager.clean_embedding_db()
                    self.embedding_manager.add_embeddings(proposal_chunks, proposal_doc_embeddings)
                    self.logger.info("Embeddings added to manager.") # Use self.logger
                else:
                    self.logger.warning("Skipping adding embeddings to manager: no chunks or embeddings to add.") # Use self.logger
            except Exception as e:
                self.logger.error(f"Error managing embeddings: {e}") # Use self.logger

            self.logger.info('--------------------------PROPOSAL CHUNKS STORED IN EMBEDDING MANAGER--------------------------------------') # Use self.logger

            # processing the RFP chunks
            self.logger.info("Starting RFP chunks processing.") # Use self.logger
            try:
                # Need to make sure this storage and the phase 1(RFP Completeness check) storage location are different
                # Ensure rfp_text is a string before processing
                if isinstance(rfp_text, list):
                    rfp_text_str = "\n\n ".join(rfp_text)
                else:
                    rfp_text_str = str(rfp_text)
                
                rfp_chunks = self.chunk_processor.create_chunks(rfp_text_str)
                self.logger.info("RFP chunks created.")
            except Exception as e:
                self.logger.error(f"Error processing RFP chunks: {e}") # Use self.logger

            # Running crewAI multi agentic system
            self.logger.info("Running CrewAI multi-agentic system.") # Use self.logger
            try:
                self.logger.info("Calling RFPRequirementExtractor.start_crew_process...") # Use self.logger
                results_aggregaated_json, results_aggregaated_json_path = self.rfp_requirement_extractor.start_crew_process(rfp_chunks=rfp_chunks, k=self.top_k_chunks,rfp_path=rfp_extracted_path)
                self.logger.info(f"RFPRequirementExtractor result: {results_aggregaated_json}") # Use self.logger
                self.logger.info(f"RFPRequirementExtractor output path: {results_aggregaated_json_path}") # Use self.logger
                print(results_aggregaated_json)

                # Add Aggregator agent here

                # Use the results directly - should be a dict or JSON-serializable object
                rfp_requirements = results_aggregaated_json

                print("going to rag eval crew:")
                self.logger.info("Calling RAGEvaluationCrew.start_crew_process...") # Use self.logger
                rag_eval_result = self.rag_eval_crew.start_crew_process(rfp_requirements=rfp_requirements,rfp_path=rfp_extracted_path,proposal_path=proposal_extracted_path)
                self.logger.info(f"RAGEvaluationCrew raw result: {rag_eval_result}") # Use self.logger
                
                # Handle the case where rag_eval_crew returns None or unexpected format
                formatted_eval = None
                formatted_eval_path = None
                Total_assigned_score = 0

                if rag_eval_result is None:
                    self.logger.error("RAGEvaluationCrew returned None. Skipping further evaluation steps for RAG.") # Use self.logger
                    # No return here, allow the process to continue with default None values
                elif isinstance(rag_eval_result, tuple) and len(rag_eval_result) == 3:
                    formatted_eval, formatted_eval_path, Total_assigned_score = rag_eval_result
                    self.logger.info(f"RAGEvaluationCrew formatted_eval: {formatted_eval}") # Use self.logger
                    self.logger.info(f"RAGEvaluationCrew formatted_eval_path: {formatted_eval_path}") # Use self.logger
                    self.logger.info(f"RAGEvaluationCrew Total_assigned_score: {Total_assigned_score}") # Use self.logger
                    
                    # Validate that formatted_eval_path is not None
                    if formatted_eval_path is None:
                        self.logger.error("RAGEvaluationCrew returned None for formatted_eval_path. Proceeding with None.") # Use self.logger
                else:
                    self.logger.error(f"Unexpected result format from rag_eval_crew: {rag_eval_result}. Expected (formatted_eval, formatted_eval_path, Total_assigned_score). Proceeding with None values.") # Use self.logger
                
                print(formatted_eval)

                print("going to analysis crew:")
                MAX_RETRIES = 3
                retry_count = 0
                final_analysis_json = None
                final_analysis_json_path = None

                if formatted_eval_path is not None:
                    while retry_count < MAX_RETRIES:
                        try:
                            self.logger.info(f"Calling AnalysisCrew.start_crew_process (Attempt {retry_count + 1}/{MAX_RETRIES}) with path: {formatted_eval_path}") # Use self.logger
                            final_analysis_json, final_analysis_json_path = self.analysis_crew.start_crew_process(formatted_eval_path)
                            if final_analysis_json and 'error' not in final_analysis_json: # Check for a valid, non-error response
                                self.logger.info(f"AnalysisCrew final_analysis_json: {final_analysis_json}") # Use self.logger
                                self.logger.info(f"AnalysisCrew final_analysis_json_path: {final_analysis_json_path}") # Use self.logger
                                break # Exit loop on success
                            else:
                                self.logger.warning(f"AnalysisCrew returned an error or empty response on attempt {retry_count + 1}: {final_analysis_json}")
                        except Exception as e:
                            self.logger.error(f"Error calling AnalysisCrew.start_crew_process (Attempt {retry_count + 1}/{MAX_RETRIES}): {e}") # Use self.logger
                        finally:
                            retry_count += 1
                            if retry_count < MAX_RETRIES: # Only sleep if we're going to retry
                                time.sleep(5) # Wait for 5 seconds before retrying
                    else:
                        self.logger.error(f"AnalysisCrew failed after {MAX_RETRIES} attempts.")
                else:
                    self.logger.warning("Skipping analysis crew as formatted_eval_path is None.") # Use self.logger

                print(final_analysis_json)

            except Exception as e:
                self.logger.error(f"Error in crew process: {e}") # Use self.logger
                # Do not return None, None here. The variables are already initialized to None or will be handled downstream.
                # This ensures the overall pipeline continues.

            # Initialize variables with default values for error handling
            output_file_path = None # Initialize to None
            score_to_write = 0 # Initialize to 0

            self.logger.info("Starting HTML report generation.") # Use self.logger
            try:
                # Removing the unscored keys from the formatted_eval_path
                formatted_eval_path_stripped = None # Initialize to None
                if formatted_eval_path is not None:
                    self.logger.info(f"Creating stripped copy of JSON from: {formatted_eval_path}") # Use self.logger
                    formatted_eval_path_stripped = self.make_copy_of_json(formatted_eval_path, "_stripped")
                    self.strip_unscored_keys(formatted_eval_path_stripped)
                    self.logger.info(f"Stripped JSON saved to: {formatted_eval_path_stripped}") # Use self.logger
                else:
                    self.logger.warning("formatted_eval_path is None, cannot strip unscored keys.") # Use self.logger
                    formatted_eval_path_stripped = None

                # Storing the english version of report for evaluation purpose
                html_report = None

                if formatted_eval_path_stripped is not None and final_analysis_json_path is not None:
                    self.logger.info(f"Generating English HTML report from {formatted_eval_path_stripped} and {final_analysis_json_path}") # Use self.logger
                    html_report, output_file_path = save_proposal_evaluation_html(formatted_eval_path_stripped, final_analysis_json_path, 'proposal_eval_final_outputs/evaluation_proposal_eval.html', "english")
                    self.logger.info('Final English HTML output stored in the html file') # Use self.logger
                else:
                    self.logger.warning("Skipping English HTML report generation due to missing paths.") # Use self.logger

                # Saving the final html output
                if rfp_language != "english":
                    if formatted_eval_path_stripped is not None:
                        self.logger.info(f"Translating formatted evaluation to Arabic from: {formatted_eval_path_stripped}") # Use self.logger
                        formatted_eval_path_eng = self.make_copy_of_json(formatted_eval_path_stripped)
                        formatted_eval_arabic = self.translator.process_json_file_for_proposal_eval(formatted_eval_path_stripped, formatted_eval_path_stripped)
                    else:
                        self.logger.warning("Skipping Arabic evaluation translation due to missing formatted_eval_path_stripped.") # Use self.logger

                    if final_analysis_json_path is not None:
                        self.logger.info(f"Translating final analysis to Arabic from: {final_analysis_json_path}") # Use self.logger
                        final_analysis_json_path_eng = self.make_copy_of_json(final_analysis_json_path)
                        formatted_analysis_arabic = self.translator.process_json_file_for_proposal_eval(final_analysis_json_path, final_analysis_json_path)
                    else:
                        self.logger.warning("Skipping Arabic analysis translation due to missing final_analysis_json_path.") # Use self.logger

                # The formatted_eval_arabic, formatted_analysis_arabic are not used as they are JSONs, so we need to pass the path that JSON, not the original JSON for generating the html report
                if formatted_eval_path_stripped is not None and final_analysis_json_path is not None:
                    self.logger.info(f"Generating final HTML report in {rfp_language} from {formatted_eval_path_stripped} and {final_analysis_json_path}") # Use self.logger
                    html_report,output_file_path = save_proposal_evaluation_html(formatted_eval_path_stripped, final_analysis_json_path, 'proposal_eval_final_outputs/proposal_eval.html', rfp_language)
                    self.logger.info("Final HTML report generated.") # Use self.logger
                else:
                    self.logger.warning("Skipping final HTML report generation due to missing paths.") # Use self.logger

                print("*****************")
                print(html_report)
                print("*****************")
                print(output_file_path)
                print("*****************")
                # End the timer
                end_time = time.time()

                # Calculate and self.logger.info elapsed time
                elapsed_minutes = (end_time - start_time) / 60
                self.logger.info(f"Execution time: {elapsed_minutes:.4f} minutes") # Use self.logger
                
                dir = "score_tracker"
                os.makedirs(dir, exist_ok=True)
                # Ensure Total_assigned_score is not None before writing
                score_to_write = Total_assigned_score if Total_assigned_score is not None else 0
                with open(f"{dir}/score_tracker.txt", "w") as f:
                    f.write(str(score_to_write))
                self.logger.info(f"Score tracker saved to {dir}/score_tracker.txt with score: {score_to_write}") # Use self.logger
                
                # return html_report_path
                self.logger.info("Exiting process_pdfs method successfully.") # Use self.logger
                return output_file_path, score_to_write # Use score_to_write to ensure it's not None

            except Exception as e:
                self.logger.error(f"Error during HTML report generation or score tracking: {e}") # Use self.logger
                return None, 0 # Return None for path and 0 for score on error

        except Exception as e:
            self.logger.error(f"An unexpected error occurred in process_pdfs: {e}") # Use self.logger
            return None, 0

if __name__ == "__main__":
    pipeline = ProposalEvalPipeline()
    pipeline.process_pdfs(rfp_pdf_path=["/home/ubuntu/Tendor_POC/custom_proposal/tendor_poc/proposal_test/sample_data/p2-1-2.pdf"], proposal_pdf_path=["/home/ubuntu/Tendor_POC/custom_proposal/tendor_poc/proposal_test/sample_data/p2-1-2.pdf"], tmp_output_path="tmp_proposal_eval/tmp_output")