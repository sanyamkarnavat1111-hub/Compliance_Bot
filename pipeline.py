#!/usr/bin/env python
import sys
import warnings
import os
import shutil
import time
from logger_config import get_logger

logger = get_logger(__file__)

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_parsing_vision import PDFParser
from chunk_processing_rfp import ChunkProcessor
from embedding_generation import EmbeddingGeneration
from embedding_management import EmbeddingManager
from is_arabic_detection import ArabicEnglishDetector
from procurement_agentic_system.crew import RFPEvaluationCrew
from procurement_agentic_system.duplicated_group_categorizer_crew import DuplicatedGroupCategorizer
from procurement_agentic_system.group_deduplicate_decision_maker_crew import GroupDeduplicateDecisionMaker
from pdf_auto_translation_experiment import PDFVisionProcessor
from posprocessing import clean_and_report
from json_to_html_convertor_rfp_correction import save_html_file
from procurement_agentic_system.arabic_translation_crew import ArabicJsonTranslator

class Pipeline:
    def __init__(self):
        try:
            self.max_chunk_size = 2048
            self.embedding_dim = 768
            self.top_k_chunks = 5
            self.pdf_parser = PDFParser()
            self.pdf_processor = PDFVisionProcessor()
            self.chunk_processor = ChunkProcessor(max_chunk_size=self.max_chunk_size)
            self.embedding_generation = EmbeddingGeneration(max_len=self.max_chunk_size)
            self.embedding_manager = EmbeddingManager(embedding_dim=self.embedding_dim, top_k_chunks=self.top_k_chunks)
            self.arabic_english_detector = ArabicEnglishDetector()
            self.duplicated_group_categorizer = DuplicatedGroupCategorizer()
            self.group_deduplicate_decision_maker = GroupDeduplicateDecisionMaker()
        except Exception as e:
            logger.info(f"Error initializing Pipeline: {e}")
            raise

    def read_entity_requirement_file(self, entity_path):
        try:
            with open(entity_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.info(f"Entity requirement file not found: {entity_path}")
            return ""
        except Exception as e:
            logger.info(f"Error reading entity requirement file: {e}")
            return ""

    def process_pdfs(self, rfp_pdf_path: list, ea_standard_pdf_path: str, entity_path: str, tmp_output_path: str, output_language: str):
        """
        Process multiple PDF files through the pipeline.
        
        Args:
            rfp_pdf_path: List of paths to the RFP PDF files to process
            ea_standard_pdf_path: Path to the EA Standard PDF file
            entity_path: Path to the entity requirement file
            tmp_output_path: Path to save the processed output
        """
        try:
            # Start the timer
            start_time = time.time()
            
            # Cleaning up output folder
            try:
                shutil.rmtree("outputs/", ignore_errors=True)
                shutil.rmtree("final_outputs/", ignore_errors=True)
                shutil.rmtree("tmp/", ignore_errors=True)
            except Exception as e:
                logger.info(f"Warning: Failed to clean up directories: {e}")
            
            # Create the tmp folder and the tmp_output.txt file
            try:
                os.makedirs('tmp', exist_ok=True)
                with open('tmp/tmp_output.txt', 'w') as f:
                    f.write('')
            except Exception as e:
                logger.info(f"Error creating temporary files: {e}")
                raise
        
            logger.info('--------------------------EA STANDARD TEXT PROCESS STARTED--------------------------------------')
            
            try:
                ea_standard_text = self.pdf_processor.process_and_save(ea_standard_pdf_path, output_filename=tmp_output_path+"_ea_standard_vision.txt")
                
                # Convert text to string if it's a list
                if isinstance(ea_standard_text, list):
                    ea_standard_text = "\n\n ".join(ea_standard_text)
                else:
                    ea_standard_text = ea_standard_text

                ea_standard_text, ea_standard_report = clean_and_report(ea_standard_text)
                logger.info(ea_standard_report)
                
                ea_standard_language = self.arabic_english_detector.detect_language("\n\n ".join(self.pdf_parser.parse_pdfs_to_text(input_pdfs = ea_standard_pdf_path, output_txt_path = tmp_output_path+"_ea_standard_pdf_extraction.txt")))
            except Exception as e:
                logger.info(f"Error processing EA standard text: {e}")
                raise

            logger.info('--------------------------EA STANDARD TEXT PROCESS COMPLETED--------------------------------------')
            logger.info(ea_standard_text)
            logger.info('--------------------------RFP TEXT PROCESS STARTED--------------------------------------')

            try:
                rfp_text = self.pdf_processor.process_and_save(rfp_pdf_path, output_filename=tmp_output_path+"_rfp_vision.txt")

                # Convert text to string if it's a list
                if isinstance(rfp_text, list):
                    rfp_text = "\n\n ".join(rfp_text)
                else:
                    rfp_text = rfp_text
                    
                rfp_text, rfp_report = clean_and_report(rfp_text)
                logger.info(rfp_report)

                #old method + with OCR
                # rfp_language = self.arabic_english_detector.detect_language("\n\n ".join(self.pdf_parser.parse_pdfs_to_text(input_pdfs = rfp_pdf_path, output_txt_path = tmp_output_path+"_rfp_pdf_extraction.txt")))

                #latest from DB 
                rfp_language = output_language
            except Exception as e:
                logger.info(f"Error processing RFP text: {e}")
                raise
            
            logger.info(rfp_text)
            
            try:
                rfp_chunks = self.chunk_processor.create_chunks(rfp_text)
                rfp_doc_embeddings = self.embedding_generation.generate_embeddings(rfp_chunks)
            except Exception as e:
                logger.info(f"Error processing chunks and embeddings: {e}")
                raise

            try:
                # Here, The auto old data delete functionality is being added, to remove the old data from the embedding manager
                self.embedding_manager.clean_embedding_db()
                self.embedding_manager.add_embeddings(rfp_chunks, rfp_doc_embeddings)
            except Exception as e:
                logger.info(f"Error managing embeddings: {e}")
                raise

            logger.info('--------------------------RFP TEXT PROCESS COMPLETED--------------------------------------')

            try:
                # Static sections + read entity sections
                additional_sections_entity = 'Scope of Work, Project Delivery, Cyber Security, ' + self.read_entity_requirement_file(entity_path)
            except Exception as e:
                logger.info(f"Error reading additional sections: {e}")
                additional_sections_entity = 'Scope of Work, Project Selivery, Cyber Security'
            
            logger.info('--------------------------- ADDITIONAL SECTIONS ENTITY ---------------------------')
            logger.info(additional_sections_entity)

            try:
                # Running crewAI multi agentic systems
               
                crew = RFPEvaluationCrew(ea_standard_text=ea_standard_text, additional_sections=additional_sections_entity)
                formatted_improvements, saved_json_path, saved_arabic_json_path = crew.start_crew_process()

                duplicated_group_categorizer_files_paths = self.duplicated_group_categorizer.process_improvements_and_save_all_outputs(input_json_path=saved_json_path, output_dir='outputs/duplicated_group_categorizer_crew')

                formatted_improvements, saved_json_path, saved_arabic_json_path = self.group_deduplicate_decision_maker.add_non_duplicate_items(duplicated_group_categorizer_files_paths['duplicate_groups_path'], duplicated_group_categorizer_files_paths['unique_items_path'])

            except Exception as e:
                logger.info(f"Error in crew process: {e}")
                raise
            
            # Storing the english version of report for evaluation purpose
            save_html_file(formatted_improvements, "final_outputs/report_for_evaluation.html", language="english")

            if rfp_language != "english":
                translator = ArabicJsonTranslator()
                formatted_improvements = translator.process_json_file_for_rfp_eval(saved_json_path, saved_arabic_json_path)

            save_html_file(formatted_improvements, "final_outputs/report.html", language=rfp_language)
            
            logger.info("final_outputs/report.html")

            # End the timer
            end_time = time.time()

            # Calculate and logger.info elapsed time
            elapsed_minutes = (end_time - start_time) / 60
            logger.info(f"Execution time: {elapsed_minutes:.4f} minutes")

            # Return the final report path for downstream use
            return "final_outputs/report.html"
        except Exception as e:
            logger.info(f"Critical error in process_pdfs: {e}")
            raise


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.process_pdfs(rfp_pdf_path=["sample_data/4.2_neel/4.2/01 RFP.pdf"], ea_standard_pdf_path=["sample_data/4.2_neel/4.2/02 EA Standards.pdf"], entity_path="sample_data/4.2_neel/4.2/03 sections.txt", tmp_output_path="tmp/tmp_output")