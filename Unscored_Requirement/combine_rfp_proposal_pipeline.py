from logger_config import get_logger
import os
from typing import Optional, Tuple
from Unscored_Requirement.rfp_requirement_extractor import RFPRequirementExtractorAgent
from Unscored_Requirement.proposal_evaluator_agent import ProposalEvaluatorAgent
from Unscored_Requirement.json_to_html import generate_html_report
import json


class UnScoredRequirementWorkflow:
    """
    A comprehensive workflow class for RFP analysis and proposal evaluation.
    
    This class orchestrates the end-to-end process of extracting requirements from an RFP,
    evaluating a proposal against those requirements, and generating a final HTML report.
    """
    
    def __init__(self, rfp_input_path: str, proposal_input_path: str, output_directory: str = "analysis_outputs"):
        """
        Initialize the RFP Analysis Workflow.
        
        Args:
            rfp_input_path (str): The file path for the input RFP document.
            proposal_input_path (str): The file path for the input proposal document.
            output_directory (str): The directory where all output files will be stored.
                                  Defaults to "analysis_outputs".
        """
        self.rfp_input_path = rfp_input_path
        self.proposal_input_path = proposal_input_path
        self.output_directory = output_directory
        
        # Define output file paths
        self.rfp_requirements_output = os.path.join(output_directory, "rfp_requirements.json")
        self.evaluation_output = os.path.join(output_directory, "proposal_evaluation.json")
        self.html_report_output = os.path.join(output_directory, "evaluation_report.html")
        
        # Initialize workflow state
        self.is_initialized = False
        self.current_step = 0
        self.total_steps = 3
    
    def _validate_inputs(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that input files exist and are accessible.
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not os.path.exists(self.rfp_input_path):
            return False, f"RFP input file not found: {self.rfp_input_path}"
        
        if not os.path.exists(self.proposal_input_path):
            return False, f"Proposal input file not found: {self.proposal_input_path}"
        
        return True, None
    
    def _create_output_directory(self) -> Tuple[bool, Optional[str]]:
        """
        Create the output directory if it doesn't exist.
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            os.makedirs(self.output_directory, exist_ok=True)
            return True, None
        except Exception as e:
            return False, f"Failed to create output directory: {e}"
    
    def _extract_rfp_requirements(self) -> Tuple[bool, Optional[str]]:
        """
        Extract requirements from the RFP document.
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        print("--- Starting Step 1: Extracting RFP Requirements ---")
        try:
            # Initialize the RFP requirement extractor agent
            requirement_extractor = RFPRequirementExtractorAgent(
                input_file_path=self.rfp_input_path,
                output_path=self.rfp_requirements_output,
            )
            
            # Execute the extraction process
            requirement_results = requirement_extractor.run_extraction()
            
            if "error" in requirement_results:
                error_message = f"RFP requirement extraction failed: {requirement_results['error']}"
                print(error_message)
                return False, error_message
            
            print(f"--- RFP Requirements extracted successfully to: {self.rfp_requirements_output} ---")
            self.current_step = 1
            return True, None
            
        except Exception as e:
            error_message = f"An unexpected error occurred during RFP extraction: {e}"
            print(error_message)
            return False, error_message
    
    def _evaluate_proposal(self) -> Tuple[bool, Optional[str]]:
        """
        Evaluate the proposal against the extracted requirements.
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        print("\n--- Starting Step 2: Evaluating Proposal ---")
        try:
            # Initialize the proposal evaluator agent
            proposal_evaluator = ProposalEvaluatorAgent(
                requirements_json_path=self.rfp_requirements_output,
                proposal_path=self.proposal_input_path,
                output_path=self.evaluation_output,
                logger_instance=get_logger(__name__)
            )
            
            # Run the evaluation
            proposal_evaluator.run_evaluation()
            print(f"--- Proposal evaluation completed. Results saved to: {self.evaluation_output} ---")
            # Read and print the evaluation results
            try:
                with open(self.evaluation_output, 'r', encoding='utf-8') as f:
                    evaluation_data = json.load(f)
                
            except Exception as e:
                print(f"Error reading evaluation results: {e}")
            self.current_step = 2
            return True, None
            
        except Exception as e:
            error_message = f"An unexpected error occurred during proposal evaluation: {e}"
            print(error_message)
            return False, error_message
    
    def _generate_html_report(self) -> Tuple[bool, Optional[str]]:
        """
        Generate the final HTML report from the evaluation results.
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message_or_json_path)
        """
        print("\n--- Starting Step 3: Generating HTML Report ---")
        try:
            # Generate the final HTML report from the evaluation results
            success, html_content, report_path = generate_html_report(
                input_json_file=self.evaluation_output,
                output_html_file=self.html_report_output
            )
            
            if not success:
                error_message = "Failed to generate the HTML report."
                print(error_message)
                return False, error_message
            
            print(f"--- HTML Report generated successfully: {report_path} ---")
            self.current_step = 3
            # Return the JSON file path instead of HTML report path
            return True, self.evaluation_output
            
        except Exception as e:
            error_message = f"An unexpected error occurred during HTML report generation: {e}"
            print(error_message)
            return False, error_message
    
    def initialize(self) -> Tuple[bool, Optional[str]]:
        """
        Initialize the workflow by validating inputs and creating output directory.
        
        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        # Validate input files
        validation_success, validation_error = self._validate_inputs()
        if not validation_success:
            return False, validation_error
        
        # Create output directory
        directory_success, directory_error = self._create_output_directory()
        if not directory_success:
            return False, directory_error
        
        self.is_initialized = True
        return True, None
    
    def run_workflow(self) -> str:
        """
        Execute the complete RFP analysis workflow.
        
        Returns:
            str: The file path of the proposal evaluation JSON file, or an error message.
        """
        # Initialize the workflow
        if not self.is_initialized:
            init_success, init_error = self.initialize()
            if not init_success:
                return init_error
        
        # Step 1: Extract RFP Requirements
        step1_success, step1_error = self._extract_rfp_requirements()
        if not step1_success:
            return step1_error
        
        # Step 2: Evaluate Proposal Against Requirements
        step2_success, step2_error = self._evaluate_proposal()
        if not step2_success:
            return step2_error
        
        # Step 3: Generate HTML Report
        step3_success, step3_result = self._generate_html_report()
        if not step3_success:
            return step3_result
        
        return step3_result
    
    def get_progress(self) -> Tuple[int, int]:
        """
        Get the current progress of the workflow.
        
        Returns:
            Tuple[int, int]: (current_step, total_steps)
        """
        return self.current_step, self.total_steps
    
    def get_output_files(self) -> dict:
        """
        Get a dictionary of all output file paths.
        
        Returns:
            dict: Dictionary containing paths to all output files.
        """
        return {
            "rfp_requirements": self.rfp_requirements_output,
            "proposal_evaluation": self.evaluation_output,
            "html_report": self.html_report_output
        }


def run_full_rfp_analysis_workflow(
    rfp_input_path: str,
    proposal_input_path: str,
    output_directory: str = "analysis_outputs"
) -> str:
    """
    Legacy function wrapper for backward compatibility.
    
    This function provides backward compatibility by wrapping the new class-based approach.
    
    Args:
        rfp_input_path (str): The file path for the input RFP document.
        proposal_input_path (str): The file path for the input proposal document.
        output_directory (str): The directory where all output files will be stored.
                                Defaults to "analysis_outputs".
    
    Returns:
        str: The file path of the proposal evaluation JSON file, or an error message.
    """
    workflow = UnScoredRequirementWorkflow(rfp_input_path, proposal_input_path, output_directory)
    return workflow.run_workflow()


if __name__ == '__main__':
    # --- Configuration for the Workflow ---
    # Define the paths for your input files
    RFP_FILE = "input_data/686fb105c518feR01u1752150277.txt"
    PROPOSAL_FILE = "input_data/687f779b35c14YH9RR1753184155.txt"
    OUTPUT_DIR = "workflow_output"

    # --- Execute the Workflow Using Class-Based Approach ---
    print("=== RFP Analysis Workflow Starting ===")
    
    # Create workflow instance
    workflow = UnScoredRequirementWorkflow(
        rfp_input_path=RFP_FILE,
        proposal_input_path=PROPOSAL_FILE,
        output_directory=OUTPUT_DIR
    )
    
    # Run the complete workflow
    evaluation_json_path = workflow.run_workflow()
    
    # --- Final Summary ---
    print("\n=== RFP Analysis Workflow Complete ===")
    if "failed" not in evaluation_json_path and "error" not in evaluation_json_path:
        print(f"The proposal evaluation JSON file is available at: {evaluation_json_path}")
        
        # Display output file information
        output_files = workflow.get_output_files()
        print("\nGenerated Files:")
        for file_type, file_path in output_files.items():
            print(f"  - {file_type}: {file_path}")
    else:
        print("The workflow encountered an error and could not be completed.")
        print(f"Error: {evaluation_json_path}")

