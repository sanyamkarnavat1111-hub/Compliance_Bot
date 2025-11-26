from Thread import Thread
import time
import json
from logger import custom_logger as llog
from send_mail import send_mail_fun
from config import SENDER_MAIL, SENDER_PASSWORD, DEVELOPER_RECEIVER_MAIL, CLIENT_RECEIVER_MAIL, PROPER_ERROR_MESSEGE_LIST_TO_SHOW_USER
from config import WORKER_IDLE_TIMEOUT_SECONDS, WORKER_POLL_INTERVAL_SECONDS
from random import choices
from check_process_db import check_rfp_status,check_proposal_status
from config import DEVELOPER_RECEIVER_MAIL, CLIENT_RECEIVER_MAIL, SENDER_MAIL, SENDER_PASSWORD
from config import DATABASE_HOST, DATABASE_USER, USER_PASSWORD, USED_DATABASE

class BaseWorker(Thread):
    """Base worker class that specialized process workers inherit from"""
    def __init__(self, manager, id):
        """
        Initializes the BaseWorker instance with reference to the manager.

        Args:
            manager (Manager): The manager instance from which jobs are fetched.
            id (str): The unique identifier for the worker.
        """
        Thread(id)
        self.name = id
        self.m_manager = manager
        self.send_mail = True
        self.last_job_time = time.time()  # Track when we last had a job

class CompletenessWorker(BaseWorker):
    """Worker specialized for RFP completeness check process"""
    def __init__(self, manager, id, rfp_completeness, model_provider=None):
        """
        Initializes the CompletenessWorker instance.

        Args:
            manager (Manager): The manager instance from which jobs are fetched.
            id (str): The unique identifier for the worker.
            rfp_completeness (RFPCompleteness): The completeness check processing module.
            model_provider (ModelProvider, optional): Shared model provider instance.
        """
        super().__init__(manager, id)
        self.rfp_completeness = rfp_completeness
        self.model_provider = model_provider
        self.process_type = "completeness_check"
        
    def run(self):
        """
        Executes the worker thread, continuously fetching and processing completeness check jobs from the manager.
        Handles any exceptions that occur during job retrieval, processing, or submission.
        """
        while self.isWait():
            try:
                # Check for idle timeout
                if WORKER_IDLE_TIMEOUT_SECONDS > 0:
                    idle_time = time.time() - self.last_job_time
                    if idle_time > WORKER_IDLE_TIMEOUT_SECONDS:
                        llog("CompletenessWorker", f"Worker {self.name} shutting down after {idle_time:.1f}s of inactivity", "worker_timeout")
                        print(f"Worker {self.name} shutting down due to inactivity")
                        self.stop()  # Stop the thread
                        break
                
                print(f"======= {self.name} going to take completeness check job ==========")
                result, log_save_file_name = self.m_manager.getCompletenessJob()
                print(f"======= run :::: {self.name} with process_type: {self.process_type} ======== with result is {result}===")
            except Exception as e:
                print(f"Error getting completeness job: {e}")
                time.sleep(WORKER_POLL_INTERVAL_SECONDS)
                continue
                
            try:
                if result is not None:
                    self.last_job_time = time.time()  # Reset idle timer
                    self._handle_completeness_check(result, log_save_file_name)
                else:
                    time.sleep(WORKER_POLL_INTERVAL_SECONDS)
            except Exception as e:
                technical_error_msg = f"Error occurred in completeness worker: {str(e)}"
                error_msg_for_user = "Something Went Wrong. Please try again later"
                new_json_output = {
                    "success": False,
                    "message": error_msg_for_user,
                    "data": {}
                }
                
                json_output = json.dumps(new_json_output)
                llog("CompletenessWorker", f"Error in worker thread: {str(e)}", log_save_file_name)
                
                if result is not None:
                    # Send mail to developers
                    subject = f"Error in {self.process_type}"
                    msg = f"""Error in {self.process_type} with id = {result[0]}, 
                            user_displayed_message = {error_msg_for_user}, 
                            technical_error_msg = {technical_error_msg}"""
                    send_mail_fun(DEVELOPER_RECEIVER_MAIL, SENDER_MAIL, SENDER_PASSWORD, subject, msg)
                    
                    # Submit error job
                    self.m_manager.submitJob(result[0], "error", log_save_file_name, json_output, self.process_type, technical_error_msg=technical_error_msg)
    
    def _handle_completeness_check(self, result, log_save_file_name):
        """
        Handle completeness check process.
        
        Args:
            result: Job data from database  
            log_save_file_name: Unique identifier for logging
        """
        id, model, rfp_file_path, ea_standard_eval_file_path, output_tokens, industry_standards, ministry_compliances, output_language = result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]
        
        llog("CompletenessWorker", f"Completeness Check - id: {id}, model: {model}, rfp_path: {rfp_file_path}, ea_path: {ea_standard_eval_file_path}", log_save_file_name)
        
        # Convert model type - gets model directly from database for each job
        if model == '2':
            model_type = 'opensource'
        else:
            model_type = 'openai'  # Default to 'openai' for any value that's not '2'
            
        llog("CompletenessWorker", f"Using model type: {model_type} for job id: {id}", log_save_file_name)
        
        # Acquire the model if using opensource
        llog("CompletenessWorker", "Calling model_aquire", log_save_file_name)
        model_acquired = False
        token = None
        
        if model_type == 'opensource' and self.model_provider:
            token=self.model_provider.acquire_model()
            model_acquired = True
            llog("CompletenessWorker", "Model acquired", log_save_file_name)
        
        try:
            llog("CompletenessWorker", "Starting completeness check functionality", log_save_file_name)
            
            # Process the completeness check - RFPCompleteness will set model type in the model provider
            try:
                rfp_id, output, error_msg_for_user, technical_error_msg = self.rfp_completeness.is_complete(
                    id=id,
                    model=model_type,
                    rfp_url=rfp_file_path,
                    ea_standard_eval_url=ea_standard_eval_file_path,
                    log_save_file_name=log_save_file_name,
                    output_tokens=output_tokens,
                    industry_standards=industry_standards,
                    ministry_compliances=ministry_compliances,
                    output_language=output_language
                )
            except Exception as e:
                technical_error_msg = f"Unhandled exception in is_complete: {str(e)}"
                error_msg_for_user = "An error occurred while processing the PDF."
                llog("CompletenessWorker", technical_error_msg, log_save_file_name)
                json_output = {
                    "success": False,
                    "message": error_msg_for_user,
                    "data": {}
                }
                self.m_manager.submitJob(id, "error", log_save_file_name, json.dumps(json_output), self.process_type, None, technical_error_msg)
                return
            llog("CompletenessWorker", "Completeness check functionality completed", log_save_file_name)
            llog("CompletenessWorker", f"RFP ID: {rfp_id}", log_save_file_name)
            llog("CompletenessWorker", f"WEB Output: {output}", log_save_file_name)
            llog("CompletenessWorker", f"WEB Error Message for User: {error_msg_for_user}", log_save_file_name)
            llog("CompletenessWorker", f"WEB Technical Error Message: {technical_error_msg}", log_save_file_name)
            
            if output:
                llog("CompletenessWorker", "Output is not None, preparing to submit job", log_save_file_name)
                # completeness_score = output.get("score", {})
                # llog("CompletenessWorker", f"Completeness Score: {completeness_score}", log_save_file_name)
                json_output = {
                    "success": True,
                    "message": "Process Completed Successfully",
                    "data": {
                        "result": output.get("result", {}),
                        "model": model_type
                    }
                }
                llog("CompletenessWorker", f"JSON Output: {json_output}", log_save_file_name)
                
                json_output_str = json.dumps(json_output)
                self.m_manager.submitJob(id, "processed", log_save_file_name, json_output_str, self.process_type, None)
            elif error_msg_for_user or technical_error_msg:
                llog("CompletenessWorker", "error block called", log_save_file_name)
                completeness_score = None
                json_output = {
                    "success": False,
                    "message": error_msg_for_user,
                    "data": {}
                }
                
                json_output_str = json.dumps(json_output)
                self.m_manager.submitJob(id, "error", log_save_file_name, json_output_str, self.process_type, completeness_score, technical_error_msg)

            # Call the function
            result = check_rfp_status(DATABASE_HOST, DATABASE_USER, USER_PASSWORD, USED_DATABASE)
            print(result)

            # Send mail to developers
            if "process completed successfully" or "technical_error" in result:
                subject = f"Successfuly finish :{self.process_type}"
                msg = f"""complete process in {self.process_type} with id = {id}, 
                        Response of  = {json_output}"""
                send_mail_fun(DEVELOPER_RECEIVER_MAIL, SENDER_MAIL, SENDER_PASSWORD, subject, msg)
                llog("completeness-Worker", f"Mail sent to developers: {subject}", log_save_file_name)
                # # Submit error job

        finally:
            # Release the model if it was acquired
            if model_acquired:
                self.model_provider.release_model()

class ProposalWorker(BaseWorker):
    """Worker specialized for proposal evaluation process"""
    def __init__(self, manager, id, proposal_evaluation, model_provider=None):
        """
        Initializes the ProposalWorker instance.

        Args:
            manager (Manager): The manager instance from which jobs are fetched.
            id (str): The unique identifier for the worker.
            proposal_evaluation (ProposalEvaluation): The proposal evaluation module.
            model_provider (ModelProvider, optional): Shared model provider instance.
        """
        super().__init__(manager, id)
        self.proposal_evaluation = proposal_evaluation
        self.model_provider = model_provider
        self.process_type = "proposal_evaluation"
        
    def run(self):
        """
        Executes the worker thread, continuously fetching and processing proposal evaluation jobs from the manager.
        Handles any exceptions that occur during job retrieval, processing, or submission.
        """
        while self.isWait():
            try:
                # Check for idle timeout
                if WORKER_IDLE_TIMEOUT_SECONDS > 0:
                    idle_time = time.time() - self.last_job_time
                    if idle_time > WORKER_IDLE_TIMEOUT_SECONDS:
                        llog("ProposalWorker", f"Worker {self.name} shutting down after {idle_time:.1f}s of inactivity", "worker_timeout")
                        print(f"Worker {self.name} shutting down due to inactivity")
                        self.stop()  # Stop the thread
                        break
                
                print(f"<<<<<<<<< {self.name} going to take proposal evaluation job >>>>>>>>>>>>")
                result, log_save_file_name = self.m_manager.getProposalJob()
                print(f"<<<<<<<<< run :::: {self.name} with process_type: {self.process_type} >>>>>>>>>>>")
            except Exception as e:
                print(f"Error getting proposal job: {e}")
                time.sleep(WORKER_POLL_INTERVAL_SECONDS)
                continue
                
            try:
                if result is not None:
                    self.last_job_time = time.time()  # Reset idle timer
                    self._handle_proposal_evaluation(result, log_save_file_name)
                else:
                    time.sleep(WORKER_POLL_INTERVAL_SECONDS)
            except Exception as e:
                technical_error_msg = f"Error occurred in proposal worker: {str(e)}"
                error_msg_for_user = "Something Went Wrong. Please try again later"
                new_json_output = {
                    "success": False,
                    "message": error_msg_for_user,
                    "data": {}
                }
                
                json_output = json.dumps(new_json_output)
                llog("ProposalWorker", f"Error in worker thread: {str(e)}", log_save_file_name)
                
                if result is not None:
                    # Send mail to developers
                    subject = f"Error in {self.process_type}"
                    msg = f"""Error in {self.process_type} with id = {result[0]}, 
                            user_displayed_message = {error_msg_for_user}, 
                            technical_error_msg = {technical_error_msg}"""
                    send_mail_fun(DEVELOPER_RECEIVER_MAIL, SENDER_MAIL, SENDER_PASSWORD, subject, msg)
                    
                    # Submit error job
                    self.m_manager.submitJob(result[0], "error", log_save_file_name, json_output, self.process_type, None, technical_error_msg)
    
    def _handle_proposal_evaluation(self, result, log_save_file_name):
        """
        Handle proposal evaluation process.
        
        Args:
            result: Job data from database
            log_save_file_name: Unique identifier for logging
        """
        id, rfp_uid, rfp_file_path, proposal_file_path, format_of_response, model, output_language = result[0], result[1], result[2], result[3], result[4], result[5], result[6]
        
        llog("ProposalWorker", f"Proposal Evaluation - id: {id}, model: {model}, rfp_path: {rfp_file_path}, proposal_path: {proposal_file_path}", log_save_file_name)
        
        # Convert model type - gets model directly from database for each job
        if model == '2':
            model_type = 'opensource'
        else:
            model_type = 'openai'  # Default to 'openai' for any value that's not '2'
            
        llog("ProposalWorker", f"Using model type: {model_type} for job id: {id}", log_save_file_name)
        
        # Acquire the model if using opensource
        model_acquired = False
        if model_type == 'opensource' and self.model_provider:
            self.model_provider.acquire_model()
            model_acquired = True
        
        try:
            llog("ProposalWorker", "Starting proposal evaluation functionality", log_save_file_name)
            
            # Process the proposal evaluation - ProposalEvaluation will set model type in the model provider

            llog("ProposalWorker", f"Value pass to Start_evaluation is  rfp_id: {rfp_uid}, model: {model_type}, rfp_file_path: {rfp_file_path}, proposal_file_path: {proposal_file_path}, log_save_file_name: {log_save_file_name}, log_save_file_name")
            
            rfp_id, output, error_msg_for_user, technical_error_msg = self.proposal_evaluation.start_evaluation(
                rfp_id=rfp_uid,
                model=model_type,
                rfp_url=rfp_file_path,
                proposal_url=proposal_file_path,
                log_save_file_name=log_save_file_name,
                output_language=output_language
            )
            
            llog("ProposalWorker", "Proposal evaluation completed", log_save_file_name)
            
            if output:
                completeness_score = output.get("score", {})
                json_output = {
                    "success": True,
                    "message": "Process Completed Successfully",
                    "data": {
                        "result": output.get("results", {}),
                        "model": model_type
                    }
                }
                
                json_output_str = json.dumps(json_output)
                self.m_manager.submitJob(id, "processed", log_save_file_name, json_output_str, self.process_type, completeness_score)
            elif error_msg_for_user or technical_error_msg:
                completeness_score = None
                json_output = {
                    "success": False,
                    "message": error_msg_for_user,
                    "data": {}
                }
                
                json_output_str = json.dumps(json_output)
                self.m_manager.submitJob(id, "error", log_save_file_name, json_output_str, self.process_type, completeness_score, technical_error_msg)


            # Set your credentials
            DATABASE_HOST = '20.46.197.51'
            DATABASE_USER = 'compliance_bot_ai'
            USER_PASSWORD = 'nQVdQsU9&F&bt'
            USED_DATABASE= 'compliance_bot'

            # Call the function
            result = check_proposal_status(DATABASE_HOST, DATABASE_USER, USER_PASSWORD, USED_DATABASE)
            print(result)

            # Send mail to developers
            if "process completed successfully" or "technical_error" in result:
                subject = f"Successfuly finish :{self.process_type}"
                msg = f"""complete process in {self.process_type} with id = {id}, 
                        Response of  = {json_output}"""
                send_mail_fun(DEVELOPER_RECEIVER_MAIL, SENDER_MAIL, SENDER_PASSWORD, subject, msg)
                llog("ProposalWorker", f"Mail sent to developers: {subject}", log_save_file_name)
                # # Submit error job
                # self.m_manager.submitJob(id, "error", log_save_file_name, json_output_str, self.process_type, completeness_score, technical_error_msg)

        finally:
            # Release the model if it was acquired
            if model_acquired:
                self.model_provider.release_model()

