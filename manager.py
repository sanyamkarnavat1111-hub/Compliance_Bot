from worker import CompletenessWorker, ProposalWorker
from database import DATABASE
from config import NUMBER_OF_WORKERS, COMPLETENESS_WORKERS, PROPOSAL_WORKERS
from logger import custom_logger as llog
import uuid
import datetime
import threading
import time

from main import RFPCompleteness
from main_proposal_eval import ProposalEvaluation
# from model_provider import ModelProvider  # Disabled - CUDA not available

class Manager():
    def __init__(self):
        print(11111111111111111111111111111111111111111111)
        initial_log_id = self.getUniqueFileNameForLogger()
        print(22222222222222222222222222222222222222222222222)
        llog("Manager", "Starting manager initialization", "Manager_tracker_"+initial_log_id)

        self.database_obj = DATABASE(self)
        print("Database object created")
        llog("Manager", "Database connection initialized", "Manager_tracker_"+initial_log_id)
        self.manager_lock = threading.Lock()
        llog("Manager", "Acquired manager lock", "Manager_tracker_"+initial_log_id)
        self.database_obj.ensureRfpColumnExists()
        self.database_obj.addEnumValue4IfNotExists()
        self.database_obj.convertUnderProcessToNeedToProcess()

        llog("Manager", "ModelProvider disabled - CUDA not available", "Manager_tracker_"+initial_log_id)
        # self.model_provider = ModelProvider(log_save_file_name="Manager_tracker_"+initial_log_id)  # Disabled - CUDA not available
        self.model_provider = None
        print("--------------------------------")
        print(f"model_provider : ", {self.model_provider})
        print("--------------------------------")
        print("model_provider type : ", type(self.model_provider))
        print("--------------------------------")

        llog("Manager", "Continuing with manager initialization", "Manager_tracker_"+initial_log_id)
        llog("Manager", "Initializing RFPCompleteness module", "Manager_tracker_"+initial_log_id)
        self.rfp_completeness = RFPCompleteness(self.model_provider)
        llog("Manager", "Initializing ProposalEvaluation module", "Manager_tracker_"+initial_log_id)
        self.proposal_evaluation = ProposalEvaluation(self.model_provider)

        # --- Modern thread startup (like new code) ---
        self.completeness_workers = [
            CompletenessWorker(self, f"c{i}", self.rfp_completeness, self.model_provider)
            for i in range(COMPLETENESS_WORKERS)
        ]
        for worker in self.completeness_workers:
            worker.start()

        self.proposal_workers = [
            ProposalWorker(self, f"p{i}", self.proposal_evaluation, self.model_provider)
            for i in range(PROPOSAL_WORKERS)
        ]
        for worker in self.proposal_workers:
            worker.start()

        llog("Manager", "Manager initialization complete, all workers started", "Manager_tracker_"+initial_log_id)
        # Wait for all workers to finish (if desired, e.g., for a batch mode or graceful shutdown)
        self.wait_for_workers()

    def wait_for_workers(self):
        try:
            while True:
                # Wait until only the main thread is left (all workers done)
                active_threads = threading.active_count()
                if active_threads <= 1:
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            llog("Manager", "Graceful shutdown initiated", "shutdown")
        finally:
            # Attempt to join all workers for clean exit
            for worker in self.completeness_workers + self.proposal_workers:
                if worker.is_alive():
                    worker.join()

    # All existing methods (unchanged)
    def getCompletenessJob(self):
        print("getCompletenessJob")
        llog("Manager", "Retrieving completeness job from database", "Manager_tracker_getCompletenessJob")
        result, log_save_file_name = self.database_obj.getCompletenessData()
        llog("Manager", f"Retrieved completeness job with ID: {result if result else None}", "Manager_tracker_getCompletenessJob")
        llog("Manager", f"Retrieved completeness job with ID: {result[0] if result else None}", "Manager_tracker_getCompletenessJob")
        return result, log_save_file_name

    def getProposalJob(self):
        result, log_save_file_name = self.database_obj.getProposalData()
        llog("Manager", f"Retrieved proposal job with ID: {result[0] if result else None}", "Manager_tracker_getProposalJob")
        return result, log_save_file_name

    def submitJob(self, id, status, log_save_file_name, encrypted_json_output, process_type, completeness_score=None, technical_error_msg=None):
        if status == "processed":
            llog("Manager", f"Updating {process_type} job ID {id} to processed", "Manager_tracker_")
            if process_type == "proposal_evaluation":
                self.database_obj.updateDatabaseProcessed(id, encrypted_json_output, completeness_score, log_save_file_name, process_type)
            else:  # completeness_check
                self.database_obj.updateDatabaseProcessed(id, encrypted_json_output, None, log_save_file_name, process_type)
        elif status == "error":
            llog("Manager", f"Updating {process_type} job ID {id} to error", "Manager_tracker_")
            if process_type == "proposal_evaluation":
                self.database_obj.updateDatabaseError(id, encrypted_json_output, completeness_score, log_save_file_name, technical_error_msg, process_type)
            else:  # completeness_check
                self.database_obj.updateDatabaseError(id, encrypted_json_output, None, log_save_file_name, technical_error_msg, process_type)

    def getUniqueFileNameForLogger(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())
        log_save_file_name = f"{timestamp}_{unique_id}"
        return log_save_file_name

if __name__ == "__main__":
    manager_obj = Manager()