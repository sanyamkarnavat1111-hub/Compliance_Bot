import mysql.connector as msc
from config import DATABASE_HOST, DATABASE_USER, USER_PASSWORD, USED_DATABASE
import threading
import uuid
import datetime
from logger import custom_logger as llog # Revert to original llog import with full path
import os
from encryption_utils import EncryptionUtil
from typing import Optional
import json

class DATABASE():
    def __init__(self, manager_obj):
        """
        Initializes the DATABASE instance, setting up the database connection attributes 
        and a lock for thread-safe operations on the job processing database.
        This unified database handler works with both completeness_check (rfps table)
        and proposal_evaluation (proposals table) processes.
        """
        self.db = None
        self.cursor = None
        self.manager_obj = manager_obj
        self.jobDatabaseLock = threading.Lock()
        # Initialize EncryptionUtil
        app_key = os.getenv("APP_KEY")
        if not app_key:
            llog("Database", "APP_KEY environment variable not set. Encryption/Decryption will not be available.")
            self.encryptor = None
        else:
            try:
                self.encryptor = EncryptionUtil(app_key)
                llog("Database", "EncryptionUtil initialized successfully.")
            except ValueError as e:
                llog("Database", f"Error initializing EncryptionUtil: {e}. Encryption/Decryption will not be available.")
                self.encryptor = None
    
    def __checkDatabaseConnection(self):
        """
        Checks if the database connection is active.

        Returns:
            bool: True if the connection is active, False otherwise.
        """
        if self.db is None or not self.db.is_connected():
            return False
        return True

    
    def __connectDatabase(self):  
        """
        Establishes a connection to the MySQL database using credentials from the config file.
        If the connection fails, the database and cursor attributes are set to None.
        """
        # print("USER_PASSWORD : ", USER_PASSWORD)
        # print("DATABASE_HOST : ", DATABASE_HOST)
        # print("DATABASE_USER : ", DATABASE_USER)
        try: 
            self.db = msc.connect(
                # unix_socket='/var/run/mysqld/mysqld.sock',
                host=DATABASE_HOST,
                user=DATABASE_USER,
                password=USER_PASSWORD,
                database=USED_DATABASE
            )
            self.cursor = self.db.cursor()
        except msc.Error as e:
            print(f"Error connecting to database: {e}")
            self.db = None
            self.cursor = None

    def convertUnderProcessToNeedToProcess(self):
        """
        Updates the status of jobs marked as '2' to '1' in both the 'rfps' and 'proposals' tables.
        This method locks the database operations to ensure thread safety.
        """
        self.jobDatabaseLock.acquire()
        self.__connectDatabase()

        if self.cursor:
            # Update completeness_check jobs
            self.cursor.execute("UPDATE rfps SET need_to_check_completeness = '1' WHERE need_to_check_completeness = '2';")
            self.db.commit()
            
            # Update proposal_evaluation jobs
            self.cursor.execute("UPDATE proposals SET need_to_check_completeness = '1' WHERE need_to_check_completeness = '2';")
            self.db.commit()
            
        self.jobDatabaseLock.release()  

    def getUniqueFileNameForLogger(self):
        """
        Generates a unique filename for logging with a timestamp prefix.
        
        Returns:
            str: A unique identifier string with timestamp.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())
        log_save_file_name = f"{timestamp}_{unique_id}"
        return log_save_file_name
    
    def getCompletenessData(self):
        """
        Retrieves the next RFP completeness job from the rfps table.
        If a job is found, it is marked as '2' (in progress).

        Returns:
            tuple: (result, log_save_file_name) where:
                  - result is the job data or None if no job is found
                  - log_save_file_name is a unique identifier for logging
        """
        self.jobDatabaseLock.acquire()
        log_save_file_name = None
        result = None
        
        try:
            self.__connectDatabase()
            if self.cursor:
                print("********* going to get completeness job from rfps table *************")
                
                # Get a job from rfps table - also retrieve industry_standards and ministry_compliances
                # query_2 = "SELECT ID, type_of_rfp, rfp_file_path, ea_standard_file_path, need_to_check_completeness FROM rfps WHERE rfp_file_path IS NOT NULL order by id asc limit 5"

                # Modified query to select file_access_path and ea_file_access_path (templates) instead of direct paths
                query = "SELECT id, type_of_rfp, file_access_path, ea_file_access_path, output_tokens, industry_standards, ministry_compliances, output_language, file_access_secret_key, ea_file_access_secret_key FROM rfps WHERE need_to_check_completeness = '1' and file_access_path IS NOT NULL order by id asc limit 1"

                self.cursor.execute(query)
                # print("query has fired : ", query)
                # self.cursor.execute(query_2)
                raw_result = self.cursor.fetchone()
                # result_2 = self.cursor.fetchall()
                print("result catch from DB is : ", raw_result)

                # Use a fixed log file name for all RFP completeness jobs
                log_save_file_name = "rfp_job_processing_log"
                llog("Database", f"Fetched completeness job:{raw_result}", log_save_file_name)
                
                if raw_result is not None:
                    (id, model, rfp_file_path_template_field, ea_file_access_path_template_field, output_tokens, industry_standards, ministry_compliances, output_language, file_access_secret_key, ea_file_access_secret_key) = raw_result

                    llog("Database", f"DB Raw - rfp_file_path_template_field: {rfp_file_path_template_field}", log_save_file_name)
                    llog("Database", f"DB Raw - ea_file_access_path_template_field: {ea_file_access_path_template_field}", log_save_file_name)
                    llog("Database", f"DB Raw - file_access_secret_key: {file_access_secret_key}", log_save_file_name)
                    llog("Database", f"DB Raw - ea_file_access_secret_key: {ea_file_access_secret_key}", log_save_file_name)

                    # Directly use secret keys and construct full URLs - no decryption needed for file access keys
                    # The secret keys themselves are not encrypted Fernet tokens, but plain text tokens
                    decrypted_rfp_file_path = f"{rfp_file_path_template_field}{{secret_key}}".format(secret_key=file_access_secret_key)
                    llog("Database", f"Constructed RFP URL for ID {id}: {decrypted_rfp_file_path}", log_save_file_name)

                    # Handle cases where ea_file_access_path_template_field or ea_file_access_secret_key might be None
                    if ea_file_access_path_template_field and ea_file_access_secret_key:
                        decrypted_ea_standard_file_path = f"{ea_file_access_path_template_field}{{secret_key}}".format(secret_key=ea_file_access_secret_key)
                        llog("Database", f"Constructed EA Standard URL for ID {id}: {decrypted_ea_standard_file_path}", log_save_file_name)
                    else:
                        decrypted_ea_standard_file_path = None
                        llog("Database", f"EA Standard URL for ID {id} could not be constructed due to missing template or secret key. Set to None.", log_save_file_name)

                    # Decrypt industry_standards and ministry_compliances using Laravel decryption if encryptor is available
                    decrypted_industry_standards = industry_standards # Initialize with original in case of no encryptor or error
                    if self.encryptor:
                        llog("Database", f"Attempting to decrypt industry_standards. Raw value from DB: {industry_standards}", log_save_file_name)
                        try:
                            decrypted_industry_standards = self.encryptor.decrypt_text(industry_standards)
                            llog("Database", f"Successfully decrypted industry_standards from DB using Laravel decryption. Decrypted content (first 50 chars): {decrypted_industry_standards[:50]}...", log_save_file_name)
                        except Exception as e:
                            llog("Database", f"Laravel decryption failed for industry_standards: {e}. Returning original.", log_save_file_name)
                            decrypted_industry_standards = industry_standards # Fallback to original if decryption fails
                    else:
                        llog("Database", "No encryptor available for industry_standards. Returning original.", log_save_file_name)

                    decrypted_ministry_compliances = ministry_compliances # Initialize with original in case of no encryptor or error
                    if self.encryptor:
                        llog("Database", f"Attempting to decrypt ministry_compliances. Raw value from DB: {ministry_compliances}", log_save_file_name)
                        try:
                            decrypted_ministry_compliances = self.encryptor.decrypt_text(ministry_compliances)
                            llog("Database", f"Successfully decrypted ministry_compliances from DB using Laravel decryption. Decrypted content (first 50 chars): {decrypted_ministry_compliances[:50]}...", log_save_file_name)
                        except Exception as e:
                            llog("Database", f"Laravel decryption failed for ministry_compliances: {e}. Returning original.", log_save_file_name)
                            decrypted_ministry_compliances = ministry_compliances # Fallback to original if decryption fails
                    else:
                        llog("Database", "No encryptor available for ministry_compliances. Returning original.", log_save_file_name)

                    # Reconstruct the result tuple with decrypted values
                    result = (
                        id, model, decrypted_rfp_file_path, decrypted_ea_standard_file_path,
                        output_tokens, decrypted_industry_standards, decrypted_ministry_compliances, output_language
                    )

                    self.__updateLogfilename(id, log_save_file_name, "rfps")
                    self.__updateDatabaseProcessing(id, "rfps", log_save_file_name)
                    
                    # Debug log the results 
                    llog("Database", f"Final Constructed RFP URL for ID {id}: {decrypted_rfp_file_path}", log_save_file_name)
                    llog("Database", f"Final Constructed EA Standard URL for ID {id}: {decrypted_ea_standard_file_path}", log_save_file_name)
                    debug_result = [
                        str(id),  # id
                        str(model) if model else 'None',  # type_of_rfp
                        str(decrypted_rfp_file_path) if decrypted_rfp_file_path else 'None',  # rfp_file_path (now constructed_rfp_file_path)
                        str(decrypted_ea_standard_file_path) if decrypted_ea_standard_file_path else 'None',  # ea_standard_file_path (now constructed_ea_standard_file_path)
                        str(output_tokens) if output_tokens else 'None',  # output_tokens
                        str(decrypted_industry_standards) if decrypted_industry_standards else 'None',  # industry_standards
                        str(decrypted_ministry_compliances) if decrypted_ministry_compliances else 'None',   # ministry_compliances
                        str(output_language) if output_language else 'None' # output_language
                    ]
                    llog("Database", f"Fetched completeness job: {', '.join(debug_result)}", log_save_file_name)
        except Exception as e:
            print(f"Error retrieving completeness job: {e}")
            if log_save_file_name:
                llog("Database", f"Error retrieving completeness job: {e}", log_save_file_name)
            result = None
        finally:
            self.jobDatabaseLock.release()
            
        return result, log_save_file_name

    def getProposalData(self):
        """
        Retrieves the next proposal evaluation job from the proposals table.
        If a job is found, it is marked as '2' (in progress).

        Returns:
            tuple: (result, log_save_file_name) where:
                  - result is the job data or None if no job is found
                  - log_save_file_name is a unique identifier for logging
        """
        self.jobDatabaseLock.acquire()
        log_save_file_name = None
        result = None
        
        try:
            self.__connectDatabase()
            if self.cursor:
                print("<<<<<<<<<<<<< going to get proposal job from proposals table >>>>>>>>>>>>>>>>>>>>>")
                
                # Get a job from proposals table
                query = "SELECT id, rfp_uid, rfp_file_access_path, proposal_file_access_path, format_of_response, model, output_language, rfp_file_access_secret_key, proposal_file_access_secret_key FROM proposals WHERE need_to_check_completeness = '1' and rfp_file_access_path IS NOT NULL order by id asc limit 1"
                
                self.cursor.execute(query)
                raw_result = self.cursor.fetchone()
                
                # Use a fixed log file name for all Proposal evaluation jobs
                log_save_file_name = "proposal_job_processing_log"
                llog("Database", f"Fetched completeness job:{raw_result}", log_save_file_name)
                
                if raw_result is not None:
                    (id, rfp_uid, rfp_file_access_path_template_field, proposal_file_access_path_template_field, format_of_response, model, output_language, rfp_file_access_secret_key, proposal_file_access_secret_key) = raw_result

                    llog("Database", f"DB Raw - rfp_file_access_path_template_field: {rfp_file_access_path_template_field}", log_save_file_name)
                    llog("Database", f"DB Raw - proposal_file_access_path_template_field: {proposal_file_access_path_template_field}", log_save_file_name)
                    llog("Database", f"DB Raw - rfp_file_access_secret_key: {rfp_file_access_secret_key}", log_save_file_name)
                    llog("Database", f"DB Raw - proposal_file_access_secret_key: {proposal_file_access_secret_key}", log_save_file_name)

                    # Directly use secret keys and construct full URLs - no decryption needed for file access keys
                    # The secret keys themselves are not encrypted Fernet tokens, but plain text tokens
                    decrypted_rfp_file_path = f"{rfp_file_access_path_template_field}{{secret_key}}".format(secret_key=rfp_file_access_secret_key)
                    llog("Database", f"Constructed RFP URL for ID {id}: {decrypted_rfp_file_path}", log_save_file_name)

                    decrypted_proposal_file_path = f"{proposal_file_access_path_template_field}{{secret_key}}".format(secret_key=proposal_file_access_secret_key)
                    llog("Database", f"Constructed Proposal URL for ID {id}: {decrypted_proposal_file_path}", log_save_file_name)

                    # Reconstruct the result tuple with constructed URLs
                    result = (
                        id, rfp_uid, decrypted_rfp_file_path, decrypted_proposal_file_path,
                        format_of_response, model, output_language
                    )

                    self.__updateLogfilename(id, log_save_file_name, "proposals")
                    self.__updateDatabaseProcessing(id, "proposals", log_save_file_name)
                    
                    # Debug log the results
                    llog("Database", f"Final Constructed RFP URL for ID {id}: {decrypted_rfp_file_path}", log_save_file_name)
                    llog("Database", f"Final Constructed Proposal URL for ID {id}: {decrypted_proposal_file_path}", log_save_file_name)
                    debug_result = [
                        str(result[0]),  # id
                        str(result[1]) if result[1] else 'None',  # rfp_uid
                        str(result[2]) if result[2] else 'None',  # rfp_file_path (now constructed_rfp_file_path)
                        str(result[3]) if result[3] else 'None',  # proposal_file_path (now constructed_proposal_file_path)
                        str(result[4]) if result[4] else 'None',  # format_of_response
                        str(result[5]) if result[5] else 'None',  # model
                        str(result[6]) if result[6] else 'None'   # final_report_language
                    ]
                    llog("Database", f"Fetched proposal job: {', '.join(debug_result)}", log_save_file_name)
        except Exception as e:
            print(f"Error retrieving proposal job: {e}")
            if log_save_file_name:
                llog("Database", f"Error retrieving proposal job: {e}", log_save_file_name)
            result = None
        finally:
            self.jobDatabaseLock.release()
            
        return result, log_save_file_name
    
    def __updateLogfilename(self, id, log_save_file_name, table):
        """
        Updates the log_file_name for a job in the specified table.
        
        Args:
            id: The job ID
            log_save_file_name: The unique log file name
            table: The table to update ('rfps' or 'proposals')
            # process_logger: The logger instance for this specific process
        """
        self.__connectDatabase()
        if self.cursor:
            llog("Database", f"Updating log file name for {table} ID: {id}", log_save_file_name)
            query = f"UPDATE {table} SET log_file_name = %s WHERE id = %s"
            val = (log_save_file_name, id)
            try:
                self.cursor.execute(query, val)
                self.db.commit()
            except msc.Error as e:
                llog("Database", f"Error updating log file name: {e}", log_save_file_name)
                print(f"Error updating log file name: {e}")

    def __updateDatabaseProcessing(self, id, table, log_save_file_name):
        """
        Updates the status of a job to '2' (in progress) in the specified table.

        Args:
            id: The job ID
            table: The table to update ('rfps' or 'proposals')
            log_save_file_name: The unique log file name
            # process_logger: The logger instance for this specific process
        """
        self.__connectDatabase()
        if self.cursor:
            print(f"<<<<<<<<<<<<< updating {table} job {id} to processing status >>>>>>>>>>>>>>>>>>>>>")
            
            query = f"UPDATE {table} SET need_to_check_completeness = '2' WHERE id = %s"
            val = (id,)
            try:
                self.cursor.execute(query, val)
                self.db.commit()
                llog("Database", f"Updated {table} job {id} to processing status", log_save_file_name)
            except msc.Error as e:
                print(f"Error updating database: {e}")
                llog("Database", f"Error updating database: {e}", log_save_file_name)

    def updateDatabaseProcessed(self, id, encrypted_json_output, completeness_score, log_save_file_name, process_type):
        """
        Updates the status of a job to 'processed' (3) and stores the JSON output in the appropriate table.
        The encrypted_json_output may contain encrypted HTML report.

        Args:
            id: The job ID
            encrypted_json_output: The processed output in JSON format (may contain encrypted HTML)
            completeness_score: Score for proposal evaluation (None for completeness check)
            log_save_file_name: The unique log file name
            process_type: Either "completeness_check" or "proposal_evaluation"
            # process_logger: The logger instance for this specific process
        """
        self.jobDatabaseLock.acquire()
        table = "rfps" if process_type == "completeness_check" else "proposals"
        
        try:
            self.__connectDatabase()
            if self.cursor:
                llog("Database", f"Updating {table} job {id} to processed status", log_save_file_name)
                
                if process_type == "proposal_evaluation":
                    query = f"UPDATE {table} SET need_to_check_completeness = '3', result = %s, completeness_score = %s WHERE id = %s"
                    val = (encrypted_json_output, completeness_score, id)
                else:  # completeness_check
                    query = f"UPDATE {table} SET need_to_check_completeness = '3', result = %s WHERE id = %s"
                    val = (encrypted_json_output, id)
                
                self.cursor.execute(query, val)
                self.db.commit()
                llog("Database", f"Successfully updated {table} job {id} to processed", log_save_file_name)
        except msc.Error as e:
            print(f"Error updating database to processed: {e}")
            llog("Database", f"Error updating database to processed: {e}", log_save_file_name)
        finally:
            self.jobDatabaseLock.release()

    def updateDatabaseError(self, id, encrypted_json_output, completeness_score, log_save_file_name, technical_error, process_type):
        """
        Updates the status of a job to 'error' (4) in the appropriate table.
        The encrypted_json_output may contain encrypted HTML report.

        Args:
            id: The job ID
            encrypted_json_output: Error output in JSON format (may contain encrypted HTML)
            completeness_score: Score for proposal evaluation (None for completeness check)
            log_save_file_name: The unique log file name
            technical_error: Technical error message
            process_type: Either "completeness_check" or "proposal_evaluation"
            # process_logger: The logger instance for this specific process
        """
        self.jobDatabaseLock.acquire()
        table = "rfps" if process_type == "completeness_check" else "proposals"
        
        try:
            self.__connectDatabase()
            if self.cursor:
                llog("Database", f"Updating {table} job {id} to error status", log_save_file_name)
                
                if process_type == "proposal_evaluation":
                    query = f"UPDATE {table} SET need_to_check_completeness = '4', result = %s, completeness_score = %s, technical_error = %s WHERE id = %s"
                    val = (encrypted_json_output, completeness_score, technical_error, id)
                else:  # completeness_check
                    query = f"UPDATE {table} SET need_to_check_completeness = '4', result = %s, technical_error = %s WHERE id = %s"
                    val = (encrypted_json_output, technical_error, id)
                
                self.cursor.execute(query, val)
                self.db.commit()
                llog("Database", f"Successfully updated {table} job {id} to error", log_save_file_name)
        except msc.Error as e:
            print(f"Error updating database to error: {e}")
            llog("Database", f"Error updating database to error: {e}", log_save_file_name)
        finally:
            self.jobDatabaseLock.release()
    
    def createTableIfNotExists(self):
        """
        Creates the 'rfps' table if it doesn't exist in the database.
        """
        self.jobDatabaseLock.acquire()
        self.__connectDatabase()
        
        if self.cursor:
            # Create rfps table if it doesn't exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS rfps (
                    id INT(11) NOT NULL AUTO_INCREMENT,
                    client_id INT(11) DEFAULT NULL,
                    rfp_file_path VARCHAR(255) DEFAULT NULL, 
                    industry_standards VARCHAR(500) DEFAULT NULL,
                    ministry_compliances VARCHAR(500) DEFAULT NULL,
                    ea_standard_file_path VARCHAR(255) DEFAULT NULL,
                    output_tokens VARCHAR(20) DEFAULT NULL,
                    need_to_check_completeness ENUM('1', '2', '3', '4') DEFAULT '1',
                    result MEDIUMTEXT DEFAULT NULL,
                    technical_error TEXT DEFAULT NULL,
                    type_of_rfp VARCHAR(50) DEFAULT NULL,
                    log_file_name VARCHAR(100) DEFAULT NULL,
                    PRIMARY KEY (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
            """)
            
            # Create proposals table if it doesn't exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS proposals (
                    id INT(11) NOT NULL AUTO_INCREMENT,
                    rfp_uid VARCHAR(255) DEFAULT NULL,
                    rfp_file_path VARCHAR(255) DEFAULT NULL,
                    proposal_file_path VARCHAR(255) DEFAULT NULL,
                    format_of_response VARCHAR(20) DEFAULT NULL,
                    model VARCHAR(20) DEFAULT NULL,
                    need_to_check_completeness ENUM('1', '2', '3', '4') DEFAULT '1',
                    result MEDIUMTEXT DEFAULT NULL,
                    technical_error TEXT DEFAULT NULL,
                    completeness_score FLOAT DEFAULT NULL,
                    log_file_name VARCHAR(100) DEFAULT NULL,
                    PRIMARY KEY (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
            """)
            
            self.db.commit()
            
        self.jobDatabaseLock.release()

    def ensureRfpColumnExists(self):
        """
        Ensures that the required columns exist in the 'rfps' table.
        """
        self.jobDatabaseLock.acquire()
        self.__connectDatabase()
        
        if self.cursor:
            # Check and add columns to rfps table if they don't exist
            self.cursor.execute("SHOW COLUMNS FROM rfps")
            columns = [column[0] for column in self.cursor.fetchall()]
            
            if 'log_file_name' not in columns:
                self.cursor.execute("ALTER TABLE rfps ADD COLUMN log_file_name VARCHAR(100) DEFAULT NULL")
                
            if 'technical_error' not in columns:
                self.cursor.execute("ALTER TABLE rfps ADD COLUMN technical_error TEXT DEFAULT NULL")
                
            # Check and add columns to proposals table if they don't exist
            self.cursor.execute("SHOW COLUMNS FROM proposals")
            columns = [column[0] for column in self.cursor.fetchall()]
            
            if 'rfp_file_access_path' not in columns:
                self.cursor.execute("ALTER TABLE proposals ADD COLUMN rfp_file_access_path VARCHAR(255) DEFAULT NULL")
            if 'rfp_file_access_secret_key' not in columns:
                self.cursor.execute("ALTER TABLE proposals ADD COLUMN rfp_file_access_secret_key VARCHAR(255) DEFAULT NULL")
            if 'proposal_file_access_path' not in columns:
                self.cursor.execute("ALTER TABLE proposals ADD COLUMN proposal_file_access_path VARCHAR(255) DEFAULT NULL")
            if 'proposal_file_access_secret_key' not in columns:
                self.cursor.execute("ALTER TABLE proposals ADD COLUMN proposal_file_access_secret_key VARCHAR(255) DEFAULT NULL")

            if 'log_file_name' not in columns:
                self.cursor.execute("ALTER TABLE proposals ADD COLUMN log_file_name VARCHAR(100) DEFAULT NULL")
                
            if 'technical_error' not in columns:
                self.cursor.execute("ALTER TABLE proposals ADD COLUMN technical_error TEXT DEFAULT NULL")
                
            if 'completeness_score' not in columns:
                self.cursor.execute("ALTER TABLE proposals ADD COLUMN completeness_score FLOAT DEFAULT NULL")
                
            self.db.commit()
            
        self.jobDatabaseLock.release()

    def addEnumValue4IfNotExists(self):   
        """
        Ensures that the 'need_to_check_completeness' column in both tables
        supports all required enum values.
        """
        self.jobDatabaseLock.acquire()
        self.__connectDatabase()
        
        if self.cursor:
            # Update enum values for rfps table
            self.cursor.execute("SHOW COLUMNS FROM rfps WHERE Field = 'need_to_check_completeness'")
            column_info = self.cursor.fetchone()
            if column_info and "enum('1','2','3')" in column_info[1]:
                self.cursor.execute("ALTER TABLE rfps MODIFY COLUMN need_to_check_completeness ENUM('1', '2', '3', '4') DEFAULT '1'")
                
            # Update enum values for proposals table
            self.cursor.execute("SHOW COLUMNS FROM proposals WHERE Field = 'need_to_check_completeness'")
            column_info = self.cursor.fetchone()
            if column_info and "enum('1','2','3')" in column_info[1]:
                self.cursor.execute("ALTER TABLE proposals MODIFY COLUMN need_to_check_completeness ENUM('1', '2', '3', '4') DEFAULT '1'")
                
            self.db.commit()
            
        self.jobDatabaseLock.release()

if __name__ == "__main__":
    db_obj = DATABASE()
    db_obj.addEnumValue4IfNotExists()
    # db.convertPrecessingToNeedToProcess()
    # data = db.getRequestDataFromDatabase()
    # if data:
    #     print(f"Job ID: {data[0]}, Model: {data[1]}, RFP URL: {data[4]}")
    # else:
    #     print("No jobs found in the database.")