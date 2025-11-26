# DATABASE_HOST = 'localhost'
# DATABASE_USER = 'rfp'
# USER_PASSWORD = 'rfps'
# USED_DATABASE = 'compliance_bot'

# DATABASE_HOST = '20.46.197.51'
# DATABASE_USER = 'compliance_bot_ai'
# USER_PASSWORD = 'nQVdQsU9&F&bt'
# USED_DATABASE= 'compliance_bot'

### Database Configuration ###
DATABASE_HOST = '20.46.197.51'
# DATABASE_USER = 'compliance_bot_ai'
# USER_PASSWORD = 'nQVdQsU9&F&bt'
DATABASE_USER= "compliance_bot_ai"
USER_PASSWORD = "4T~vAkmF)'[5H-J}wc1D"
USED_DATABASE = 'compliance_bot'

### Worker Configuration ###
NUMBER_OF_WORKERS = 2 # Legacy setting - total workers
COMPLETENESS_WORKERS = 1 # Number of workers for RFP completeness checks
# PROPOSAL_WORKERS = 1  # Number of workers for proposal evaluations
PROPOSAL_WORKERS = 1  # Number of workers for proposal evaluations

# Worker idle timeout configuration
WORKER_IDLE_TIMEOUT_SECONDS = 0  # Workers will exit after 5 minutes of inactivity (set to 0 to disable)
WORKER_POLL_INTERVAL_SECONDS = 1   # How often workers check for new jobs

### Logging Configuration ###
LOG_FILES_DIR_PATH = 'logs/'

### Language Detection Thresholds ###
arabic_threshold = 0.01  # 1% threshold for Arabic
other_threshold = 0.03   # 3% threshold for other languages

### Document Processing Configuration ###
MAX_TOC_PAGES = 5  # Max pages to check for Table of Contents

### Email Configuration ###
# Email addresses for notifications
DEVELOPER_RECEIVER_MAIL = ["jaydip.z@neuramonks.com","kaushik.p@neuramonks.com","ketan@neuramonks.com"] 
CLIENT_RECEIVER_MAIL = ["jaydip.z@neuramonks.com","kaushik.p@neuramonks.com","ketan@neuramonks.com"] 
#["piyush@neuramonks.com", "ketan@neuramonks.com", "shrey.s@neuramonks.com"]

# Email sender credentials
SENDER_MAIL = "developer.kshatra@gmail.com" #yash
SENDER_PASSWORD = "ustdztntzvtwvcsx" #yash

show_addressed=True
show_partially_addressed=True
show_contradicted=True
show_not_found=False

# SENDER_MAIL = "pbsonani@gmail.com" #piyush sir
# SENDER_PASSWORD = "luxu dyyy oonh umqt" #piyush sir

### Process Configuration ###
# Process types
PROCESS_TYPES = {
    "COMPLETENESS_CHECK": "completeness_check",
    "PROPOSAL_EVALUATION": "proposal_evaluation"
}

### Error Messages ###
# List of error messages that don't require developer notification
PROPER_ERROR_MESSEGE_LIST_TO_SHOW_USER = [
    "No valid topics provided. Please enter at least one topic.",
    "We encountered an issue while accessing the file. Please check if the file is accessible and try again.",
    "We encountered an issue while processing the file. Please try again.",
    "We encountered an issue while reading the file. Please ensure you've uploaded a valid PDF file and try again.",
    "The uploaded file is empty.",
    "The PDF file is empty or contains no pages.",
    "The PDF file appears to be blank or contains no readable text.",
    "The PDF file is password-protected or encrypted.",
    "The PDF file is corrupted or damaged.",
    "Error opening the file. Please ensure it's a valid PDF.",
    "Open-Source model is under development",
    "Invalid Choice of Model"
]


# Language_Threshold = 0  # 0 for arabic, 1 for english
ARABIC_PERCENT = 30