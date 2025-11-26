import pymysql
from config import DATABASE_HOST, DATABASE_USER, USER_PASSWORD, USED_DATABASE
from logger_config import get_logger

module_logger = get_logger(__name__)

def check_proposal_status(
    db_host: str,
    db_user: str,
    db_password: str,
    db_name: str
) -> str:
    """
    Connects to the MySQL database, checks the 'success' value from the latest row in 'rfps',
    and returns an appropriate status message.

    Returns:
        str: 'process completed successfully', 'technical_error', or a default message.
    """
    try:
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        
        with connection.cursor() as cursor:
            query = """
                SELECT 
                JSON_UNQUOTE(JSON_EXTRACT(result, '$.success')) AS success
                FROM proposals
                ORDER BY id DESC
                LIMIT 1;
            """
            cursor.execute(query)
            row = cursor.fetchone()

            if row:
                success_value = row[0]
                # success_value = 'true'
                if success_value == 'true':
                    return "process completed successfully"
                elif success_value == 'false':
                    return "technical_error"
                else:
                    return "No valid success value found."
            else:
                return "No rows in table."

    except Exception as e:
        module_logger.error(f"Database error in check_proposal_status: {e}")
        return f"Database error: {e}"

    finally:
        if 'connection' in locals():
            connection.close()


def check_rfp_status(
    db_host: str,
    db_user: str,
    db_password: str,
    db_name: str
) -> str:
    """
    Connects to the MySQL database, checks the 'success' value from the latest row in 'rfps',
    and returns an appropriate status message.

    Returns:
        str: 'process completed successfully', 'technical_error', or a default message.
    """
    try:
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        
        with connection.cursor() as cursor:
            query = """
            SELECT 
            JSON_UNQUOTE(JSON_EXTRACT(result, '$.success')) AS success
            FROM rfps
            ORDER BY id DESC
            LIMIT 1;
            """
            cursor.execute(query)
            row = cursor.fetchone()

            if row:
                module_logger.info(row)
                success_value = row[0]
                # success_value = 'true'
                if success_value == 'true':
                    return "process completed successfully"
                elif success_value == 'false':
                    module_logger.info("==============================")
                    return "technical_error"
                else:
                    return "No valid success value found."
            else:
                return "No rows in table."

    except Exception as e:
        module_logger.error(f"Database error in check_rfp_status: {e}")
        return f"Database error: {e}"

    finally:
        if 'connection' in locals():
            connection.close()


# Set your credentials
# Call the function
result = check_rfp_status(DATABASE_HOST, DATABASE_USER, USER_PASSWORD, USED_DATABASE)
module_logger.info(result)
