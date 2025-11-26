
from dotenv import load_dotenv
import os
import json
import re
from typing import List
import tiktoken
from logger import custom_logger as llog
from reactive_agent import run_react_agent

# Constants
MAX_TOKENS = 120000  # Maximum context length

def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception as e:
        llog("EA_Standard_Eval", f"Error using tiktoken: {str(e)}", "ea_standard_eval.log")
        return len(text.split())

def chunk_text(text: str, tokens_per_chunk: int = 60000) -> List[str]:
    """Split text into chunks of approximately tokens_per_chunk tokens."""
    if not text:
        return []
    
    approx_total_tokens = count_tokens(text)
    if approx_total_tokens <= tokens_per_chunk:
        return [text]
    
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0
    
    paragraphs = re.split(r'\n\s*\n', text)
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        paragraph_tokens = count_tokens(paragraph + "\n\n")
        
        if paragraph_tokens > tokens_per_chunk:
            lines = paragraph.split('\n')
            for line in lines:
                if not line.strip():
                    continue
                line_tokens = count_tokens(line + "\n")
                if current_chunk_tokens + line_tokens > tokens_per_chunk:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = line + "\n"
                    current_chunk_tokens = line_tokens
                else:
                    current_chunk += line + "\n"
                    current_chunk_tokens += line_tokens
        else:
            if current_chunk_tokens + paragraph_tokens > tokens_per_chunk:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph + "\n\n"
                current_chunk_tokens = paragraph_tokens
            else:
                current_chunk += paragraph + "\n\n"
                current_chunk_tokens += paragraph_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
        
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens > tokens_per_chunk:
            print(f"Warning: Chunk still too large ({chunk_tokens} tokens), applying token-based splitting")
            words = chunk.split()
            sub_chunk = ""
            sub_chunk_tokens = 0
            for word in words:
                word_tokens = count_tokens(word + " ")
                if sub_chunk_tokens + word_tokens > tokens_per_chunk:
                    final_chunks.append(sub_chunk)
                    sub_chunk = word + " "
                    sub_chunk_tokens = word_tokens
                else:
                    sub_chunk += word + " "
                    sub_chunk_tokens += word_tokens
            if sub_chunk:
                final_chunks.append(sub_chunk)
        else:
            final_chunks.append(chunk)
            
    print(f"Text chunked into {len(final_chunks)} chunks")
    return final_chunks

    
#old logic for model provider : aquire and release method 
# def evaluation(file_path: str, ea_file_path: str, topics: List[str], detected_language: str, output_tokens: str, log_save_file_name: str, model_provider=None):
#     """
#     Main evaluation function that reads files and runs evaluation using ReAct agent.
    
#     Args:
#         file_path: Path to the RFP text file
#         ea_file_path: Path to the EA standard text file
#         topics: List of additional topics to be included in the evaluation
#         detected_language: The detected language code ('ar' or 'en')
#         output_tokens: Either 'medium' or 'large' to control output size
#         log_save_file_name: The log file name
#         model_provider: Optional model provider from rfp_proposal_eval
        
#     Returns:
#         Evaluation result as JSON
#     """
#     llog("EA_Standard_Eval", "Starting evaluation", "demo_log")
#     load_dotenv()
#     llog("EA_Standard_Eval", f"MOdel provider value is : {model_provider}", "demo_log")
#     is_using_shared_model = model_provider is not None and model_provider.model_type == "opensource"
#     if is_using_shared_model:
#         llog("EA_Standard_Eval", "Using shared opensource model for evaluation", "demo_log")
#         model_provider.ensure_model_loaded('opensource')
#     else:
#         llog("EA_Standard_Eval", "Using OpenRouter with Qwen 3 32B for evaluation", "demo_log")
    
#     with open(file_path, 'r') as file:
#         rfp_text = file.read()
#     with open(ea_file_path, 'r') as file:
#         ea_standard_text = file.read()
    
#     token_limit = 100000 if is_using_shared_model else 15000
#     rfp_tokens = count_tokens(rfp_text)
#     llog("EA_Standard_Eval", f"RFP document contains {rfp_tokens} tokens", "demo_log")
#     print(f"\n📄 RFP document contains {rfp_tokens} tokens")
    
#     result = None
#     if rfp_tokens > token_limit:
#         llog("EA_Standard_Eval", f"RFP exceeds token limit of {token_limit}. Processing as single document with vector store.", log_save_file_name)
#         print(f"\n🔄 RFP exceeds token limit of {token_limit}. Processing with vector store...")
        
#         # Pass full RFP text to ReAct agent; vector store handles chunking internally
#         llog("EA_Standard_Eval", f"Fix topic are {topics}", "demo_log")
#         result = run_react_agent(rfp_text, ea_standard_text, detected_language, topics, "demo_log")
#         llog("EA_Standard_Eval", f"Evaluation completed result is {result}", "demo_log")
#     else:
#         print(f"\n🔄 RFP within token limits. Using standard processing path...")
#         result = run_react_agent(rfp_text, ea_standard_text, detected_language, topics , "demo_log")
    
#     llog("EA_Standard_Eval", f"Evaluation completed result is {result}", "demo_log")
#     return result


# openai and Opensource : synchronous model handling
def evaluation(file_path: str, ea_file_path: str, topics: List[str], detected_language: str,log_save_file_name: str, model_provider=None):
    
    llog("EA_Standard_Eval", "Starting evaluation", "demo_log")
    load_dotenv()
    llog("EA_Standard_Eval", f"MOdel provider value is : {model_provider}", "demo_log")
    
    with open(file_path, 'r') as file:
        rfp_text = file.read()
    with open(ea_file_path, 'r') as file:
        ea_standard_text = file.read()
    
    rfp_tokens = count_tokens(rfp_text)
    
    llog("EA_Standard_Eval", f"RFP document contains {rfp_tokens} tokens", "demo_log")
    print(f"\n📄 RFP document contains {rfp_tokens} tokens")
    
    # Pass full RFP text to ReAct agent; vector store handles chunking internally
    llog("EA_Standard_Eval", f"Fix topic are {topics}", "demo_log")
    result = run_react_agent(rfp_text, ea_standard_text, topics, detected_language,  log_save_file_name, model_provider)
    llog("EA_Standard_Eval", f"Evaluation completed result is {result}", log_save_file_name)
    
    return result



if __name__ == "__main__":
    sample_topics = [
        "Data Governance", 
        "Sustainability Considerations", 
        "Preferred Application Frameworks", 
        "Accepted Database",
        "Accepted Operating System",
        "Requirement of SSL Certificate"
    ]
    result = evaluation('translated.txt', 'ea_standard.txt', sample_topics, 'en', 'large', 'rfp_z.txt')
    print(json.dumps(result, ensure_ascii=False, indent=2))