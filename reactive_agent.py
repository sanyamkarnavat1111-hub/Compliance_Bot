
import os
import uuid
import json
import re
from datetime import datetime
from typing import List, Dict, Any, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from logger import custom_logger as llog
from dotenv import load_dotenv
import tiktoken
import psutil
from Prompt_LLM import get_llm, task1_prompt, task2_intent_prompt, task2_description_prompt, task2_translation_prompt, task3_prompt,task3_prompt_3, standard_English_refiner_prompt, task3_translation_prompt,task3_strict_translation_prompt
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import time
import gc
import shutil
 

load_dotenv()

DIR = "Json_object"
                    
if not os.path.exists(DIR):
    os.makedirs(DIR)
                        
# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)

# State for LangGraph
class AgentState(TypedDict):
    input_text: str
    section_topics: List[Dict[str, str]]
    standard_text: str
    mapped_sections: Dict[str, str]
    html_table: str
    language: str
    iteration: int
    max_iterations: int
    error: str
    dynamic_topics: List[str]
    log_file: str
    model_provider : str
    json_report: list

def count_tokens(text: str, log_file: str = None) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        llog("ReAct_Agent", f"Error counting tokens: {str(e)}", log_file)
        return len(text.split())  # Fallback approximation

def chunk_text(text: str, tokens_per_chunk: int = 750, log_file: str = None) -> List[str]:
    """Split text into chunks of approximately tokens_per_chunk tokens."""
    if not text:
        return []
    
    approx_total_tokens = count_tokens(text,log_file=log_file)
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
            llog("ReAct_Agent", f"Chunk too large ({chunk_tokens} tokens), applying word-based splitting", log_file)
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
            
    llog("ReAct_Agent", f"Text chunked into {len(final_chunks)} chunks", log_file)
    return final_chunks    

def query_vector_store(query: str, vector_store: FAISS, k: int = 12, log_file: str = None) -> List[str]:
    """Query the FAISS vector store for relevant context."""
    try:
        results = vector_store.similarity_search(query, k=k)
        return [result.page_content for result in results]
    except Exception as e:
        llog("ReAct_Agent", f"Error querying vector store: {str(e)}", log_file)
        return []

def initialize_rfp_vector_store(rfp_text: str, log_file: str = None) -> FAISS:
    """Initialize FAISS vector store with RFP text."""
    try:
        chunks = chunk_text(rfp_text, tokens_per_chunk=500, log_file= log_file)
        llog("ReAct_Agent", f"Created {len(chunks)} chunks for vector store", log_file)
        vector_store = FAISS.from_texts(chunks, embeddings)
        llog("ReAct_Agent", "RFP FAISS vector store initialized successfully", log_file)
        return vector_store
    except Exception as e:
        llog("ReAct_Agent", f"Error initializing RFP vector store: {str(e)}", log_file)
        raise
    
def standard_English_refiner(state:AgentState)-> Dict[str, Any]:
    
    llog("ReAct_Agent", "Starting standard refiner", state["log_file"])
    prompt = standard_English_refiner_prompt    
    llm = get_llm(state["model_provider"])
    
    system_promt = f"You are an expert in RFP analysis. Your task is to JUST convert the give Standard Document into English Language."
    
    llog("ReAct_Agent", f"Prompt constructed", state["log_file"])
    human_prompt = prompt.format(standard_text=state["standard_text"])
    message = [
        SystemMessage(content=system_promt),
        HumanMessage(content=human_prompt)
    ]
    llog("ReAct_Agent", f"Prompt sent to LLM", state["log_file"])
    
    try:
        response = llm.invoke(message, timeout=30)
        llog("ReAct_Agent", f"Response received from LLM: {response}", state["log_file"])
        
        # Validate response content
        if response.content and isinstance(response.content, str) and response.content.strip():
            state["standard_text"] = response.content
            llog("ReAct_Agent", f"Response for standard refiner: {response.content}", state["log_file"])
        else:
            llog("ReAct_Agent", "LLM returned empty or invalid response", state["log_file"])
            llog("ReAct_Agent", "Falling back to original standard_text", state["log_file"])
            # Fallback: Keep original text
            state["standard_text"] = state["standard_text"]
            
    except Exception as e:
        # Handle any exception during LLM invocation
        llog("ReAct_Agent", f"Error during LLM invocation: {str(e)}", state["log_file"])
        llog("ReAct_Agent", "Falling back to original standard_text", state["log_file"])
        # Fallback: Keep original text
        state["standard_text"] = state["standard_text"]
    
    return state
    
def task1_extract_topics(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct agent to extract section topics and descriptions from standard document with robust fallback and retry handling.
    
    Args:
        state: AgentState dictionary containing standard_text, model_provider, log_file, language, max_iterations, etc.
    
    Returns:
        Dictionary with section_topics (list), error message (str), and iteration count (int).
    """
    llog("ReAct_Agent", "Starting task 1: Extract Section Topics", state["log_file"])
    llog("ReAct_Agent", f"Standard Text: {state['standard_text']}", state["log_file"])
    # Input validation
    if not state.get("standard_text") or not isinstance(state["standard_text"], str):
        llog("ReAct_Agent", "Invalid or empty standard_text", state["log_file"])
        return {"section_topics": [], "error": "Invalid or empty standard_text", "iteration": state.get("iteration", 0)}
    
    if not state.get("model_provider"):
        llog("ReAct_Agent", "Missing model_provider", state["log_file"])
        return {"section_topics": [], "error": "Missing model_provider", "iteration": state.get("iteration", 0)}
    
    # Initialize variables
    state["iteration"] = state.get("iteration", 0)
    max_retries = 3  # Maximum retries for LLM and JSON parsing
    parser = JsonOutputParser()
    
    try:
        llm = get_llm(state["model_provider"])
        llog("ReAct_Agent", f"LLM initialized with model provider: {llm}", state["log_file"])
    except Exception as e:
        llog("ReAct_Agent", f"Failed to initialize LLM: {str(e)}", state["log_file"])
        return {"section_topics": [], "error": f"Failed to initialize LLM: {str(e)}", "iteration": state["iteration"]}
    
    # Prompt setup
    prompt = task1_prompt
    llog("ReAct_Agent", "Prompt is ready for LLM", state["log_file"])
    
    for _ in range(state.get("max_iterations", 5)):
        llog("ReAct_Agent", f"Starting iteration {state['iteration'] + 1}", state["log_file"])
        
        # Prepare system message and language
        language = 'Arabic' if state.get("language") == 'ar' else 'English'
        system_message = f"You are an expert specializing in RFP analysis and compliance evaluation. Provide all output in {language}."
        llog("ReAct_Agent", f"System message ready, language: {language}", state["log_file"])
        
        try:
            # Format prompt
            prompt_filled = prompt.format(
                standard_text=state["standard_text"],
                language=language,
                max_iterations=state.get("max_iterations", 5)
            )
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt_filled)
            ]
            
            # Retry loop for LLM invocation
            for attempt in range(max_retries):
                try:
                    llog("ReAct_Agent", f"LLM invocation attempt {attempt + 1}", state["log_file"])
                    response = llm.invoke(messages, timeout=30)
                    
                    # Validate LLM response
                    if not hasattr(response, "content") or not response.content or not isinstance(response.content, str):
                        llog("ReAct_Agent", "Invalid LLM response", state["log_file"])
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        raise ValueError("Invalid LLM response after max retries")
                    
                    llog("ReAct_Agent", f"LLM response received {response.content}", state["log_file"])
                    
                    # Clean and parse JSON response
                    cleaned_response = re.sub(r'^```json\n', '', response.content)
                    cleaned_response = re.sub(r'\n```$', '', cleaned_response)
                    cleaned_response = cleaned_response.strip()
                    llog("ReAct_Agent", f"Cleaned response length: {len(cleaned_response)}", state["log_file"])
                    llog("ReAct_Agent", f"Cleaned response: {cleaned_response}", state["log_file"])
                    
                    # Retry loop for JSON parsing
                    for parse_attempt in range(max_retries):
                        try:
                            topics = parser.parse(cleaned_response)
                            llog("ReAct_Agent", f"Parsed topics lenght: {len(topics)} items", state["log_file"])
                            llog("ReAct_Agent", f"Parsed topics: {topics}", state["log_file"])
                            break
                        except Exception as parse_e:
                            llog("ReAct_Agent", f"JSON parsing attempt {parse_attempt + 1} failed: {str(parse_e)}", state["log_file"])
                            if parse_attempt < max_retries - 1:
                                cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
                                cleaned_response = re.sub(r',\s*\]', ']', cleaned_response)
                                time.sleep(1)
                                continue
                            try:
                                topics = json.loads(cleaned_response)
                                llog("ReAct_Agent", f"Fallback JSON parsing succeeded: {len(topics)} items", state["log_file"])
                            except Exception as fallback_e:
                                llog("ReAct_Agent", f"Fallback JSON parsing failed: {str(fallback_e)}", state["log_file"])
                                topics = []
                                break
                    
                    # Validate parsed topics
                    if not isinstance(topics, list):
                        llog("ReAct_Agent", "Parsed topics is not a list", state["log_file"])
                        topics = []
                    
                    llog("ReAct_Agent", f"Topics extracted: {len(topics)} items", state["log_file"])
                    
                    # Combine and deduplicate topics
                    seen = set()
                    unique_topics = [
                        t for t in topics
                        if isinstance(t, dict) and "topic" in t and
                        not (t["topic"].lower() in seen or seen.add(t["topic"].lower()))
                    ]
                    llog("ReAct_Agent", f"Unique topics: {len(unique_topics)} items", state["log_file"])
                    
                    return {"section_topics": unique_topics, "error": "", "iteration": state["iteration"]}
                
                except Exception as llm_e:
                    llog("ReAct_Agent", f"LLM invocation attempt {attempt + 1} failed: {str(llm_e)}", state["log_file"])
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    llog("ReAct_Agent", "Max LLM retries reached", state["log_file"])
                    return {
                        "section_topics": [],
                        "error": f"LLM invocation failed after {max_retries} retries: {str(llm_e)}",
                        "iteration": state["iteration"]
                    }
        
        except Exception as e:
            state["error"] = f"Error in topic extraction: {str(e)}"
            llog("ReAct_Agent", state["error"], state["log_file"])
            state["iteration"] += 1
            if state["iteration"] >= state.get("max_iterations", 5):
                llog("ReAct_Agent", "Max iterations reached", state["log_file"])
                return {
                    "section_topics": [],
                    "error": f"Topic extraction failed after {state['iteration']} iterations",
                    "iteration": state["iteration"]
                }
    
    llog("ReAct_Agent", "Exhausted iterations", state["log_file"])
    return {
        "section_topics": [],
        "error": "Topic extraction failed after maximum iterations",
        "iteration": state["iteration"]
    }


def task2_map_sections(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct agent to deduplicate dynamic topics and combine with section topics with robust fallback handling.
    
    Args:
        state: AgentState dictionary containing section_topics, dynamic_topics, model_provider, log_file, language, standard_text.
    
    Returns:
        Dictionary with section_topics (list), error (str), iteration (int), mapped_sections (dict).
    """
    llog("ReAct_Agent", "Starting task 2: Map Sections", state["log_file"])
    
    # Input validation
    if not state.get("section_topics") or not isinstance(state["section_topics"], list):
        llog("ReAct_Agent", "Invalid or missing section_topics", state["log_file"])
        return {"section_topics": [], "error": "Invalid or missing section_topics", "iteration": 0, "mapped_sections": {}}
    
    if not state.get("model_provider"):
        llog("ReAct_Agent", "Missing model_provider", state["log_file"])
        return {"section_topics": state["section_topics"], "error": "Missing model_provider", "iteration": 0, "mapped_sections": {}}
    
    # Initialize LLM
    max_retries = 3  # Maximum retries for LLM and JSON parsing
    try:
        llm = get_llm(state["model_provider"])
        llog("ReAct_Agent", f"LLM initialized with model provider: {llm}", state["log_file"])
    except Exception as e:
        llog("ReAct_Agent", f"Failed to initialize LLM: {str(e)}", state["log_file"])
        return {"section_topics": state["section_topics"], "error": f"Failed to initialize LLM: {str(e)}", "iteration": 0, "mapped_sections": {}}
    
    # Extract existing topic names
    existing_topics = [{"topic": t["topic"]} for t in state["section_topics"] if isinstance(t, dict) and "topic" in t]
    llog("ReAct_Agent", f"Existing topics length: {len(existing_topics)} items", state["log_file"])
    llog("ReAct_Agent", f"Existing topics: {existing_topics} items", state["log_file"])
    
    # Deduplicate dynamic topics (case-insensitive)
    seen = set()
    deduped_dynamic_topics = [t for t in state.get("dynamic_topics", []) if not (t.lower() in seen or seen.add(t.lower()))]
    llog("ReAct_Agent", f"Initial deduplicated dynamic topics: {len(deduped_dynamic_topics)} items", state["log_file"])
    llog("ReAct_Agent", f"Initial deduplicated dynamic topics: {deduped_dynamic_topics} items", state["log_file"])
    
    # Fixed topics
    fixed_topics = ["Scope of Work", "Project Delivery", "Cybersecurity"]
    
    # Remove duplicates between dynamic and fixed topics
    duplicate_remover_dynamic_fix = PromptTemplate(
        input_variables=["fixed_topics", "deduped_dynamic_topics"],
        template="""
        You are an expert in RFP analysis and compliance evaluation. Your task is to check whether dynamic topics contains any Duplicate topics from the fixed topics.
        - If any duplicate topics are found, remove them from the dynamic topics list.
        - Duplicate means: semantically similar or identical topics, might be redundant or overlapping, meaning can be Similar or Identical, This all're count as the Duplicate.
        
        Fixed Topics: {fixed_topics}
        Dynamic Topics: {deduped_dynamic_topics}
        
        - Strictly NOT INCLUDE any content from or end side and Thinking or Reasoning, Just return the deduplicated dynamic topics as a list of strings.
        """
    )
    
    # LLM call for duplicate removal
    language = 'Arabic' if state.get("language") == 'ar' else 'English'
    messages = [
        SystemMessage(content=f"You are an expert in RFP analysis and compliance evaluation. Provide all output in {language} Language Only."),
        HumanMessage(content=duplicate_remover_dynamic_fix.format(
            fixed_topics=json.dumps(fixed_topics, ensure_ascii=False),
            deduped_dynamic_topics=json.dumps(deduped_dynamic_topics, ensure_ascii=False),
        ))
    ]
    
    cleaned_response = []
    for attempt in range(max_retries):
        try:
            llog("ReAct_Agent", f"Duplicate removal LLM attempt {attempt + 1}", state["log_file"])
            response = llm.invoke(messages, timeout=30)
            
            # Validate response
            if not hasattr(response, "content") or not response.content or not isinstance(response.content, str):
                llog("ReAct_Agent", "Invalid LLM response for duplicate removal", state["log_file"])
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                cleaned_response = deduped_dynamic_topics  # Fallback to original dynamic topics
                break
            
            # Parse response
            match = re.search(r'\[\s*{.*?}\s*\]|\[\s*".*?"\s*(?:,\s*".*?")*\]', response.content, re.DOTALL)
            if match:
                json_like_list = match.group(0)
                for parse_attempt in range(max_retries):
                    try:
                        cleaned_response = json.loads(json_like_list)
                        llog("ReAct_Agent", f"JSON parsed successfully: {len(cleaned_response)} items", state["log_file"])
                        llog("ReAct_Agent", f"JSON parsed successfully: {cleaned_response} items", state["log_file"])
                        break
                    except json.JSONDecodeError as parse_e:
                        llog("ReAct_Agent", f"JSON parsing attempt {parse_attempt + 1} failed: {str(parse_e)}", state["log_file"])
                        if parse_attempt < max_retries - 1:
                            json_like_list = re.sub(r',\s*}', '}', json_like_list)
                            json_like_list = re.sub(r',\s*\]', ']', json_like_list)
                            time.sleep(1)
                            continue
                        cleaned_response = deduped_dynamic_topics  # Fallback to original dynamic topics
                        break
            else:
                llog("ReAct_Agent", "No JSON list found in response", state["log_file"])
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                cleaned_response = deduped_dynamic_topics  # Fallback to original dynamic topics
                break
        
        except Exception as e:
            llog("ReAct_Agent", f"Duplicate removal LLM attempt {attempt + 1} failed: {str(e)}", state["log_file"])
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            cleaned_response = deduped_dynamic_topics  # Fallback to original dynamic topics
            break
    
    # Combine with fixed topics
    deduped_dynamic_topics = cleaned_response + fixed_topics
    llog("ReAct_Agent", f"Dynamic + fixed topics: {len(deduped_dynamic_topics)} items", state["log_file"])
    llog("ReAct_Agent", f"Dynamic + fixed topics: {deduped_dynamic_topics} items", state["log_file"])
    
    # Filter dynamic topics by intent
    def check_intent_with_retry(dynamic_topic, existing_topics, intent_prompt, llm):
        messages = [
            SystemMessage(content="You are an expert in RFP analysis. Provide output in JSON."),
            HumanMessage(content=intent_prompt.format(
                dynamic_topic=dynamic_topic,
                existing_topics=json.dumps(existing_topics, ensure_ascii=False)
            ))
        ]
        for attempt in range(max_retries):
            try:
                response = llm.invoke(messages, timeout=30)
                if not hasattr(response, "content") or not response.content or not isinstance(response.content, str):
                    llog("ReAct_Agent", f"Invalid intent check response for '{dynamic_topic}'", state["log_file"])
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None  # Signal fallback
                return response
            except Exception as e:
                llog("ReAct_Agent", f"Intent check attempt {attempt + 1} for '{dynamic_topic}' failed: {str(e)}", state["log_file"])
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None  # Signal fallback
    
    unique_dynamic_topics = []
    intent_prompt = task2_intent_prompt
    
    for dynamic_topic in deduped_dynamic_topics:
        response = check_intent_with_retry(dynamic_topic, existing_topics, intent_prompt, llm)
        if response is None:
            llog("ReAct_Agent", f"Fallback: Keeping '{dynamic_topic}' due to intent check failure", state["log_file"])
            unique_dynamic_topics.append(dynamic_topic)
            continue
        
        match = re.search(r'\{.*?\}', response.content, re.DOTALL)
        if match:
            cleaned_response = match.group(0)
            for parse_attempt in range(max_retries):
                try:
                    result = json.loads(cleaned_response)
                    if not result.get("is_duplicate", False):
                        unique_dynamic_topics.append(dynamic_topic)
                        llog("ReAct_Agent", f"Dynamic topic '{dynamic_topic}' is unique", state["log_file"])
                    else:
                        llog("ReAct_Agent", f"Dynamic topic '{dynamic_topic}' is duplicate, removed", state["log_file"])
                    break
                except json.JSONDecodeError as parse_e:
                    llog("ReAct_Agent", f"JSON parsing attempt {parse_attempt + 1} for '{dynamic_topic}' failed: {str(parse_e)}", state["log_file"])
                    if parse_attempt < max_retries - 1:
                        cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
                        cleaned_response = re.sub(r',\s*\]', ']', cleaned_response)
                        time.sleep(1)
                        continue
                    llog("ReAct_Agent", f"Fallback: Keeping '{dynamic_topic}' due to JSON parsing failure", state["log_file"])
                    unique_dynamic_topics.append(dynamic_topic)
                    break
        else:
            llog("ReAct_Agent", f"No valid JSON found for '{dynamic_topic}'", state["log_file"])
            unique_dynamic_topics.append(dynamic_topic)
    
    llog("ReAct_Agent", f"Unique dynamic topics after intent check: {len(unique_dynamic_topics)} items", state["log_file"])
    
    # Translate dynamic topics if language is Arabic
    translated_dynamic_topics = []
    # if state.get("language") == "ar":
    #     for topic in unique_dynamic_topics:
    #         for attempt in range(max_retries):
    #             try:
    #                 def translate_topic():
    #                     translation_prompt = task2_translation_prompt.format(topic=topic)
    #                     response = llm.invoke([HumanMessage(content=translation_prompt)], timeout=30)
    #                     if not hasattr(response, "content") or not response.content or not isinstance(response.content, str):
    #                         raise ValueError("Invalid translation response")
    #                     return response.content.strip()
                    
    #                 translated = translate_topic()
    #                 llog("ReAct_Agent", f"Translated dynamic topic '{topic}' to '{translated}'", state["log_file"])
    #                 translated_dynamic_topics.append(translated)
    #                 break
    #             except Exception as e:
    #                 llog("ReAct_Agent", f"Translation attempt {attempt + 1} for '{topic}' failed: {str(e)}", state["log_file"])
    #                 if attempt < max_retries - 1:
    #                     time.sleep(2)
    #                     continue
    #                 llog("ReAct_Agent", f"Fallback: Keeping untranslated topic '{topic}'", state["log_file"])
    #                 translated_dynamic_topics.append(topic)
    #                 break
    # else:
    translated_dynamic_topics = unique_dynamic_topics
    
    llog("ReAct_Agent", f"Translated dynamic topics: {len(translated_dynamic_topics)} items", state["log_file"])
    
    # Generate descriptions for dynamic topics
    description_prompt = task2_description_prompt
    dynamic_topics_with_desc = []
    
    for topic in translated_dynamic_topics:
        for attempt in range(max_retries):
            try:
                def generate_description():
                    messages = [
                        SystemMessage(content=f"Provide output in English language."),
                        HumanMessage(content=description_prompt.format(
                            topic=topic,
                            language=language,
                            standard_text=state["standard_text"],
                        ))
                    ]
                    response = llm.invoke(messages, timeout=30)
                    if not hasattr(response, "content") or not response.content or not isinstance(response.content, str):
                        raise ValueError("Invalid description response")
                    return response
                
                response = generate_description()
                cleaned_response = re.sub(r'^```json\n', '', response.content)
                cleaned_response = re.sub(r'\n```$', '', cleaned_response)
                cleaned_response = cleaned_response.strip()
                
                for parse_attempt in range(max_retries):
                    try:
                        topic_info = json.loads(cleaned_response)
                        if isinstance(topic_info, dict) and "description" in topic_info:
                            dynamic_topics_with_desc.append(topic_info)
                            llog("ReAct_Agent", f"Generated description for '{topic}'", state["log_file"])
                            llog("ReAct_Agent", f"Generated description for '{topic}' is: {topic_info}", state["log_file"])
                            break
                        else:
                            raise json.JSONDecodeError("Invalid topic_info format", cleaned_response, 0)
                    except json.JSONDecodeError as parse_e:
                        llog("ReAct_Agent", f"Description JSON parsing attempt {parse_attempt + 1} for '{topic}' failed: {str(parse_e)}", state["log_file"])
                        if parse_attempt < max_retries - 1:
                            cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
                            cleaned_response = re.sub(r',\s*\]', ']', cleaned_response)
                            time.sleep(1)
                            continue
                        llog("ReAct_Agent", f"Fallback: Using default description for '{topic}'", state["log_file"])
                        dynamic_topics_with_desc.append({"topic": topic, "description": f"Details for {topic} to be extracted from the RFP."})
                        break
                break
            except Exception as e:
                llog("ReAct_Agent", f"Description generation attempt {attempt + 1} for '{topic}' failed: {str(e)}", state["log_file"])
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                llog("ReAct_Agent", f"Fallback: Using default description for '{topic}'", state["log_file"])
                dynamic_topics_with_desc.append({"topic": topic, "description": f"Details for {topic} to be extracted from the RFP."})
                break
    
    # Combine all topics
    all_topics = state["section_topics"] + dynamic_topics_with_desc
    seen = set()
    unique_topics = [t for t in all_topics if isinstance(t, dict) and "topic" in t and not (t["topic"].lower() in seen or seen.add(t["topic"].lower()))]
    llog("ReAct_Agent", f"Final unique topics length: {len(unique_topics)} items", state["log_file"])
    llog("ReAct_Agent", f"Final unique topics: {unique_topics} items", state["log_file"])
    return {"section_topics": unique_topics, "error": "", "iteration": 0, "mapped_sections": {}}


# each field of topic step by step 

# def task3_generate_json(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     ReAct agent to generate JSON for each section using RFP vector store with robust fallback handling.
#     Args:
#         state: AgentState dictionary containing section_topics, input_text, model_provider, log_file, language, max_iterations.
    
#     Returns:
#         Dictionary with json_report (list), error (str), iteration (int).
#     """
#     llog("Task3_ReAct_Agent", "Starting task: Generate JSON", state["log_file"])
    
#     json_objects = []
#     language_instruction = "Arabic" if state.get("language", "") == "ar" else "English"
#     required_fields = ["topic", "standard_requirement", "rfp_status", "impact", "recommended_improvement", "implementation_notes"]
#     chinese_pattern = r'[\u4e00-\u9fff]'
#     state["iteration"] = state.get("iteration", 0)
    
#     # Input validation
#     if not state.get("section_topics") or not isinstance(state["section_topics"], list):
#         llog("ReAct_Agent", "Invalid or missing section_topics", state["log_file"])
#         return {"error": "Invalid or empty section topics", "json_report": []}
    
#     if not state.get("input_text") or not isinstance(state["input_text"], str):
#         llog("ReAct_Agent", "Invalid or missing input_text", state["log_file"])
#         return {"error": "Invalid or empty input_text", "json_report": []}
    
#     if not state.get("model_provider"):
#         llog("ReAct_Agent", "Missing model_provider", state["log_file"])
#         return {"error": "Missing model provider", "json_report": []}
    
#     vector_store = None
#     try:
#         # Log memory before vector store
#         process = psutil.Process()
#         mem_before = process.memory_info().rss / 1024 / 1024
#         llog("ReAct_Agent", f"Memory usage before vector store: {mem_before:.2f} MB", state["log_file"])
        
#         # Initialize vector store
#         max_retries = 5  # Maximum retries for LLM and vector store
#         for attempt in range(max_retries):
#             try:
#                 vector_store = initialize_rfp_vector_store(state["input_text"], log_file=state["log_file"])
#                 llog("ReAct_Agent", "RFP vector store initialized", state["log_file"])
#                 break
#             except Exception as e:
#                 llog("ReAct_Agent", f"attempt  {attempt + 1} failed: {str(e)}", state["log_file"])
#                 if attempt < max_retries - 1:
#                     time.sleep(1)  # Wait before retrying
#                     continue
#                 llog("ReAct_Agent", "Max retries reached for vector store initialization", state["log_file"])
#                 return {"error": "Failed to initialize vector store after {max_retries} retries", "json_report": []}
        
#         mem_after = process.memory_info().rss / 1024 / 1024
#         llog("ReAct_Agent", f"Memory usage after vector store: {mem_after:.2f} MB", state["log_file"])
        
#         # Process each topic
#         for topic_dict in state["section_topics"]:
#             if not isinstance(topic_dict, dict) or not topic_dict.get("topic"):
#                 llog("ReAct_Agent", "Invalid topic format, skipping", state["log_file"])
#                 continue
                
#             topic = topic_dict["topic"]
#             description = topic_dict.get("description", "")
#             llog("ReAct_Agent", f"Processing topic: {topic}", state["log_file"])
            
#             # Initialize LLM
#             try:
#                 llm = get_llm(state["model_provider"])
#                 llog("ReAct_Agent", "LLM initialized successfully", state["log_file"])
#             except Exception as e:
#                 llog("ReAct_Agent", f"Failed to initialize LLM for {topic}: {str(e)}", state["log_file"])
#                 json_objects.append({
#                     "topic": topic,
#                     "standard_requirement": description,
#                     "rfp_status": f"Failed to generate data due to LLM initialization failure.",
#                     "impact": "Unable to assess compliance.",
#                     "recommended_improvement": "Manually review topic {topic} in RFP.",
#                     "implementation_notes": "Ensure LLM provider is configured correctly."
#                 })
#                 continue
            
#             valid_json = False
#             json_obj = None
            
#             for _ in range(state.get("max_iterations", 7)):
#                 try:
#                     query = f"{topic}: {description}"
#                     # Query vector store with retry
#                     for attempt in range(max_retries):
#                         try:
#                             rfp_context = query_vector_store(query=query, vector_store=vector_store, k=12, log_file=state["log_file"])
#                             rfp_context_text = "\n".join(rfp_context) if rfp_context else ""
#                             llog("ReAct_Agent", f"RFP context tokens: {count_tokens(rfp_context_text)}", state["log_file"])
#                             break
#                         except Exception as e:
#                             llog("ReAct_Agent", f"Vector store query attempt {attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
#                             if attempt < max_retries - 1:
#                                 time.sleep(1)
#                                 continue
#                             rfp_context_text = ""
#                             llog("ReAct_Agent", f"Fallback: Empty context for {topic}", state["log_file"])
#                             break
                    
#                     # Prepare LLM prompt
#                     llog("ReAct_Agent", f"Generating prompt for {topic} and des : {description}", state["log_file"])
#                     prompt = task3_prompt
#                     system_message = f"You are an expert in RFP analysis and compliance evaluation. Provide all output EXCLUSIVELY in {language_instruction} LANGUAGE ONLY."
#                     messages = [
#                         SystemMessage(content=system_message),
#                         HumanMessage(content=prompt.format(
#                             topic=topic,
#                             description=description,
#                             rfp_context=rfp_context_text,
#                             language=language_instruction
#                         ))
#                     ]
#                     llog("ReAct_Agent", f"Sending prompt for {topic} to LLM", state["log_file"])
                    
#                     # LLM invocation with retry
#                     def invoke_with_retry():
#                         for attempt in range(max_retries):
#                             try:
#                                 response = llm.invoke(messages, timeout=30)
#                                 if not hasattr(response, "content") or not isinstance(response.content, str) or not response.content.strip():
#                                     raise ValueError("Invalid LLM response")
#                                 return response
#                             except Exception as e:
#                                 llog("ReAct_Agent", f"LLM attempt {attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
#                                 if attempt < max_retries - 1:
#                                     time.sleep(2)
#                                     continue
#                                 raise ValueError("LLM invocation failed after max retries")
                    
#                     response = invoke_with_retry()
                    
#                     # Clean response
#                     cleaned_response = re.sub(r'^```json\s*\n?', '', response.content.strip())
#                     cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
#                     cleaned_response = cleaned_response.strip()
                    
#                     # Check for Chinese characters
#                     if re.search(chinese_pattern, cleaned_response):
#                         llog("ReAct_Agent", f"Chinese characters detected for {topic}: {re.findall(chinese_pattern, cleaned_response)[:10]}", state["log_file"])
#                         continue
                    
#                     # Parse JSON with retry
#                     for parse_attempt in range(max_retries):
#                         try:
#                             json_obj = json.loads(cleaned_response)
#                             llog("ReAct_Agent", f"JSON generated for {topic}", state["log_file"])
#                             break
#                         except json.JSONDecodeError as parse_e:
#                             llog("ReAct_Agent", f"JSON parsing attempt {parse_attempt + 1} for {topic} failed: {str(parse_e)}", state["log_file"])
#                             if parse_attempt < max_retries - 1:
#                                 cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
#                                 cleaned_response = re.sub(r',\s*\]', ']', cleaned_response)
#                                 time.sleep(1)
#                                 continue
#                             json_obj = None
#                             break
                    
#                     if json_obj is None:
#                         llog("ReAct_Agent", f"Fallback: Invalid JSON for {topic}", state["log_file"])
#                         continue
                    
#                     # Validate required fields
#                     missing_fields = [field for field in required_fields if field not in json_obj or not json_obj[field]]
#                     if missing_fields:
#                         llog("ReAct_Agent", f"Missing or empty fields for {topic}: {missing_fields}", state["log_file"])
#                         continue
                    
#                     # Validate topic matches
#                     if json_obj["topic"] != topic:
#                         llog("ReAct_Agent", f"Topic mismatch for {topic}: got {json_obj['topic']}", state["log_file"])
#                         continue
                    
#                     llog("ReAct_Agent", f"Valid JSON generated for {topic}", state["log_file"])
#                     llog("ReAct_Agent", f"JSON: {json_obj}", state["log_file"])
#                     json_objects.append(json_obj)
#                     valid_json = True
#                     break
                
#                 except Exception as e:
#                     llog("ReAct_Agent", f"Error generating JSON for {topic}: {str(e)}", state["log_file"])
#                     state["iteration"] += 1
#                     if state["iteration"] >= state.get("max_iterations", 7):
#                         break
            
#             if not valid_json:
#                 llog("ReAct_Agent", f"Fallback: Failed to generate valid JSON for {topic}", state["log_file"])
#                 json_objects.append({
#                     "topic": topic,
#                     "standard_requirement": description,
#                     "rfp_status": f"No valid RFP alignment data generated after {state.get('max_iterations', 7)} attempts.",
#                     "impact": "Unable to assess compliance due to generation failure.",
#                     "recommended_improvement": "Re-evaluate RFP context manually for this topic.",
#                     "implementation_notes": "Consider increasing max_iterations or refining RFP context."
#                 })
    
#     except Exception as e:
#         llog("ReAct_Agent", f"Critical error in task: {str(e)}", state["log_file"])
#         return {"error": f"Task failed: {str(e)}", "json_report": json_objects}
    
#     finally:
#         if vector_store is not None:
#             del vector_store
#             gc.collect()
#             mem_after_cleanup = process.memory_info().rss / 1024 / 1024
#             llog("ReAct_Agent", f"Memory usage after cleanup: {mem_after_cleanup:.2f} MB", state["log_file"])
    
#     llog("ReAct_Agent", f"Generated JSON report with {len(json_objects)} entries", state["log_file"])
#     return {"json_report": json_objects, "error": "", "iteration": 0}


# def task3_generate_json(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     ReAct agent to generate JSON for each section using RFP vector store with robust fallback handling.
#     Generates JSON in English first (except for topic, which retains input language), then translates to the decided language (Arabic or English).
    
#     Args:
#         state: AgentState dictionary containing section_topics, input_text, model_provider, log_file, language, max_iterations.
    
#     Returns:
#         Dictionary with json_report (list), error (str), iteration (int).
#     """
#     llog("Task3_ReAct_Agent", "Starting task: Generate JSON", state["log_file"])
    
#     json_objects = []
#     required_fields = ["topic", "standard_requirement", "rfp_status", "impact", "recommended_improvement", "implementation_notes"]
#     chinese_pattern = r'[\u4e00-\u9fff]'
#     state["iteration"] = state.get("iteration", 0)
#     language_instruction = "English"  # Force English for non-topic fields
    
#     # Input validation
#     if not state.get("section_topics") or not isinstance(state["section_topics"], list):
#         llog("ReAct_Agent", "Invalid or missing section_topics", state["log_file"])
#         return {"error": "Invalid or empty section topics", "json_report": []}
    
#     if not state.get("input_text") or not isinstance(state["input_text"], str):
#         llog("ReAct_Agent", "Invalid or missing input_text", state["log_file"])
#         return {"error": "Invalid or empty input_text", "json_report": []}
    
#     if not state.get("model_provider"):
#         llog("ReAct_Agent", "Missing model_provider", state["log_file"])
#         return {"error": "Missing model provider", "json_report": []}
    
#     vector_store = None
#     try:
#         # Log memory before vector store
#         process = psutil.Process()
#         mem_before = process.memory_info().rss / 1024 / 1024
#         llog("ReAct_Agent", f"Memory usage before vector store: {mem_before:.2f} MB", state["log_file"])
        
#         # Initialize vector store
#         max_retries = 5  # Maximum retries for LLM and vector store
#         for attempt in range(max_retries):
#             try:
#                 vector_store = initialize_rfp_vector_store(state["input_text"], log_file=state["log_file"])
#                 llog("ReAct_Agent", "RFP vector store initialized", state["log_file"])
#                 break
#             except Exception as e:
#                 llog("ReAct_Agent", f"Vector store initialization attempt {attempt + 1} failed: {str(e)}", state["log_file"])
#                 if attempt < max_retries - 1:
#                     time.sleep(1)
#                     continue
#                 llog("ReAct_Agent", "Max retries reached for vector store initialization", state["log_file"])
#                 return {"error": f"Failed to initialize vector store after {max_retries} retries", "json_report": []}
        
#         mem_after = process.memory_info().rss / 1024 / 1024
#         llog("ReAct_Agent", f"Memory usage after vector store: {mem_after:.2f} MB", state["log_file"])
        
#         # Process each topic
#         for topic_dict in state["section_topics"]:
#             if not isinstance(topic_dict, dict) or not topic_dict.get("topic"):
#                 llog("ReAct_Agent", "Invalid topic format, skipping", state["log_file"])
#                 continue
                
#             topic = topic_dict["topic"]
#             description = topic_dict.get("description", "")
#             llog("ReAct_Agent", f"Processing topic: {topic}", state["log_file"])
            
#             # Initialize LLM
#             try:
#                 llm = get_llm(state["model_provider"])
#                 llog("ReAct_Agent", "LLM initialized successfully", state["log_file"])
#             except Exception as e:
#                 llog("ReAct_Agent", f"Failed to initialize LLM for {topic}: {str(e)}", state["log_file"])
#                 json_objects.append({
#                     "topic": topic,
#                     "standard_requirement": description,
#                     "rfp_status": "Failed to generate data due to LLM initialization failure.",
#                     "impact": "Unable to assess compliance.",
#                     "recommended_improvement": f"Manually review topic {topic} in RFP.",
#                     "implementation_notes": "Ensure LLM provider is configured correctly."
#                 })
#                 continue
            
#             valid_json = False
#             json_obj = None
            
#             for _ in range(state.get("max_iterations", 7)):
#                 try:
#                     query = f"{topic}: {description}"
#                     # Query vector store with retry
#                     for attempt in range(max_retries):
#                         try:
#                             rfp_context = query_vector_store(query=query, vector_store=vector_store, k=12, log_file=state["log_file"])
#                             rfp_context_text = "\n".join(rfp_context) if rfp_context else ""
#                             llog("ReAct_Agent", f"RFP context tokens: {count_tokens(rfp_context_text)}", state["log_file"])
#                             break
#                         except Exception as e:
#                             llog("ReAct_Agent", f"Vector store query attempt {attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
#                             if attempt < max_retries - 1:
#                                 time.sleep(1)
#                                 continue
#                             rfp_context_text = ""
#                             llog("ReAct_Agent", f"Fallback: Empty context for {topic}", state["log_file"])
#                             break
                    
#                     # Prepare LLM prompt
#                     prompt = task3_prompt
#                     system_message = f"You are an expert in RFP analysis and compliance evaluation. Provide all output EXCLUSIVELY in {language_instruction} LANGUAGE ONLY for fields other than topic. The topic field MUST use the exact input topic '{topic}' verbatim."
#                     messages = [
#                         SystemMessage(content=system_message),
#                         HumanMessage(content=prompt.format(
#                             topic=topic,
#                             description=description,
#                             rfp_context=rfp_context_text,
#                             language=language_instruction
#                         ))
#                     ]
#                     llog("ReAct_Agent", f"Sending prompt for {topic} to LLM in English (except topic)", state["log_file"])
                    
#                     # LLM invocation with retry
#                     def invoke_with_retry():
#                         for attempt in range(max_retries):
#                             try:
#                                 response = llm.invoke(messages, timeout=30)
#                                 if not hasattr(response, "content") or not isinstance(response.content, str) or not response.content.strip():
#                                     raise ValueError("Invalid LLM response")
#                                 return response
#                             except Exception as e:
#                                 llog("ReAct_Agent", f"LLM attempt {attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
#                                 if attempt < max_retries - 1:
#                                     time.sleep(2)
#                                     continue
#                                 raise ValueError("LLM invocation failed after max retries")
                    
#                     response = invoke_with_retry()
                    
#                     # Clean response
#                     cleaned_response = re.sub(r'^```json\s*\n?', '', response.content.strip())
#                     cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
#                     cleaned_response = cleaned_response.strip()
                    
#                     # Check for Chinese characters
#                     if re.search(chinese_pattern, cleaned_response):
#                         llog("ReAct_Agent", f"Chinese characters detected for {topic}: {re.findall(chinese_pattern, cleaned_response)[:10]}", state["log_file"])
#                         continue
                    
#                     # Parse JSON with retry
#                     for parse_attempt in range(max_retries):
#                         try:
#                             json_obj = json.loads(cleaned_response)
#                             llog("ReAct_Agent", f"JSON generated for {topic} in English (except topic)", state["log_file"])
#                             break
#                         except json.JSONDecodeError as parse_e:
#                             llog("ReAct_Agent", f"JSON parsing attempt {parse_attempt + 1} for {topic} failed: {str(parse_e)}", state["log_file"])
#                             if parse_attempt < max_retries - 1:
#                                 cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
#                                 cleaned_response = re.sub(r',\s*\]', ']', cleaned_response)
#                                 time.sleep(1)
#                                 continue
#                             json_obj = None
#                             break
                    
#                     if json_obj is None:
#                         llog("ReAct_Agent", f"Fallback: Invalid JSON for {topic}", state["log_file"])
#                         continue
                    
#                     # Validate required fields
#                     missing_fields = [field for field in required_fields if field not in json_obj or not json_obj[field]]
#                     if missing_fields:
#                         llog("ReAct_Agent", f"Missing or empty fields for {topic}: {missing_fields}", state["log_file"])
#                         continue
                    
#                     # Validate topic matches
#                     if json_obj["topic"] != topic:
#                         llog("ReAct_Agent", f"Topic mismatch for {topic}: got {json_obj['topic']}", state["log_file"])
#                         continue
                    
#                     # Store JSON    
#                     path_to_store_json = os.path.join(DIR, f"EA_{topic}.json")
                    
#                     with open(path_to_store_json, "a") as f:
#                         json.dump(json_obj, f, indent=4, ensure_ascii=False)
                    
#                     llog("ReAct_Agent", f"english JSON Store for {topic}", state["log_file"])
#                     llog("ReAct_Agent", f"JSON: {json_obj}", state["log_file"])
                    
                    
#                     # Translate to target language if Arabic
#                     target_language = state.get("language", "en")
#                     if target_language == "ar":
#                         llog("ReAct_Agent", f"Translating JSON fields (except topic) for {topic} to Arabic", state["log_file"])
#                         translation_prompt = task3_translation_prompt
#                         translated_json = {"topic": topic}  # Retain original topic
#                         for field in required_fields[1:]:  # Skip topic
#                             for trans_attempt in range(max_retries):
#                                 try:
#                                     messages = [
#                                         SystemMessage(content="You are an expert in translation. Provide output in Arabic ONLY."),
#                                         HumanMessage(content=translation_prompt.format(
#                                             text=json_obj[field]
#                                         ))
#                                     ]
#                                     response = llm.invoke(messages, timeout=30)
#                                     if not hasattr(response, "content") or not isinstance(response.content, str) or not response.content.strip():
#                                         raise ValueError("Invalid translation response")
#                                     translated_json[field] = response.content.strip()
#                                     llog("ReAct_Agent", f"Translated {field} for {topic}", state["log_file"])
#                                     break
#                                 except Exception as e:
#                                     llog("ReAct_Agent", f"Translation attempt {trans_attempt + 1} for {field} of {topic} failed: {str(e)}", state["log_file"])
#                                     if trans_attempt < max_retries - 1:
#                                         time.sleep(2)
#                                         continue
#                                     llog("ReAct_Agent", f"Fallback: Using English {field} for {topic}", state["log_file"])
#                                     translated_json[field] = json_obj[field]
#                         json_obj = translated_json
                        

#                         if not os.path.exists(DIR):
#                             os.makedirs(DIR)
#                         path_to_store_json = os.path.join(DIR, f"EA_{topic}.json")
                        
#                         with open(path_to_store_json, "a") as f:
#                             json.dump(json_obj, f, indent=4, ensure_ascii=False)
                            
#                     else:
#                         llog("ReAct_Agent", f"No translation needed for {topic}, language is English", state["log_file"])
                    
#                     llog("ReAct_Agent", f"Valid JSON generated for {topic} in target language", state["log_file"])
#                     llog("ReAct_Agent", f"JSON: {json_obj}", state["log_file"])
#                     json_objects.append(json_obj)
#                     valid_json = True
#                     break
                
#                 except Exception as e:
#                     llog("ReAct_Agent", f"Error generating JSON for {topic}: {str(e)}", state["log_file"])
#                     state["iteration"] += 1
#                     if state["iteration"] >= state.get("max_iterations", 7):
#                         break
            
#             if not valid_json:
#                 llog("ReAct_Agent", f"Fallback: Failed to generate valid JSON for {topic}", state["log_file"])
#                 json_objects.append({
#                     "topic": topic,
#                     "standard_requirement": description,
#                     "rfp_status": f"No valid RFP alignment data generated after {state.get('max_iterations', 7)} attempts.",
#                     "impact": "Unable to assess compliance due to generation failure.",
#                     "recommended_improvement": "Re-evaluate RFP context manually for this topic.",
#                     "implementation_notes": "Consider increasing max_iterations or refining RFP context."
#                 })
    
#     except Exception as e:
#         llog("ReAct_Agent", f"Critical error in task: {str(e)}", state["log_file"])
#         return {"error": f"Task failed: {str(e)}", "json_report": json_objects}
    
#     finally:
#         if vector_store is not None:
#             del vector_store
#             gc.collect()
#             mem_after_cleanup = process.memory_info().rss / 1024 / 1024
#             llog("ReAct_Agent", f"Memory usage after cleanup: {mem_after_cleanup:.2f} MB", state["log_file"])
    
#     llog("ReAct_Agent", f"Generated JSON report with {len(json_objects)} entries", state["log_file"])
#     return {"json_report": json_objects, "error": "", "iteration": 0}



#little bit english but can take the final result with little english
# def task3_generate_json(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     ReAct agent to generate JSON for each section using RFP vector store with robust fallback handling.
#     Generates JSON in English first, translates topics to Arabic if needed, and ensures the entire JSON is in Arabic when required.
    
#     Args:
#         state: AgentState dictionary containing section_topics, input_text, model_provider, log_file, language, max_iterations.
    
#     Returns:
#         Dictionary with json_report (list), error (str), iteration (int).
#     """
#     llog("Task3_ReAct_Agent", "Starting task: Generate JSON", state["log_file"])
    
#     json_objects = []
#     required_fields = ["topic", "standard_requirement", "rfp_status", "impact", "recommended_improvement", "implementation_notes"]
#     chinese_pattern = r'[\u4e00-\u9fff]'
#     # Relaxed Arabic pattern to allow proper nouns and transliterated terms
#     arabic_pattern = r'^[\u0600-\u06FF\s\d.,;:!?()"\']+[\u0600-\u06FF\s\d.,;:!?()"\']*$'
#     state["iteration"] = state.get("iteration", 0)
#     language_instruction = "English"  # Force English for initial JSON generation
    
#     # Input validation
#     if not state.get("section_topics") or not isinstance(state["section_topics"], list):
#         llog("ReAct_Agent", "Invalid or missing section_topics", state["log_file"])
#         return {"error": "Invalid or empty section topics", "json_report": []}
    
#     if not state.get("input_text") or not isinstance(state["input_text"], str):
#         llog("ReAct_Agent", "Invalid or missing input_text", state["log_file"])
#         return {"error": "Invalid or empty input_text", "json_report": []}
    
#     if not state.get("model_provider"):
#         llog("ReAct_Agent", "Missing model_provider", state["log_file"])
#         return {"error": "Missing model provider", "json_report": []}
    
#     vector_store = None
#     try:
#         # Log memory before vector store
#         process = psutil.Process()
#         mem_before = process.memory_info().rss / 1024 / 1024
#         llog("ReAct_Agent", f"Memory usage before vector store: {mem_before:.2f} MB", state["log_file"])
        
#         # Initialize vector store with retries
#         max_retries = 5
#         for attempt in range(max_retries):
#             try:
#                 vector_store = initialize_rfp_vector_store(state["input_text"], log_file=state["log_file"])
#                 llog("ReAct_Agent", "RFP vector store initialized", state["log_file"])
#                 break
#             except Exception as e:
#                 llog("ReAct_Agent", f"Vector store initialization attempt {attempt + 1} failed: {str(e)}", state["log_file"])
#                 if attempt < max_retries - 1:
#                     time.sleep(1)
#                     continue
#                 llog("ReAct_Agent", f"Max retries reached for vector store initialization", state["log_file"])
#                 return {"error": f"Failed to initialize vector store after {max_retries} retries", "json_report": []}
        
#         mem_after = process.memory_info().rss / 1024 / 1024
#         llog("ReAct_Agent", f"Memory usage after vector store: {mem_after:.2f} MB", state["log_file"])
        
#         # Process each topic
#         for topic_dict in state["section_topics"]:
#             if not isinstance(topic_dict, dict) or not topic_dict.get("topic"):
#                 llog("ReAct_Agent", "Invalid topic format, skipping", state["log_file"])
#                 continue
                
#             topic = topic_dict["topic"]
#             description = topic_dict.get("description", "")
#             llog("ReAct_Agent", f"Processing topic: {topic}", state["log_file"])
#             llog("ReAct_Agent", f"Description: {description}", state["log_file"])
            
#             # Initialize LLM
#             try:
#                 llm = get_llm(state["model_provider"])
#                 llog("ReAct_Agent", "LLM initialized successfully", state["log_file"])
#             except Exception as e:
#                 llog("ReAct_Agent", f"Failed to initialize LLM for {topic}: {str(e)}", state["log_file"])
#                 json_objects.append({
#                     "topic": topic,
#                     "standard_requirement": description,
#                     "rfp_status": "فشل في إنشاء البيانات بسبب فشل تهيئة نموذج اللغة.",
#                     "impact": "غير قادر على تقييم الامتثال.",
#                     "recommended_improvement": f"راجع يدويًا الموضوع {topic} في طلب العرض.",
#                     "implementation_notes": "تأكد من تهيئة مزود نموذج اللغة بشكل صحيح."
#                 })
#                 continue
            
#             valid_json = False
#             json_obj = None
            
#             for _ in range(state.get("max_iterations", 7)):
#                 try:
#                     query = f"{topic}: {description}"
#                     # Query vector store with retry
#                     for attempt in range(max_retries):
#                         try:
#                             rfp_context = query_vector_store(query=query, vector_store=vector_store, k=12, log_file=state["log_file"])
#                             rfp_context_text = "\n".join(rfp_context) if rfp_context else ""
#                             llog("ReAct_Agent", f"RFP context tokens: {count_tokens(rfp_context_text)}", state["log_file"])
#                             break
#                         except Exception as e:
#                             llog("ReAct_Agent", f"Vector store query attempt {attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
#                             if attempt < max_retries - 1:
#                                 time.sleep(1)
#                                 continue
#                             rfp_context_text = ""
#                             llog("ReAct_Agent", f"Fallback: Empty context for {topic}", state["log_file"])
#                             break
                    
#                     # Prepare LLM prompt
#                     prompt = task3_prompt
#                     system_message = f"You are an expert in RFP analysis and compliance evaluation. Provide all output EXCLUSIVELY in {language_instruction} LANGUAGE ONLY for fields other than topic. The topic field MUST use the exact input topic '{topic}' verbatim."
#                     messages = [
#                         SystemMessage(content=system_message),
#                         HumanMessage(content=prompt.format(
#                             topic=topic,
#                             description=description,
#                             rfp_context=rfp_context_text,
#                             language=language_instruction
#                         ))
#                     ]
#                     llog("ReAct_Agent", f"Sending prompt for {topic} to LLM in English (except topic)", state["log_file"])
                    
#                     # LLM invocation with retry
#                     def invoke_with_retry():
#                         for attempt in range(max_retries):
#                             try:
#                                 response = llm.invoke(messages, timeout=30)
#                                 if not hasattr(response, "content") or not isinstance(response.content, str) or not response.content.strip():
#                                     raise ValueError("Invalid LLM response")
#                                 return response
#                             except Exception as e:
#                                 llog("ReAct_Agent", f"LLM attempt {attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
#                                 if attempt < max_retries - 1:
#                                     time.sleep(2)
#                                     continue
#                                 raise ValueError("LLM invocation failed after max retries")
                    
#                     response = invoke_with_retry()
                    
#                     # Clean response
#                     cleaned_response = re.sub(r'^```json\s*\n?', '', response.content.strip())
#                     cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
#                     cleaned_response = cleaned_response.strip()
                    
#                     # Check for Chinese characters
#                     if re.search(chinese_pattern, cleaned_response):
#                         llog("ReAct_Agent", f"Chinese characters detected for {topic}: {re.findall(chinese_pattern, cleaned_response)[:10]}", state["log_file"])
#                         continue
                    
#                     # Parse JSON with retry
#                     for parse_attempt in range(max_retries):
#                         try:
#                             json_obj = json.loads(cleaned_response)
#                             llog("ReAct_Agent", f"JSON generated for {topic} in English (except topic)", state["log_file"])
#                             break
#                         except json.JSONDecodeError as parse_e:
#                             llog("ReAct_Agent", f"JSON parsing attempt {parse_attempt + 1} for {topic} failed: {str(parse_e)}", state["log_file"])
#                             if parse_attempt < max_retries - 1:
#                                 cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
#                                 cleaned_response = re.sub(r',\s*\]', ']', cleaned_response)
#                                 time.sleep(1)
#                                 continue
#                             json_obj = None
#                             break
                    
#                     if json_obj is None:
#                         llog("ReAct_Agent", f"Fallback: Invalid JSON for {topic}", state["log_file"])
#                         continue
                    
#                     # Validate required fields
#                     missing_fields = [field for field in required_fields if field not in json_obj or not json_obj[field]]
#                     if missing_fields:
#                         llog("ReAct_Agent", f"Missing or empty fields for {topic}: {missing_fields}", state["log_file"])
#                         continue
                    
#                     # Validate topic matches
#                     if json_obj["topic"] != topic:
#                         llog("ReAct_Agent", f"Topic mismatch for {topic}: got {json_obj['topic']}", state["log_file"])
#                         continue
                    
#                     # Store English JSON
#                     if not os.path.exists(DIR):
#                         os.makedirs(DIR)
#                     path_to_store_json = os.path.join(DIR, f"EA_{topic}_en.json")
#                     with open(path_to_store_json, "w", encoding='utf-8') as f:
#                         json.dump(json_obj, f, indent=4, ensure_ascii=False)
#                     llog("ReAct_Agent", f"English JSON stored for {topic}", state["log_file"])
                    
#                     # Translate to Arabic if target language is Arabic
#                     target_language = "ar"
#                     if target_language == "ar":
#                         llog("ReAct_Agent", f"Translating entire JSON for {topic} to Arabic", state["log_file"])
#                         translation_prompt = task3_translation_prompt
#                         json_input = json.dumps(json_obj, ensure_ascii=False)
#                         translated_json = {"topic": topic}  # Initialize with translated topic
#                         for trans_attempt in range(max_retries):
#                             try:
#                                 messages = [
#                                     SystemMessage(content="You are an expert in translation. Provide output in Arabic ONLY for all fields."),
#                                     HumanMessage(content=translation_prompt.format(json_text=json_input))
#                                 ]
#                                 response = llm.invoke(messages, timeout=30)
#                                 llog("ReAct_Agent", f"Entire EN to AR translation for:{response.content} ", state["log_file"])
#                                 if not hasattr(response, "content") or not isinstance(response.content, str) or not response.content.strip():
#                                     raise ValueError("Invalid translation response")
#                                 # Clean and parse translated response
#                                 cleaned_trans_response = re.sub(r'^```json\s*\n?', '', response.content.strip())
#                                 cleaned_trans_response = re.sub(r'\n?```$', '', cleaned_trans_response)
#                                 translated_json = json.loads(cleaned_trans_response)
#                                 # Validate translated JSON
#                                 if not all(field in translated_json for field in required_fields):
#                                     raise ValueError("Missing required fields in translated JSON")
#                                 # Check for non-Arabic text in all fields
#                                 for field in required_fields:
#                                     llog("ReAct_Agent", f"Checking for non-Arabic text in field {field}", state["log_file"])
#                                     if not re.match(arabic_pattern, translated_json[field], re.UNICODE):
#                                         raise ValueError(f"Non-Arabic text detected in field {field}")
#                                 llog("ReAct_Agent", f"Successfully translated JSON for {topic}", state["log_file"])
#                                 break
#                             except Exception as e:
#                                 llog("ReAct_Agent", f"Translation attempt {trans_attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
#                                 if trans_attempt < max_retries - 1:
#                                     time.sleep(2)
#                                     continue
#                                 llog("ReAct_Agent", f"Fallback: Using partially translated JSON for {topic}", state["log_file"])
#                                 translated_json = {
#                                     "topic": topic,
#                                     "standard_requirement": json_obj["standard_requirement"],
#                                     "rfp_status": json_obj["rfp_status"],
#                                     "impact": json_obj["impact"],
#                                     "recommended_improvement": json_obj["recommended_improvement"],
#                                     "implementation_notes": json_obj["implementation_notes"]
#                                 }
#                                 break
                        
#                         # Second strict translation step
#                         llog("ReAct_Agent", f"Performing strict Arabic translation for {topic}", state["log_file"])
#                         json_input = json.dumps(translated_json, ensure_ascii=False)
#                         for strict_trans_attempt in range(max_retries):
#                             try:
#                                 messages = [
#                                     SystemMessage(content="You are an expert in translation. Provide output in Arabic ONLY for all fields, preserving technical terms as proper nouns or transliterating them."),
#                                     HumanMessage(content=task3_strict_translation_prompt.format(json_text=json_input))
#                                 ]
#                                 response = llm.invoke(messages, timeout=30)
#                                 llog("ReAct_Agent", f"Strict EN to AR translation for:{response.content}", state["log_file"])
#                                 if not hasattr(response, "content") or not isinstance(response.content, str) or not response.content.strip():
#                                     raise ValueError("Invalid strict translation response")
#                                 # Clean and parse strict translated response
#                                 cleaned_strict_trans_response = re.sub(r'^```json\s*\n?', '', response.content.strip())
#                                 cleaned_strict_trans_response = re.sub(r'\n?```$', '', cleaned_strict_trans_response)
#                                 strict_translated_json = json.loads(cleaned_strict_trans_response)
#                                 # Validate strict translated JSON
#                                 if not all(field in strict_translated_json for field in required_fields):
#                                     raise ValueError("Missing required fields in strict translated JSON")
#                                 # Relaxed validation for Arabic to allow proper nouns
#                                 for field in required_fields:
#                                     llog("ReAct_Agent", f"Checking for non-Arabic text in strict translated field {field}", state["log_file"])
#                                     if not re.match(arabic_pattern, strict_translated_json[field], re.UNICODE):
#                                         raise ValueError(f"Non-Arabic text detected in strict translated field {field}")
#                                 llog("ReAct_Agent", f"Successfully strict translated JSON for {topic}", state["log_file"])
#                                 translated_json = strict_translated_json
#                                 break
#                             except Exception as e:
#                                 llog("ReAct_Agent", f"Strict translation attempt {strict_trans_attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
#                                 if strict_trans_attempt < max_retries - 1:
#                                     time.sleep(2)
#                                     continue
#                                 llog("ReAct_Agent", f"Fallback: Using first translated JSON for {topic}", state["log_file"])
#                                 break
                        
#                         json_obj = translated_json
                        
#                         # Store Arabic JSON
#                         path_to_store_json = os.path.join(DIR, f"EA_{topic}_ar.json")
#                         with open(path_to_store_json, "w", encoding='utf-8') as f:
#                             json.dump(json_obj, f, indent=4, ensure_ascii=False)
#                         llog("ReAct_Agent", f"Arabic JSON stored for {topic}", state["log_file"])
#                     else:
#                         llog("ReAct_Agent", f"No translation needed for {topic}, language is English", state["log_file"])
                    
#                     llog("ReAct_Agent", f"Valid JSON generated for {topic} in target language", state["log_file"])
#                     llog("ReAct_Agent", f"JSON: {json_obj}", state["log_file"])
#                     json_objects.append(json_obj)
#                     valid_json = True
#                     break
                
#                 except Exception as e:
#                     llog("ReAct_Agent", f"Error generating JSON for {topic}: {str(e)}", state["log_file"])
#                     state["iteration"] += 1
#                     if state["iteration"] >= state.get("max_iterations", 7):
#                         break
            
#             if not valid_json:
#                 llog("ReAct_Agent", f"Fallback: Failed to generate valid JSON for {topic}", state["log_file"])
#                 json_objects.append({
#                     "topic": topic,
#                     "standard_requirement": description,
#                     "rfp_status": f"لم يتم إنشاء بيانات محاذاة طلب العرض الصحيحة بعد {state.get('max_iterations', 7)} محاولات.",
#                     "impact": "غير قادر على تقييم الامتثال بسبب فشل الإنشاء.",
#                     "recommended_improvement": f"إعادة تقييم سياق طلب العرض يدويًا لهذا الموضوع {topic}.",
#                     "implementation_notes": "فكر في زيادة الحد الأقصى للتكرارات أو تحسين سياق طلب العرض."
#                 })
    
#     except Exception as e:
#         llog("ReAct_Agent", f"Critical error in task: {str(e)}", state["log_file"])
#         return {"error": f"Task failed: {str(e)}", "json_report": json_objects}
    
#     finally:
#         if vector_store is not None:
#             del vector_store
#             gc.collect()
#             mem_after_cleanup = process.memory_info().rss / 1024 / 1024
#             llog("ReAct_Agent", f"Memory usage after cleanup: {mem_after_cleanup:.2f} MB", state["log_file"])
    
#     llog("ReAct_Agent", f"Generated JSON report with {len(json_objects)} entries", state["log_file"])
#     return {"json_report": json_objects, "error": "", "iteration": 0}
def task3_generate_json(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct agent to generate JSON for each section using RFP vector store with robust fallback handling.
    Generates JSON in Arabic first, applies translation to ensure all fields are Arabic, and uses strict translation to convert any remaining English words.
    
    Args:
        state: AgentState dictionary containing section_topics, input_text, model_provider, log_file, language, max_iterations.
    
    Returns:
        Dictionary with json_report (list), error (str), iteration (int).
    """
    llog("Task3_ReAct_Agent", "Starting task: Generate JSON", state["log_file"])
    
    json_objects = []
    required_fields = ["topic", "standard_requirement", "rfp_status", "impact", "recommended_improvement", "implementation_notes"]
    chinese_pattern = r'[\u4e00-\u9fff]'
    # Relaxed Arabic pattern to allow proper nouns and transliterated terms
    arabic_pattern = r'^[\u0600-\u06FF\s\d.,;:!?()"\']+[\u0600-\u06FF\s\d.,;:!?()"\']*$'
    state["iteration"] = state.get("iteration", 0)
    # Set language instruction based on target language
    target_language = "ar"
    language_instruction = "Arabic" if target_language == "ar" else "English"
    
    # Input validation
    if not state.get("section_topics") or not isinstance(state["section_topics"], list):
        llog("ReAct_Agent", "Invalid or missing section_topics", state["log_file"])
        return {"error": "Invalid or empty section topics", "json_report": []}
    
    if not state.get("input_text") or not isinstance(state["input_text"], str):
        llog("ReAct_Agent", "Invalid or missing input_text", state["log_file"])
        return {"error": "Invalid or empty input_text", "json_report": []}
    
    if not state.get("model_provider"):
        llog("ReAct_Agent", "Missing model_provider", state["log_file"])
        return {"error": "Missing model provider", "json_report": []}
    
    vector_store = None
    try:
        # Log memory before vector store
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        llog("ReAct_Agent", f"Memory usage before vector store: {mem_before:.2f} MB", state["log_file"])
        
        # Initialize vector store with retries
        max_retries = 2
        for attempt in range(max_retries):
            try:
                vector_store = initialize_rfp_vector_store(state["input_text"], log_file=state["log_file"])
                llog("ReAct_Agent", "RFP vector store initialized", state["log_file"])
                break
            except Exception as e:
                llog("ReAct_Agent", f"Vector store initialization attempt {attempt + 1} failed: {str(e)}", state["log_file"])
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                llog("ReAct_Agent", f"Max retries reached for vector store initialization", state["log_file"])
                return {"error": f"Failed to initialize vector store after {max_retries} retries", "json_report": []}
        
        mem_after = process.memory_info().rss / 1024 / 1024
        llog("ReAct_Agent", f"Memory usage after vector store: {mem_after:.2f} MB", state["log_file"])
        
        # Process each topic
        for topic_dict in state["section_topics"]:
            if not isinstance(topic_dict, dict) or not topic_dict.get("topic"):
                llog("ReAct_Agent", "Invalid topic format, skipping", state["log_file"])
                continue
                
            topic = topic_dict["topic"]
            description = topic_dict.get("description", "")
            llog("ReAct_Agent", f"Processing topic: {topic}", state["log_file"])
            llog("ReAct_Agent", f"Description: {description}", state["log_file"])
            
            # Initialize LLM
            try:
                llm = get_llm(state["model_provider"])
                llog("ReAct_Agent", "LLM initialized successfully", state["log_file"])
            except Exception as e:
                llog("ReAct_Agent", f"Failed to initialize LLM for {topic}: {str(e)}", state["log_file"])
                json_objects.append({
                    "topic": topic,
                    "standard_requirement": description,
                    "rfp_status": "فشل في إنشاء البيانات بسبب فشل تهيئة نموذج اللغة.",
                    "impact": "غير قادر على تقييم الامتثال.",
                    "recommended_improvement": f"راجع يدويًا الموضوع {topic} في طلب العرض.",
                    "implementation_notes": "تأكد من تهيئة مزود نموذج اللغة بشكل صحيح."
                })
                continue
            
            valid_json = False
            json_obj = None
            
            for _ in range(state.get("max_iterations", 7)):
                try:
                    query = f"{topic}: {description}"
                    # Query vector store with retry
                    for attempt in range(max_retries):
                        try:
                            rfp_context = query_vector_store(query=query, vector_store=vector_store, k=12, log_file=state["log_file"])
                            rfp_context_text = "\n".join(rfp_context) if rfp_context else ""
                            llog("ReAct_Agent", f"RFP context tokens: {count_tokens(rfp_context_text)}", state["log_file"])
                            break
                        except Exception as e:
                            llog("ReAct_Agent", f"Vector store query attempt {attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
                            if attempt < max_retries - 1:
                                time.sleep(1)
                                continue
                            rfp_context_text = ""
                            llog("ReAct_Agent", f"Fallback: Empty context for {topic}", state["log_file"])
                            break
                    
                    # Prepare LLM prompt
                    prompt = task3_prompt_3
                    system_message = f"You are an expert in RFP analysis and compliance evaluation. Provide all output EXCLUSIVELY in English LANGUAGE ONLY for all fields. The topic field MUST use the exact input topic '{topic}' verbatim."
                    messages = [
                        SystemMessage(content=system_message),
                        HumanMessage(content=prompt.format(
                            topic=topic,
                            description=description,
                            rfp_context=rfp_context_text,
                            language=language_instruction
                        ))
                    ]
                    llog("ReAct_Agent", f"Sending prompt for {topic} to LLM in {language_instruction}", state["log_file"])
                    
                    # LLM invocation with retry
                    def invoke_with_retry():
                        for attempt in range(max_retries):
                            try:
                                response = llm.invoke(messages, timeout=30)
                                if not hasattr(response, "content") or not isinstance(response.content, str) or not response.content.strip():
                                    raise ValueError("Invalid LLM response")
                                return response
                            except Exception as e:
                                llog("ReAct_Agent", f"LLM attempt {attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
                                if attempt < max_retries - 1:
                                    time.sleep(2)
                                    continue
                                raise ValueError("LLM invocation failed after max retries")
                    
                    response = invoke_with_retry()
                    
                    # Clean response
                    cleaned_response = re.sub(r'^```json', '', response.content)
                    cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
                    cleaned_response = cleaned_response.strip()
                    
                    # Check for Chinese characters
                    if re.search(chinese_pattern, cleaned_response):
                        llog("ReAct_Agent", f"Chinese characters detected for {topic}: {re.findall(chinese_pattern, cleaned_response)[:10]}", state["log_file"])
                        continue
                    
                    # Parse JSON with retry
                    for parse_attempt in range(max_retries):
                        try:
                            json_obj = json.loads(cleaned_response)
                            llog("ReAct_Agent", f"JSON generated for {topic} in {language_instruction}", state["log_file"])
                            break
                        except json.JSONDecodeError as parse_e:
                            llog("ReAct_Agent", f"JSON parsing attempt {parse_attempt + 1} for {topic} failed: {str(parse_e)}", state["log_file"])
                            if parse_attempt < max_retries - 1:
                                cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
                                cleaned_response = re.sub(r',\s*\]', ']', cleaned_response)
                                time.sleep(1)
                                continue
                            json_obj = None
                            break
                    
                    if json_obj is None:
                        llog("ReAct_Agent", f"Fallback: Invalid JSON for {topic}", state["log_file"])
                        continue
                    
                    # Validate required fields
                    missing_fields = [field for field in required_fields if field not in json_obj or not json_obj[field]]
                    if missing_fields:
                        llog("ReAct_Agent", f"Missing or empty fields for {topic}: {missing_fields}", state["log_file"])
                        continue
                    
                    # Validate topic matches
                    if json_obj["topic"] != topic:
                        llog("ReAct_Agent", f"Topic mismatch for {topic}: got {json_obj['topic']}", state["log_file"])
                        continue
                    
                    # Store initial JSON
                    if not os.path.exists(DIR):
                        os.makedirs(DIR)
                    path_to_store_json = os.path.join(DIR, f"EA_{topic}_initial.json")
                    with open(path_to_store_json, "w", encoding='utf-8') as f:
                        json.dump(json_obj, f, indent=4, ensure_ascii=False)
                    llog("ReAct_Agent", f"Initial JSON stored for {topic}", state["log_file"])
                    
                    # Handle Arabic output
                    if target_language == "ar":
                        # First translation step
                        llog("ReAct_Agent", f"Translating JSON for {topic} to Arabic", state["log_file"])
                        json_input = json.dumps(json_obj, ensure_ascii=False)
                        translated_json = json_obj  # Initialize with initial JSON
                        is_fully_arabic = False
                        for trans_attempt in range(max_retries):
                            try:
                                messages = [
                                    SystemMessage(content="You are an expert in translation. Provide output in Arabic ONLY for all fields."),
                                    HumanMessage(content=task3_translation_prompt.format(json_text=json_input))
                                ]
                                response = llm.invoke(messages, timeout=30)
                                llog("ReAct_Agent", f"Translation attempt {trans_attempt + 1} for {topic}: {response.content}", state["log_file"])
                                if not hasattr(response, "content") or not isinstance(response.content, str) or not response.content.strip():
                                    raise ValueError("Invalid translation response")
                                # Clean and parse translated response
                                cleaned_trans_response = re.sub(r'^```json','', response.content)
                                cleaned_trans_response = re.sub(r'\n?```$', '', cleaned_trans_response)
                                translated_json = json.loads(cleaned_trans_response)
                                # Check if translated JSON is fully Arabic
                                is_fully_arabic = True
                                for field in required_fields:
                                    llog("ReAct_Agent", f"Checking for non-Arabic text in translated field {field}", state["log_file"])
                                    if not re.match(arabic_pattern, translated_json[field], re.UNICODE):
                                        llog("ReAct_Agent", f"Non-Arabic text detected in translated field {field} for {topic}", state["log_file"])
                                        is_fully_arabic = False
                                        break
                                if is_fully_arabic:
                                    llog("ReAct_Agent", f"Translation attempt {trans_attempt + 1} for {topic} produced fully Arabic JSON", state["log_file"])
                                    break
                            except Exception as e:
                                llog("ReAct_Agent", f"Translation attempt {trans_attempt + 1} for {topic} failed: {str(e)}", state["log_file"])
                                if trans_attempt < max_retries - 1:
                                    time.sleep(2)
                                    continue
                                llog("ReAct_Agent", f"Reached max translation retries for {topic}, proceeding to strict translation", state["log_file"])
                                break
                        
                        # If not fully Arabic after max_retries, perform strict translation Seantranslation
                        if not is_fully_arabic:
                            llog("ReAct_Agent", f"Performing strict Arabic translation for {topic}", state["log_file"])
                            json_input = json.dumps(translated_json, ensure_ascii=False)
                            try:
                                messages = [
                                    SystemMessage(content="You are an expert in translation. Provide output in Arabic ONLY for all fields, preserving technical terms as proper nouns or transliterating them."),
                                    HumanMessage(content=task3_strict_translation_prompt.format(json_text=json_input))
                                ]
                                response = llm.invoke(messages, timeout=30)
                                llog("ReAct_Agent", f"Strict translation for {topic}: {response.content}", state["log_file"])
                                if not hasattr(response, "content") or not isinstance(response.content, str) or not response.content.strip():
                                    raise ValueError("Invalid strict translation response")
                                # Clean and parse strict translated response
                                cleaned_strict_trans_response = re.sub(r'^```json','', response.content)
                                cleaned_strict_trans_response = re.sub(r'\n?```$', '', cleaned_strict_trans_response)
                                translated_json = json.loads(cleaned_strict_trans_response)
                                llog("ReAct_Agent", f"Strict translation for {topic} completed, using as final JSON", state["log_file"])
                            except Exception as e:
                                llog("ReAct_Agent", f"Strict translation for {topic} failed: {str(e)}, using last translated JSON", state["log_file"])
                        
                        json_obj = translated_json
                        
                        # Store Arabic JSON
                        path_to_store_json = os.path.join(DIR, f"EA_{topic}_ar.json")
                        with open(path_to_store_json, "w", encoding='utf-8') as f:
                            json.dump(json_obj, f, indent=4, ensure_ascii=False)
                        llog("ReAct_Agent", f"Arabic JSON stored for {topic}", state["log_file"])
                    else:
                        llog("ReAct_Agent", f"No translation needed for {topic}, language is English", state["log_file"])
                    
                    llog("ReAct_Agent", f"Valid JSON generated for {topic} in target language", state["log_file"])
                    llog("ReAct_Agent", f"JSON: {json_obj}", state["log_file"])
                    json_objects.append(json_obj)
                    valid_json = True
                    break
                
                except Exception as e:
                    llog("ReAct_Agent", f"Error generating JSON for {topic}: {str(e)}", state["log_file"])
                    state["iteration"] += 1
                    if state["iteration"] >= state.get("max_iterations", 7):
                        break
            
            if not valid_json:
                llog("ReAct_Agent", f"Fallback: Failed to generate valid JSON for {topic}", state["log_file"])
                json_objects.append({
                    "topic": topic,
                    "standard_requirement": description,
                    "rfp_status": f"لم يتم إنشاء بيانات محاذاة طلب العرض الصحيحة بعد {state.get('max_iterations', 7)} محاولات.",
                    "impact": "غير قادر على تقييم الامتثال بسبب فشل الإنشاء.",
                    "recommended_improvement": f"إعادة تقييم سياق طلب العرض يدويًا لهذا الموضوع {topic}.",
                    "implementation_notes": "فكر في زيادة الحد الأقصى للتكرارات أو تحسين سياق طلب العرض."
                })
    
    except Exception as e:
        llog("ReAct_Agent", f"Critical error in task: {str(e)}", state["log_file"])
        return {"error": f"Task failed: {str(e)}", "json_report": json_objects}
    
    finally:
        if vector_store is not None:
            del vector_store
            gc.collect()
            mem_after_cleanup = process.memory_info().rss / 1024 / 1024
            llog("ReAct_Agent", f"Memory usage after cleanup: {mem_after_cleanup:.2f} MB", state["log_file"])
    
    llog("ReAct_Agent", f"Generated JSON report with {len(json_objects)} entries", state["log_file"])
    return {"json_report": json_objects, "error": "", "iteration": 0}

def task4_generate_html(state: Dict[str, Any]) -> Dict[str, Any]:
    """ReAct agent to convert aggregated JSON into an HTML report."""
    llog("ReAct_Agent", "Starting task 4: Generate HTML Report", state["log_file"])
    
    language_instruction = "Arabic" if state["language"] == "ar" else "English"
    table_dir = 'rtl' if state["language"] == "ar" else 'ltr'
    table_tag = f'<table class="rfp-report-table" dir="{table_dir}">'
    json_report = state.get("json_report", [])
    
    if not json_report:
        llog("ReAct_Agent", "Error: No JSON report available", state["log_file"])
        return {
            "html_table": f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>RFP Compliance Report</title>
                <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
                <style>
                    .rfp-report-container * {{
                        box-sizing: border-box;
                        margin: 0;
                        padding: 0;
                        font-family: 'Roboto', sans-serif;
                    }}
                    .rfp-report-container {{
                        max-width: 1200px;
                        margin: 20px auto;
                        background-color: #fff;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        direction: {table_dir};
                    }}
                    .rfp-report-table {{
                        border-collapse: collapse;
                        width: 100%;
                        font-size: 14px;
                        margin: 20px 0;
                    }}
                    .rfp-report-table th, .rfp-report-table td {{
                        border: 1px solid #ddd;
                        padding: 12px;
                        text-align: {'right' if state["language"] == "ar" else 'left'};
                        vertical-align: top;
                    }}
                    .rfp-report-table th {{
                        background-color: #4a6fa5;
                        color: #fff;
                        font-weight: 700;
                    }}
                    .rfp-report-table td:nth-child(2) {{
                        padding: 15px;
                    }}
                    .rfp-report-table tr:not(:last-child) {{
                        border-bottom: 2px solid #e5e7eb;
                    }}
                    .rfp-report-section-header {{
                        font-size: 24px;
                        font-weight: 700;
                        margin: 30px 0 15px;
                        color: #1a3c6e;
                        text-align: center;
                    }}
                    .rfp-report-element {{
                        background-color: #f5f7fa;
                        padding: 10px;
                        margin-bottom: 8px;
                        border-radius: 4px;
                        transition: all 0.2s ease;
                    }}
                    .rfp-report-element:hover {{
                        background-color: #e8ecef;
                        transform: scale(1.01);
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .rfp-report-element.error {{
                        background-color: #fef2f2;
                        color: #b91c1c;
                    }}
                </style>
            </head>
            <body>
                <div class="rfp-report-container">
                    <div class="rfp-report-section-header">RFP Compliance Report</div>
                    {table_tag}
                        <tr><th>Elements</th><th>Analysis</th></tr>
                        <tr><td colspan="2">No data available due to missing JSON report.</td></tr>
                    </table>
                </div>
            </body>
            </html>
            """,
            "error": "No JSON report available"
        }

    html_rows = []
    required_fields = ["topic", "standard_requirement", "rfp_status", "impact", "recommended_improvement", "implementation_notes"]

    for entry in json_report:
        try:
            # Validate JSON entry
            missing_fields = [field for field in required_fields if field not in entry or not entry[field]]
            if missing_fields:
                llog("ReAct_Agent", f"Invalid JSON entry for topic {entry.get('topic', 'unknown')}: Missing fields {missing_fields}", state["log_file"])
                html_rows.append(f"""
                    <tr>
                        <td>{entry.get('topic', 'Unknown Topic')}</td>
                        <td>
                            <div class="rfp-report-element error">
                                <b>Error:</b> Missing or empty fields: {', '.join(missing_fields)}<br>
                                <b>Fallback:</b> Please review the JSON generation for this topic.
                            </div>
                        </td>
                    </tr>
                """)
                continue
            
            # Generate HTML row
            html_rows.append(f"""
                <tr>
                    <td>{entry['topic']}</td>
                    <td>
                        <div class="rfp-report-element">
                            <b>Standard Requirement:</b> {entry['standard_requirement']}
                        </div>
                        <div class="rfp-report-element">
                            <b>RFP Status:</b> {entry['rfp_status']}
                        </div>
                        <div class="rfp-report-element">
                            <b>Impact:</b> {entry['impact']}
                        </div>
                        <div class="rfp-report-element">
                            <b>Recommended Improvement:</b> {entry['recommended_improvement']}
                        </div>
                        <div class="rfp-report-element">
                            <b>Implementation Notes:</b> {entry['implementation_notes']}
                        </div>
                    </td>
                </tr>
            """)
            llog("ReAct_Agent", f"Generated HTML row for topic {entry['topic']}", state["log_file"])
        
        except Exception as e:
            llog("ReAct_Agent", f"Error processing JSON entry for topic {entry.get('topic', 'unknown')}: {str(e)}", state["log_file"])
            html_rows.append(f"""
                <tr>
                    <td>{entry.get('topic', 'Unknown Topic')}</td>
                    <td>
                        <div class="rfp-report-element error">
                            <b>Error:</b> Failed to process entry: {str(e)}<br>
                            <b>Fallback:</b> Please review the JSON data for this topic.
                        </div>
                    </td>
                </tr>
            """)

    html_report = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RFP Compliance Report</title>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
            <style>
                .rfp-report-container * {{
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                    font-family: 'Roboto', sans-serif;
                }}
                .rfp-report-container {{
                    max-width: 1200px;
                    margin: 20px auto;
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    direction: {table_dir};
                }}
                .rfp-report-table {{
                    border-collapse: collapse;
                    width: 100%;
                    font-size: 14px;
                    margin: 20px 0;
                }}
                .rfp-report-table th, .rfp-report-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: {'right' if state["language"] == "ar" else 'left'};
                    vertical-align: top;
                }}
                .rfp-report-table th {{
                    background-color: #4a6fa5;
                    color: #fff;
                    font-weight: 700;
                }}
                .rfp-report-table td:nth-child(2) {{
                    padding: 15px;
                }}
                .rfp-report-table tr:not(:last-child) {{
                    border-bottom: 2px solid #e5e7eb;
                }}
                .rfp-report-section-header {{
                    font-size: 24px;
                    font-weight: 700;
                    margin: 30px 0 15px;
                    color: #1a3c6e;
                    text-align: center;
                }}
                .rfp-report-element {{
                    background-color: #f5f7fa;
                    padding: 10px;
                    margin-bottom: 8px;
                    border-radius: 4px;
                    transition: all 0.2s ease;
                }}
                .rfp-report-element:hover {{
                    background-color: #e8ecef;
                    transform: scale(1.01);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .rfp-report-element.error {{
                    background-color: #fef2f2;
                    color: #b91c1c;
                }}
            </style>
        </head>
        <body>
            <div class="rfp-report-container">
                <div class="rfp-report-section-header">RFP Compliance Report</div>
                {table_tag}
                    <tr><th>Elements</th><th>Analysis</th></tr>
                    {''.join(html_rows)}
                </table>
            </div>
        </body>
        </html>
    """
    llog("ReAct_Agent", f"Generated HTML report with {len(html_rows)} rows", state["log_file"])

    # if os.path.exists(DIR):
    #     shutil.rmtree(DIR)
    #     print(f"Directory '{DIR}' has been deleted.")
    # else:
    #     print(f"Directory '{DIR}' does not exist.")

    return {"html_table": html_report, "error": ""}


def build_workflow(state: Dict[str, Any]) -> StateGraph:
    """Build the LangGraph workflow for the ReAct agent system."""
    llog("ReAct_Agent", "Building LangGraph", state["log_file"])
    workflow = StateGraph(AgentState)
    # workflow.add_node("Start_flow", standard_English_refiner)
    workflow.add_node("extract_topics", task1_extract_topics)
    workflow.add_node("map_sections", task2_map_sections)
    workflow.add_node("generate_json", task3_generate_json)
    workflow.add_node("generate_html", task4_generate_html)
    # workflow.add_edge(START, "Start_flow")
    workflow.add_edge(START, "extract_topics")
    workflow.add_edge("extract_topics", "map_sections")
    workflow.add_edge("map_sections", "generate_json")
    workflow.add_edge("generate_json", "generate_html")
    workflow.add_edge("generate_html", END)
    return workflow.compile()

def run_react_agent(rfp_text: str, standard_text: str, topics: List[str], language: str, log_save_file_name: str, model_provider: str) -> Dict[str, Any]:
    """Run the ReAct agent workflow."""
    llog("ReAct_Agent", "Starting ReAct agent workflow", log_save_file_name)
    llog("ReAct_Agent", f"Dynamic topics received: {topics}", log_save_file_name)
    initial_state = {
        "input_text": rfp_text,
        "section_topics": [],
        "standard_text": standard_text,
        "mapped_sections": {},
        "html_table": "",
        "language": language,
        "iteration": 0,
        "max_iterations": 5,
        "error": "",
        "dynamic_topics": topics,
        "log_file": log_save_file_name,
        "model_provider": model_provider
    }

    try:
        llog("ReAct_Agent", "Main Running ", log_save_file_name)
        workflow = build_workflow(initial_state)
        result = workflow.invoke(initial_state)
        llog("ReAct_Agent", "Main Finished", log_save_file_name)
        return {"ea_standard_eval": {"report": result["html_table"]}}

    except Exception as e:
        llog("ReAct_Agent", f"Error running workflow: {str(e)}", log_save_file_name)
        return {"error": str(e)}
    finally:
        llog("ReAct_Agent", "Vector store cleared (in-memory, no persistence)", log_save_file_name)


if __name__ == "__main__":
    
    log_file = f"z_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open("/home/khantil/kaushik/RFP_agent/my_agent_rag_rfp/translated.txt", "r") as f:
        rfp_text = f.read()
    with open("/home/khantil/kaushik/RFP_agent/my_agent_rag_rfp/ea_standard.txt", "r") as f:
        standard_text = f.read()
        
    run_react_agent(
        rfp_text,
        standard_text,
        ["Preferred Application Frameworks","Accepted Database","Accepted Operating System","Requirement of SSL Certificate","Non-Functional","Scope of Work",  "Project Delivery", "Cybersecurity"],
        "ar",
        log_file,
        "opensource"
    )