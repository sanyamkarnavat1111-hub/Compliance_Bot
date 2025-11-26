# import torch  # Disabled - CUDA not available
import time
import multiprocessing
import httpx
# from vllm import LLM, SamplingParams
# from vllm.transformers_utils.tokenizer import get_tokenizer
from langchain_openai import ChatOpenAI
from groq import Groq
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from logger import custom_logger as llog
import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from prompts import *
from dotenv import load_dotenv
# from model_provider import ModelProvider  # Disabled - CUDA not available
from langchain_core.caches import BaseCache
from pydantic import BaseModel
import datetime
import uuid 
import re
import os
import warnings
import json
import tiktoken  # Import tiktoken for OpenAI token counting
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks  # Add this import
warnings.filterwarnings("ignore")

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues in forked processes
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set
    pass

class AgentState(TypedDict):
    detected_language: str
    rfp_text: str
    proposal_text: str
    rfp_requirements: str
    proposal_eval_report_json: str
    proposal_eval_report_html: str

class AgentCluster():   
    def __init__(self, model_provider=None):
        """
        Initialize the AgentCluster for proposal evaluation
        
        Args:
            model_provider: Optional shared model provider instance
        """
        load_dotenv()
        # Initialize OpenAI client for openai model type
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.log_save_file_name = 'rfp_proposal_eval_llog.txt'
        # self.model_provider = model_provider

        # initial_log_id = self.getUniqueFileNameForLogger()  
        llog("MAIN", "Initializing MAIN ", "rfp_proposal_eval_log")
        # self.model_provider = ModelProvider(log_save_file_name=initial_log_id)
        

        
        # # Use the shared model provider if available, otherwise initialize locally
        # if not model_provider:
        #     # Initialize the tokenizer
        #     print("Initializing local tokenizer...")
        #     self.tokenizer = None
        #     try:
        #         self.tokenizer = get_tokenizer("Qwen/Qwen2.5-14B-Instruct")
        #         print("Tokenizer initialized successfully")
        #     except Exception as e:
        #         print(f"Error initializing tokenizer: {str(e)}")
        #         raise
            
        #     # Initialize the model
        #     print("Initializing local Qwen model...")
        #     self.model = None
        #     try:
        #         gpu_count = torch.cuda.device_count()
        #         print(f"Using {gpu_count} GPUs for tensor parallelism")
                
        #         self.model = LLM(
        #             model="Qwen/Qwen2.5-14B-Instruct",  
        #             gpu_memory_utilization=0.9,
        #             tensor_parallel_size=gpu_count,
        #             dtype="auto",
        #             max_model_len=131072,
        #             enforce_eager=True
        #         )
        #         print("Model initialized successfully")
        #     except Exception as e:
        #         print(f"Error initializing model: {str(e)}")
        #         raise
        
        # Initialize model provider
        self.model = None
        self.tokenizer = None
        self.model_provider = model_provider
        
        if not model_provider:
            print("Initializing ModelProvider...")
            # try:
            #     self.model_provider = ModelProvider(log_save_file_name="rfp_proposal_eval_log")
            #     llog("AgentCluster", "ModelProvider initialized successfully", "rfp_proposal_eval_log")
            #     print("ModelProvider initialized successfully")
            # except Exception as e:
            #     # error_msg = f"Error initializing ModelProvider: {str(e)}"
            #     # print(f"Error: {error_msg}")
            #     # llog("AgentCluster", error_msg, "rfp_proposal_eval_log")
            #     # raise
            #     print(f"Warning: {e}")
            #     print("Continuing without ModelProvider - will use CrewAI/OpenRouter for LLM tasks")
            #     llog("AgentCluster", f"ModelProvider initialization failed: {e}", "rfp_proposal_eval_log")
            #     llog("AgentCluster", "Continuing without ModelProvider - will use CrewAI/OpenRouter", "rfp_proposal_eval_log")
            #     self.model_provider = None  # Set to None instead of raising
                # ModelProvider disabled - CUDA not available, using CrewAI/OpenRouter instead
            print("ModelProvider disabled - using CrewAI/OpenRouter for all LLM tasks")
            llog("AgentCluster", "ModelProvider disabled - using CrewAI/OpenRouter for all LLM tasks", "rfp_proposal_eval_log")
            self.model_provider = None
        else:
            llog("AgentCluster", "Using provided ModelProvider", "rfp_proposal_eval_log")
        
        # Preload the open-source model
        # try:
        #     success = self.model_provider.preload_opensource_model()
        #     if not success or not self.model_provider.model:
        #         error_msg = "Failed to preload open-source model"
        #         llog("AgentCluster", error_msg, "rfp_proposal_eval_log")
        #         raise RuntimeError(error_msg)
        #     print("Open-source model preloaded successfully")
        #     llog("AgentCluster", "ModelProvider state: model_type=%s, model=%s" % (
        #         getattr(self.model_provider, 'model_type', 'None'),
        #         "set" if getattr(self.model_provider, 'model', None) else "None"
        #     ), initial_log_id)
        # except Exception as e:
        #     error_msg = f"Error preloading open-source model: {str(e)}"
        #     print(f"Error: {error_msg}")
        #     llog("AgentCluster", error_msg, "rfp_proposal_eval_log")
        #     raise
                
        # # Set up OpenAI model if needed
        # try:
        #     self.openai_model = ChatOpenAI(
        #         model="gpt-4o", 
        #         temperature=0, 
        #         request_timeout=120, 
        #         max_retries=3,
        #         api_key=os.getenv("OPENAI_API_KEY")
        #     )
        #     print("LangChain OpenAI client initialized successfully")
        # except Exception as e:
        #     print(f"Error initializing LangChain OpenAI client: {str(e)}")
        #     self.openai_model = None
        
        # Set the maximum tokens
        self.max_tokens = 120000  # Maximum context length for Qwen2.5-14B
        
        # Initialize openai_model attribute to None
        # self.openai_model = None
        
        self.graph = self.__setup_graph()

        # Set up OpenAI model
        self.openai_model = None
        if self.api_key:
            try:
                print("Debug: Attempting to access BaseCache and Callbacks")
 
                print("Debug: BaseCache and Callbacks imported successfully")
                BaseCache._pydantic_model = BaseModel
                print("Debug: Set BaseCache._pydantic_model")
                # Ensure Callbacks is recognized by Pydantic
                if not hasattr(ChatOpenAI, '_pydantic_model'):
                    ChatOpenAI._pydantic_model = BaseModel
                ChatOpenAI.model_rebuild()
                print("Debug: ChatOpenAI model rebuilt")
                self.openai_model = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    request_timeout=120,
                    max_retries=3,
                    api_key=self.api_key
                )
                print("LangChain OpenAI client initialized successfully")
            except Exception as e:
                print(f"Error initializing LangChain OpenAI client: {str(e)}")
                llog("AgentCluster", f"Error initializing LangChain OpenAI client: {str(e)}", self.log_save_file_name)
                self.openai_model = None
        else:
            print("No OPENAI_API_KEY found. OpenAI model will not be available.")

        # router
        # if self.api_key:
        #     try:
        #         from openai import OpenAI
        #         print("Debug: Initializing OpenRouter client with OpenAI")
        #         self.openai_model = OpenAI(
        #             base_url="https://openrouter.ai/api/v1",
        #             api_key=self.api_key,
        #             extra_headers={
        #                 "HTTP-Referer": "https://localhost/rfp-eval-test",
        #                 "X-Title": "RFP Evaluation Test"
        #             }
        #         )
        #         print("Debug: OpenRouter client initialized successfully")
        #     except Exception as e:
        #         print(f"Error initializing OpenRouter client: {str(e)}")
        #         llog("AgentCluster", f"Error initializing OpenRouter client: {str(e)}", self.log_save_file_name)
        #         self.openai_model = None
        # else:
        #     print("No OPENROUTER_API_KEY found. OpenRouter model will not be available.")
        
        # groq
        # if self.api_key:
        #     try:
        #         print("Debug: Initializing Groq client")
        #         self.openai_model = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        #         print("Debug: Groq client initialized successfully")
        #     except Exception as e:
        #         error_msg = f"Error initializing Groq client: {str(e)}"
        #         print(error_msg)
        #         llog("AgentCluster", error_msg, self.log_save_file_name)
        #         self.groq_client = None
        # else:
        #     print("No GROQ_API_KEY found. Groq model will not be available.")

        self.max_tokens = getattr(self, 'max_tokens', 120000)  # Fallback if not set
        print(f"Debug: max_tokens set to {self.max_tokens}")
        self.graph = self.__setup_graph()
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        # If using OpenAI, use tiktoken for accurate counting
        if (self.model_provider and self.model_provider.model_type == "openai") or (not self.model_provider and self.openai_model):
            try:
                # Use tiktoken for OpenAI models
                encoding = tiktoken.encoding_for_model("gpt-4o")
                return len(encoding.encode(text))
            except Exception as e:
                print(f"Error using tiktoken: {str(e)}")
                # Fall back to rough estimate
                return len(text.split())
        
        # For non-OpenAI models, use the appropriate tokenizer
        if self.model_provider and hasattr(self.model_provider, 'count_tokens'):
            return self.model_provider.count_tokens(text)
        elif self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            print("Warning: No tokenizer available for token counting")
            return len(text.split())  # Fallback to rough estimate
    
    # def generate(self, prompt: str) -> str:
    #     """Generate a response from the model."""
    #     # If we have a model provider and it's set to opensource, use it
    #     if self.model_provider and self.model_provider.model_type == "opensource":
    #         try:
    #             # Acquire the model only if it's opensource
    #             # The model acquisition/release is now handled by the worker class
    #             # We don't need to acquire/release here
    #             return self.model_provider.generate(prompt)
    #         except Exception as e:
    #             error_msg = f"Error using shared model provider: {str(e)}"
    #             print(f"Error: {error_msg}")
    #             llog("AgentCluster", error_msg, self.log_save_file_name)
    #             return f"Error: {error_msg}"
    
    #     # Handle OpenAI model type
    #     if (self.model_provider and self.model_provider.model_type == "openai") or (not self.model_provider and self.openai_model):
    #         try:
    #             # Log that we're making an OpenAI API call
    #             llog("AgentCluster", f"Making OpenAI API call with prompt length: {len(prompt)}", self.log_save_file_name)
                
    #             # Make API call with LangChain ChatOpenAI
    #             if not self.openai_model:
    #                 # If for some reason the client wasn't initialized properly, initialize it now
    #                try:
    #                     llog("rfp_proposal_eval", f"Initializing LangChain OpenAI client", self.log_save_file_name)
    #                     self.openai_model = ChatOpenAI(
    #                         model="gpt-4o", 
    #                         temperature=0, 
    #                         request_timeout=120, 
    #                         max_retries=3,
    #                         api_key=os.getenv("OPENAI_API_KEY")
    #                     )
    #                     print("LangChain OpenAI client initialized successfully")
    #                     llog("rfp_proposal_eval", f"process Done LangChain OpenAI client", self.log_save_file_name)
    #                except Exception as e:
    #                     print(f"Error initializing LangChain OpenAI client: {str(e)}")
    #                     self.openai_model = None
                
    #             # Count tokens for logging
    #             llog("rfp_proposal_eval", f"counting tokesns of prompt..", self.log_save_file_name)
    #             token_count = self.count_tokens(prompt)
    #             print(f"Sending prompt to OpenAI (length: {token_count} tokens)")
                
    #             # Create messages array
    #             messages = [
    #                 SystemMessage(content="You are an expert in RFP and proposal evaluation."),
    #                 HumanMessage(content=prompt)
    #             ]
                
    #             # Make the API call
    #             response = self.openai_model.invoke(messages)
                
    #             # Extract the content
    #             return response.content
                
    #         except Exception as e:
    #             error_msg = f"Error using OpenAI API: {str(e)}"
    #             llog("AgentCluster", error_msg, self.log_save_file_name)
    #             print(f"Error: {error_msg}")
    #             return f"Error: {error_msg}"
        
    #     # Otherwise use the local model    if available
    #     try:
    #         if not self.model_provider and not self.tokenizer:
    #             raise ValueError("No model or tokenizer available")
                
    #         messages = [
    #             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #             {"role": "user", "content": prompt}
    #         ]

    #         # Use local tokenizer if no model provider
    #         if not self.model_provider:
    #             text = self.tokenizer.apply_chat_template(
    #                 messages, tokenize=False, add_generation_prompt=True
    #             )
    #         else:
    #             # Use model provider's tokenizer
    #             text = self.model_provider.tokenizer.apply_chat_template(
    #                 messages, tokenize=False, add_generation_prompt=True
    #             )
            
    #         # Print a progress message
    #         token_count = self.count_tokens(text)
    #         print(f"Sending prompt to model (length: {token_count} tokens)")
            
    #         sampling_params = SamplingParams(
    #             max_tokens=8192,
    #             temperature=0.7,
    #             repetition_penalty=1,
    #             top_p=0.8,
    #             top_k=20
    #         )

    #         start_time = time.time()
    #         try:
    #             # Use local model if no model provider
    #             if not self.model_provider:
    #                 outputs = self.model.generate([text], sampling_params)
    #             else:
    #                 # Use model provider's model - no need to acquire/release here
    #                 # as it's handled by the worker class
    #                 outputs = self.model_provider.model.generate([text], sampling_params)
                
    #             # More robust error checking for empty outputs
    #             if not outputs:
    #                 raise ValueError("Model returned empty outputs list")
                
    #             if len(outputs) == 0:
    #                 raise ValueError("Model returned outputs list of length 0")
                    
    #             if len(outputs[0].outputs) == 0:
    #                 raise ValueError("Model returned empty outputs[0].outputs list")
                    
    #             # Check if the output text is empty
    #             response = outputs[0].outputs[0].text
    #             if not response or response.strip() == "":
    #                 raise ValueError("Model returned empty response text")
                    
    #             elapsed = time.time() - start_time
    #             print(f"Response generated in {elapsed:.2f} seconds")
    #             return response
    #         except IndexError as ie:
    #             error_msg = f"Model generation failed - likely due to input being too large: {str(ie)}"
    #             print(f"Error: {error_msg}")
    #             return f"Error: {error_msg}"
    #         except ValueError as ve:
    #             error_msg = f"Model generation failed: {str(ve)}"
    #             print(f"Error: {error_msg}")
    #             return f"Error: {error_msg}"
    #         except Exception as e:
    #             error_msg = f"Unexpected error in model generation: {str(e)}"
    #             print(f"Error: {error_msg}")
    #             return f"Error: {error_msg}"
    #     except Exception as e:
    #         error_msg = f"Error in generate method: {str(e)}"
    #         print(f"Error: {error_msg}")
    #         llog("AgentCluster", error_msg, self.log_save_file_name)
    #         return f"Error: {error_msg}"
    
    # def generate(self, prompt: str) -> str:
    #     """Generate a response from the model."""
    #     # If we have a model provider and it's set to opensource, use it
    #     if self.model_provider and self.model_provider.model_type == "opensource":
    #         try:
    #             return self.model_provider.generate(prompt)
    #         except Exception as e:
    #             error_msg = f"Error using shared model provider: {str(e)}"
    #             print(f"Error: {error_msg}")
    #             llog("AgentCluster", error_msg, self.log_save_file_name)
    #             return f"Error: {error_msg}"

    #     # Handle OpenAI model type
    #     if (self.model_provider and self.model_provider.model_type == "openai") or (not self.model_provider and self.openai_model):
    #         if not self.openai_model:
    #             # Attempt to reinitialize if possible
    #             if self.api_key:
    #                 try:
    #                     llog("rfp_proposal_eval", f"Initializing LangChain OpenAI client", self.log_save_file_name)
    #                     if not hasattr(BaseCache, '_pydantic_model'):
    #                         BaseCache._pydantic_model = BaseModel
    #                     ChatOpenAI.model_rebuild()
    #                     self.openai_model = ChatOpenAI(
    #                         model="gpt-4o",
    #                         temperature=0,
    #                         request_timeout=120,
    #                         max_retries=3,
    #                         api_key=self.api_key
    #                     )
    #                     print("LangChain OpenAI client initialized successfully")
    #                     llog("rfp_proposal_eval", f"process Done LangChain OpenAI client", self.log_save_file_name)
    #                 except Exception as e:
    #                     print(f"Error initializing LangChain OpenAI client: {str(e)}")
    #                     self.openai_model = None

    #         if self.openai_model:
    #             try:
    #                 llog("AgentCluster", f"Making OpenAI API call with prompt length: {len(prompt)}", self.log_save_file_name)
    #                 token_count = self.count_tokens(prompt)
    #                 print(f"Sending prompt to OpenAI (length: {token_count} tokens)")
    #                 messages = [
    #                     SystemMessage(content="You are an expert in RFP and proposal evaluation."),
    #                     HumanMessage(content=prompt)
    #                 ]
    #                 response = self.openai_model.invoke(messages)
    #                 return response.content
    #             except Exception as e:
    #                 error_msg = f"Error using OpenAI API: {str(e)}"
    #                 llog("AgentCluster", error_msg, self.log_save_file_name)
    #                 print(f"Error: {error_msg}")
    #                 return f"Error: {error_msg}"
    #         else:
    #             error_msg = "OpenAI model unavailable. Falling back to local model if available."
    #             print(f"Error: {error_msg}")
    #             llog("AgentCluster", error_msg, self.log_save_file_name)

    #     # Fallback to local model if available
    #     if not self.model_provider and self.model and self.tokenizer:
    #         try:
    #             messages = [
    #                 {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #                 {"role": "user", "content": prompt}
    #             ]
    #             text = self.tokenizer.apply_chat_template(
    #                 messages, tokenize=False, add_generation_prompt=True
    #             )
    #             token_count = self.count_tokens(text)
    #             print(f"Sending prompt to local model (length: {token_count} tokens)")
    #             sampling_params = SamplingParams(
    #                 max_tokens=8192,
    #                 temperature=0.7,
    #                 repetition_penalty=1,
    #                 top_p=0.8,
    #                 top_k=20
    #             )
    #             outputs = self.model.generate([text], sampling_params)
    #             if not outputs or len(outputs) == 0 or len(outputs[0].outputs) == 0:
    #                 raise ValueError("Model returned empty response")
    #             response = outputs[0].outputs[0].text
    #             if not response or response.strip() == "":
    #                 raise ValueError("Model returned empty response text")
    #             elapsed = time.time() - start_time
    #             print(f"Response generated in {elapsed:.2f} seconds")
    #             return response
    #         except Exception as e:
    #             error_msg = f"Error using local model: {str(e)}"
    #             print(f"Error: {error_msg}")
    #             llog("AgentCluster", error_msg, self.log_save_file_name)
    #             return f"Error: {error_msg}"

    #     error_msg = "No valid model available for generation"
    #     print(f"Error: {error_msg}")
    #     llog("AgentCluster", error_msg, self.log_save_file_name)
    #     return f"Error: {error_msg}"

    # def generate(self, prompt: str) -> str:
    #     """Generate a response from the model using only the selected model type."""
    #     start_time = time.time()
    #     model_type = self.model_provider.model_type if self.model_provider else "opensource"
    #     llog("AgentCluster", f"Generating with model type: {model_type}", self.log_save_file_name)
    #     print(f"Debug: Using model type: {model_type}")

    #     # # Handle opensource model
    #     # if model_type == "opensource":
    #     #     if self.model_provider and self.model_provider.model_type == "opensource":
    #     #         try:
    #     #             response = self.model_provider.generate(prompt)
    #     #             elapsed = time.time() - start_time
    #     #             print(f"Response generated in {elapsed:.2f} seconds")
    #     #             return response
    #     #         except Exception as e:
    #     #             error_msg = f"Error using shared model provider: {str(e)}"
    #     #             print(f"Error: {error_msg}")
    #     #             llog("AgentCluster", error_msg, self.log_save_file_name)
    #     #             return f"Error: {error_msg}"
    #     #     elif self.model and self.tokenizer:
    #     #         try:
    #     #             messages = [
    #     #                 {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #     #                 {"role": "user", "content": prompt}
    #     #             ]
    #     #             text = self.tokenizer.apply_chat_template(
    #     #                 messages, tokenize=False, add_generation_prompt=True
    #     #             )
    #     #             token_count = self.count_tokens(text)
    #     #             print(f"Sending prompt to local model (length: {token_count} tokens)")
    #     #             sampling_params = SamplingParams(
    #     #                 max_tokens=8192,
    #     #                 temperature=0.7,
    #     #                 repetition_penalty=1,
    #     #                 top_p=0.8,
    #     #                 top_k=20
    #     #             )
    #     #             outputs = self.model.generate([text], sampling_params)
    #     #             if not outputs or len(outputs) == 0 or len(outputs[0].outputs) == 0:
    #     #                 raise ValueError("Model returned empty response")
    #     #             response = outputs[0].outputs[0].text
    #     #             if not response or response.strip() == "":
    #     #                 raise ValueError("Model returned empty response text")
    #     #             elapsed = time.time() - start_time
    #     #             print(f"Response generated in {elapsed:.2f} seconds")
    #     #             return response
    #     #         except Exception as e:
    #     #             error_msg = f"Error using local model: {str(e)}"
    #     #             print(f"Error: {error_msg}")
    #     #             llog("AgentCluster", error_msg, self.log_save_file_name)
    #     #             return f"Error: {error_msg}"
    #     #     else:
    #     #         error_msg = "Opensource model not available"
    #     #         print(f"Error: {error_msg}")
    #     #         llog("AgentCluster", error_msg, self.log_save_file_name)
    #     #         return f"Error: {error_msg}"

    #     # Handle opensource model
    #     if model_type == "opensource":
    #         try:
    #             if self.model_provider:
    #                 response = self.model_provider.generate(prompt)
    #                 elapsed = time.time() - start_time
    #                 print(f"Response generated in {elapsed:.2f} seconds")
    #                 return response
    #             elif self.model and self.tokenizer:
    #                 messages = [
    #                     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #                     {"role": "user", "content": prompt}
    #                 ]
    #                 text = self.tokenizer.apply_chat_template(
    #                     messages, tokenize=False, add_generation_prompt=True
    #                 )
    #                 token_count = self.count_tokens(text)
    #                 print(f"Sending prompt to local model (length: {token_count} tokens)")
    #                 sampling_params = SamplingParams(
    #                     max_tokens=8192,
    #                     temperature=0.7,
    #                     repetition_penalty=1,
    #                     top_p=0.8,
    #                     top_k=20
    #                 )
    #                 outputs = self.model.generate([text], sampling_params)
    #                 if not outputs or len(outputs) == 0 or len(outputs[0].outputs) == 0:
    #                     raise ValueError("Model returned empty response")
    #                 response = outputs[0].outputs[0].text
    #                 if not response or response.strip() == "":
    #                     raise ValueError("Model returned empty response text")
    #                 elapsed = time.time() - start_time
    #                 print(f"Response generated in {elapsed:.2f} seconds")
    #                 return response
    #             else:
    #                 error_msg = "Opensource model not available"
    #                 print(f"Error: {error_msg}")
    #                 llog("AgentCluster", error_msg, self.log_save_file_name)
    #                 return f"Error: {error_msg}"
    #         except Exception as e:
    #             error_msg = f"Error using opensource model: {str(e)}"
    #             print(f"Error: {error_msg}")
    #             llog("AgentCluster", error_msg, self.log_save_file_name)
    #             return f"Error: {error_msg}"

    #     # Handle OpenAI model
    #     if model_type == "openai":
    #         if not self.openai_model and self.api_key:
    #             try:
    #                 print("Debug: Reinitializing OpenAI client")
    #                 from langchain_core.caches import BaseCache
    #                 from langchain_core.callbacks import Callbacks
    #                 BaseCache._pydantic_model = BaseModel
    #                 if not hasattr(ChatOpenAI, '_pydantic_model'):
    #                     ChatOpenAI._pydantic_model = BaseModel
    #                 ChatOpenAI.model_rebuild()
    #                 self.openai_model = ChatOpenAI(
    #                     model="gpt-4o",
    #                     temperature=0,
    #                     request_timeout=120,
    #                     max_retries=3,
    #                     api_key=self.api_key
    #                 )
    #                 print("LangChain OpenAI client reinitialized successfully")
    #             except Exception as e:
    #                 error_msg = f"Error reinitializing LangChain OpenAI client: {str(e)}"
    #                 print(f"Error: {error_msg}")
    #                 llog("AgentCluster", error_msg, self.log_save_file_name)
    #                 return f"Error: {error_msg}"

    #         if self.openai_model:
    #             try:
    #                 token_count = self.count_tokens(prompt)
    #                 print(f"Sending prompt to OpenAI (length: {token_count} tokens)")
    #                 messages = [
    #                     SystemMessage(content="You are an expert in RFP and proposal evaluation."),
    #                     HumanMessage(content=prompt)
    #                 ]
    #                 response = self.openai_model.invoke(messages)
    #                 elapsed = time.time() - start_time
    #                 print(f"Response generated in {elapsed:.2f} seconds")
    #                 return response.content
    #             except Exception as e:
    #                 error_msg = f"Error using OpenAI API: {str(e)}"
    #                 print(f"Error: {error_msg}")
    #                 llog("AgentCluster", error_msg, self.log_save_file_name)
    #                 return f"Error: {error_msg}"
    #         else:
    #             error_msg = "OpenAI model not available"
    #             print(f"Error: {error_msg}")
    #             llog("AgentCluster", error_msg, self.log_save_file_name)
    #             return f"Error: {error_msg}"

    #     error_msg = f"Invalid model type: {model_type}"
    #     print(f"Error: {error_msg}")
    #     llog("AgentCluster", error_msg, self.log_save_file_name)
    #     return f"Error: {error_msg}"

    def generate(self, prompt: str) -> str:
        """Generate a response from the model using only the selected model type."""
        start_time = time.time()
        # Check if we should use openai based on API key presence
        if not self.model_provider and self.api_key:
            model_type = "openai"
        else:
            model_type = self.model_provider.model_type if self.model_provider else "opensource"
        llog("AgentCluster", f"Generating with model type: {model_type}", self.log_save_file_name)
        print(f"Debug: Using model type: {model_type}")

       # Handle opensource model
        # if model_type == "opensource":
        #     try:
        #         if not self.model_provider:
        #             error_msg = "ModelProvider is None in generate"
        #             print(f"Error: {error_msg}")
        #             llog("AgentCluster", error_msg, self.log_save_file_name)
        #             return f"Error: {error_msg}"
        #         token = self.model_provider.acquire_model()
        #         try:
        #             if not getattr(self.model_provider, 'model', None):
        #                 error_msg = "Open-source model not initialized in ModelProvider"
        #                 llog("AgentCluster", error_msg, self.log_save_file_name)
        #                 raise RuntimeError(error_msg)
        #             response = self.model_provider.generate(prompt)
        #             elapsed = time.time() - start_time
        #             print(f"Response generated in {elapsed:.2f} seconds")
        #             return response
        #         finally:
        #             self.model_provider.release_model()
        #     except Exception as e:
        #         error_msg = f"Error using opensource model: {str(e)}"
        #         print(f"Error: {error_msg}")
        #         llog("AgentCluster", error_msg, self.log_save_file_name)
        #         return f"Error: {error_msg}"

        # Handle OpenAI model
        # if model_type == "openai":
            # if not self.openai_model and self.api_key:
        try:
            llog("rfp_proposal_eval", f"Initializing LangChain OpenAI client", self.log_save_file_name)
            print("Debug: Reinitializing OpenAI client")
            from langchain_core.caches import BaseCache
            from langchain_core.callbacks import Callbacks
            BaseCache._pydantic_model = BaseModel
            
            llog("rfp_proposal_eval", f"checking _pydantic_model", self.log_save_file_name)
            if not hasattr(ChatOpenAI, '_pydantic_model'):
                ChatOpenAI._pydantic_model = BaseModel

            ChatOpenAI.model_rebuild()
            llog("rfp_proposal_eval", f"start Initializing LangChain OpenAI client", self.log_save_file_name)
            self.openai_model = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                request_timeout=120,
                max_retries=3,
                api_key=self.api_key
            )
            print("LangChain OpenAI client reinitialized successfully")
            llog("rfp_proposal_eval", f"process Done LangChain OpenAI client", self.log_save_file_name)
        except Exception as e:
            error_msg = f"Error reinitializing LangChain OpenAI client: {str(e)}"
            print(f"Error: {error_msg}")
            llog("AgentCluster", error_msg, self.log_save_file_name)
            return f"Error: {error_msg}"

        if self.openai_model:
            try:
                token_count = self.count_tokens(prompt)
                print(f"Sending prompt to OpenAI (length: {token_count} tokens)")
                messages = [
                    SystemMessage(content="You are an expert in RFP and proposal evaluation."),
                    HumanMessage(content=prompt)
                ]
                llog("AgentCluster", f"in subprocess : Sending prompt to OpenAI (length: {token_count} tokens)", self.log_save_file_name)
                response = self.openai_model.invoke(messages)
                elapsed = time.time() - start_time
                print(f"Response generated in {elapsed:.2f} seconds")
                return response.content
            except Exception as e:
                error_msg = f"Error using OpenAI API: {str(e)}"
                print(f"Error: {error_msg}")
                llog("AgentCluster", error_msg, self.log_save_file_name)
                return f"Error: {error_msg}"
        else:
            error_msg = "OpenAI model not available"
            print(f"Error: {error_msg}")
            llog("AgentCluster", error_msg, self.log_save_file_name)
            return f"Error: {error_msg}"

        # error_msg = f"Invalid model type: {model_type}"
        # print(f"Error: {error_msg}")
        # llog("AgentCluster", error_msg, self.log_save_file_name)
        # return f"Error: {error_msg}"

        # # openrouter
        # if model_type == "openai":
        #     try:
        #         print("Debug: Initializing OpenRouter client")
        #         from openai import OpenAI
        #         self.openai_model = OpenAI(
        #             base_url="https://openrouter.ai/api/v1",
        #             api_key=os.getenv("OPENROUTER_API_KEY")  # Use your OpenRouter API key
        #         )
        #         print("OpenRouter client initialized successfully")
        #     except Exception as e:
        #         error_msg = f"Error initializing OpenRouter client: {str(e)}"
        #         print(f"Error: {error_msg}")
        #         llog("AgentCluster", error_msg, self.log_save_file_name)
        #         return f"Error: {error_msg}"

        #     if self.openai_model:
        #         try:
        #             token_count = self.count_tokens(prompt)  # Reuse your token counting method
        #             print(f"Sending prompt to OpenRouter (length: {token_count} tokens)")
        #             llog("AgentCluster", f"in subprocess : Sending prompt to OpenRouter (length: {token_count} tokens)", self.log_save_file_name)
        #             messages = [
        #                 {"role": "system", "content": "You are an expert in RFP and proposal evaluation."},
        #                 {"role": "user", "content": prompt}
        #             ]
        #             response = self.openai_model.chat.completions.create(
        #                 # model="deepseek/deepseek-r1-distill-qwen-14b:free", # short response 
        #                 # model="qwen/qwq-32b-preview:free",                  # Json formate
        #                 # model="meta-llama/llama-4-scout:free",              # similar to OPENAI
        #                 # model="google/gemma-3-27b-it:free",                 # mistake in assignment scorebut go response
        #                 # model="deepseek/deepseek-r1-zero:free",
        #                 # model="qwen/qwq-32b:free",
        #                 model="qwen/qwen3-32b",
        #                 messages=messages,
        #                 temperature=0,  # Match ChatOpenAI's temperature
        #                 max_tokens=4000,  # Adjust based on your needs or model limits
        #                 extra_headers={
        #                     # "HTTP-Referer": "YOUR_SITE_URL",  # Replace with your site URL
        #                     "X-Title": "RFP_Proposal_model_testing"       # Replace with your site name
        #                 },
        #                 extra_body={}
        #             )
        #             elapsed = time.time() - start_time
        #             print(f"Response generated in {elapsed:.2f} seconds")
        #             print(f"response isssss :: {response}")
        #             print(f"response isssss :: {response.choices[0].message.content}")
        #             return response.choices[0].message.content
        #         except Exception as e:
        #             error_msg = f"Error using OpenRouter API: {str(e)}"
        #             print(f"Error: {error_msg}")
        #             llog("AgentCluster", error_msg, self.log_save_file_name)
        #             return f"Error: {error_msg}"
        #     else:
        #         error_msg = "OpenRouter client not available"
        #         print(f"Error: {error_msg}")
        #         llog("AgentCluster", error_msg, self.log_save_file_name)
        #         return f"Error: {error_msg}"

        #groq
        # if model_type == "openai":
        #     # if not self.openai_model and self.api_key:
        #     llog("AgentCluster", "Initializing groq client", self.log_save_file_name)
        #     try:
        #         self.openai_model = Groq(api_key=os.getenv("GROQ_API_KEY"))
        #         print("groq client initialized successfully")
        #     except Exception as e:
        #         error_msg = f"Error initializing OpenRouter client: {str(e)}"
        #         print(f"Error: {error_msg}")
        #         llog("AgentCluster", error_msg, self.log_save_file_name)
        #         return f"Error: {error_msg}"
        #     llog("AgentCluster", "groq client initialized successfully", self.log_save_file_name)
        #     llog("AgentCluster", "start prediction", self.log_save_file_name)
        #     if self.openai_model:
        #         max_attempts = 3 
        #         for attempt in range(1, max_attempts + 1):
        #             try:
        #                 token_count = self.count_tokens(prompt)
        #                 # print(f"Debug: Sending prompt this : {prompt} to Groq (length: {token_count} tokens, attempt {attempt})")
                        
        #                 llog("AgentCluster", f"Sending prompt this : {prompt} to Groq (length: {token_count} tokens, attempt {attempt})", self.log_save_file_name)

        #                 messages = [
        #                     {"role": "system", "content": "You are an expert in RFP and proposal evaluation. and MustReturn output in JSON format along with proper structure without anymistake."},
        #                     {"role": "user", "content": prompt}
        #                 ]
        #                 llog("AgentCluster", f"Sending prompt to Groq (length: {token_count} tokens, attempt {attempt})", self.log_save_file_name)

        #                 response = self.openai_model.chat.completions.create(
        #                     #model="qwen-qwq-32b",  # Valid Groq model
        #                     # model = "deepseek-r1-distill-llama-70b",
        #                     model = "llama-3.3-70b-versatile",
        #                     messages=messages,
        #                     temperature=0.01,
        #                     max_tokens=8096,
        #                     top_p=0.95,
        #                     stop=None,
        #                     # response_format={"type": "json_object"}
        #                 )
        #                 print("Debug: Response received from Groq")
        #                 llog("AgentCluster", f"Response received from Groq {response}", self.log_save_file_name)
        #                 if not response or not response.choices or not response.choices[0].message:
        #                     error_msg = "Error: Empty or invalid Groq API response"
        #                     print(error_msg)
        #                     llog("AgentCluster", error_msg, self.log_save_file_name)
        #                     if attempt == max_attempts:
        #                         return {}
        #                     continue
        #                 raw_response = response.choices[0].message.content


        #                 llog("AgentCluster", f"Raw response return from generate with Groq {raw_response}", self.log_save_file_name)

        #                 return raw_response
        #                 # parsed_json = self.extract_json(raw_response)
        #                 # if parsed_json:
        #                 #     elapsed = time.time() - start_time
        #                 #     print(f"Response generated in {elapsed:.2f} seconds")
        #                 #     return parsed_json
        #                 # else:
        #                 #     error_msg = "Error: Failed to extract valid JSON from Groq response"
        #                 #     print(error_msg)
        #                 #     llog("AgentCluster", error_msg, self.log_save_file_name)
        #                 #     if attempt == max_attempts:
        #                 #         return {}
        #                 #     continue
        #             except Exception as e:
        #                 error_msg = f"Error using Groq API: {str(e)}"
        #                 print(f"Error: {error_msg}")
        #                 llog("AgentCluster", error_msg, self.log_save_file_name)
        #                 if attempt == max_attempts:
        #                     error_msg = f"Error: Max attempts ({max_attempts}) reached"
        #                     print(error_msg)
        #                     llog("AgentCluster", error_msg, self.log_save_file_name)
        #                     return {}
        #             continue
        #     return None
        
    
    # def process_proposal(self, rfp_content, proposal_content, language, model_type, log_save_file_name):
    #     """
    #     Process a proposal evaluation using the specified model type
    #     Run the evaluation process on the provided RFP and proposal texts
        
    #     Args:
    #         rfp_content: The content of the RFP document
    #         proposal_content: The content of the proposal document
    #         language: The detected language ('English' or 'Arabic')
    #         model_type: The model type to use ('openai' or 'opensource')
    #         log_save_file_name: The log file name for tracking
            
    #     Returns:
    #         dict: The evaluation result in JSON format
    #     """
    #     self.log_save_file_name = log_save_file_name
        
    #     # Set the model type if using a shared model provider
    #     if self.model_provider and model_type:
    #         llog("AgentCluster", f"Setting model type to {model_type}", log_save_file_name)
    #         self.model_provider.ensure_model_loaded(model_type)
        
    #     # Run the evaluation
    #     llog("AgentCluster", "Starting proposal evaluation", log_save_file_name)
    #     html_result, score = self.run(rfp_content, proposal_content, language, log_save_file_name)
        
    #     # Format the result
    #     formatted_json = {"results": html_result, "score": score}
    #     llog("AgentCluster", "Proposal evaluation completed", log_save_file_name)
        
    #     return formatted_json

    # def process_proposal(self, rfp_content, proposal_content, language, model_type, log_save_file_name):
    #     """
    #     Process a proposal evaluation using the specified model type
    #     Run the evaluation process on the provided RFP and proposal texts
        
    #     Args:
    #         rfp_content: The content of the RFP document
    #         proposal_content: The content of the proposal document
    #         language: The detected language ('English' or 'Arabic')
    #         model_type: The model type to use ('openai' or 'opensource')
    #         log_save_file_name: The log file name for tracking
            
    #     Returns:
    #         dict: The evaluation result in JSON format
    #     """
    #     self.log_save_file_name = log_save_file_name
        
    #     # Log the model type
    #     llog("AgentCluster", f"Using model type: {model_type}", log_save_file_name)
        
    #     # Run the evaluation
    #     llog("AgentCluster", "Starting proposal evaluation", log_save_file_name)
    #     html_result, score = self.run(rfp_content, proposal_content, language, log_save_file_name)
        
    #     # Format the result
    #     formatted_json = {"results": html_result, "score": score}
    #     llog("AgentCluster", "Proposal evaluation completed", log_save_file_name)
        
    #     return formatted_json

    # def process_proposal(self, rfp_content, proposal_content, language, model_type, log_save_file_name):
    #     """
    #     Process a proposal evaluation using the specified model type
    #     Run the evaluation process on the provided RFP and proposal texts
        
    #     Args:
    #         rfp_content: The content of the RFP document
    #         proposal_content: The content of the proposal document
    #         language: The detected language ('English' or 'Arabic')
    #         model_type: The model type to use ('openai' or 'opensource')
    #         log_save_file_name: The log file name for tracking
        
    #     Returns:
    #         dict: The evaluation result in JSON format
    #     """
    #     self.log_save_file_name = log_save_file_name
        
    #     # Validate model type
    #     if model_type not in ["openai", "opensource"]:
    #         error_msg = f"Invalid model type: {model_type}"
    #         llog("AgentCluster", error_msg, log_save_file_name)
    #         return {"error": error_msg}
        
    #     # Set model type in ModelProvider
    #     if self.model_provider and self.model_provider.model_type != model_type:
    #         llog("AgentCluster", f"Setting model type to {model_type}", log_save_file_name)
    #         self.model_provider.model_type = model_type
        
    #     # Log the model type
    #     llog("AgentCluster", f"Using model type: {model_type}", log_save_file_name)
        
    #     # Run the evaluation
    #     llog("AgentCluster", "Starting proposal evaluation", log_save_file_name)
    #     html_result, score = self.run(rfp_content, proposal_content, language, log_save_file_name)
        
    #     # Format the result
    #     formatted_json = {"results": html_result, "score": score}
    #     llog("AgentCluster", "Proposal evaluation completed", log_save_file_name)
        
    #     return formatted_json

    def process_proposal(self, rfp_content, proposal_content, language, model_type, log_save_file_name):
        """
        Process a proposal evaluation using the specified model type
        Run the evaluation process on the provided RFP and proposal texts
        
        Args:
            rfp_content: The content of the RFP document
            proposal_content: The content of the proposal document
            language: The detected language ('English' or 'Arabic')
            model_type: The model type to use ('openai' or 'opensource')
            log_save_file_name: The log file name for tracking
        
        Returns:
            dict: The evaluation result in JSON format
        """
        self.log_save_file_name = log_save_file_name
        
        # Validate model type
        if model_type not in ["openai", "opensource"]:
            error_msg = f"Invalid model type: {model_type}"
            llog("AgentCluster", error_msg, log_save_file_name)
            return {"error": error_msg}
        
        # Check ModelProvider (only required for opensource model)
        if model_type == "opensource" and not self.model_provider:
            error_msg = "ModelProvider is None in process_proposal for opensource model"
            llog("AgentCluster", error_msg, log_save_file_name)
            return {"error": error_msg}
        
        # Set model type in ModelProvider (only if model_provider exists)
        if self.model_provider and getattr(self.model_provider, 'model_type', None) != model_type:
            llog("AgentCluster", f"Setting model type to {model_type}", log_save_file_name)
            self.model_provider.model_type = model_type
        
        # Log the model type
        llog("AgentCluster", f"Using model type: {model_type}", log_save_file_name)
        
        # Run the evaluation
        llog("AgentCluster", "Starting proposal evaluation", log_save_file_name)
        html_result, score = self.run(rfp_content, proposal_content, language, log_save_file_name)
        
        # Format the result
        formatted_json = {"results": html_result, "score": score}
        llog("AgentCluster", "Proposal evaluation completed", log_save_file_name)
        
        return formatted_json
    
    def __rfp_requirements_extraction_node(self, state: AgentState):
        output_dir = "Chunk_output"
        os.makedirs(output_dir, exist_ok=True)
        # Check if RFP text needs to be chunked
        rfp_tokens = self.count_tokens(state["rfp_text"])
        print(f"\n📄 RFP text contains {rfp_tokens} tokens")
        
        if rfp_tokens > self.max_tokens:
            llog("RFP", f"RFP text exceeds token limit of {self.max_tokens} tokens", self.log_save_file_name)
            print(f"\n🔄 RFP exceeds token limit of {self.max_tokens}. Splitting into chunks...")
            rfp_chunks = self.chunk_text(state["rfp_text"], self.max_tokens)
            print(f"\n✂️ Split RFP into {len(rfp_chunks)} chunks")
            
            # Process each chunk and combine results
            all_requirements = []
            previous_requirements = ""
            
            for i, chunk in enumerate(rfp_chunks):
                print(f"\n{'-'*80}\nPROCESSING RFP CHUNK {i+1}/{len(rfp_chunks)}\n{'-'*80}")
                print(f"Chunk {i+1} token count: {self.count_tokens(chunk)} tokens")
                # print(f"Chunk {i+1} first 200 chars: {chunk}...")
                
                # For the first chunk, use the standard prompt
                if i == 0:
                    prompt = RFP_REQUIREMENT_EXTRACTION_PROMPT.format(language=state["detected_language"]) + "\n\n" + chunk
                else:
                    # For subsequent chunks, include previous requirements for context
                    prompt = RFP_REQUIREMENT_WITH_CONTEXT_EXTRACTION_PROMPT.format(
                        language=state["detected_language"],
                        previous_extractions=previous_requirements,
                        chunk_number=i+1,
                        total_chunks=len(rfp_chunks),
                        rfp_chunk=chunk
                    )
                
                response = self.generate(prompt)
                print(f"\nREQUIREMENTS FROM CHUNK {i+1}:\n{'-'*40}")
                print("\033[1;32m" + response + "...\033[0m")  # Print first 200 chars in bright green
                chunk_filename = os.path.join(output_dir, f"chunk_{i+1}.txt")
                with open(chunk_filename, 'w', encoding='utf-8') as f:
                    f.write(response)
                
                # Update previous requirements for next chunk
                if i < len(rfp_chunks):  # Don't need to update for the last chunk
                    previous_requirements = response
                    print(f"Previous requirements: {previous_requirements}")
                    print(f"📝 Saved requirements from chunk {i+1} for context in next chunk")
                
                all_requirements.append(response)
            
            # Combine all requirements
            if len(rfp_chunks) > 1:
                print(f"\n{'-'*80}\nCOMBINING REQUIREMENTS FROM {len(rfp_chunks)} CHUNKS\n{'-'*80}")
                combined_requirements = "\n\n".join(all_requirements)
                
                prompt = COMBINE_RFP_CHUNK_RESULTS_PROMPT.format(language=state["detected_language"]) + "\n\n" + combined_requirements
                
                response = self.generate(prompt)
                print(f"\nCOMBINED REQUIREMENTS:\n{'-'*40}")
                print(response)  # Print first 200 chars in bright green
                print(f"📝 Saved combined requirements")
                chunk_filename = os.path.join(output_dir, f"combined_requirements.txt")
                with open(chunk_filename, 'w', encoding='utf-8') as f:
                    f.write(response)
                state["rfp_requirements"] = response
            else:
                # If there was only one chunk, use its requirements directly
                state["rfp_requirements"] = all_requirements[0]
        else:
            # Original code for when RFP is within token limits
            prompt = RFP_REQUIREMENT_EXTRACTION_PROMPT.format(language=state["detected_language"]) + "\n\n" + state["rfp_text"]
            response = self.generate(prompt)
            llog("RFP", f"RFP requirements extracted. Length: {len(response)} characters", self.log_save_file_name)

            #while use groq [just for groq not for openai]
            # import re
            # regex_pattern = r'<table\b[^>]*>[\s\S]*?</table>'

            # # Find all matches in the input text
            # response = re.findall(regex_pattern, response, re.DOTALL)
            # print(response[0])

            print("\033[1;32m" + response + "\033[0m")  # Print in bright green
            state["rfp_requirements"] = response
        
        print(f"✅ RFP requirements extracted. Length: {len(state['rfp_requirements'])} characters")
        return state
    
    def __proposal_eval_node(self, state: AgentState):
        try:
            llog("Proposal Eval", "Starting proposal evaluation", self.log_save_file_name)
            print("*"*50+ "Detected Language for proposal_evaluation_json_node: "+ state["detected_language"]+ "*"*50)
            # Handle case-insensitive language detection
            detected_lang = state["detected_language"].lower()
            if detected_lang == "english":
                format = format_2
            elif detected_lang == "arabic":
                format = format_2_arabic
            
            # Log the inputs
            llog("Proposal Eval", f"RFP Requirements: {state['rfp_requirements'][:100]}...", self.log_save_file_name)
            llog("Proposal Eval", f"Proposal Text Length: {len(state['proposal_text'])}", self.log_save_file_name)
            
            prompt = PROPOSAL_EVAL_PROMPT.format(
                proposal_text=state["proposal_text"], 
                format=format,
                language=state["detected_language"],
                rfp_requirements= state["rfp_requirements"]
            ) + "\n\n" + state["rfp_requirements"]
            
            # Log that we're making the model call
            llog("Proposal Eval", "Making model call...", self.log_save_file_name)
            response = self.generate(prompt)
            
            # regex_pattern = r'\{[\s\S]*\}'
            # matches = re.finditer(regex_pattern, response)
            # response = matches[0]
            # llog("Proposal Eval", f"Gregax applied for SHORT FORMATE proposal : {response}", self.log_save_file_name)

            llog("Proposal Eval", "Groq generate and pass for Json formate", self.log_save_file_name)
            json_format_prompt = FORMAT_OUTPUT_TO_JSON_PROMPT.format(
                evaluation_content=response,
                json_schema=json.dumps(format, ensure_ascii=False, indent=2),
                language=state["detected_language"]
            )
            
            llog("Proposal Eval", "Converting response to JSON format...", self.log_save_file_name)
            json_formatted_response = self.generate(json_format_prompt)
            llog("Proposal Eval", "json_formatted_response - Model response received", self.log_save_file_name)
            # Clean the response content
            content = json_formatted_response
            # Remove markdown code block syntax
            content = re.sub(r'^```json\n', '', content)  # Remove opening ```json
            content = re.sub(r'\n```$', '', content)      # Remove closing ```
            content = content.strip()

            dir_path = "groq_testing"
            os.makedirs(dir_path, exist_ok=True)  
            # Save the cleaned response to a file
            
            file_path = os.path.join(dir_path, "Proposal_eval_output.txt")

            # Fix with statement syntax (remove incorrect commas)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            llog("Proposal Eval", f"json_formatted_response - Cleaned response with strip: {content[:200]}...", self.log_save_file_name)
            
            # Validate JSON
            try:
                llog("Proposal Eval", "Validating JSON...", self.log_save_file_name)
                json.loads(content)  # Test if it's valid JSON
                state["proposal_eval_report_json"] = content
                return state
            except json.JSONDecodeError as e:
                llog("Proposal Eval", f"Invalid JSON received: {str(e)}", self.log_save_file_name)
                llog("Proposal Eval", f"First 200 chars of response: {json_formatted_response[:200]}", self.log_save_file_name)
                
                # Try to extract JSON from the response using regex
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', json_formatted_response)
                if json_match:
                    extracted_json = json_match.group(1).strip()
                    try:
                        json.loads(extracted_json)  # Test if extracted content is valid JSON
                        state["proposal_eval_report_json"] = extracted_json
                        llog("Proposal Eval", "Successfully extracted JSON using regex", self.log_save_file_name)
                        return state
                    except json.JSONDecodeError:
                        llog("Proposal Eval", "Extracted JSON is still invalid", self.log_save_file_name)
                
                # If we get here, we need to retry with a more explicit prompt
                llog("Proposal Eval", "Retrying with explicit JSON completion prompt", self.log_save_file_name)
                retry_prompt = "You previously started generating a JSON response but it was incomplete or had formatting issues. " \
                               "Please generate a complete and valid JSON response following the required format. " \
                               "Do not include any introductory text or concluding messages. " \
                               "The response should be a single valid JSON object starting with '{' and ending with '}'. " \
                               "Do not include any markdown code block syntax or other formatting.\n\n" \
                               f"Complete this JSON: {json_formatted_response}"
                
                retry_response = self.generate(retry_prompt)
                retry_content = retry_response.strip()
                
                # Clean up the retry response
                retry_content = re.sub(r'^```json\n', '', retry_content)
                retry_content = re.sub(r'\n```$', '', retry_content)
                
                try:
                    json.loads(retry_content)  # Test if retry content is valid JSON
                    state["proposal_eval_report_json"] = retry_content
                    llog("Proposal Eval", "Successfully retrieved valid JSON on retry", self.log_save_file_name)
                    return state
                except json.JSONDecodeError as retry_error:
                    llog("Proposal Eval", f"Retry also failed: {str(retry_error)}", self.log_save_file_name)
                    raise ValueError(f"Model returned invalid JSON and retry failed: {str(e)}")
        
        except Exception as e:
            llog("Proposal Eval", f"Error in proposal evaluation: {str(e)}", self.log_save_file_name)
            raise
    # old and based worked 
    # def __json_to_html(self, state: AgentState):
    #     proposal_eval_data = json.loads(state["proposal_eval_report_json"])
    #     print(f"After Json laod starting Html ::: ")
    #     html = "<table class='table table-bordered'>"
        
        
    #     # Scored Requirements
    #     print(f"Score requirements : HTML")
    #     html += "<tr><th colspan='2'>Scored Requirements</th></tr>"
    #     html += "<tr><th>Evaluation Criteria</th><th>Details</th></tr>"
    #     for criterion in proposal_eval_data['scored_requirements']['evaluation_criteria']:
    #         html += f"""
    #         <tr>
    #             <td>{criterion['criterion_name']}</td>
    #             <td>
    #                 <strong>Requirements:</strong> {criterion['requirements']}<br>
    #                 <strong>Proposal Compliance:</strong> {criterion['proposal_compliance']}<br>
    #                 <strong>Required Score:</strong> {criterion['required_score']}<br>
    #                 <strong>Assigned Score:</strong> {criterion['assigned_score']}<br>
    #                 <strong>Justification:</strong> {criterion['justification']}<br>
    #             </td>
    #         </tr>
    #         """
        
    #     # Overall Assessment
    #     print(f"Overall Assessment : HTML")
    #     assessment = proposal_eval_data['scored_requirements']['overall_assessment']
    #     html += "<tr><th colspan='2'>Overall Assessment</th></tr>"
    #     html += f"<tr><td>Total Technical Score</td><td>{assessment['total_technical_score']}</td></tr>"
        
    #     html += "<tr><td>Technical Strengths</td><td><ul>"
    #     for strength in assessment['technical_strengths']:
    #         html += f"<li>{strength}</li>"
    #     html += "</ul></td></tr>"
        
    #     html += "<tr><td>Technical Weaknesses</td><td><ul>"
    #     for weakness in assessment['technical_weaknesses']:
    #         html += f"<li>{weakness}</li>"
    #     html += "</ul></td></tr>"
        
    #     html += f"<tr><td>Summary</td><td>{assessment['summary']}</td></tr>"
    #     html += "</table>"
        
    #     # Unscored Requirements
    #     print(f"Unscored Requirements")
        
    #     html += "<table class='table table-bordered'><tr><th colspan='2'>Unscored Requirements</th></tr>"
    #     html += "<tr><th>Category</th><th>Details</th></tr>"
    #     for category in proposal_eval_data['unscored_requirements']['requirements']:
    #         html += f"<tr><td>{category['category']}</td><td>"
    #         for req in category['requirements']:
    #             html += f"""
    #             <strong>{req['name']}:</strong> {req['description']} <p>(Compliance Status: {req['evaluation']['compliance_status']})</p><br>
    #             """
    #         html += f"<strong>Category Assessment:</strong> {category['category_assessment']}</td></tr>"
    #     html += "</table>"
        
    #     print("analysis : html ")
    #     # Analysis
    #     html += "<table class='table table-bordered'><tr><th colspan='2'>Analysis</th></tr>"
    #     analysis = proposal_eval_data['analysis']
    #     html += "<tr><td>Strengths</td><td><ul>"
    #     for strength in analysis['strengths']:
    #         html += f"<li>{strength}</li>"
    #     html += "</ul></td></tr>"
        
    #     html += "<tr><td>Concerns</td><td><ul>"
    #     for concern in analysis['concerns']:
    #         html += f"<li>{concern}</li>"
    #     html += "</ul></td></tr>"
        
    #     html += "<tr><td>Risks</td><td><ul>"
    #     for risk in analysis['risks']:
    #         html += f"<li>{risk}</li>"
    #     html += "</ul></td></tr>"
    #     html += "</table>"
        
    #     # Conclusion
    #     print("conclusion : html")
    #     conclusion = proposal_eval_data['conclusion']
    #     html += "<table class='table table-bordered'><tr><th colspan='2'>Conclusion</th></tr>"
    #     html += f"<tr><td>Overall Assessment</td><td>{conclusion['overall_assessment']}</td></tr>"
    #     html += f"<tr><td>Recommendation</td><td>{conclusion['recommendation']}</td></tr>"
        
    #     html += "<tr><td>Next Steps</td><td><ul>"
    #     for step in conclusion['next_steps']:
    #         html += f"<li>{step}</li>"
    #     html += "</ul></td></tr>"
    #     html += "</table>"

    #     print("finally store the html in to report : html")
    #     state["proposal_eval_report_html"] = html
    #     print(f"HTML from Json to HTML converter :{html}")
    #     print(state["proposal_eval_report_html"])
    #     return state
    

    def __json_to_html(self, state: AgentState):
        """Convert JSON proposal evaluation report to HTML table format"""
        try:
            # Log raw input
            print(f"Raw state['proposal_eval_report_json']: {state['proposal_eval_report_json'][:200]}...")
            
            # Clean input: remove Markdown code blocks if present
            json_input = state["proposal_eval_report_json"]
            json_input = re.sub(r'^```json\s*|\s*```$', '', json_input, flags=re.MULTILINE)
            print(f"Cleaned JSON input: {json_input[:200]}...")

            # Parse JSON
            
            try:
                proposal_eval_data = json.loads(json_input)
                if isinstance(proposal_eval_data, str):
                    print("Detected double-encoded JSON, parsing again")
                    proposal_eval_data = json.loads(proposal_eval_data)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                raise ValueError(f"Invalid JSON: {json_input}")

            if not isinstance(proposal_eval_data, dict):
                raise TypeError(f"Expected dictionary, got {type(proposal_eval_data)}: {proposal_eval_data}")
            print(f"After first json.loads, proposal_eval_data type: {type(proposal_eval_data)}")
            print(f"proposal_eval_data: {str(proposal_eval_data)[:200]}...")

            # Handle double-encoded JSON
            if isinstance(proposal_eval_data, str):
                print(f"Detected string after first json.loads, attempting second parse")
                proposal_eval_data = json.loads(proposal_eval_data)
                print(f"After second json.loads, proposal_eval_data type: {type(proposal_eval_data)}")
                print(f"proposal_eval_data: {str(proposal_eval_data)[:200]}...")

            # Validate top-level structure
            if not isinstance(proposal_eval_data, dict):
                error_msg = f"Expected dictionary for proposal_eval_data, got {type(proposal_eval_data)}"
                print(error_msg)
                raise ValueError(error_msg)

            # Check for expected keys
            expected_keys = ['scored_requirements', 'unscored_requirements', 'analysis', 'conclusion']
            missing_keys = [key for key in expected_keys if key not in proposal_eval_data]
            if missing_keys:
                print(f"Warning: Missing keys in proposal_eval_data: {missing_keys}")

            html = "<table class='table table-bordered'>"

            # Scored Requirements
            print(f"Score requirements : HTML")
            scored_requirements = proposal_eval_data.get('scored_requirements', {})
            print(f"scored_requirements type: {type(scored_requirements)}")
            print(f"scored_requirements: {str(scored_requirements)[:200]}...")

            if not isinstance(scored_requirements, dict):
                error_msg = f"Expected dictionary for scored_requirements, got {type(scored_requirements)}"
                print(error_msg)
                raise ValueError(error_msg)

            evaluation_criteria = scored_requirements.get('evaluation_criteria', [])
            print(f"evaluation_criteria type: {type(evaluation_criteria)}")
            print(f"evaluation_criteria: {str(evaluation_criteria)[:200]}...")

            if not isinstance(evaluation_criteria, list):
                error_msg = f"Expected list for evaluation_criteria, got {type(evaluation_criteria)}"
                print(error_msg)
                raise ValueError(error_msg)

            html += "<tr><th colspan='2'>Scored Requirements</th></tr>"
            html += "<tr><th>Evaluation Criteria</th><th>Details</th></tr>"
            for criterion in evaluation_criteria:
                if not isinstance(criterion, dict):
                    error_msg = f"Expected dictionary for criterion, got {type(criterion)}"
                    print(error_msg)
                    raise ValueError(error_msg)

                html += f"""
                <tr>
                    <td>{criterion.get('criterion_name', 'N/A')}</td>
                    <td>
                        <strong>Requirements:</strong> {criterion.get('requirements', 'N/A')}<br>
                        <strong>Proposal Compliance:</strong> {criterion.get('proposal_compliance', 'N/A')}<br>
                        <strong>Required Score:</strong> {criterion.get('required_score', 'N/A')}<br>
                        <strong>Assigned Score:</strong> {criterion.get('assigned_score', 'N/A')}<br>
                        <strong>Justification:</strong> {criterion.get('justification', 'N/A')}<br>
                    </td>
                </tr>
                """

            # Overall Assessment
            print(f"Overall Assessment : HTML")
            assessment = scored_requirements.get('overall_assessment', {})
            print(f"overall_assessment type: {type(assessment)}")
            print(f"overall_assessment: {str(assessment)[:200]}...")

            if not isinstance(assessment, dict):
                error_msg = f"Expected dictionary for overall_assessment, got {type(assessment)}"
                print(error_msg)
                raise ValueError(error_msg)

            html += "<tr><th colspan='2'>Overall Assessment</th></tr>"
            html += f"<tr><td>Total Technical Score</td><td>{assessment.get('total_technical_score', 'N/A')}</td></tr>"

            html += "<tr><td>Technical Strengths</td><td><ul>"
            technical_strengths = assessment.get('technical_strengths', [])
            if not isinstance(technical_strengths, list):
                error_msg = f"Expected list for technical_strengths, got {type(technical_strengths)}"
                print(error_msg)
                raise ValueError(error_msg)
            for strength in technical_strengths:
                html += f"<li>{strength}</li>"
            html += "</ul></td></tr>"

            html += "<tr><td>Technical Weaknesses</td><td><ul>"
            technical_weaknesses = assessment.get('technical_weaknesses', [])
            if not isinstance(technical_weaknesses, list):
                error_msg = f"Expected list for technical_weaknesses, got {type(technical_weaknesses)}"
                print(error_msg)
                raise ValueError(error_msg)
            for weakness in technical_weaknesses:
                html += f"<li>{weakness}</li>"
            html += "</ul></td></tr>"

            html += f"<tr><td>Summary</td><td>{assessment.get('summary', 'N/A')}</td></tr>"
            html += "</table>"

            # Unscored Requirements
            print(f"Unscored Requirements")
            unscored_requirements = proposal_eval_data.get('unscored_requirements', {})
            print(f"unscored_requirements type: {type(unscored_requirements)}")
            print(f"unscored_requirements: {str(unscored_requirements)[:200]}...")

            if not isinstance(unscored_requirements, dict):
                error_msg = f"Expected dictionary for unscored_requirements, got {type(unscored_requirements)}"
                print(error_msg)
                raise ValueError(error_msg)

            requirements = unscored_requirements.get('requirements', [])
            if not isinstance(requirements, list):
                error_msg = f"Expected list for unscored requirements, got {type(requirements)}"
                print(error_msg)
                raise ValueError(error_msg)

            html += "<table class='table table-bordered'><tr><th colspan='2'>Unscored Requirements</th></tr>"
            html += "<tr><th>Category</th><th>Details</th></tr>"
            for category in requirements:
                if not isinstance(category, dict):
                    error_msg = f"Expected dictionary for category, got {type(category)}"
                    print(error_msg)
                    raise ValueError(error_msg)

                html += f"<tr><td>{category.get('category', 'N/A')}</td><td>"
                category_requirements = category.get('requirements', [])
                if not isinstance(category_requirements, list):
                    error_msg = f"Expected list for category requirements, got {type(category_requirements)}"
                    print(error_msg)
                    raise ValueError(error_msg)

                for req in category_requirements:
                    if not isinstance(req, dict):
                        error_msg = f"Expected dictionary for requirement, got {type(req)}"
                        print(error_msg)
                        raise ValueError(error_msg)
                    evaluation = req.get('evaluation', {})
                    html += f"""
                    <strong>{req.get('name', 'N/A')}:</strong> {req.get('description', 'N/A')} <p>(Compliance Status: {evaluation.get('compliance_status', 'N/A')})</p><br>
                    """
                html += f"<strong>Category Assessment:</strong> {category.get('category_assessment', 'N/A')}</td></tr>"
            html += "</table>"

            # Analysis
            print(f"Analysis : HTML")
            analysis = proposal_eval_data.get('analysis', {})
            print(f"analysis type: {type(analysis)}")
            print(f"analysis: {str(analysis)[:200]}...")

            if not isinstance(analysis, dict):
                error_msg = f"Expected dictionary for analysis, got {type(analysis)}"
                print(error_msg)
                raise ValueError(error_msg)

            html += "<table class='table table-bordered'><tr><th colspan='2'>Analysis</th></tr>"
            html += "<tr><td>Strengths</td><td><ul>"
            strengths = analysis.get('strengths', [])
            if not isinstance(strengths, list):
                error_msg = f"Expected list for strengths, got {type(strengths)}"
                print(error_msg)
                raise ValueError(error_msg)
            for strength in strengths:
                html += f"<li>{strength}</li>"
            html += "</ul></td></tr>"

            html += "<tr><td>Concerns</td><td><ul>"
            concerns = analysis.get('concerns', [])
            if not isinstance(concerns, list):
                error_msg = f"Expected list for concerns, got {type(concerns)}"
                print(error_msg)
                raise ValueError(error_msg)
            for concern in concerns:
                html += f"<li>{concern}</li>"
            html += "</ul></td></tr>"

            html += "<tr><td>Risks</td><td><ul>"
            risks = analysis.get('risks', [])
            if not isinstance(risks, list):
                error_msg = f"Expected list for risks, got {type(risks)}"
                print(error_msg)
                raise ValueError(error_msg)
            for risk in risks:
                html += f"<li>{risk}</li>"
            html += "</ul></td></tr>"
            html += "</table>"

            # Conclusion
            print(f"Conclusion : HTML")
            conclusion = proposal_eval_data.get('conclusion', {})
            print(f"conclusion type: {type(conclusion)}")
            print(f"conclusion: {str(conclusion)[:200]}...")

            if not isinstance(conclusion, dict):
                error_msg = f"Expected dictionary for conclusion, got {type(conclusion)}"
                print(error_msg)
                raise ValueError(error_msg)

            html += "<table class='table table-bordered'><tr><th colspan='2'>Conclusion</th></tr>"
            html += f"<tr><td>Overall Assessment</td><td>{conclusion.get('overall_assessment', 'N/A')}</td></tr>"
            html += f"<tr><td>Recommendation</td><td>{conclusion.get('recommendation', 'N/A')}</td></tr>"

            html += "<tr><td>Next Steps</td><td><ul>"
            next_steps = conclusion.get('next_steps', [])
            if not isinstance(next_steps, list):
                error_msg = f"Expected list for next_steps, got {type(next_steps)}"
                print(error_msg)
                raise ValueError(error_msg)
            for step in next_steps:
                html += f"<li>{step}</li>"
            html += "</ul></td></tr>"
            html += "</table>"

            # Store HTML in state
            print(f"Finally store the HTML in report")
            state["proposal_eval_report_html"] = html
            print(f"HTML from JSON to HTML converter: {html[:200]}...")
            print(f"Stored HTML in state: {state['proposal_eval_report_html'][:200]}...")
            return state

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error in JSON to HTML conversion: {str(e)}"
            print(error_msg)
            raise
    
    def __json_to_html_arabic(self, state: AgentState):
        llog("Proposal Eval", "Converting JSON to HTML for Arabic", self.log_save_file_name)
        proposal_eval_json = state["proposal_eval_report_json"].replace("```json", "").replace("```", "").strip()
        
        proposal_eval_data = json.loads(proposal_eval_json)
        llog("Proposal Eval", f"After first json.loads, type of proposal_eval_data: {type(proposal_eval_data)}", self.log_save_file_name)
        llog("Proposal Eval", f"Content of proposal_eval_data: {str(proposal_eval_data)[:200]}...", self.log_save_file_name)
        
        if isinstance(proposal_eval_data, str):
            llog("Proposal Eval", "Detected double-encoded JSON, parsing again", self.log_save_file_name)
            proposal_eval_data = json.loads(proposal_eval_data)
            llog("Proposal Eval", f"After second json.loads, type of proposal_eval_data: {type(proposal_eval_data)}", self.log_save_file_name)
            llog("Proposal Eval", f"Content of proposal_eval_data: {str(proposal_eval_data)[:200]}...", self.log_save_file_name)
        
        if not isinstance(proposal_eval_data, dict):
            error_msg = f"Expected dictionary for proposal_eval_data, got {type(proposal_eval_data)}"
            llog("Proposal Eval", error_msg, self.log_save_file_name)
            raise TypeError(error_msg)
        
        
        llog("Proposal Eval", "Converting JSON loaded", self.log_save_file_name)
        # Main container with RTL direction
        html = "<div dir='rtl'>"
        html += "<table class='table table-bordered' style='text-align: right;'>"
        llog("Proposal Eval", "Table tag finish", self.log_save_file_name)
        # المتطلبات المُسجَّلة بالدرجات
        html += "<tr><th colspan='2' style='text-align: right;'>المتطلبات المُسجَّلة بالدرجات</th></tr>"
        html += "<tr><th style='text-align: right;'>معايير التقييم</th><th style='text-align: right;'>التفاصيل</th></tr>"
        llog("Proposal Eval", "Style also finished", self.log_save_file_name)
        for criterion in proposal_eval_data['المتطلبات_المُسجَّلة_بالدرجات']['معايير_التقييم']:
            llog("Proposal Eval", f"Processing criterion:", self.log_save_file_name)
            html += f"""
            {llog("Proposal Eval", f"Processing criterion:", self.log_save_file_name)}
            <tr>
                <td style='text-align: right;'>{criterion['اسم_المعيار']}</td>
                <td style='text-align: right;'>
                    <strong>المتطلبات:</strong> {criterion['المتطلبات']}<br>
                    <strong>امتثال الاقتراح:</strong> {criterion['امتثال_الاقتراح']}<br>
                    <strong>الدرجة المطلوبة:</strong> {criterion['الدرجة_المطلوبة']}<br>
                    <strong>الدرجة المخصصة:</strong> {criterion['الدرجة_المخصصة']}<br>
                    <strong>التبرير:</strong> {criterion['التبرير']}<br>
                </td>
            </tr>
            """
        
        # التقييم العام
        assessment = proposal_eval_data['المتطلبات_المُسجَّلة_بالدرجات']['التقييم_العام']
        html += "<tr><th colspan='2' style='text-align: right;'>التقييم العام</th></tr>"
        html += f"<tr><td style='text-align: right;'>إجمالي الدرجة الفنية</td><td style='text-align: right;'>{assessment['إجمالي_الدرجة_الفنية']}</td></tr>"
        
        html += "<tr><td style='text-align: right;'>نقاط القوة الفنية</td><td style='text-align: right;'><ul style='padding-right: 20px;'>"
        for strength in assessment['نقاط_القوة_الفنية']:
            html += f"<li style='text-align: right;'>{strength}</li>"
        html += "</ul></td></tr>"
        
        html += "<tr><td style='text-align: right;'>نقاط الضعف الفنية</td><td style='text-align: right;'><ul style='padding-right: 20px;'>"
        for weakness in assessment['نقاط_الضعف_الفنية']:
            html += f"<li style='text-align: right;'>{weakness}</li>"
        html += "</ul></td></tr>"
        
        html += f"<tr><td style='text-align: right;'>الملخص</td><td style='text-align: right;'>{assessment['الملخص']}</td></tr>"
        html += "</table>"
        
        # المتطلبات غير المُسجَّلة بالدرجات
        html += "<table class='table table-bordered' style='text-align: right;'><tr><th colspan='2' style='text-align: right;'>المتطلبات غير المُسجَّلة بالدرجات</th></tr>"
        html += "<tr><th style='text-align: right;'>الفئة</th><th style='text-align: right;'>التفاصيل</th></tr>"
        for category in proposal_eval_data['المتطلبات_غير_المُسجَّلة_بالدرجات']['المتطلبات']:
            html += f"<tr><td style='text-align: right;'>{category['الفئة']}</td><td style='text-align: right;'>"
            for req in category['المتطلبات']:
                html += f"""
                <strong>{req['الاسم']}:</strong> {req['الوصف']} <p>(حالة الامتثال: {req['تقييم']['حالة_الامتثال']})</p><br>
                """
            html += f"<strong>تقييم الفئة:</strong> {category['تقييم_الفئة']}</td></tr>"
        html += "</table>"
        
        # التحليل
        html += "<table class='table table-bordered' style='text-align: right;'><tr><th colspan='2' style='text-align: right;'>التحليل</th></tr>"
        analysis = proposal_eval_data['التحليل']
        html += "<tr><td style='text-align: right;'>نقاط القوة</td><td style='text-align: right;'><ul style='padding-right: 20px;'>"
        for strength in analysis['نقاط_القوة']:
            html += f"<li style='text-align: right;'>{strength}</li>"
        html += "</ul></td></tr>"
        
        html += "<tr><td style='text-align: right;'>المخاوف</td><td style='text-align: right;'><ul style='padding-right: 20px;'>"
        for concern in analysis['المخاوف']:
            html += f"<li style='text-align: right;'>{concern}</li>"
        html += "</ul></td></tr>"
        
        html += "<tr><td style='text-align: right;'>المخاطر</td><td style='text-align: right;'><ul style='padding-right: 20px;'>"
        for risk in analysis['المخاطر']:
            html += f"<li style='text-align: right;'>{risk}</li>"
        html += "</ul></td></tr>"
        html += "</table>"
        
        # الاستنتاج
        conclusion = proposal_eval_data['الاستنتاج']
        html += "<table class='table table-bordered' style='text-align: right;'><tr><th colspan='2' style='text-align: right;'>الاستنتاج</th></tr>"
        html += f"<tr><td style='text-align: right;'>التقييم العام</td><td style='text-align: right;'>{conclusion['التقييم_العام']}</td></tr>"
        html += f"<tr><td style='text-align: right;'>التوصية</td><td style='text-align: right;'>{conclusion['التوصية']}</td></tr>"
        
        html += "<tr><td style='text-align: right;'>الخطوات التالية</td><td style='text-align: right;'><ul style='padding-right: 20px;'>"
        for step in conclusion['الخطوات_التالية']:
            html += f"<li style='text-align: right;'>{step}</li>"
        html += "</ul></td></tr>"
        html += "</table>"
        html += "</div>"  # Close the RTL container

        state["proposal_eval_report_html"] = html
        return state
    
    def __json_to_html_node(self, state: AgentState):
        print("*"*50+ "Detected Language for json to html node: "+ state["detected_language"]+ "*"*50)
        # Handle case-insensitive language detection
        detected_lang = state["detected_language"].lower()
        if detected_lang == "english":
            return self.__json_to_html(state)
        elif detected_lang == "arabic":
            return self.__json_to_html_arabic(state)
        else:
            # Default to English if language not recognized
            llog("Proposal Eval", f"Unknown language '{state['detected_language']}', defaulting to English HTML format", self.log_save_file_name)
            return self.__json_to_html(state)
    
    def __setup_graph(self):
        builder = StateGraph(AgentState)
        llog("Proposal Eval", "Setting up graph...", self.log_save_file_name)   

        llog("Proposal Eval", "Adding rfp_requirements_extraction node to graph", self.log_save_file_name)
        builder.add_node("rfp_requirements_extraction", self.__rfp_requirements_extraction_node)

        llog("Proposal Eval", "Adding proposal_eval node to graph", self.log_save_file_name)
        builder.add_node("proposal_eval", self.__proposal_eval_node)

        llog("Proposal Eval", "Adding json_to_html_converter node to graph", self.log_save_file_name)
        builder.add_node("json_to_html_converter", self.__json_to_html_node)

        llog("Proposal Eval", "Setting rfp_requirements_extraction as entry point", self.log_save_file_name)
        builder.set_entry_point("rfp_requirements_extraction")

        llog("Proposal Eval", "Adding edge from rfp_requirements_extraction to proposal_eval", self.log_save_file_name)
        builder.add_edge("rfp_requirements_extraction", "proposal_eval")

        llog("Proposal Eval", "Adding edge from proposal_eval to json_to_html_converter", self.log_save_file_name)
        builder.add_edge("proposal_eval", "json_to_html_converter")

        return builder.compile()

    def split_text_into_chunks(self, text, max_tokens):
        """Split text into chunks that don't exceed max_tokens."""
        if not self.tokenizer:
            print("Warning: Tokenizer not initialized for chunking")
            # Fallback to simple character-based chunking
            chunk_size = max_tokens * 4  # Rough estimate: 1 token ≈ 4 chars
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            print(f"Splitting text into chunk {len(chunks)}: {len(chunk_tokens)} tokens")
            
        return chunks
    
    def chunk_text(self, text: str, tokens_per_chunk: int = 100000) -> List[str]:
        """Split text into chunks of approximately tokens_per_chunk tokens.
        
        This uses several approaches to ensure sensible chunk boundaries:
        1. Tries to split on paragraph boundaries (double newlines)
        2. If paragraphs are too big, splits on single newlines
        3. If still too big, splits on sentence boundaries
        4. As a last resort, splits on token count
        """
        if not text:
            return []
        
        # First try: get a rough estimate of total tokens to see if chunking is needed
        approx_total_tokens = self.count_tokens(text)
        if approx_total_tokens <= tokens_per_chunk:
            return [text]  # No chunking needed
        
        # Initialize list to store chunks and counters
        chunks = []
        current_chunk = ""
        current_chunk_tokens = 0
        
        # First attempt: Split on paragraph boundaries (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            paragraph_tokens = self.count_tokens(paragraph + "\n\n")
            
            # If this single paragraph exceeds chunk size, we'll need to split it further
            if paragraph_tokens > tokens_per_chunk:
                # Split the paragraph by newlines first
                lines = paragraph.split('\n')
                
                for line in lines:
                    if not line.strip():
                        continue
                        
                    line_tokens = self.count_tokens(line + "\n")
                    
                    # If adding this line would exceed the chunk size, start a new chunk
                    if current_chunk_tokens + line_tokens > tokens_per_chunk:
                        if current_chunk:  # Don't add empty chunks
                            chunks.append(current_chunk)
                        current_chunk = line + "\n"
                        current_chunk_tokens = line_tokens
                    else:
                        current_chunk += line + "\n"
                        current_chunk_tokens += line_tokens
            else:
                # Check if adding this paragraph would exceed the chunk size
                if current_chunk_tokens + paragraph_tokens > tokens_per_chunk:
                    # Start a new chunk
                    if current_chunk:  # Don't add empty chunks
                        chunks.append(current_chunk)
                    current_chunk = paragraph + "\n\n"
                    current_chunk_tokens = paragraph_tokens
                else:
                    # Add to current chunk
                    current_chunk += paragraph + "\n\n"
                    current_chunk_tokens += paragraph_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        # Verify chunks aren't too large
        final_chunks = []
        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk)
            if chunk_tokens > tokens_per_chunk:
                # This is rare, but could happen with very long paragraphs
                # Further split using a simpler approach - just divide by token count
                print(f"Warning: Chunk still too large ({chunk_tokens} tokens), applying token-based splitting")
                words = chunk.split()
                sub_chunk = ""
                sub_chunk_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word + " ")
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
    
    def process_large_proposal(self, rfp_requirements, proposal_chunks, detected_language):
        """Process a large proposal by handling each chunk separately, with context from previous chunks."""
        llog("Agent", f"Processing large proposal in {len(proposal_chunks)} chunks", self.log_save_file_name)
        print(f"\n{'='*80}\nPROCESSING LARGE PROPOSAL IN {len(proposal_chunks)} CHUNKS\n{'='*80}")
        
        all_results = []
        previous_evaluations_json = ""
        
        for i, chunk in enumerate(proposal_chunks):
            llog("Agent", f"Processing chunk {i+1}/{len(proposal_chunks)}", self.log_save_file_name)
            print(f"\n{'-'*80}\nPROCESSING CHUNK {i+1}/{len(proposal_chunks)}\n{'-'*80}")
            print(f"Chunk {i+1} token count: {self.count_tokens(chunk)} tokens")
            print(f"Lenght of the chunk {len(proposal_chunks)}")
            # print(f"Chunk {i+1} first 200 chars: {chunk[:200]}...")
            # print(f"Chunk {i+1} first 200 chars: {chunk}...")
            
            # Handle case-insensitive language detection
            detected_lang_lower = detected_language.lower()
            if detected_lang_lower == "english":
                format = format_2
            elif detected_lang_lower == "arabic":
                format = format_2_arabic
            else:
                # Default to English format if language not recognized
                format = format_2
                llog("Proposal Eval", f"Unknown language '{detected_language}', defaulting to English format", self.log_save_file_name)

            # storing chunk Proposal chunk into chunks folder with unique name
            os.makedirs("Chunk_output", exist_ok=True)
            chunk_filename = os.path.join("Chunk_output", f"chunk_{i+1}.txt")
            with open(chunk_filename, 'w', encoding='utf-8') as f:
                f.write(chunk)
            print(f"Chunk {i+1} saved to {chunk_filename}")
            # Log the inputs
            llog("Proposal Eval", f"RFP Requirements: {rfp_requirements[:100]}...", self.log_save_file_name)
            llog("Proposal Eval", f"Proposal Text Length: {len(proposal_chunks)}", self.log_save_file_name)
            llog("Proposal Eval", f"Chunk {i+1} Text Length: {len(chunk)}", self.log_save_file_name)
            llog("Proposal Eval", f"Chunk {i+1} Filename: {chunk_filename}", self.log_save_file_name)
            llog("Proposal Eval", f"Chunk {i+1} detected language: {detected_language}", self.log_save_file_name)
            # Log that we're making the model call
            llog("Proposal Eval", "Making model call...", self.log_save_file_name)
            print(f"Making model call for chunk {i+1}...")

            # For the first chunk, use the standard prompt
            if i == 0:
                prompt = PROPOSAL_EVAL_PROMPT.format(
                    language=detected_language,
                    rfp_requirements=rfp_requirements,
                    proposal_text=chunk,
                    format=format,
                ) + "\n\n" + rfp_requirements
            else:
                # For subsequent chunks, include full previous evaluation results
                prompt = CHUNK_WITH_FULL_CONTEXT_EVAL_PROMPT.format(
                    proposal_chunk=chunk,
                    chunk_number=i+1,
                    total_chunks=len(proposal_chunks),
                    previous_evaluations_json=previous_evaluations_json,
                    # format=format,
                    language=detected_language
                ) + "\n\n" + rfp_requirements
            
            try:
                print(f"Sending chunk {i+1} to model for evaluation...")
                llog("Agent", f"Sending chunk {i+1} to model for evaluation", self.log_save_file_name)

                response = self.generate(prompt)
                llog("Proposal Eval", f"Received response for chunk {i+1}", self.log_save_file_name)
                llog("Proposal Eval", f"Received response for chunk {i+1} is {response}", self.log_save_file_name)

                # Convert the unstructured response to JSON format
                json_format_prompt = FORMAT_OUTPUT_TO_JSON_PROMPT.format(
                    evaluation_content=response,
                    json_schema=json.dumps(format, ensure_ascii=False, indent=2),
                    language=detected_language
                )
                
                llog("Agent", f"Converting chunk {i+1} response to JSON format", self.log_save_file_name)
                json_formatted_response = self.generate(json_format_prompt)

                print("=================== lenght of chunk proposal eval is =====================")
                print(f"Chunk {i+1} response length: {len(json_formatted_response)} characters")

                # Clean and parse the response
                content = json_formatted_response
                # print(f"\nRESPONSE FOR CHUNK {i+1}:\n{'-'*40}\n{content[:500]}...\n{'-'*40}")
                print(f"\nRESPONSE FOR CHUNK {i+1}:\n{'-'*40}\n{content}...\n{'-'*40}")
                llog("Agent", f"Received response for chunk {i+1}", self.log_save_file_name)
                
                content = re.sub(r'^```json\n', '', content)
                content = re.sub(r'\n```$', '', content)
                content = content.strip()
                
                try:
                    chunk_result = json.loads(content)
                    all_results.append(chunk_result)
                    llog("Proposal Eval", f"Successfully processed chunk {i+1}", self.log_save_file_name)
                    print(f"✅ Successfully processed chunk {i+1} - JSON parsed correctly")
                    print(f"📊 Chunk {i+1} data summary: {len(str(chunk_result))} characters, {len(chunk_result.keys() if isinstance(chunk_result, dict) else [])} top-level keys")
                    
                    # Update the previous evaluations for the next chunk
                    if i < len(proposal_chunks) - 1:  # Don't need to update for the last chunk
                        previous_evaluations_json = json.dumps(chunk_result, ensure_ascii=False, indent=2)
                        print(f"📝 Saved full evaluation results for next chunk context")
                    
                except json.JSONDecodeError as e:
                    llog("Proposal Eval", f"Invalid JSON for chunk {i+1}: {str(e)}", self.log_save_file_name)
                    print(f"❌ Invalid JSON for chunk {i+1}: {str(e)}")
                    # print(f"🔍 First 100 characters of problematic content: {content[:100]}")
                    # print(f"🔍 First 100 characters of problematic content: {content}")
                    print(f"🔍 Last 100 characters of problematic content: {content[-100:] if len(content) > 100 else content}")
                    
                    # Try to clean up the JSON and retry
                    try:
                        fixed_content = self.fix_json_content(content)
                        chunk_result = json.loads(fixed_content)
                        all_results.append(chunk_result)
                        llog("Proposal Eval", f"Fixed and processed chunk {i+1}", self.log_save_file_name)
                        print(f"🔧 Fixed and processed chunk {i+1} - JSON now parses correctly")
                        print(f"📊 Fixed chunk {i+1} data summary: {len(str(chunk_result))} characters, {len(chunk_result.keys() if isinstance(chunk_result, dict) else [])} top-level keys")
                        
                        # Update the previous evaluations for the next chunk
                        if i < len(proposal_chunks) - 1:
                            previous_evaluations_json = json.dumps(chunk_result, ensure_ascii=False, indent=2)
                            print(f"📝 Saved full evaluation results for next chunk context")
                    except Exception as e2:
                        llog("Proposal Eval", f"Failed to fix JSON for chunk {i+1}: {str(e2)}", self.log_save_file_name)
                        print(f"❌ Failed to fix JSON for chunk {i+1}: {str(e2)}")
                        print(f"⚠️ Storing raw content for chunk {i+1} and continuing with next chunk")
                        # Store the raw content as a fallback
                        fallback_result = {"chunk_number": i+1, "raw_content": content[:1000] + "...", "parsing_failed": True}
                        all_results.append(fallback_result)
                        print(f"📄 Added fallback result for chunk {i+1} to results list")
            except Exception as e:
                llog("Proposal Eval", f"Error processing chunk {i+1}: {str(e)}", self.log_save_file_name)
                print(f"❌ Error processing chunk {i+1}: {str(e)}")
                # Instead of raising an error, continue with the next chunk
                print(f"⚠️ Continuing with next chunk despite processing error")
                llog("Proposal Eval", f"Continuing with next chunk despite processing error", self.log_save_file_name)
                # Store information about the error
                error_result = {"chunk_number": i+1, "error": str(e), "processing_failed": True}
                all_results.append(error_result)
                print(f"📄 Added error information for chunk {i+1} to results list")
        
        # Check if we have any results to combine
        if not all_results:
            llog("Proposal Eval", "No results from any chunks to combine", self.log_save_file_name)
            raise ValueError("Failed to process any chunks successfully")
        
        # Print summary of all chunks processed
        print(f"\n{'='*80}")
        print(f"SUMMARY OF PROCESSED CHUNKS")
        print(f"{'='*80}")
        for i, result in enumerate(all_results):
            if "parsing_failed" in result:
                print(f"Chunk {i+1}: ❌ JSON parsing failed - included as raw content")
            elif "processing_failed" in result:
                print(f"Chunk {i+1}: ❌ Processing error - {result.get('error', 'Unknown error')}")
            else:
                print(f"Chunk {i+1}: ✅ Successfully processed - {len(str(result))} characters")
        print(f"{'='*80}\n")
        
        # Combine results from all chunks
        print(f"\n{'='*80}\nCOMBINING RESULTS FROM {len(all_results)} CHUNKS\n{'='*80}")
        print("========== lenght is checking fort all_chunk_result ============")
        llog("Proposal Eval", f"Length of all_results: {len(all_results)}", self.log_save_file_name)
        llog("Proposal_eval", f"types of all_result is {type(all_results)}", self.log_save_file_name)
        print(f"Length of all_results: {len(all_results)}")
        print(f"type of all_result is :  {type(all_results)}")
        print("================================================================")
     

        return self.combine_chunk_results(all_results, detected_language,rfp_requirements)

    def fix_json_content(self, content):
        """Attempt to fix common JSON issues"""
        # Remove any markdown artifacts
        content = re.sub(r'```.*\n', '', content)
        content = re.sub(r'\n```', '', content)
        
        # Fix trailing commas in arrays and objects
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*\]', ']', content)
        
        # Fix missing quotes around keys
        content = re.sub(r'(\w+):', r'"\1":', content)
        
        # Fix single quotes used instead of double quotes
        content = re.sub(r"'([^']*)':", r'"\1":', content)  # For keys
        
        # Try to fix unescaped quotes in strings
        content = re.sub(r':\s*"([^"]*)"([^,}]*)"([^"]*)"', r': "\1\2\3"', content)
        
        # Fix missing commas between objects in arrays
        content = re.sub(r'}\s*{', '},{', content)
        
        # Fix extra commas
        content = re.sub(r',\s*,', ',', content)
        
        return content
    
    def combine_chunk_results(self, chunk_results, detected_language,rfp_requirements):
        """Combine results from multiple chunks into a single evaluation."""
        llog("Agent", "Combining results from multiple chunks", self.log_save_file_name)
        
        # Create a prompt to combine the results
        combined_results_json = json.dumps(chunk_results, ensure_ascii=False, indent=2)
        print(f"Combined chunk results JSON size: {len(combined_results_json)} characters")
        
        # Handle case-insensitive language detection
        detected_lang_lower = detected_language.lower()
        if detected_lang_lower == "english":
            format = format_2
        elif detected_lang_lower == "arabic":
            format = format_2_arabic
        else:
            # Default to English format if language not recognized
            format = format_2
            llog("Agent", f"Unknown language '{detected_language}', defaulting to English format", self.log_save_file_name)
        
        prompt = COMBINE_CHUNK_RESULTS_PROMPT.format(
            rfp_text = rfp_requirements,
            language=detected_language,
            format=format
        ) + "\n\n" + combined_results_json
        
        try:
            print("Sending combined chunks to model for final evaluation...")
            llog("Agent", "Sending combined chunks to model for final evaluation", self.log_save_file_name)
            response = self.generate(prompt)

            llog("Agent", f"combined evaluation response ::: {response}", self.log_save_file_name)
            # Convert the unstructured response to JSON format
            json_format_prompt = FORMAT_OUTPUT_TO_JSON_PROMPT.format(
                evaluation_content=response,
                json_schema=json.dumps(format, ensure_ascii=False, indent=2),
                language=detected_language
            )
            
            llog("Agent", "Converting combined response to JSON format", self.log_save_file_name)
            json_formatted_response = self.generate(json_format_prompt)
            llog("Agent", f"Received JSON format response {json_formatted_response}", self.log_save_file_name)


            regex_pattern = r'```json\n([\s\S]*?)\n```'
            match = re.search(regex_pattern, json_formatted_response, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
                try:
                    # Parse to validate JSON
                    llog("Proposal Eval", f"Parsing JSON content: {json_content[:200]}...", self.log_save_file_name)
                    parsed_json = json.loads(json_content)
                    # Store the cleaned JSON
                    response = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                    llog("Proposal Eval", f"Parsed JSON response: {response}", self.log_save_file_name)
                except json.JSONDecodeError as e:
                    llog("Proposal Eval", f"Failed to parse JSON: {str(e)}", self.log_save_file_name)
                    response = response  # Fallback to raw response
            else:
                llog("Proposal Eval", "No JSON block found in response, using raw response", self.log_save_file_name)
                response = response  # Fallback to raw response

            # # Clean the response content
            # content = json_formatted_response


            # print(f"\nCOMBINED EVALUATION RESPONSE:\n{'-'*40}\n{content[:500]}...\n{'-'*40}")
            print(f"\nCOMBINED EVALUATION RESPONSE:\n{'-'*40}\n{response}...\n{'-'*40}")
            llog("Agent", "Received combined evaluation response", self.log_save_file_name)
            
            # content = re.sub(r'^```json\n', '', content)
            # content = re.sub(r'\n```$', '', content)
            # content = content.strip()
            
            try:
                # final_result = json.loads(content)
                print(f"✅ Successfully combined and processed all chunks")
                return json.dumps(response, ensure_ascii=False)
            except json.JSONDecodeError as e:
                llog("Proposal Eval", f"Invalid JSON in combined result: {str(e)}", self.log_save_file_name)
                print(f"❌ Invalid JSON in combined result: {str(e)}")
                
                # Try to fix the JSON
                try:
                    fixed_content = self.fix_json_content(response)
                    final_result = json.loads(fixed_content)
                    print(f"🔧 Fixed and processed combined result")
                    return json.dumps(final_result, ensure_ascii=False)
                except Exception as e2:
                    llog("Proposal Eval", f"Failed to fix combined JSON: {str(e2)}", self.log_save_file_name)
                    llog("Proposal Eval", f"Full response: {response[:1000]}", self.log_save_file_name)
                    raise ValueError(f"Model returned invalid JSON when combining results: {str(e)}")
        except Exception as e:
            llog("Proposal Eval", f"Error in combine_chunk_results: {str(e)}", self.log_save_file_name)
            print(f"❌ Error in combine_chunk_results: {str(e)}")
            raise ValueError(f"Error combining chunk results: {str(e)}")
    
    def run(self, rfp_txt, proposal_text, detected_language, log_save_file_name):
        """
        Run the evaluation process on the provided RFP and proposal texts.
        
        Args:
            rfp_txt (str): The RFP text content
            proposal_text (str): The proposal text content
            detected_language (str): The detected language of the texts
            log_save_file_name (str): The log file name for tracking
            
        Returns:
            tuple: (HTML evaluation report, JSON data, score)
        """
        self.log_save_file_name = log_save_file_name
        
        # Count tokens in the proposal text
        proposal_tokens = self.count_tokens(proposal_text)
        llog("Agent", f"Proposal text contains {proposal_tokens} tokens", log_save_file_name)
        print(f"\n📄 Proposal text contains {proposal_tokens} tokens")
        
        if proposal_tokens > self.max_tokens:
            # Large document handling path
            llog("Agent", f"Proposal exceeds token limit. Splitting into chunks.", log_save_file_name)
            print(f"\n🔄 Proposal exceeds token limit of {self.max_tokens}. Splitting into chunks...")
            
            # First extract RFP requirements
            initial_state = {
                "detected_language": detected_language,
                "rfp_text": rfp_txt,
                "proposal_text": "",  # Not usedin this step 
                "rfp_requirements": "",
                "proposal_eval_report_json": "",
                "proposal_eval_report_html": ""
            }
            
            # Run only the RFP requirements extraction
            print(f"\n📋 Extracting RFP requirements...")
            requirements_state = self.__rfp_requirements_extraction_node(initial_state)
            rfp_requirements = requirements_state["rfp_requirements"]


            # temparory set just for fast debugging remove this after working good.
            # print(f"\n📋 BYPASSS RFP requirements...")
            # # rfp_file_path = "/workspace/efp_opensource_prompt_testing/rfp_3.txt"
            # rfp_file_path = "/home/khantil/kaushik/RFP_2/translated.txt"
            # with open(rfp_file_path, 'r', encoding='utf-8') as f:
            #     rfp_requirements = f.read()

            print(f"✅ RFP requirements extracted. Length: {len(rfp_requirements)} characters")
            
            # Split proposal into chunks
            proposal_chunks = self.chunk_text(proposal_text, tokens_per_chunk=75000) 
            print(f"\n✂️ Split proposal into {len(proposal_chunks)} chunks")
            
            # Process chunks and combine results
            proposal_eval_report_json = self.process_large_proposal(
                rfp_requirements, 
                proposal_chunks, 
                detected_language
            )
            llog("AgenClustert", f"chunk from large = Proposal Evaluation Report JSon: {proposal_eval_report_json}", log_save_file_name)
            
            # print("===================== proposal_chunk =======================")
            # print(f" proposal chunk {proposal_chunks}")

            # Convert JSON to HTML
            print(f"\n🔄 Converting JSON to HTML...")
            state = {
                "detected_language": detected_language,
                "rfp_text": rfp_txt,
                "proposal_text": proposal_text,
                "rfp_requirements": rfp_requirements,
                "proposal_eval_report_json": proposal_eval_report_json,
                "proposal_eval_report_html": ""
            }
            
            result = self.__json_to_html_node(state)
            proposal_eval_report_html = result["proposal_eval_report_html"]
            print(f"✅ JSON converted to HTML. Length: {len(proposal_eval_report_html)} characters")
            
            # print(f"✅ JSON converted to HTML. {proposal_eval_report_html} ")
            # Calculate the technical score
            try:
                json_data = json.loads(proposal_eval_report_json)
                if isinstance(json_data, str):
                    print("Detected double-encoded JSON, parsing again")
                    json_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                raise ValueError(f"Invalid JSON: {proposal_eval_report_json}")

            # Validate JSON structure
            if not isinstance(json_data, dict):
                error_msg = f"Expected dictionary for json_data, got {type(json_data)}: {json_data}"
                print(error_msg)
                raise TypeError(error_msg)
            score = self.calculate_technical_score(json_data, detected_language)
            
            # Return both the HTML and the JSON data
            return proposal_eval_report_html, score
        
        else:
            # Standard processing path for documents within token limits
            llog("Agent", f"Proposal within token limits. Using standard processing path.", log_save_file_name)
            print(f"\n🔄 Proposal within token limits. Using standard processing path...")
            initial_state = {
                "detected_language": detected_language,
                "rfp_text": rfp_txt,
                "proposal_text": proposal_text,
                "rfp_requirements": "",
                "proposal_eval_report_json": "",
                "proposal_eval_report_html": ""
            }
            llog("Agent", f"Initial state:", log_save_file_name)
            result = self.graph.invoke(initial_state)
            llog("Agent", f"Proposal Eval Result ::::: {result}", log_save_file_name)
            #store result in file
            output_dir = "TEMP_TEST_DIR"
            os.makedirs(output_dir, exist_ok=True)
            # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"proposal_result_.json"
            
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result["proposal_eval_report_json"], f, indent=2, ensure_ascii=False)

            # Calculate the technical score
            json_data = json.loads(result["proposal_eval_report_json"])
            score = self.calculate_technical_score(json_data, detected_language)
            
            # Return both the HTML and the JSON data
            return result["proposal_eval_report_html"], score

    def calculate_technical_score(self, json_data, language):
        """
        Calculate the technical score from the evaluation JSON result.
        
        Args:
            json_data (dict): The evaluation result in JSON format
            language (str): The detected language ('English' or 'Arabic')
            
        Returns:
            float: The calculated technical score as a percentage
        """
        try:
            # Handle case-insensitive language detection
            lang_lower = language.lower() if isinstance(language, str) else ""
            if lang_lower == "arabic":
                # Handle Arabic JSON format
                criteria = json_data.get("المتطلبات_المُسجَّلة_بالدرجات", {}).get("معايير_التقييم", [])
                
                total_required = 0
                total_awarded = 0
                
                for criterion in criteria:
                    # Convert string values to float if needed
                    required_score = criterion.get("الدرجة_المطلوبة", 0)
                    assigned_score = criterion.get("الدرجة_المخصصة", 0)
                    
                    # Handle if scores are strings
                    if isinstance(required_score, str):
                        try:
                            required_score = float(required_score)
                        except ValueError:
                            required_score = 0
                    
                    if isinstance(assigned_score, str):
                        try:
                            assigned_score = float(assigned_score)
                        except ValueError:
                            assigned_score = 0
                    
                    total_required += required_score
                    total_awarded += assigned_score
            else:
                # Handle English JSON format
                criteria = json_data.get("scored_requirements", {}).get("evaluation_criteria", [])
                
                total_required = 0
                total_awarded = 0
                
                for criterion in criteria:
                    # Convert string values to float if needed
                    required_score = criterion.get("required_score", 0)
                    assigned_score = criterion.get("assigned_score", 0)
                    
                    # Handle if scores are strings
                    if isinstance(required_score, str):
                        try:
                            required_score = float(required_score)
                        except ValueError:
                            required_score = 0
                    
                    if isinstance(assigned_score, str):
                        try:
                            assigned_score = float(assigned_score)
                        except ValueError:
                            assigned_score = 0
                    
                    total_required += required_score
                    total_awarded += assigned_score

            llog("Agent", f"Total required score: {total_required}, Total awarded score: {total_awarded}", self.log_save_file_name)
            # Calculate percentage score
            if total_required > 0:
                percentage_score = (total_awarded / total_required) * 100
                return round(percentage_score, 2)
            else:
                print("Warning: Total required score is zero")
                llog("Agent", "Warning: Total required score is zero", self.log_save_file_name)
                return 0
            
        except Exception as e:
            print(f"Error calculating technical score: {str(e)}")
            llog("Agent", f"Error calculating technical score: {str(e)}", self.log_save_file_name)
            return 0

    # def getUniqueFileNameForLogger(self):
    #     """
    #     Generates a unique filename for logging with a timestamp prefix.
        
    #     Returns:
    #         str: A unique identifier string with timestamp.
    #     """
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    #     unique_id = str(uuid.uuid4())
    #     log_save_file_name = f"{timestamp}_{unique_id}"
    #     return log_save_file_name

if __name__ == "__main__":
    def text_file_read(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    # # rfp_txt = text_file_read("/root/RFP/tendor_poc/src/rfp_text.txt")
    # rfp_txt = text_file_read("/root/tendor_poc/src/rfp.txt")
    # # proposal_text = text_file_read("/root/RFP/tendor_poc/src/translated_proposal_2.txt")
    # proposal_text = text_file_read("/root/tendor_poc/src/propsal.txt")
    rfp_text = text_file_read("/root/RFP/SOW_for_NAFIS_PLATFROM_Developmen_1.txt")
    proposal_text = text_file_read("/root/RFP/Nafis_Proposal_IDC_Final.txt")
    evaluator = AgentCluster()
    evaluator.run(rfp_text, proposal_text, "English", "new_log.txt")
    # evaluator.run(rfp_text, proposal_text, "English", "log.txt")
