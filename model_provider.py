# import torch
# import time
# import multiprocessing
# from vllm import LLM, SamplingParams
# from vllm.transformers_utils.tokenizer import get_tokenizer
# from logger import custom_logger as llog
# import threading
# import queue
# import os

# # Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues in forked processes
# try:
#     multiprocessing.set_start_method('spawn', force=True)
# except RuntimeError:
#     # Method already set
#     pass

# class ModelProvider:
#     """
#     Singleton class to provide shared access to a loaded model across different modules.
#     This ensures the model is only loaded once during the application lifecycle.
#     Model loading is done on-demand based on job requirements.
#     """
#     _instance = None
#     _initialized = False
    
#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super(ModelProvider, cls).__new__(cls)
#             # Initialize class variables here to ensure they exist on the first instance
#             cls._instance.model = None
#             cls._instance.model_type = None
#             cls._instance.tokenizer = None
#             cls._instance.log_save_file_name = None
#             cls._instance.model_lock = None
#             cls._instance.model_in_use = False
#             cls._instance.model_queue = None
#         return cls._instance
    
#     def __init__(self, model_type=None, log_save_file_name=None):
#         # If already initialized, just update the log file name if needed
#         if ModelProvider._initialized:
#             if log_save_file_name:
#                 self.log_save_file_name = log_save_file_name
#             return
            
#         self.log_save_file_name = log_save_file_name
#         # Don't set model_type in init, it will be set during preload or ensure_model_loaded
#         self.model_type = None
#         self.model = None
#         self.tokenizer = None
        
#         # Add model resource lock and queue
#         self.model_lock = threading.Lock()
#         self.model_in_use = False
#         self.model_queue = queue.Queue()
#         self.model_queue.put(1)  # Initialize the queue with a single token
        
#         # Initialize tokenizer for token counting
#         try:
#             llog("ModelProvider", "Initializing tokenizer...", self.log_save_file_name)
#             self.tokenizer = get_tokenizer("Qwen/Qwen2.5-14B-Instruct")
#             llog("ModelProvider", "Tokenizer initialized successfully for token counting", self.log_save_file_name)
#         except Exception as e: 
#             llog("ModelProvider", f"Error initializing tokenizer: {str(e)}", self.log_save_file_name)
            
#         ModelProvider._initialized = True
#         llog("ModelProvider", "ModelProvider initialized successfully", self.log_save_file_name)
    
#     def preload_opensource_model(self):
#         """
#         Preload the opensource model during initialization.
#         This ensures the model is ready when the first job arrives.
#         """
#         llog("ModelProvider", "Preloading opensource model...", self.log_save_file_name)
#         try:
#             # Always set model type to opensource for preloading
#             self.model_type = 'opensource'
            
#             # Directly initialize the model
#             if self.model is None:
#                 llog("ModelProvider", "Model is None, initializing it now", self.log_save_file_name)
#                 self._initialize_opensource_model()
#                 llog("ModelProvider", "Opensource model preloaded successfully", self.log_save_file_name)
#                 return True
#             else:
#                 llog("ModelProvider", "Model already initialized", self.log_save_file_name)
#                 return True
                
#         except Exception as e:
#             error_msg = f"Error preloading opensource model: {str(e)}"
#             llog("ModelProvider", error_msg, self.log_save_file_name)
#             # Reset model_type to None if initialization failed
#             self.model_type = None
#             return False
    
#     def acquire_model(self):
#         """
#         Acquire the model resource for exclusive use.
#         This will block until the model is available.
#         Only applies to opensource model - OpenAI doesn't need queuing.
#         """
#         # Only queue for opensource model
#         if self.model_type == "opensource":
#             # Wait for the model to be available
#             token = self.model_queue.get(block=True)
#             return token
#         # For OpenAI, no need to queue
#         return None

#     def release_model(self):
#         """
#         Release the model resource so others can use it.
#         Only applies to opensource model - OpenAI doesn't need queuing.
#         """
#         # Only release for opensource model
#         if self.model_type == "opensource":
#             # Return the token to the queue
#             try:
#                 self.model_queue.put(1)
#             except:
#                 # If there's an error, ensure we don't deadlock
#                 if self.model_queue.empty():
#                     self.model_queue.put(1)
    
#     # def ensure_model_loaded(self, model_type):
#     #     """
#     #     Ensure the appropriate model is loaded based on the requested model type.
#     #     This method is called before any generation operation.
        
#     #     Args:
#     #         model_type: The model type to use ('opensource' or 'openai')
#     #     """
#     #     # Handle case where model_type parameter is None
#     #     if model_type is None:
#     #         llog("ModelProvider", "Warning: ensure_model_loaded called with None model_type", self.log_save_file_name)
#     #         return
            
#     #     # Always update model_type to what's requested
#     #     if self.model_type != model_type:
#     #         llog("ModelProvider", f"Model type changed from {self.model_type or 'None'} to {model_type}", self.log_save_file_name)
#     #         self.model_type = model_type
            
#     #     # If requesting opensource model and it's not loaded yet, initialize it
#     #     if model_type == "opensource" and self.model is None:
#     #         llog("ModelProvider", "Initializing opensource model on demand", self.log_save_file_name)
#     #         self._initialize_opensource_model()

#     # def ensure_model_loaded(self, model_type):
#     #     if model_type is None:
#     #         llog("ModelProvider", "Warning: ensure_model_loaded called with None model_type", self.log_save_file_name)
#     #         return
#     #     if self.model_type == model_type and self.model is not None:
#     #         llog("ModelProvider", f"Model {model_type} already loaded, skipping initialization", self.log_save_file_name)
#     #         return
#     #     llog("ModelProvider", f"Model type changed from {self.model_type or 'None'} to {model_type}", self.log_save_file_name)
#     #     self.model_type = model_type
#     #     if model_type == "opensource":
#     #         llog("ModelProvider", "Initializing opensource model on demand", self.log_save_file_name)
#     #         self._initialize_opensource_model()

#     def ensure_model_loaded(self, model_type):
#         """
#         Ensure the appropriate model is loaded based on the requested model type.
#         Avoid reinitializing if the model is already loaded.
        
#         Args:
#             model_type: The model type to use ('opensource' or 'openai')
#         """
#         if model_type is None:
#             llog("ModelProvider", "Warning: ensure_model_loaded called with None model_type", self.log_save_file_name)
#             return
            
#         # If the requested model type is already loaded, skip initialization
#         if self.model_type == model_type and self.model is not None and model_type == "opensource":
#             llog("ModelProvider", f"Model {model_type} already loaded, skipping initialization", self.log_save_file_name)
#             return
            
#         # If the model type is changing, reset the model (only for opensource, OpenAI doesn't need model loading)
#         if self.model_type != model_type:
#             llog("ModelProvider", f"Model type changed from {self.model_type or 'None'} to {model_type}", self.log_save_file_name)
#             self.model_type = model_type
#             if model_type == "opensource" and self.model is None:
#                 llog("ModelProvider", "Initializing opensource model on demand", self.log_save_file_name)
#                 self._initialize_opensource_model()
    
#     def _initialize_opensource_model(self):
#         """Initialize the open source model using vLLM"""
#         try:
#             llog("ModelProvider", "Initializing Qwen model...", self.log_save_file_name)
            
#             # Check if CUDA is available with detailed information
#             gpu_available = torch.cuda.is_available()
#             llog("ModelProvider", f"CUDA available: {gpu_available}", self.log_save_file_name)
            
#             if not gpu_available:
#                 error_msg = "CUDA is not available. Cannot initialize the Qwen model."
#                 llog("ModelProvider", error_msg, self.log_save_file_name)
#                 raise RuntimeError(error_msg)
                
#             # Get detailed GPU information
#             gpu_count = torch.cuda.device_count()
#             llog("ModelProvider", f"Number of GPU devices detected: {gpu_count}", self.log_save_file_name)
            
#             if gpu_count == 0:
#                 error_msg = "No GPU devices found. Cannot initialize the Qwen model."
#                 llog("ModelProvider", error_msg, self.log_save_file_name)
#                 raise RuntimeError(error_msg)
                
#             # Log GPU device details
#             for i in range(gpu_count):
#                 device_name = torch.cuda.get_device_name(i)
#                 device_cap = torch.cuda.get_device_capability(i)
#                 llog("ModelProvider", f"GPU {i}: {device_name}, compute capability: {device_cap}", self.log_save_file_name)
                
#             llog("ModelProvider", f"Using {gpu_count} GPUs for tensor parallelism", self.log_save_file_name)
            
#             # Set environment variable to allow longer context if needed
#             os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
#             llog("ModelProvider", "Set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1", self.log_save_file_name)
            
#             llog("ModelProvider", "Creating LLM instance with vLLM...", self.log_save_file_name)
#             try:
#                 self.model = LLM(
#                     model="Qwen/Qwen2.5-14B-Instruct",
#                     gpu_memory_utilization=0.9,
#                     tensor_parallel_size=gpu_count,
#                     dtype="auto", 
#                     max_model_len=131072,  # Use a safer value that matches the model's config
#                     enforce_eager=False,  # Enable CUDA graphs for better performance
#                     disable_custom_all_reduce=True  # Silence P2P warning
#                 )
#                 llog("ModelProvider", "Model initialized successfully", self.log_save_file_name)
#             except Exception as model_e:
#                 llog("ModelProvider", f"Error creating LLM instance: {str(model_e)}", self.log_save_file_name)
#                 raise
                
#         except Exception as e:
#             error_msg = f"Error initializing model: {str(e)}"
#             llog("ModelProvider", error_msg, self.log_save_file_name)
#             self.model = None
#             self.model_type = None
#             raise
            
#     def count_tokens(self, text):
#         """Count tokens in a text string using the loaded tokenizer"""
#         if self.tokenizer:
#             return len(self.tokenizer.encode(text))
#         else:
#             # Rough estimate if tokenizer not available
#             return len(text.split())
    
#     def generate(self, prompt, max_tokens=8192, temperature=0.7, top_p=0.8, top_k=20):
#         """
#         Generate a response using either the local model or OpenAI API.
#         This method will ensure the correct model is loaded based on the current model_type.
        
#         Args:
#             prompt: The input prompt text
#             max_tokens: Maximum number of tokens to generate
#             temperature: Temperature parameter for generation
#             top_p: Top-p sampling parameter
#             top_k: Top-k sampling parameter
            
#         Returns:
#             Generated text response
#         """
#         # Check if model_type is set
#         if not self.model_type:
#             error_msg = "Model type not set before calling generate()"
#             llog("ModelProvider", error_msg, self.log_save_file_name)
#             raise ValueError(error_msg)
            
#         if self.model_type == "openai":
#             # OpenAI model doesn't need queuing, so we can call it directly
#             return self._generate_openai(prompt, max_tokens, temperature, top_p, top_k)
#         else:
#             # For opensource model, we need to make sure we've acquired the model first
#             # Note: The acquire/release should be handled by the caller
#             return self._generate_opensource(prompt, max_tokens, temperature, top_p, top_k)
    
#     def _generate_opensource(self, prompt, max_tokens=8192, temperature=0.7, top_p=0.8, top_k=20):
#         """Generate a response using the local Qwen model"""
#         try:
#             # Check if model is initialized
#             if not self.model:
#                 llog("ModelProvider", "Model not initialized, attempting to initialize now", self.log_save_file_name)
#                 self._initialize_opensource_model()
                
#             if not self.tokenizer:
#                 llog("ModelProvider", "Tokenizer not initialized, initializing now", self.log_save_file_name)
#                 self.tokenizer = get_tokenizer("Qwen/Qwen2.5-14B-Instruct")
                
#             if not self.model or not self.tokenizer:
#                 raise ValueError("Failed to initialize model or tokenizer")
                
#             messages = [
#                 {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ]

#             text = self.tokenizer.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )
            
#             token_count = self.count_tokens(text)
#             llog("ModelProvider", f"Sending prompt to model (length: {token_count} tokens)", self.log_save_file_name)
            
#             sampling_params = SamplingParams(
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 repetition_penalty=1,
#                 top_p=top_p,
#                 top_k=top_k
#             )

#             start_time = time.time()
#             outputs = self.model.generate([text], sampling_params)
            
#             if not outputs or len(outputs) == 0 or len(outputs[0].outputs) == 0:
#                 raise ValueError("Model returned empty output")
                
#             response = outputs[0].outputs[0].text
#             if not response or response.strip() == "":
#                 raise ValueError("Model returned empty response text")
                
#             elapsed = time.time() - start_time
#             llog("ModelProvider", f"Response generated in {elapsed:.2f} seconds", self.log_save_file_name)
#             return response
            
#         except Exception as e:
#             error_msg = f"Error in model generation: {str(e)}"
#             llog("ModelProvider", error_msg, self.log_save_file_name)
#             return f"Error: {error_msg}" 

import torch
import time
import multiprocessing
# from vllm import LLM, SamplingParams
# from vllm.transformers_utils.tokenizer import get_tokenizer
from logger import custom_logger as llog
import threading
import queue
import os
from typing import List
import re

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues in forked processes
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set
    pass

class ModelProvider:
    """
    Singleton class to provide shared access to a loaded model across different modules.
    This ensures the model is only loaded once during the application lifecycle.
    Model loading is done on-demand based on job requirements.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelProvider, cls).__new__(cls)
            # Initialize class variables here to ensure they exist on the first instance
            cls._instance.model = None
            cls._instance.model_type = None
            cls._instance.tokenizer = None
            cls._instance.log_save_file_name = None
            cls._instance.model_lock = None
            cls._instance.model_in_use = False
            cls._instance.model_queue = None
            cls._instance._thread_local = threading.local()  # Thread-local storage
        return cls._instance
    
    def __init__(self, model_type=None, log_save_file_name=None):
        # If already initialized, just update the log file name if needed
        if ModelProvider._initialized:
            if log_save_file_name:
                self.log_save_file_name = log_save_file_name
            return
            
        self.log_save_file_name = log_save_file_name
        self.model_type = None
        self.model = None
        self.tokenizer = None
        
        # Add model resource lock and queue
        self.model_lock = threading.Lock()
        self.model_in_use = False
        self.model_queue = queue.Queue()
        self.model_queue.put(1)  # Initialize the queue with a single token
        self._thread_local = threading.local()  # Per-thread storage
        
        # Initialize tokenizer for token counting
        try:
            llog("ModelProvider", "Initializing tokenizer...", self.log_save_file_name)
            # self.tokenizer = get_tokenizer("Qwen/Qwen2.5-14B-Instruct") Qwen/QwQ-32B
            # self.tokenizer = get_tokenizer("Qwen/QwQ-32B") # meta-llama/Llama-3.3-70B-Instruct
            # self.tokenizer = get_tokenizer("meta-llama/Llama-3.3-70B-Instruct") # deepseek-ai/DeepSeek-V3
            # self.tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3") #deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
            self.tokenizer = get_tokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") #deepseek-ai/DeepSeek-R1-Distill-Llama-70B
            llog("ModelProvider", "Tokenizer initialized successfully for token counting", self.log_save_file_name)
        except Exception as e: 
            llog("ModelProvider", f"Error initializing tokenizer: {str(e)}", self.log_save_file_name)
            self.tokenizer = None
            
        ModelProvider._initialized = True
        llog("ModelProvider", "ModelProvider initialized successfully", self.log_save_file_name)
    
    def preload_opensource_model(self):
        """
        Preload the opensource model during initialization.
        This ensures the model is ready when the first job arrives.
        """
        llog("ModelProvider", "Preloading opensource model...", self.log_save_file_name)
        try:
            # Always set model type to opensource for preloading
            self.model_type = 'opensource'
            
            # Directly initialize the model
            # if self.model is None:
            #     llog("ModelProvider", "Model is None, initializing it now", self.log_save_file_name)
            #     self._initialize_opensource_model()
            #     llog("ModelProvider", "Opensource model preloaded successfully", self.log_save_file_name)
            #     return True
            # else:
            #     llog("ModelProvider", "Model already initialized", self.log_save_file_name)
            #     return True
                
        except Exception as e:
            error_msg = f"Error preloading opensource model: {str(e)}"
            llog("ModelProvider", error_msg, self.log_save_file_name)
            self.model_type = None
            return False
    
    def acquire_model(self):
        """
        Acquire the model resource for exclusive use.
        This will block until the model is available.
        Only applies to opensource model - OpenAI doesn't need queuing.
        """

       
        # if self.model_type == "opensource":
        #     try:
        #         llog("ModelProvider", "Acquiring model resource", self.log_save_file_name)
        #         token = self.model_queue.get(block=True, timeout=30)  # 30-second timeout
        #     except queue.Empty:
        #         error_msg = "Timed out waiting for model resource"
        #         llog("ModelProvider", error_msg, self.log_save_file_name)
        #         raise RuntimeError(error_msg)
        #     llog("ModelProvider", "Model resource acquired", self.log_save_file_name)
        #     return token
        # return None

        # if self.model_type == "opensource":
        #     llog("ModelProvider", "Acquiring model resource", self.log_save_file_name)
        #     if threading.current_thread() in self.model_queue._getters:  # Check if thread already holds token
        #         llog("ModelProvider", "Model already acquired by this thread", self.log_save_file_name)
        #         return None
        #     token = self.model_queue.get(block=True, timeout=30)
        #     llog("ModelProvider", "Model resource acquired", self.log_save_file_name)
        #     return token
        # return None
        
        if self.model_type == "opensource":
            llog("ModelProvider", "Acquiring model resource", self.log_save_file_name)
            # Check if the current thread already holds the token
            if hasattr(self._thread_local, 'has_token') and self._thread_local.has_token:
                llog("ModelProvider", "Model already acquired by this thread", self.log_save_file_name)
                return None
            try:
                token = self.model_queue.get(block=True, timeout=30)
                self._thread_local.has_token = True  # Mark thread as holding token
                llog("ModelProvider", "Model resource acquired", self.log_save_file_name)
                return token
            except queue.Empty:
                error_msg = "Timed out waiting for model resource"
                llog("ModelProvider", error_msg, self.log_save_file_name)
                raise RuntimeError(error_msg)
        return None

    def release_model(self):
        """
        Release the model resource so others can use it.
        Only applies to opensource model - OpenAI doesn't need queuing.
        """
        if self.model_type == "opensource":
            llog("ModelProvider", "Releasing model resource", self.log_save_file_name)
            try:
                self.model_queue.put(1)
                llog("ModelProvider", "Model resource released", self.log_save_file_name)
            except Exception as e:
                llog("ModelProvider", f"Error releasing model resource: {str(e)}", self.log_save_file_name)
                if self.model_queue.empty():
                    self.model_queue.put(1)
    
    def ensure_model_loaded(self, model_type):
        """
        Ensure the appropriate model is loaded based on the requested model type.
        Avoid reinitializing if the model is already loaded.
        
        Args:
            model_type: The model type to use ('opensource' or 'openai')
        """
        if model_type is None:
            llog("ModelProvider", "Warning: ensure_model_loaded called with None model_type", self.log_save_file_name)
            return
            
        # If the requested model type is already loaded, skip initialization
        if self.model_type == model_type and self.model is not None and model_type == "opensource":
            llog("ModelProvider", f"Model {model_type} already loaded, skipping initialization", self.log_save_file_name)
            return
            
        # If the model type is changing, reset the model (only for opensource, OpenAI doesn't need model loading)
        if self.model_type != model_type:
            llog("ModelProvider", f"Model type changed from {self.model_type or 'None'} to {model_type}", self.log_save_file_name)
            self.model_type = model_type
            if model_type == "opensource" and self.model is None:
                llog("ModelProvider", "Initializing opensource model on demand", self.log_save_file_name)
                # self._initialize_opensource_model()
    
    def _initialize_opensource_model(self):
        """Initialize the open source model using vLLM"""
        try:
            llog("ModelProvider", "Initializing Qwen model...", self.log_save_file_name)
            
            # Check CUDA and GPU availability
            gpu_available = torch.cuda.is_available()
            llog("ModelProvider", f"CUDA available: {gpu_available}", self.log_save_file_name)
            if not gpu_available:
                error_msg = "CUDA is not available. Skipping local Qwen model initialization - will use API instead."
                llog("ModelProvider", error_msg, self.log_save_file_name)
                # Don't raise error, just log and skip local model initialization
                return False
                
            gpu_count = torch.cuda.device_count()
            llog("ModelProvider", f"Number of GPU devices detected: {gpu_count}", self.log_save_file_name)
            if gpu_count == 0:
                error_msg = "No GPU devices found. Skipping local Qwen model initialization - will use API instead."
                llog("ModelProvider", error_msg, self.log_save_file_name)
                #raise RuntimeError(error_msg)

                # Don't raise error, just log and skip local model initialization
                return False
                
            # Log GPU details
            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                device_cap = torch.cuda.get_device_capability(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)
                llog("ModelProvider", f"GPU {i}: {device_name}, compute capability: {device_cap}, total memory: {total_memory:.2f} GB, free memory: {free_memory:.2f} GB", self.log_save_file_name)
                
            # Verify sufficient memory
            required_memory = 28  # Approx memory for Qwen2.5-14B
            for i in range(gpu_count):
                free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)
                if free_memory < required_memory:
                    error_msg = f"Insufficient GPU memory on GPU {i}: {free_memory:.2f} GB available, {required_memory} GB required."
                    llog("ModelProvider", error_msg, self.log_save_file_name)
                    raise RuntimeError(error_msg)
                    
            llog("ModelProvider", f"Using {gpu_count} GPUs for tensor parallelism", self.log_save_file_name)
            
            # Clear GPU memory and synchronize
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            llog("ModelProvider", "Cleared GPU memory cache and synchronized", self.log_save_file_name)
            
            # Set environment variables
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            os.environ["VLLM_WORKER_LOG_LEVEL"] = "DEBUG"  # Enable debug logs for vLLM
            llog("ModelProvider", "Set VLLM environment variables", self.log_save_file_name)
            
            llog("ModelProvider", "Creating LLM instance with vLLM...", self.log_save_file_name)
            try:
                self.model = LLM(
                    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                    gpu_memory_utilization=0.85,  # Reduced from 0.9 to avoid fragmentation
                    tensor_parallel_size=gpu_count,
                    dtype="auto",
                    max_model_len=120000,  # Keep as required
                    enforce_eager=True,  # Disable CUDA graphs to avoid kernel issues
                    disable_custom_all_reduce=True,
                    enable_chunked_prefill=True,  # Enable chunked prefill for large contexts
                    max_num_seqs=1  # Limit to one sequence to reduce memory pressure
                )
                llog("ModelProvider", "Model initialized successfully", self.log_save_file_name)
            except Exception as model_e:
                llog("ModelProvider", f"Error creating LLM instance: {str(model_e)}", self.log_save_file_name)
                raise
                
        except Exception as e:
            error_msg = f"Error initializing model: {str(e)}"
            llog("ModelProvider", error_msg, self.log_save_file_name)
            self.model = None
            self.model_type = None
            raise
            
    def count_tokens(self, text):
        """Count tokens in a text string using the loaded tokenizer"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                llog("ModelProvider", f"Error counting tokens: {str(e)}", self.log_save_file_name)
                return len(text.split())  # Fallback
        else:
            llog("ModelProvider", "Tokenizer not available, using word count estimate", self.log_save_file_name)
            return len(text.split())
    
    # chinise char banned
    # def _remove_chinese_characters(self, text: str) -> str:
    #     """
    #     Remove Chinese characters from the text.
        
    #     Args:
    #         text: Input text to process
            
    #     Returns:
    #         Text with Chinese characters removed
    #     """
    #     chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    #     cleaned_text = chinese_pattern.sub('', text)
    #     if chinese_pattern.search(text):
    #         llog("ModelProvider", f"Removed Chinese characters from input: {text[:100]}...", self.log_save_file_name)
    #     return cleaned_text

    # def _identify_chinese_tokens(self):
    #     """
    #     Identify token IDs that correspond to Chinese characters.
        
    #     Returns:
    #         List of token IDs that produce Chinese characters
    #     """
    #     if self.banned_token_ids is not None:
    #         return self.banned_token_ids
            
    #     chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    #     banned_token_ids = []
        
    #     for token_id in range(len(self.tokenizer)):
    #         token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
    #         if chinese_pattern.search(token_text):
    #             banned_token_ids.append(token_id)
        
    #     self.banned_token_ids = banned_token_ids
    #     llog("ModelProvider", f"Identified {len(banned_token_ids)} Chinese token IDs", self.log_save_file_name)
    #     return banned_token_ids

    # def _has_chinese_tokens(self, token_ids: list) -> bool:
    #     """
    #     Check if the generated token IDs include Chinese tokens.
        
    #     Args:
    #         token_ids: List of token IDs from the model's output
            
    #     Returns:
    #         bool: True if Chinese tokens are present, False otherwise
    #     """
    #     banned_tokens = set(self._identify_chinese_tokens())
    #     return any(token_id in banned_tokens for token_id in token_ids)

    def generate(self, prompt, max_tokens=8192, temperature=0.1, top_p=0.95, top_k=30):
        """
        Generate a response using the local model.
        This method will ensure the correct model is loaded based on the current model_type.
        
        Args:
            prompt: The input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text response
        """
        if not self.model_type:
            error_msg = "Model type not set before calling generate()"
            llog("ModelProvider", error_msg, self.log_save_file_name)
            raise ValueError(error_msg)
            
        if self.model_type != "opensource":
            error_msg = "Generate called with non-opensource model type"
            llog("ModelProvider", error_msg, self.log_save_file_name)
            raise ValueError(error_msg)
            
        return self._generate_opensource(prompt, max_tokens, temperature, top_p, top_k)
    
    def _generate_opensource(self, prompt, max_tokens=8192, temperature=0.1, top_p=0.90, top_k=30):
        """Generate a response using the local Qwen model"""
        # print("============ opensource prompt lenght ================")
        # print(f"  LENGHT IS  ::   {len(prompt)}")
        # print(f"  Prompt IS  ::   {prompt}")
        try:
            # Check model and tokenizer
            if not self.model:
                llog("ModelProvider", "Model not initialized, attempting to initialize now", self.log_save_file_name)
                # self._initialize_opensource_model()
                
            if not self.tokenizer:
                llog("ModelProvider", "Tokenizer not initialized, initializing now", self.log_save_file_name)
                self.tokenizer = get_tokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
                
            if not self.model or not self.tokenizer:
                raise ValueError("Failed to initialize model or tokenizer")
            
            # Log prompt details
            token_count = self.count_tokens(prompt)
            llog("ModelProvider", f"Prompt length: {token_count} tokens, max_tokens: {max_tokens}, total: {token_count + max_tokens}", self.log_save_file_name)
            if token_count + max_tokens > 100000:
                error_msg = f"Prompt length ({token_count} tokens) plus max_tokens ({max_tokens}) exceeds model limit (100000)"
                llog("ModelProvider", error_msg, self.log_save_file_name)
                raise ValueError(error_msg)
                
            # Check GPU memory before generation
            for i in range(torch.cuda.device_count()):
                free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)
                llog("ModelProvider", f"GPU {i} free memory before generation: {free_memory:.2f} GB", self.log_save_file_name)

            max_attempts = 3
            attempt = 0
            response = None
            
            while attempt < max_attempts:
                attempt += 1

                messages = [
                    {"role": "system", "content": "You are a helpful assistant specializing in RFP analysis and compliance evaluation. Generate responses in languages suitable such as Arabic or English Only, strictly avoid generating any Chinese characters (e.g., 汉字) or Chinese text under any circumstances"},
                    {"role": "user", "content": f"Do not include any Chinese characters or text in the response.{prompt}"}
                ]

                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                llog("ModelProvider", f"Processed prompt length: {self.count_tokens(text)} tokens", self.log_save_file_name)
                
                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    repetition_penalty=1.0,
                    top_p=top_p,
                    top_k=top_k,
                )

                # Synchronize GPUs before generation
                torch.cuda.synchronize()
                llog("ModelProvider", "Synchronized GPUs before generation", self.log_save_file_name)
                
                start_time = time.time()
                outputs = self.model.generate([text], sampling_params)
                
                if not outputs or len(outputs) == 0 or len(outputs[0].outputs) == 0:
                    error_msg = "Model returned empty output"
                    llog("ModelProvider", error_msg, self.log_save_file_name)
                    raise ValueError(error_msg)
                    
                response = outputs[0].outputs[0].text
                if not response or response.strip() == "":
                    error_msg = "Model returned empty response text"
                    llog("ModelProvider", error_msg, self.log_save_file_name)
                    raise ValueError(error_msg)
                    
                elapsed = time.time() - start_time
                llog("ModelProvider", f"Response generated in {elapsed:.2f} seconds", self.log_save_file_name)
                
                # Synchronize after generation
                torch.cuda.synchronize()
                llog("ModelProvider", "Synchronized GPUs after generation", self.log_save_file_name)
                
                return response
            
        except Exception as e:
            error_msg = f"Error in model generation: {str(e)}"
            llog("ModelProvider", error_msg, self.log_save_file_name)
            raise RuntimeError(error_msg)

    # def _generate_opensource(self, prompt, max_tokens=8192, temperature=0.1, top_p=0.8, top_k=20):
    #         """Generate a response using the local Qwen model without Chinese characters"""
    #         try:
    #             # Check model and tokenizer
    #             if not self.model:
    #                 llog("ModelProvider", "Model not initialized, attempting to initialize now", self.log_save_file_name)
    #                 self._initialize_opensource_model()
                    
    #             if not self.tokenizer:
    #                 llog("ModelProvider", "Tokenizer not initialized, initializing now", self.log_save_file_name)
    #                 self.tokenizer = get_tokenizer("Qwen/Qwen2.5-14B-Instruct")
                    
    #             if not self.model or not self.tokenizer:
    #                 raise ValueError("Failed to initialize model or tokenizer")
                    
    #             # Remove Chinese characters from input prompt
    #             cleaned_prompt = self._remove_chinese_characters(prompt)
    #             if cleaned_prompt != prompt:
    #                 llog("ModelProvider", "Input prompt modified to remove Chinese characters", self.log_save_file_name)
                    
    #             # Log prompt details
    #             token_count = self.count_tokens(cleaned_prompt)
    #             llog("ModelProvider", f"Prompt length: {token_count} tokens, max_tokens: {max_tokens}, total: {token_count + max_tokens}", self.log_save_file_name)
    #             if token_count + max_tokens > 100000:
    #                 error_msg = f"Prompt length ({token_count} tokens) plus max_tokens ({max_tokens}) exceeds model limit (100000)"
    #                 llog("ModelProvider", error_msg, self.log_save_file_name)
    #                 raise ValueError(error_msg)
                    
    #             # Check GPU memory before generation
    #             for i in range(torch.cuda.device_count()):
    #                 free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)
    #                 llog("ModelProvider", f"GPU {i} free memory before generation: {free_memory:.2f} GB", self.log_save_file_name)
                    
    #             max_attempts = 3
    #             attempt = 0
    #             response = None
                
    #             while attempt < max_attempts:
    #                 attempt += 1
    #                 llog("ModelProvider", f"Generation attempt {attempt}/{max_attempts}", self.log_save_file_name)
                    
    #                 messages = [
    #                     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant specializing in RFP analysis and compliance evaluation for clients in Dubai. Generate responses in languages suitable for Dubai, such as Arabic or English, but strictly avoid generating any Chinese characters (e.g., 汉字) or Chinese text under any circumstances."},
    #                     {"role": "user", "content": f"Do not include any Chinese characters or text in the response. Use Arabic, English, or other relevant languages for Dubai:\n{cleaned_prompt}"}
    #                 ]

    #                 text = self.tokenizer.apply_chat_template(
    #                     messages, tokenize=False, add_generation_prompt=True
    #                 )
                    
    #                 llog("ModelProvider", f"Processed prompt length: {self.count_tokens(text)} tokens", self.log_save_file_name)
                    
    #                 sampling_params = SamplingParams(
    #                     max_tokens=max_tokens,
    #                     temperature=temperature,
    #                     repetition_penalty=1.0,
    #                     top_p=top_p,
    #                     top_k=top_k
    #                 )

    #                 torch.cuda.synchronize()
    #                 llog("ModelProvider", "Synchronized GPUs before generation", self.log_save_file_name)
                    
    #                 start_time = time.time()
    #                 outputs = self.model.generate([text], sampling_params)
                    
    #                 if not outputs or len(outputs) == 0 or len(outputs[0].outputs) == 0:
    #                     error_msg = "Model returned empty output"
    #                     llog("ModelProvider", error_msg, self.log_save_file_name)
    #                     raise ValueError(error_msg)
                    
    #                 output = outputs[0].outputs[0]
    #                 token_ids = output.token_ids
    #                 response_text = output.text
                    
    #                 # Check for Chinese tokens
    #                 if self._has_chinese_tokens(token_ids):
    #                     llog("ModelProvider", f"Chinese tokens detected in attempt {attempt}, regenerating...", self.log_save_file_name)
    #                     continue
                    
    #                 if not response_text or response_text.strip() == "":
    #                     error_msg = "Model returned empty response text"
    #                     llog("ModelProvider", error_msg, self.log_save_file_name)
    #                     raise ValueError(error_msg)
                    
    #                 response = response_text
    #                 break
                    
    #             if response is None:
    #                 error_msg = f"Failed to generate response without Chinese characters after {max_attempts} attempts"
    #                 llog("ModelProvider", error_msg, self.log_save_file_name)
    #                 raise RuntimeError(error_msg)
                    
    #             elapsed = time.time() - start_time
    #             llog("ModelProvider", f"Response generated in {elapsed:.2f} seconds", self.log_save_file_name)
                
    #             torch.cuda.synchronize()
    #             llog("ModelProvider", "Synchronized GPUs after generation", self.log_save_file_name)
                
    #             return response
                
    #         except Exception as e:
    #             error_msg = f"Error in model generation: {str(e)}"
    #             llog("ModelProvider", error_msg, self.log_save_file_name)
    #             raise RuntimeError(error_msg)
