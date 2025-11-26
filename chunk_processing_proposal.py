# The first one is the original one which is tobe used, but due to RAM limitations it would not work on developers laptop, so the developer is suggested to use second version of it for proposal eval. For RFP eval the first one would not take such a more RAM.

# FIRST ONE (FOR PRODUCTION), REQUIRES HIGH RAM

# from typing import List
# import tiktoken
# import re
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.tokenize import sent_tokenize
# from logger_config import get_logger
# import torch

# logger = get_logger(__file__)

# # Download required NLTK data (run once)
# try:
#     nltk.download('punkt_tab')
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     try:
#         nltk.download('punkt')
#     except Exception as e:
#         logger.error(f"Failed to download NLTK data: {str(e)}")
#         raise


# class ChunkProcessor:
#     def __init__(self, max_chunk_size: int = 1024, model_name: str = 'nomic-ai/nomic-embed-text-v1', shared_model=None):
#         """Initialize the chunk processor.
        
#         Args:
#             max_chunk_size: Maximum chunk size in tokens
#             model_name: Name of the model to use
#             shared_model: Optional pre-loaded SentenceTransformer model to share
#         """
#         self.max_chunk_size = max_chunk_size
#         self.similarity_threshold = 0.7  # Threshold for semantic similarity
        
#         # Initialize sentence transformer model
#         try:
#             if shared_model is not None:
#                 logger.info("Using shared SentenceTransformer model")
#                 self.model = shared_model
#             else:
#                 logger.info(f"Loading new SentenceTransformer model: {model_name}")
#                 self.model = SentenceTransformer(model_name, trust_remote_code=True)
#         except Exception as e:
#             logger.warning(f"Failed to load primary model {model_name}: {str(e)}")
#             raise

#     def count_tokens(self, text: str, model_name: str = "gpt-4") -> int:
#         """Count the number of tokens in a text string."""
#         try:
#             # Use tiktoken for OpenAI models
#             encoding = tiktoken.encoding_for_model(model_name)
#             return len(encoding.encode(text))
#         except Exception as e:
#             logger.warning(f"Error using tiktoken: {str(e)}")
#             # Fall back to rough estimate
#             return len(text.split())

#     def get_sentences(self, text: str) -> List[str]:
#         """Split text into sentences using NLTK."""
#         try:
#             sentences = sent_tokenize(text)
#             return [s.strip() for s in sentences if s.strip()]
#         except Exception as e:
#             logger.warning(f"Error in sentence tokenization: {str(e)}")
#             # Fallback to simple splitting
#             return [s.strip() for s in text.split('.') if s.strip()]

#     def compute_sentence_similarities(self, sentences: List[str]) -> np.ndarray:
#         """Compute cosine similarities between consecutive sentences."""
#         if len(sentences) < 2:
#             return np.array([])
        
#         try:
#             # Get embeddings for all sentences
#             with torch.no_grad():  # Add no_grad to reduce memory usage
#                 embeddings = self.model.encode(sentences, convert_to_tensor=True)
#                 # Move to CPU to free GPU memory
#                 embeddings = embeddings.cpu()
            
#             # Compute similarities between consecutive sentences
#             similarities = []
#             for i in range(len(sentences) - 1):
#                 sim = cosine_similarity([embeddings[i].numpy()], [embeddings[i + 1].numpy()])[0][0]
#                 similarities.append(sim)
            
#             # Clear GPU memory if using CUDA
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
            
#             return np.array(similarities)
#         except Exception as e:
#             logger.error(f"Error computing similarities: {str(e)}")
#             # Fallback to simple similarity (all sentences considered different)
#             return np.zeros(len(sentences) - 1)

#     def find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
#         """Find semantic boundaries where topic shifts occur."""
#         if len(sentences) <= 1:
#             return []
        
#         similarities = self.compute_sentence_similarities(sentences)
        
#         # Find positions where similarity drops below threshold
#         boundaries = []
#         for i, sim in enumerate(similarities):
#             if sim < self.similarity_threshold:
#                 boundaries.append(i + 1)  # +1 because we want to split after the sentence
        
#         return boundaries

#     def create_semantic_chunks(self, sentences: List[str], boundaries: List[int]) -> List[str]:
#         """Create chunks based on semantic boundaries while respecting token limits."""
#         if not sentences:
#             return []
        
#         chunks = []
#         current_chunk_sentences = []
#         current_tokens = 0
        
#         # Add sentence indices for boundary checking
#         boundary_set = set(boundaries)
        
#         for i, sentence in enumerate(sentences):
#             sentence_tokens = self.count_tokens(sentence)
            
#             # Check if adding this sentence would exceed token limit
#             if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk_sentences:
#                 # Finalize current chunk
#                 chunks.append(' '.join(current_chunk_sentences))
#                 current_chunk_sentences = [sentence]
#                 current_tokens = sentence_tokens
#             else:
#                 current_chunk_sentences.append(sentence)
#                 current_tokens += sentence_tokens
            
#             # Check if we hit a semantic boundary and have a reasonable chunk size
#             if (i + 1) in boundary_set and current_chunk_sentences:
#                 # Only create boundary if we have enough content or are forced to
#                 if current_tokens >= self.max_chunk_size * 0.3 or current_tokens + self.count_tokens(sentences[i + 1] if i + 1 < len(sentences) else "") > self.max_chunk_size:
#                     chunks.append(' '.join(current_chunk_sentences))
#                     current_chunk_sentences = []
#                     current_tokens = 0
        
#         # Add remaining sentences as final chunk
#         if current_chunk_sentences:
#             chunks.append(' '.join(current_chunk_sentences))
        
#         return chunks

#     def chunk_text(self, text: str) -> List[str]:
#         """Split text into semantic chunks while respecting token limits."""
#         if not text:
#             return []
        
#         # First check if chunking is needed
#         total_tokens = self.count_tokens(text)
#         if total_tokens <= self.max_chunk_size:
#             return [text]
        
#         # Split into sentences
#         sentences = self.get_sentences(text)
        
#         if len(sentences) <= 1:
#             # Fallback to word-based chunking for single sentence
#             return self.fallback_word_chunking(text)
        
#         # Find semantic boundaries
#         boundaries = self.find_semantic_boundaries(sentences)
        
#         # Create chunks based on semantic boundaries
#         chunks = self.create_semantic_chunks(sentences, boundaries)
        
#         # Verify and fix oversized chunks
#         final_chunks = []
#         for chunk in chunks:
#             chunk_tokens = self.count_tokens(chunk)
#             if chunk_tokens > self.max_chunk_size:
#                 # Split oversized chunks further
#                 sub_sentences = self.get_sentences(chunk)
#                 if len(sub_sentences) > 1:
#                     # Use smaller semantic chunks
#                     sub_boundaries = self.find_semantic_boundaries(sub_sentences)
#                     sub_chunks = self.create_semantic_chunks(sub_sentences, sub_boundaries)
#                     for sub_chunk in sub_chunks:
#                         if self.count_tokens(sub_chunk) > self.max_chunk_size:
#                             final_chunks.extend(self.fallback_word_chunking(sub_chunk))
#                         else:
#                             final_chunks.append(sub_chunk)
#                 else:
#                     # Fallback to word chunking
#                     final_chunks.extend(self.fallback_word_chunking(chunk))
#             else:
#                 final_chunks.append(chunk)
        
#         return final_chunks

#     def fallback_word_chunking(self, text: str) -> List[str]:
#         """Fallback word-based chunking for edge cases."""
#         words = text.split()
#         chunks = []
#         current_chunk = ""
#         current_tokens = 0
        
#         for word in words:
#             word_tokens = self.count_tokens(word + " ")
#             if current_tokens + word_tokens > self.max_chunk_size and current_chunk:
#                 chunks.append(current_chunk.strip())
#                 current_chunk = word + " "
#                 current_tokens = word_tokens
#             else:
#                 current_chunk += word + " "
#                 current_tokens += word_tokens
        
#         if current_chunk.strip():
#             chunks.append(current_chunk.strip())
        
#         return chunks

#     def create_chunks(self, input_text: str) -> List[str]:
#         """
#         Split a large text file into chunks based on semantic boundaries and token size.
        
#         Args:
#             input_text: Input text string

#         Returns:
#             List of text chunks
#         """
#         try:
#             text = input_text
            
#             token_count = self.count_tokens(text)
#             logger.info(f"Document contains {token_count} tokens")
            
#             if token_count <= self.max_chunk_size:
#                 logger.info("Document size is within limit. No chunking required.")
#                 return [text]
            
#             logger.info(f"Document exceeds token limit of {self.max_chunk_size}. Splitting into semantic chunks...")
            
#             chunks = self.chunk_text(text)
#             logger.info(f"Split document into {len(chunks)} chunks using semantic boundaries")
            
#             # logger.info chunk statistics
#             for i, chunk in enumerate(chunks):
#                 chunk_tokens = self.count_tokens(chunk)
#                 logger.info(f"Chunk {i+1}: {chunk_tokens} tokens")
            
#             return chunks
#         except Exception as e:
#             logger.error(f"Error in chunk creation: {str(e)}")
#             # Fallback to simple chunking
#             return self.fallback_word_chunking(text)


#SECOND ONE (FOR DEVLOPER), REQUIRES LOW RAM FOR TESTING ONLY FOR PROPOSAL EVAL

from typing import List
import tiktoken
import re
import json
import numpy as np
from logger_config import get_logger

# Removed module-level logger: logger = get_logger(__file__)


class ChunkProcessor:
    def __init__(self, max_chunk_size: int = 1024, model_name: str = 'nomic-ai/nomic-embed-text-v1', logger_instance=None):
        self.logger = logger_instance if logger_instance is not None else get_logger(__name__) # Use passed logger or default
        self.max_chunk_size = max_chunk_size
        # Remove sentence transformer model to save RAM
        self.logger.info("Using simple chunking mode - no semantic analysis") # Use self.logger

    def count_tokens(self, text: str, model_name: str = "gpt-4") -> int:
        """Count the number of tokens in a text string."""
        try:
            # Use tiktoken for OpenAI models
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except Exception as e:
            self.logger.warning(f"Error using tiktoken: {str(e)}") # Use self.logger
            # Fall back to rough estimate
            return len(text.split())

    def simple_chunk_text(self, text: str) -> List[str]:
        """Simple word-based chunking without semantic analysis."""
        if not text:
            return []
        
        # First check if chunking is needed
        total_tokens = self.count_tokens(text)
        if total_tokens <= self.max_chunk_size:
            return [text]
        
        # Split text into words
        words = text.split()
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for word in words:
            # Add space before word (except for first word)
            word_with_space = f" {word}" if current_chunk else word
            word_tokens = self.count_tokens(word_with_space)
            
            # Check if adding this word would exceed the limit
            if current_tokens + word_tokens > self.max_chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(current_chunk.strip())
                current_chunk = word
                current_tokens = self.count_tokens(word)
            else:
                # Add word to current chunk
                current_chunk += word_with_space
                current_tokens += word_tokens
        
        # Add remaining content as final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def create_chunks(self, input_text: str) -> List[str]:
        """
        Split a large text file into chunks using simple word-based chunking.
        
        Args:
            input_text: Input text string

        Returns:
            List of text chunks
        """
        try:
            text = input_text
            
            token_count = self.count_tokens(text)
            self.logger.info(f"Document contains {token_count} tokens") # Use self.logger
            
            if token_count <= self.max_chunk_size:
                self.logger.info("Document size is within limit. No chunking required.") # Use self.logger
                return [text]
            
            self.logger.info(f"Document exceeds token limit of {self.max_chunk_size}. Splitting into simple chunks...") # Use self.logger
            
            chunks = self.simple_chunk_text(text)
            self.logger.info(f"Split document into {len(chunks)} chunks using simple word-based chunking") # Use self.logger
            
            # Log chunk statistics
            for i, chunk in enumerate(chunks):
                chunk_tokens = self.count_tokens(chunk)
                self.logger.info(f"Chunk {i+1}: {chunk_tokens} tokens") # Use self.logger
            
            return chunks
        except Exception as e:
            self.logger.error(f"Error in chunk creation: {str(e)}") # Use self.logger
            # Fallback to even simpler chunking
            return self.fallback_word_chunking(text)

    def fallback_word_chunking(self, text: str) -> List[str]:
        """Fallback word-based chunking for edge cases."""
        words = text.split()
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for word in words:
            word_tokens = self.count_tokens(word + " ")
            if current_tokens + word_tokens > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word + " "
                current_tokens = word_tokens
            else:
                current_chunk += word + " "
                current_tokens += word_tokens
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


if __name__ == "__main__":
    chunk_processor = ChunkProcessor(logger_instance=get_logger(__name__)) # Pass module logger for example
    text ="""Order Index
Table of Contents
|                                                                                     |   1. |
|-------------------------------------------------------------------------------------|-----|
| Overview of the Kingdom                                                             | 4   |
| Overview of Vision 2030                                                             | 4   |
| About the Ministry                                                                  | 5   |
| Introduction                                                                        | 5   |
| Terminology                                                                         | 6   |
| Intellectual Property and Information Confidentiality                               | 6   |
| Disclaimer                                                                          | 7   |
| Responsible Authority                                                               | 7   |
| Considerations Before Preparing the Bid                                             | 7   |
| Considerations When Preparing and Submitting Offers                                 | 9   |
| General Terms and Conditions                                                        | 10  |
| Offer Management                                                                    | 12  |
| Scope of Work                                                                       | 13  |
| Project Objectives                                                                  | 14  |
| Project Duration                                                                    | 14  |
| Rewards Table                                                                       | 14  |
| Evaluation Criteria                                                                 | 14  |
Table (1): Key Dates, Deadlines, and Contact Information
| Event                                           | Day and Date   |
|-------------------------------------------------|----------------|
| Date of Issuance of Request for Quotations       | 2022-02-16     |
| Date of Sending the Non-Disclosure Agreement     | 2022-02-19     |
| Last Date for Receiving Inquiries                | 2022-02-19     |
| Last Date for Submitting Offers (4:00 PM)        | 2022-02-23     |
Email: tendering@moc.gov.sa
### Graphics:
- Located at the top left corner:
- "VISION 2030" logo with both English and Arabic text: "رؤية المملكة العربية السعودية 2030"
- Located at the top right corner:"""
    chunks = chunk_processor.create_chunks(text)

    for idx, chunk in enumerate(chunks):
        self.logger.info("\n") # Use self.logger
        self.logger.info(f"-------------------chunk {idx+1}----------------------------") # Use self.logger
        self.logger.info("\n") # Use self.logger
        self.logger.info(chunk) # Use self.logger
        self.logger.info("\n") # Use self.logger
        self.logger.info("\n") # Use self.logger

    self.logger.info(f"\n Created {len(chunks)} chunks") # Use self.logger