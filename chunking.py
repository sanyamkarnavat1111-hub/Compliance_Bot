from logger import custom_logger as llog
import tiktoken
from typing import List
import re

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        llog("ReAct_Agent", f"Error counting tokens: {str(e)}", "reactic_agent.log")
        return len(text.split())  # Fallback approximation


def chunk_text(text: str, tokens_per_chunk: int = 600) -> List[str]:
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
            llog("ReAct_Agent", f"Chunk too large ({chunk_tokens} tokens), applying word-based splitting", "reactic_agent.log")
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
            
    llog("ReAct_Agent", f"Text chunked into {len(final_chunks)} chunks", "reactic_agent.log")
    return final_chunks    