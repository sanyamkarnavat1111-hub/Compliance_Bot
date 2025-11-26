import faiss
import numpy as np
import torch
from typing import List, Dict, Optional, Union
import json
import os
from datetime import datetime
from logger_config import get_logger

# Removed module-level logger: logger = get_logger(__file__)

class EmbeddingManager:
    def __init__(self, embedding_dim: int = 768, top_k_chunks: int=10, db_path: str = "embedding_db", logger_instance=None):
        """
        Initialize the EmbeddingManager.
        
        Args:
            embedding_dim: Dimension of the embeddings
            db_path: Path to store the embedding database
        """
        self.logger = logger_instance if logger_instance is not None else get_logger(__name__) # Use passed logger or default
        self.embedding_dim = embedding_dim
        self.metadata = []
        self.top_k_chunks = top_k_chunks
        
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            assert self.index.d == self.embedding_dim, f"New index dimension mismatch: {self.index.d} != {self.embedding_dim}"
        except Exception as e:
            self.logger.error(f"Error initializing FAISS index: {str(e)}") # Use self.logger
            raise
        
        # Create database directory if it doesn't exist
        try:
            os.makedirs(db_path, exist_ok=True)
            self.db_path = db_path
            self.embeddings_path = os.path.join(db_path, "embeddings.npy")
            self.metadata_path = os.path.join(db_path, "metadata.json")
        except Exception as e:
            self.logger.error(f"Error creating database directory: {str(e)}") # Use self.logger
            raise
        
        # Load existing embeddings and metadata if they exist
        self.load_embeddings()

    def add_embeddings(self, chunks: List[str], embeddings: Union[np.ndarray, torch.Tensor], metadata: Optional[List[Dict]] = None):
        """
        Add new embeddings to the index along with their corresponding chunks.
        
        Args:
            chunks: List of text chunks corresponding to each embedding
            embeddings: Array or tensor of shape (n, embedding_dim)
            metadata: Optional list of metadata dictionaries corresponding to each embedding.
                      If not provided, will generate default chunk-based metadata.
        """
        try:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            if embeddings.shape[1] != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embeddings.shape[1]}")
                
            if len(chunks) != embeddings.shape[0]:
                raise ValueError(f"Number of chunks ({len(chunks)}) must match number of embeddings ({embeddings.shape[0]})")
                
            # Generate default metadata if not provided
            if metadata is None:
                current_count = self.get_embedding_count()
                metadata = [
                    {"id": f"chunk_{i + current_count}", "text": chunks[i]}
                    for i in range(embeddings.shape[0])
                ]
            else:
                # Ensure each metadata entry has the chunk text
                for i, meta in enumerate(metadata):
                    meta['text'] = chunks[i]
            
            # Add to the index
            self.index.add(embeddings.astype('float32'))
            self.metadata.extend(metadata)
            
            # Save embeddings to disk
            self.save_embeddings()
        except Exception as e:
            self.logger.error(f"Error adding embeddings: {str(e)}") # Use self.logger
            raise

    def get_current_index(self) -> faiss.IndexFlatL2:
        """
        Get a fresh copy of the current FAISS index.
        This allows users to perform searches independently without affecting the main index.
        
        Returns:
            A fresh copy of the FAISS index
        """
        try:
            # Create a fresh copy of the index
            new_index = faiss.IndexFlatL2(self.embedding_dim)
            # Add all existing embeddings to the new index
            if self.index.ntotal > 0:
                embeddings = self.index.reconstruct_n(0, self.index.ntotal)
                new_index.add(embeddings)
            return new_index
        except Exception as e:
            self.logger.error(f"Error creating fresh index: {str(e)}") # Use self.logger

    def search(self, query_embedding: np.ndarray, use_fresh_index: bool = True) -> List[Dict]:
        """
        Search for similar embeddings using FAISS.
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,)
            use_fresh_index: If True, creates a fresh copy of the index for this search
            
        Returns:
            List of dictionaries containing metadata and similarity scores
        """
        try:
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Use fresh index for search if requested
            if use_fresh_index:
                index = self.get_current_index()
            else:
                index = self.index
            
            self.logger.info("query_embedding shape: %s", query_embedding.shape) # Use self.logger

            scores, indices = index.search(query_embedding, self.top_k_chunks)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # Skip padding indices
                    meta = self.metadata[idx].copy()
                    meta['score'] = float(score)
                    results.append(meta)
            
            # Sort results by score in descending order
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}") # Use self.logger
            return []  # Return empty results as fallback

    def calculate_similarities(self, query_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between query embeddings and all stored embeddings using matrix multiplication.
        
        Args:
            query_embeddings: Array of shape (num_queries, embedding_dim)
            
        Returns:
            Array of shape (num_queries, num_documents) containing similarity scores
        """
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
            
        query_embeddings = query_embeddings.astype('float32')
        
        # Get all stored embeddings
        if self.index.ntotal == 0:
            return np.zeros((query_embeddings.shape[0], 0))
            
        stored_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
        
        # Calculate similarities using matrix multiplication
        similarities = (query_embeddings @ stored_embeddings.T) * 100
        return similarities

    def get_embedding_count(self) -> int:
        """Get the number of stored embeddings."""
        return self.index.ntotal

    def save_embeddings(self):
        """
        Save current embeddings and metadata to disk.
        """
        try:
            if self.index.ntotal > 0:
                embeddings = self.index.reconstruct_n(0, self.index.ntotal)
                np.save(self.embeddings_path, embeddings)
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.metadata, f)
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}") # Use self.logger
            
    def load_embeddings(self):
        """
        Load embeddings and metadata from disk if they exist.
        """
        try:
            if os.path.exists(self.embeddings_path) and os.path.exists(self.metadata_path):
                embeddings = np.load(self.embeddings_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Verify dimensions match
                if embeddings.shape[1] != self.embedding_dim:
                    raise ValueError(f"Loaded embedding dimension mismatch. Expected {self.embedding_dim}, got {embeddings.shape[1]}")
                
                # Add loaded embeddings to index
                self.index.add(embeddings.astype('float32'))
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}") # Use self.logger

    def clean_embedding_db(self):
        """
        Clean up the embedding database by removing stored files and resetting the index.
        """
        try:
            # Remove stored files
            if os.path.exists(self.embeddings_path):
                os.remove(self.embeddings_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            
            # Reset the index and metadata
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
        except Exception as e:
            self.logger.error(f"Error cleaning embedding database: {str(e)}") # Use self.logger

if __name__ == "__main__":
    try:
        # Example 3: Checking the number of documents present in the database
        manage3 = EmbeddingManager(embedding_dim=768, logger_instance=get_logger(__name__)) # Pass module logger for example
        manage3.logger.info(f"Number of documents in database: {manage3.get_embedding_count()}") # Use self.logger
    except Exception as e:
        manage3.logger.error(f"Error in main execution: {str(e)}") # Use self.logger