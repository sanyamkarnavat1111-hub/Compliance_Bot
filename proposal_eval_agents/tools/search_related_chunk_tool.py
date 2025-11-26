from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict
import os
import json

from embedding_generation import EmbeddingGeneration
from embedding_management import EmbeddingManager

class SearchRelatedChunk(BaseTool):
    name: str = "SearchRelatedChunk"
    description: str = (
        "Retrieves the top 8 semantically relevant chunks from proposal PDFs based on a query string. "
    )
    
    embedding_manager: EmbeddingManager = Field(default_factory=lambda: EmbeddingManager(embedding_dim=768, db_path="proposal_eval_embedding_db"))
    embedding_generation: EmbeddingGeneration = Field(default_factory=EmbeddingGeneration)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, query: str) -> List[Dict]:
        """Search for semantically relevant chunks based on the input query."""

        print('\n')
        print('\n')
        print('\n')
        print('---------------------------------')
        print('the search related chunk tool is running')
        print(query)
        print('---------------------------------')
        print('\n')
        print('\n')
        print('\n')

        # Generate embedding for the query
        query_embedding = self.embedding_generation.generate_embeddings([query])[0]
        
        # Convert tensor to numpy array and perform search
        query_embedding_np = query_embedding.detach().numpy()
        results = self.embedding_manager.search(query_embedding_np)
        
        # Format results
        formatted_results = []
        for result in results:
            #  formatted_results.append({
            #     'score': result['score'],
            #     'id': result['id'],
            #     'text': result['text']
            # })
            formatted_results.append(result['text'])   
        return formatted_results

        # return ["my", "name", "is", "neel", "shah"]

if __name__ == "__main__":
    tool = SearchRelatedChunk()
    print(tool._run("What is the scope of work?"))