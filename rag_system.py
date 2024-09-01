import requests
from models import ModelNames
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from notebooks.utils import initialize_qdrant, CodeChunk, get_embeddings

MODEL = ModelNames.DEEPSEEK_CODER_V2.value


class RAGSystem:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        codestral_url: str = "http://localhost:1234/v1",
        query_understanding_model: str = MODEL,
        reranking_model: str = MODEL,
    ):
        self.qdrant_client = qdrant_client
        self.codestral_url = codestral_url
        self.query_understanding_model = query_understanding_model
        self.reranking_model = reranking_model

    def query_understanding(self, query: str) -> str:
        """
        Process the user's query using the local Codestral model to extract key information.

        Parameters:
        -----------
        query : str
                        The user's original query.

        Returns:
        --------
        str
                        The processed query with extracted key information.
        """
        response = requests.post(
            f"{self.codestral_url}/completions",
            json={
                "model": self.query_understanding_model,
                "prompt": f"Extract key information from this query: {query}\nKey information:",
                "max_tokens": 100,
                "temperature": 0.3,
            },
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            raise Exception(f"Error in query understanding: {response.text}")

    def initial_retrieval(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant code chunks from Qdrant based on the query.

        Parameters:
        -----------
        query : str
                        The processed query.
        top_k : int
                        The number of top results to retrieve.

        Returns:
        --------
        List[Dict[str, Any]]
                        A list of dictionaries containing the retrieved code chunks and their metadata.
        """
        query_vector = get_embeddings(
            [
                CodeChunk(
                    content=query,
                    chunk_type="query",
                    start_byte=0,
                    end_byte=0,
                    start_point=(0, 0),
                    end_point=(0, 0),
                    file_path="",
                )
            ]
        )[0]

        search_result = self.qdrant_client.search(
            collection_name="IntoTheDeep",
            query_vector=query_vector.tolist(),
            limit=top_k,
        )

        return [
            {
                "content": result.payload["content"],
                "chunk_type": result.payload["chunk_type"],
                "file_path": result.payload["file_path"],
                "score": result.score,
            }
            for result in search_result
        ]

    def re_rank(
        self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank the retrieved chunks using the local Codestral model.

        Parameters:
        -----------
        query : str
                        The original query.
        chunks : List[Dict[str, Any]]
                        The list of retrieved code chunks.
        top_k : int
                        The number of top results to return after re-ranking.

        Returns:
        --------
        List[Dict[str, Any]]
                        A list of re-ranked code chunks.
        """
        reranked_chunks = []
        for chunk in chunks:
            prompt = f"Query: {query}\n\nCode Chunk:\n{chunk['content']}\n\nRate the relevance of this code chunk to the query on a scale of 0 to 10:"
            response = requests.post(
                f"{self.codestral_url}/completions",
                json={
                    "model": self.reranking_model,
                    "prompt": prompt,
                    "max_tokens": 5,
                    "temperature": 0.3,
                },
            )
            if response.status_code == 200:
                relevance_score = float(response.json()["choices"][0]["text"].strip())
                chunk["relevance_score"] = relevance_score
                reranked_chunks.append(chunk)
            else:
                raise Exception(f"Error in re-ranking: {response.text}")

        return sorted(
            reranked_chunks, key=lambda x: x["relevance_score"], reverse=True
        )[:top_k]

    def prepare_context(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare the context by combining the most relevant chunks.

        Parameters:
        -----------
        query : str
                        The original query.
        chunks : List[Dict[str, Any]]
                        The list of re-ranked code chunks.

        Returns:
        --------
        str
                        The prepared context combining the most relevant chunks.
        """
        context = f"Query: {query}\n\nRelevant Code Chunks:\n\n"
        for i, chunk in enumerate(chunks, 1):
            context += f"Chunk {i} (Relevance: {chunk['relevance_score']:.2f}):\n"
            context += f"File: {chunk['file_path']}\n"
            context += f"Type: {chunk['chunk_type']}\n"
            context += f"Content:\n{chunk['content']}\n\n"
        return context

    def process_query(self, query: str) -> str:
        """
        Process a user query through the RAG system.

        Parameters:
        -----------
        query : str
                        The user's original query.

        Returns:
        --------
        str
                        The prepared context for the Claude API.
        """
        processed_query = self.query_understanding(query)
        initial_chunks = self.initial_retrieval(processed_query)
        reranked_chunks = self.re_rank(query, initial_chunks)
        context = self.prepare_context(query, reranked_chunks)
        return context


if __name__ == "__main__":
    qdrant_client = initialize_qdrant()  # You need to implement this function
    rag_system = RAGSystem(qdrant_client)
    context = rag_system.process_query("How does the embedding generation work?")
