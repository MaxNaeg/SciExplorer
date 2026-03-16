
import numpy as np
from openai import OpenAI


def create_embeddings(
    text_chunks: list[str],
    client: OpenAI,
    model_name: str = "text-embedding-3-small",
    batch_size: int = 1
) -> tuple[np.ndarray, dict]:
    """
    Create embeddings from the provided text chunks using OpenAI, batching requests.
    
    Args:
        text_chunks (list[str]): List of text chunks to be embedded.
        client (OpenAI): An instance of the OpenAI client.
        model_name (str): The embedding model to use.
        batch_size (int): Number of chunks per API request.
        
    Returns:
        tuple[np.ndarray, dict]: Embeddings array and a mapping from index to text chunk.
    """
    embeddings = []
    index_to_text = {}

    for start_idx in range(0, len(text_chunks), batch_size):
        batch = text_chunks[start_idx:start_idx + batch_size]

        response = client.embeddings.create(
            model=model_name,
            input=batch
        )

        # response.data will have embeddings in the same order as `batch`
        for i, item in enumerate(response.data):
            embeddings.append(np.array(item.embedding, dtype=np.float32))
            index_to_text[start_idx + i] = batch[i]

    embeddings_array = np.array(embeddings, dtype=np.float32)
    return embeddings_array, index_to_text