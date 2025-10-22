"""
retrieve_chunks.py

Example script to test FAISS-based retrieval of text chunks using
SentenceTransformers embeddings and a FAISS index.
"""

import pickle
import faiss
from sentence_transformers import SentenceTransformer


def retrieve_chunks(query_text, faiss_index, chunk_data, embedding_model, top_k=5):
    """
    Retrieve the most relevant text chunks for a query using FAISS and embeddings.

    Args:
        query_text (str): The natural language query.
        faiss_index (faiss.Index): FAISS index containing embeddings.
        chunk_data (List[dict]): Original chunks with 'text' field.
        embedding_model (SentenceTransformer): Pretrained embedding model.
        top_k (int, optional): Number of top chunks to retrieve. Defaults to 5.

    Returns:
        List[str]: Top-k most relevant text chunks.
    """
    # Encode the query
    query_embedding = embedding_model.encode(
        [query_text], convert_to_numpy=True
    ).astype('float32')

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search the FAISS index
    _, indices = faiss_index.search(query_embedding, top_k)

    # Fetch corresponding text chunks
    results = [chunk_data[i]['text'] for i in indices[0]]
    return results


def main():
    """Main function to test retrieval flow."""
    # Load the FAISS index
    faiss_index = faiss.read_index("embeddings.faiss")

    # Load the chunks
    chunk_data = pickle.load(open("chunks.pkl", "rb"))

    # Initialize embedding model
    embedding_model = SentenceTransformer('all-mpnet-base-v2')

    # Example query
    query_text = "How do I insert a new customer into the database?"

    # Retrieve top chunks
    top_chunks = retrieve_chunks(
        query_text, faiss_index, chunk_data, embedding_model, top_k=5
    )

    # Print results
    print("Top retrieved chunks:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"{i}. {chunk}\n")


if __name__ == "__main__":
    main()
