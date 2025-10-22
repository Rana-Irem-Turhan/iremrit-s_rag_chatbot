"""
Create embeddings for SQL chunks using SentenceTransformers and FAISS w/ my chossing.
This is a key step: I turn the cleaned SQL Q&A chunks into vector embeddings
so that my chatbot can search and retrieve relevant answers efficiently.
"""
import json
import pickle
import math

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# File paths and constants

DATA_FILE = 'processed_chunks.json'
FAISS_INDEX_FILE = 'embeddings.faiss'
CHUNKS_PKL_FILE = 'chunks.pkl'
BATCH_SIZE = 1024

# Load processed chunks
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    chunks = json.load(f)
"""At this point chunks is a list of dicts: {'text': ..., 'answer': ...}.
These are exactly what my chatbot will later query against"""
# Initialize embedding model
model = SentenceTransformer('all-mpnet-base-v2')  
""" This sounded more accurate but slightly heavier than the other version
I checked. Plus this model produces embeddings that capture semantic meaning, 
not just exact words; Which will be handy for my chatbot."""


# Function to compute embeddings in batches
def compute_embeddings_in_batches(text_list, batch_size=BATCH_SIZE):
    """
    Compute embeddings in batches to avoid memory issues.

    Args:
        text_list (List[str]): List of text strings to embed.
        batch_size (int): Number of texts per batch.

    Returns:
        np.ndarray: Embeddings for all texts.
    """
    embeddings_list = []
    total_batches = math.ceil(len(text_list) / batch_size)
    for i in range(total_batches):
        batch_texts = text_list[i*batch_size: (i+1)*batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings_list.append(batch_embeddings)
        # Stacking all batches vertically to get a single embeddings array
    return np.vstack(embeddings_list)



# Prepare texts
all_texts = [chunk['text'] for chunk in chunks]
"""I am only embedding the 'text' field (context + question),
because answers are stored separately and used after retrieval."""

# Compute embeddings for all chunks (text field) in batches
embeddings = compute_embeddings_in_batches(all_texts, batch_size=BATCH_SIZE)

# Convert to float32 for FAISS
embeddings = embeddings.astype('float32')

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)


# Initialize FAISS index using inner product (cosine after normalization)
dimension = embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatIP(dimension)  # pylint: disable=E1120 


# Add vectors to the index
index.add(embeddings)  # pylint: disable=E1120
""" Now that the FAISS index is ready for fast semantic search. 
Query embeddings can be compared with these vectors for similarity """

# Saving the FAISS index
faiss.write_index(index, "embeddings.faiss")


# Save the chunks separately for later retrieval
with open('chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)
""" Keeping original chunks handy to return the 
answers after retrieving embeddings"""

print(f"Created and saved embeddings for {len(chunks)} chunks")
print(f"Embedding dimension: {dimension}")
