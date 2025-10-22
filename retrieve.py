"""Retriever module for fetching relevant SQL schema chunks using FAISS and SentenceTransformer."""

import os
import sys
import pickle
import argparse

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from generate import SQLGenerator


class Retriever:
    """Retriever class for fetching top-k schema chunks and generating SQL."""

    def __init__(
        self,
        index_path='embeddings.faiss',
        chunks_path='chunks.pkl',
        model_name='all-mpnet-base-v2'
    ):
        """Initialize retriever with FAISS index, chunks, and embedding model."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        self.index = faiss.read_index(index_path)

        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        self.model = SentenceTransformer(model_name)
        self.generator = SQLGenerator()

    def get_response(self, query: str, k: int = 3):
        """Return SQL generation result with relevant schema contexts."""
        contexts = self.get_relevant_chunks(query, k)
        result = self.generator.generate_query(
            query, [c['chunk'] for c in contexts]
        )

        return {
            "contexts": contexts,
            "generated_sql": result["sql"],
            "valid": result["valid"],
            "validation_message": result["validation_message"]
        }

    def get_relevant_chunks(self, query, k=3):
        """Retrieve top-k schema chunks most relevant to the query."""
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, k)
        results = []

        for rank, idx in enumerate(indices[0]):
            results.append({
                'rank': rank + 1,
                'index': int(idx),
                'score': float(distances[0][rank]),
                'chunk': self.chunks[idx]
            })

        return results


def interactive_mode(retriever):
    """Start interactive mode for manual SQL query generation."""
    print("\nSQL Query Assistant - Interactive Mode")
    print("Enter 'quit' or 'exit' to end the session")
    print("Enter 'k=N' to change the number of results (default k=3)\n")

    k = 3
    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() in ['quit', 'exit']:
            break

        if query.startswith('k='):
            try:
                k = int(query[2:])
                print(f"Number of results set to {k}")
                continue
            except ValueError:
                print("Invalid k value. Using previous value.")
                continue

        if not query:
            continue

        results = retriever.get_relevant_chunks(query, k=k)
        print(f"\nQuery: {query}\n")

        for r in results:
            chunk = r['chunk']
            print(f"Rank {r['rank']} | score={r['score']:.4f}")
            print(f"Schema: {chunk.get('text')}")
            print(f"SQL: {chunk.get('answer')}\n")


def main():
    """Command-line entry point for retriever."""
    parser = argparse.ArgumentParser(
        description="Retrieve top-k context chunks for a query"
    )
    parser.add_argument(
        '-q', '--query',
        help='Query text (optional, if not provided enters interactive mode)'
    )
    parser.add_argument(
        '-k', '--k', type=int, default=3,
        help='Number of results to return'
    )
    args = parser.parse_args()

    try:
        retriever = Retriever()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if args.query:
        results = retriever.get_relevant_chunks(args.query, k=args.k)
        print(f"\nQuery: {args.query}\n")

        for r in results:
            chunk = r['chunk']
            print(f"Rank {r['rank']} | score={r['score']:.4f}")
            print(f"Schema: {chunk.get('text')}")
            print(f"SQL: {chunk.get('answer')}\n")
    else:
        interactive_mode(retriever)


if __name__ == "__main__":
    main()
