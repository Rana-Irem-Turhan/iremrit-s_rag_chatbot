"""app.py
Main entry point for the SQL RAG chatbot.
This script combines retrieval, SQL generation, and an optional Streamlit UI."""
import os
import sys
import re
import pickle
from typing import Tuple, List, Dict
import numpy as np
import faiss
import sqlparse
from sqlparse.exceptions import SQLParseError
from sentence_transformers import SentenceTransformer
import streamlit as st
from dotenv import load_dotenv

"""Load environment variables (API KEYS etc.)
I use dotenv so I can keep my Gemini API key private and not hard-coded."""

load_dotenv()

# ===== validate.py =====
""" A small helper section for SQL validation.
Prevents any destructive queries (like DROP, DELETE) from being executed accidentally."""
_FORBIDDEN = re.compile(
    r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|EXEC|MERGE)\b',
    re.IGNORECASE
)
   
    """
    Checking if the SQL is safe and syntactically valid.
    Returns (is_valid, message, formatted_sql)
    """
def validate_sql(query: str) -> Tuple[bool, str, str]:
    if not query or not query.strip():
        return False, "Empty query", ""
    if _FORBIDDEN.search(query):
        return False, "Forbidden or potentially destructive statement detected", ""
    try:
        formatted = sqlparse.format(query, reindent=True, keyword_case='upper')
        parsed = sqlparse.parse(formatted)
        if not parsed:
            return False, "Unable to parse SQL", formatted
        return True, "OK", formatted
    except SQLParseError as e:
        return False, f"SQL parse/format error: {e}", ""


# ===== generate.py =====
""" This section I created to handle communication with the Gemini API (if available ofc.)
 and constructs the prompt for SQL generation accordingly."""
try:
    from google import genai
    from google.genai import types 
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class SQLGenerator:
    """ This class handles SQL generation using Gemini + validation."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")

        if GENAI_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def _build_prompt(self, user_question: str, contexts: List[Dict], max_context_chars: int = 3000) -> str:
        """
        Builds the input prompt for Gemini.
        Includes the most relevant schemas and examples from the retriever.
        """
        lines = [
            "You are a SQL expert. Given these schemas and examples, generate a single SQL SELECT statement. Output only the SQL query and nothing else.\n"
        ]
        used = 0
        for ctx in contexts:
            text = ctx.get("text", "")
            example = ctx.get("answer", "")
            if used >= max_context_chars:
                break
            remaining = max_context_chars - used
            text_trunc = text[:remaining]
            used += len(text_trunc)
            lines.extend([
                f"Schema: {text_trunc}",
                f"Example: {example}",
                ""
            ])
        lines.append(f"User question: {user_question}")
        lines.append("SQL:")
        return "\n".join(lines)

    def generate_query(self, user_question: str, contexts: List[Dict]) -> Dict:
        """
        Generate SQL from the retrieved context using Gemini as I mentioned earlier this section.
        In the case of failed API it falls back to example answers.
        """
        prompt = self._build_prompt(user_question, contexts)

        sql_text = ""
        try:
            if GENAI_AVAILABLE:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2)
                )
                sql_text = response.text.strip()
                # Since Gemini sometimes wraps code in markdown, I clean that up here.
                if "```sql" in sql_text.lower():
                    sql_text = sql_text.split("```sql")[1].split("```")[0].strip()
                elif "```" in sql_text:
                    sql_text = sql_text.split("```")[1].split("```")[0].strip()
            else:
                 # Fallback: return example SQL if Gemini is not available
                sql_text = contexts[0].get("answer", "") if contexts else ""
        except Exception:
            sql_text = contexts[0].get("answer", "") if contexts else ""
            return {
                "sql": sql_text,
                "valid": False,
                "validation_message": "Gemini error. Fallback used."
            }

        is_valid, message, formatted = validate_sql(sql_text)
        return {
            "sql": formatted if is_valid else sql_text,
            "valid": is_valid,
            "validation_message": message,
        }

def ask_gemini(prompt: str) -> str:
    """
    A simpler helper for getting Gemini responses in natural language (not just SQL).
    Used mostly in the chat or Streamlit interface.
    """
    if not GENAI_AVAILABLE:
        return "[Gemini not available]"
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        return response.text
    except Exception as e:
        return f"[Error communicating with Gemini API: {e}]"


# ===== retrieve.py =====
""" The retriever part where the chatbot finds the most relevant schema chunks based on user question."""

class Retriever:
    """This class is for fetching top-k schema chunks and then generates SQL."""

    def __init__(self, index_path='embeddings.faiss', chunks_path='chunks.pkl', model_name='all-mpnet-base-v2'):
     # Checking if embeddings and chunks exist
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        # Now its time to Load FAISS index and the chunks from disk
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Loading now the same SentenceTransformer model I used for embeddings
        self.model = SentenceTransformer(model_name)
        self.generator = SQLGenerator()

    def get_response(self, query: str, k: int = 3):
        """Retrieving here the relevant context and generate SQL query."""
        contexts = self.get_relevant_chunks(query, k)
        result = self.generator.generate_query(query, [c['chunk'] for c in contexts])
        return {
            "contexts": contexts,
            "generated_sql": result["sql"],
            "valid": result["valid"],
            "validation_message": result["validation_message"]
        }

    def get_relevant_chunks(self, query: str, k: int = 3):
        """Performing semantic search in the FAISS index."""
        query_embedding = np.array(self.model.encode([query])).astype('float32')
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


# ===== CLI / main =====
""" I included a simple command-line interface for local testing
 — helps verify retriever and SQL generation logic before using Streamlit."""

def main():
    retriever = Retriever()
    print("Welcome! Choose mode:\n1. Retriever-only\n2. RAG Gemini Chatbot")
    mode = input("Enter 1 or 2: ").strip()
    if mode not in ["1", "2"]:
        mode = "2"
    print("\nType 'exit' or 'quit' to stop\n")
    k = 3
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue

        #Changing k on the go
        if query.startswith("k="):
            try:
                k = int(query[2:])
                print(f"Number of results set to {k}")
                continue
            except ValueError:
                continue

        if mode == "1":
            # Retriever-only mode:  which just shows retrieved contexts and example SQL
            results = retriever.get_relevant_chunks(query, k=k)
            print(f"\nQuery: {query}\n")
            for r in results:
                chunk = r['chunk']
                print(f"Rank {r['rank']} | score={r['score']:.4f}")
                print(f"Schema: {chunk.get('text')}")
                print(f"SQL: {chunk.get('answer')}\n")
        else:
            # Full RAG pipeline mode on: context retrieval + Gemini SQL generation
            response_data = retriever.get_response(query, k=k)
            generated_sql = response_data.get("generated_sql", "[No SQL generated]")
            contexts = response_data.get("contexts", [])
            top_chunks_text = [c['chunk']['text'] for c in contexts]
            context_text = "\n".join(top_chunks_text)
            # Asking Gemini for a natural language answer too
            gemini_answer = ask_gemini(f"Use the following context to answer the question:\n{context_text}\nQuestion: {query}")
            print("\n" + "-" * 50)
            print("Gemini:", gemini_answer)
            print("Generated SQL:", generated_sql)
            print("-" * 50 + "\n")


# ===== Streamlit App =====
""" A friendly UI so others can interact with the chatbot visually.
Note-> to myslef-> this part i will imporve def"""
def run_streamlit():
    st.set_page_config(page_title="RAG Gemini Chatbot", layout="wide")
    st.title("RAG Gemini Chatbot")

    st.markdown(
        "Enter a natural language question about your database, "
        "and the model will retrieve relevant context, generate SQL, and answer using Gemini."
    )

    retriever = Retriever()

    query = st.text_input("Your question:")
    if query:
        with st.spinner("Processing..."):
            response_data = retriever.get_response(query)
            generated_sql = response_data.get("generated_sql", "[No SQL generated]")
            contexts = response_data.get("contexts", [])
            top_chunks_text = [c['chunk']['text'] for c in contexts if 'chunk' in c and 'text' in c['chunk']]
            context_text = "\n".join(top_chunks_text)
            gemini_answer = ask_gemini(f"Use the following context to answer the question:\n{context_text}\nQuestion: {query}")

        st.subheader("Retrieved Contexts")
        for i, text in enumerate(top_chunks_text, 1):
            st.markdown(f"**Rank {i}:** {text[:200]}{'...' if len(text) > 200 else ''}")

        st.subheader("Generated SQL")
        st.code(generated_sql, language="sql")

        st.subheader("Gemini Answer")
        st.markdown(gemini_answer)

""" 
Finally we have come to the end of my Project thank you for your inspiring mentor tea-time & Webinars.
Her sey icin tesekkur ederim. Ingilizce yazmayi aliskanlik haline getirmisim :)) . 
Sizleri tanimak ve bu projede bulunmak ufkumu acti acikcasi en azindan ben oyle dusunuyorum.
O nedenle Inshallah daha nicelerine diyorum. Gorusmek Uzere!!! 
"""
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        run_streamlit()
    else:
        main()
