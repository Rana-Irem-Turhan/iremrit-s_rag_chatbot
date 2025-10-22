"""
RAG Gemini Chatbot integrating Retriever and SQL generation.
"""

import os
from retrieve import Retriever  # your retrieve.py module

# Try to import google-generativeai
try:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def ask_gemini(prompt: str) -> str:
    """
    Attempt to send a prompt to Gemini API and return response.
    Falls back gracefully if SDK is old or missing.
    """
    if not GENAI_AVAILABLE:
        return "[Gemini not available]"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2  # adjust as needed
            )
        )
        return response.text
    except Exception as e:
        return f"[Error communicating with Gemini API: {e}]"


def main():
    """Start interactive RAG chatbot session."""
    retriever = Retriever()
    print("RAG Gemini Chatbot (type 'exit' or 'quit' to stop)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not query:
            continue

        # Step 1: Get response from Retriever (includes SQL)
        response_data = retriever.get_response(query)
        generated_sql = response_data.get("generated_sql", "[No SQL generated]")
        contexts = response_data.get("contexts", [])

        # Step 2: Prepare context text for Gemini
        top_chunks_text = [c['chunk']['text'] for c in contexts]
        context_text = "\n".join(top_chunks_text)

        # Step 3: Ask Gemini
        prompt = (
            f"Use the following context to answer the question:\n"
            f"{context_text}\nQuestion: {query}"
        )
        gemini_answer = ask_gemini(prompt)

        # Step 4: Display both Gemini answer and SQL
        print("\n" + "-" * 50)
        print("Gemini:", gemini_answer)
        print("Generated SQL:", generated_sql)
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
