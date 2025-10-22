# app.py
import os
import streamlit as st
from retrieve import Retriever
from rag_chatbot import ask_gemini  # reuse your existing function

# Initialize Retriever once
retriever = Retriever()

st.set_page_config(page_title="RAG Gemini Chatbot", layout="wide")
st.title("RAG Gemini Chatbot")

st.markdown(
    "Enter a natural language question about your database, "
    "and the model will retrieve relevant context, generate SQL, and answer using Gemini."
)

# User input
query = st.text_input("Your question:")

if query:
    with st.spinner("Processing..."):
        # Step 1: Retrieve context
        response_data = retriever.get_response(query)
        generated_sql = response_data.get("generated_sql", "[No SQL generated]")
        contexts = response_data.get("contexts", [])

        # Safely extract text
        top_chunks_text = [c['chunk']['text'] for c in contexts if 'chunk' in c and 'text' in c['chunk']]
        context_text = "\n".join(top_chunks_text)

        # Step 2: Ask Gemini
        gemini_answer = ask_gemini(
            f"Use the following context to answer the question:\n{context_text}\nQuestion: {query}"
        )

    # Display results
    st.subheader("Retrieved Contexts")
    for i, text in enumerate(top_chunks_text, 1):
        st.markdown(f"**Rank {i}:** {text[:200]}{'...' if len(text) > 200 else ''}")

    st.subheader("Generated SQL")
    st.code(generated_sql, language="sql")

    st.subheader("Gemini Answer")
    st.markdown(gemini_answer)
