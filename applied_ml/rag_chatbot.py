"""
rag_chatbot.py — ML Research Paper RAG Assistant
=================================================
A Retrieval-Augmented Generation (RAG) chatbot built with:
  - Streamlit      : web UI
  - LangChain      : RAG pipeline orchestration
  - HuggingFace    : local sentence embeddings (no API key needed)
  - ChromaDB       : local vector database for document retrieval
  - Ollama + Llama2: fully local LLM inference (no API key needed)

How RAG works here:
  1. PDFs are loaded and split into small overlapping text chunks
  2. Each chunk is converted into a vector (embedding) and stored in ChromaDB
  3. When a user asks a question, the top 3 most similar chunks are retrieved
  4. Those chunks are injected into the prompt as context for Llama 2
  5. Llama 2 answers using ONLY that context — no hallucination
"""

import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate


# ============================================================
# STEP 1 — DATA INGESTION & VECTOR STORE
# ============================================================

@st.cache_resource
def build_vector_database():
    """
    Loads PDFs, splits them into chunks, embeds them, and stores in ChromaDB.

    @st.cache_resource ensures this runs only ONCE per session — not on every
    user interaction. Without this, the database would be rebuilt on every query,
    making the app extremely slow.
    """

    # Load every PDF inside the ./data folder.
    # PyPDFDirectoryLoader automatically handles multi-page PDFs and
    # attaches metadata (filename, page number) to each document.
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()

    # Split documents into smaller overlapping chunks.
    # chunk_size=1000 : each chunk is at most 1000 characters
    # chunk_overlap=200: consecutive chunks share 200 characters of context
    #                    so sentences at chunk boundaries aren't cut off mid-thought
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Convert each text chunk into a numerical vector (embedding).
    # all-MiniLM-L6-v2 is a lightweight but effective sentence embedding model
    # that runs entirely locally — no HuggingFace API key required.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Store all embedded chunks in a local ChromaDB vector database.
    # persist_directory saves the database to disk so it survives app restarts
    # without needing to re-embed all documents from scratch.
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectordb


# ============================================================
# STEP 2 — STREAMLIT UI SETUP
# ============================================================

st.set_page_config(page_title="ML Research RAG Assistant", layout="centered")
st.title("📚 ML Research Paper RAG Assistant")
st.write("Ask me questions about Transformers, BERT, GPT, RAG, LoRA, and Llama!")

# Build (or load from cache) the vector database on startup
with st.spinner("Loading papers and building database... (first run takes a moment)"):
    vectordb = build_vector_database()

# Connect to the locally running Ollama server which serves Llama 2.
# Ollama must be running in the background: `ollama serve` + `ollama pull llama2`
llm = Ollama(model="llama2")


# ============================================================
# STEP 3 — PROMPT TEMPLATE
# ============================================================

# Llama 2's instruction-tuned variant expects a specific chat format:
#   [INST] <<SYS>> system message <</SYS>> user message [/INST]
# Using this exact format is important — without it, the model may ignore
# the system instructions and answer from its own training data (hallucinate).
llama_prompt_template = """<s>[INST] <<SYS>>
You are an AI assistant helping a student understand Machine Learning research papers.
Answer the user's question using ONLY the provided pieces of retrieved context below.

Strict Guardrails:
1. If the context contains the answer, explain it clearly in simple, student-friendly terms.
2. If you genuinely cannot find the answer in the context below, simply say: "I cannot answer this based on the provided papers."
3. Do not make up facts, do not hallucinate, and do not use any external knowledge.
<</SYS>>

Context:
{context}

Question: {input} [/INST]"""

prompt = PromptTemplate.from_template(llama_prompt_template)

# Configure the retriever to fetch the top 3 most semantically similar chunks.
# k=3 balances providing enough context without overloading the prompt.
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


# ============================================================
# STEP 4 — INTERACTIVE CHAT INTERFACE
# ============================================================

user_query = st.text_input("Enter your question:")

if user_query:
    with st.spinner("Thinking..."):

        # A: Retrieve the 3 most relevant chunks from ChromaDB using
        #    cosine similarity between the query embedding and stored embeddings
        retrieved_docs = retriever.invoke(user_query)

        # B: Concatenate the retrieved chunk texts into one context string
        #    to inject into the prompt
        context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # C: Fill in the prompt template with the retrieved context and user query
        formatted_prompt = prompt.format(context=context_str, input=user_query)

        # D: Send the fully formatted prompt to the local Llama 2 model via Ollama
        answer = llm.invoke(formatted_prompt)

    # Display the generated answer
    st.subheader("Answer:")
    st.write(answer)

    # Display which papers and pages the answer was sourced from.
    # This makes the RAG pipeline transparent and verifiable.
    st.subheader("Source Papers Used:")
    sources = set()
    for doc in retrieved_docs:
        source_file = os.path.basename(doc.metadata['source'])
        page_num = doc.metadata.get('page', 'Unknown')

        # PyPDFLoader indexes pages from 0; add 1 for human-readable page numbers
        if isinstance(page_num, int):
            page_num = page_num + 1

        sources.add(f"{source_file} (Page {page_num})")

    for source in sorted(sources):
        st.markdown(f"- `{source}`")
