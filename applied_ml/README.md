# Applied ML Domain — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about ML research papers using a fully local pipeline — no API keys required.

## What is RAG?

RAG stands for Retrieval-Augmented Generation. Instead of relying purely on an LLM's training data, RAG first **retrieves** relevant text from a document database and feeds it as context to the LLM before generating an answer. This grounds the model's response in real source material and prevents hallucination.

## Pipeline Overview

```
User Question
     ↓
Embed question  →  Search ChromaDB  →  Top 3 relevant chunks
                                              ↓
                                    Inject into Llama 2 prompt
                                              ↓
                                       Generate answer
                                              ↓
                                  Show answer + source pages
```

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| PDF Loading | LangChain PyPDFDirectoryLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector Store | ChromaDB (local) |
| LLM | Llama 2 via Ollama (local) |

## Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
Download Ollama from https://ollama.com, then run:
```bash
ollama pull llama2
ollama serve
```

### 3. Add your research papers
Place your PDF files inside the `data/` folder:
```
chatbot_code/
└── data/
    ├── attention_is_all_you_need.pdf
    ├── bert.pdf
    └── ...
```

### 4. Run the chatbot
```bash
streamlit run rag_chatbot.py
```

Then open `http://localhost:8501` in your browser.

## Papers Used
- Attention Is All You Need (Transformers)
- BERT
- GPT
- RAG (Lewis et al.)
- LoRA
- Llama
