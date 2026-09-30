# Cyber Crime Agent

This project builds a local retrieval-augmented generation (RAG) system for answering questions about Pakistan cyber-crime law and NCCIA-related information using documents stored in the repository.

The code loads a PDF knowledge base, breaks it into searchable chunks, embeds those chunks with a sentence-transformer model, retrieves the most relevant passages for a query, and sends the context to a Groq-hosted language model to answer the question.

## What the project does

- Reads and splits legal knowledge from `data/NCCIA_Knowledge_Base.pdf`
- Creates embeddings using `BAAI/bge-small-en-v1.5`
- Stores vectors in memory or Chroma for similarity search
- Uses a Groq LLM to answer based only on retrieved passages
- Includes two versions of the pipeline:
  - `cyber_agent.py` — a simpler direct RAG chain
  - `Agentic_Cyber_Agent.py` — an agent with a `retriever_tool` that follows a stricter retrieval-first prompt

## Project files

- `cyber_agent.py` — basic RAG implementation using a prompt template
- `Agentic_Cyber_Agent.py` — tool-based agent powered by LangChain and Groq
- `requirements.txt` — project dependencies
- `data/` — local PDF knowledge base files
- `.env` — local environment secrets (not committed)

## Requirements

- Python 3.10+
- A Groq API key
- Local PDF files in `data/`
- Internet access during the first run to download the embedding model and access Groq

## Setup

Create and activate a virtual environment on Windows:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

## Run the project

Run the simpler version:

```powershell
python .\cyber_agent.py
```

Run the agent-based version:

```powershell
python .\Agentic_Cyber_Agent.py
```

The first execution may take time because the embedding model is downloaded automatically and the LLM call is made through Groq.

## Customizing the query

In `cyber_agent.py`, change the query string passed to the RAG chain:

```python
response = rag_chain.invoke("where is cyber crime office gilgit")
```

In `Agentic_Cyber_Agent.py`, change the `query` variable near the bottom:

```python
query = "someone has created a fake account on instagram using my pictures. what should i do and also NCCIA has blocked my account what should i do"
```

## Architecture overview

```text
PDF files in data/
        |
        v
PyPDFLoader + text splitting
        |
        v
Embedding model (BAAI/bge-small-en-v1.5)
        |
        v
Vector similarity search
        |
        v
Groq LLM
        |
        v
Grounded answer based on retrieved document context
```

## Important notes

- The system is designed to answer based on the local knowledge base only.
- It is intended for informational use and should not be treated as legal advice.
- Critical legal or formal matters should be verified through official NCCIA, PECA, or qualified legal sources.
- Local secrets and the virtual environment should not be committed to source control.

## Example use cases

- Asking about cybercrime reporting procedures
- Looking up NCCIA office or enforcement information
- Searching the stored legal material for complaint-related guidance
- Understanding PECA/NCCIA references mentioned in the PDF documents
