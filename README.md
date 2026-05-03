# University RAG Assistant

A Retrieval-Augmented Generation (RAG) system for university academic policy Q&A.
Ask questions in natural language and get accurate, cited answers from the knowledge base.

## Live Demo
> Deployed on Streamlit Cloud — [Add your URL here after deployment]

---

## System Architecture

```
              ┌───────────────────────────────┐
              │       User Input              │
              │ Streamlit Chat Interface      │
              │ (or CLI: python app.py)       │
              └─────────────┬─────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │   Knowledge Base Loading       │
              │ - Load MD files & CSV rows     │
              │ - Metadata added for citations │
              │ - 38 documents loaded          │
              └─────────────┬─────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │   Chunking                     │
              │ - 700 tokens, 100 overlap      │
              │ - Preserves context, avoids    │
              │   splitting rules              │
              │ - 46 chunks created            │
              └─────────────┬─────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │   Embeddings                   │
              │ - HuggingFace all-MiniLM-L6-v2│
              │ - Lightweight, CPU-friendly    │
              │ - Strong semantic representation│
              └─────────────┬─────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │   Vector Store (Chroma)        │
              │ - Persisted locally            │
              │ - Stores chunk embeddings      │
              │ - Auto-built on first run      │
              └─────────────┬─────────────────┘
                            │
                            ▼
  ┌───────────────────────────────────────────────┐
  │ RAG Question-Answering                        │
  │------------------------------------------------│
  │ **Streamlit Interface (streamlit_app.py)**    │
  │ - Full chat UI with conversation history      │
  │ - Auto-builds vectorstore on first deploy     │
  │ - Deployed on Streamlit Cloud                 │
  │                                               │
  │ **CLI Interface (app.py)**                    │
  │ - Terminal-based Q&A loop                     │
  │ - Same RAG chain as Streamlit version         │
  │                                               │
  │ **LLM: Groq API (LLaMA 3.3 70B)**            │
  │ - Fast cloud inference, no local RAM needed   │
  │ - Free tier sufficient for demo usage         │
  │ - Replaced local LLaMA (RAM limits) and       │
  │   Gemini (model deprecation issues)           │
  └─────────────┬─────────────────────────────────┘
                            │
                            ▼
              ┌───────────────────────────────┐
              │   Answer + Citations           │
              │ - Only from retrieved context  │
              │ - Metadata: filename/CSV row   │
              │ - Rule IDs cited when relevant │
              │ - Fallback: "I don't know"     │
              └───────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| RAG Framework | LangChain (LCEL) |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| LLM | LLaMA 3.3 70B via Groq API |
| Web Interface | Streamlit |
| Document Types | Markdown + CSV |

---

## Features

- Natural language Q&A over university policy documents
- Source citations for every answer (filename + topic)
- Rule ID references from structured CSV knowledge base
- Conversation history maintained across questions
- "I don't know" fallback when context is insufficient
- Auto-builds vectorstore on first deployment — no manual setup

---

## Knowledge Base

| File | Contents |
|---|---|
| `handbook.md` | Attendance, grading, academic standing, graduation, integrity, withdrawal, appeals, leave policies |
| `operations.md` | Credit system, probation monitoring, suspension reinstatement, internship, exam eligibility, misconduct review |
| `rules.csv` | Structured rule database with unique Rule IDs (ATT, GPA, GRD, INT, WDL, APL, LV, DUR, CRD, PRO, SUS, EXM, INTN, REC) |

---

## Project Structure

```
university-rag/
├── streamlit_app.py      # Streamlit chat interface + RAG chain
├── app.py                # CLI version of RAG assistant
├── ingest.py             # Document loading, chunking, embedding pipeline
├── requirements.txt      # Python dependencies
└── knowledge_base/
    ├── handbook.md       # Academic policies
    ├── operations.md     # Operational guidelines
    └── rules.csv         # Structured rules with IDs
```

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/university-rag.git
cd university-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key (free at console.groq.com)
export GROQ_API_KEY="your_key_here"

# 4. Build the vectorstore
python ingest.py

# 5. Run Streamlit interface
streamlit run streamlit_app.py

# OR run CLI version
python app.py
```

---

## Deployment

Deployed on **Streamlit Cloud** with vectorstore built automatically on first run.

To deploy your own instance:
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `GROQ_API_KEY` in Streamlit secrets
5. Deploy — vectorstore builds automatically

---

## Development Notes

**LLM selection:** Initially attempted local LLaMA 3.2-1B (GGUF) — failed due to RAM/CPU constraints on development machine. Attempted Google Gemini API — failed due to model deprecation (`gemini-1.5-flash-latest` returned 404). Final solution uses Groq API with LLaMA 3.3 70B — faster than local inference, free tier available, no hardware constraints.

**Chunking strategy:** 700 token chunks with 100 token overlap chosen to preserve rule context while avoiding splitting related policy clauses across chunks.

**Retrieval:** k=5 chunks retrieved per query to improve recall on broad questions that span multiple policy areas.

**Citations:** Rule IDs embedded directly in document content during ingestion so the LLM can surface them in answers when relevant.
