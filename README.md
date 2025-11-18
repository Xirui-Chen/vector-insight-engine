# Vector Insight Engine

Vector Insight Engine is a practical research and analysis tool that turns unstructured documents into searchable insights.  

It combines Google Gemini for reasoning and Qdrant Cloud for vector search to deliver evidence based answers that analysts and data scientists can trust.

This project simulates a real production workflow used in many AI teams, and is designed to be easy to run, inspect, and extend.

---

## 🚀 Live Demo
Try the interactive app here:  
➡️ https://vector-insight-engine.streamlit.app

## 🎥 Demo Video
Watch the walkthrough presentation:  
➡️ https://drive.google.com/file/d/187vfLKc9OPogECo1U1WLDdzJ3DAYkMz1/view?usp=sharing

## 📊 Slide Deck
Full slide presentation (PDF):  
➡️ https://drive.google.com/file/d/1ecfZCPXDcVx3oVqHrWAqq5CTGEURlVb5/view?usp=sharing

---

## Why this project matters

Analysts, data scientists, and product teams handle large amounts of unstructured text: research notes, specifications, reports, client documents.  

The common problems:

- Key facts are hidden inside long paragraphs  
- Keyword search misses relevant context  
- Most AI summarizers do not show where their answers come from  
- Collaboration becomes slow when information is buried  

Vector Insight Engine solves this with a transparent, retrieval augmented pipeline:

- Every answer is backed by real evidence  
- All context snippets are displayed with similarity scores  
- Documents can be grouped by project or client  
- Text and PDFs can be ingested instantly  

---

## Features

### Document ingestion
- Paste raw text or upload TXT / PDF files  
- Automatic chunking with configurable size and overlap  
- Embedding via Google `text-embedding-004`  
- Stored in Qdrant Cloud with searchable payload fields:
  - `project`
  - `document_name`
  - `chunk_index`
  - `text`  

### Automatic summaries
Each document is summarized by Gemini into:
- 3 to 5 key insights  
- Actionable findings  
- Risks or important considerations  

### Multi project organization
Documents and queries can be scoped to a chosen project (e.g., `client-a`, `experiment-2`, `biomed-research`).  
This enables separate workspaces inside the same vector database.

### Evidence based question answering
- Question is embedded  
- Top K chunks retrieved from Qdrant  
- Gemini answers *only* using retrieved evidence  
- Citations `[1]`, `[2]`, etc. appear in the answer  
- Context is fully visible in the UI

### Clean Streamlit interface
- One click ingestion  
- One click question answering  
- Expandable panels for summaries, context, history  
- Real time project switching  

### Production ready patterns
- RAG pipeline  
- Vector search  
- Payload keyword indexing  
- Environment based configuration  
- Modular Python code  

---

## Tech stack

- **Language model**: Gemini `gemini-2.0-flash`  
- **Embeddings**: Google `text-embedding-004`  
- **Vector database**: Qdrant Cloud  
- **Web app**: Streamlit  
- **Language**: Python 3.11  

---

## Project structure

```text
vector-insight-engine/
  .env.example         # Template for environment variables
  .gitignore
  ingest.py            # Ingest and embed text/PDFs into Qdrant
  query_engine.py      # RAG pipeline: embed query, search, ask Gemini
  app.py               # Streamlit UI (upload, ingest, summarize, search)
  smoke_test.py        # Quick test for Gemini + Qdrant connectivity
  debug_ingest.py      # Debug script with detailed logging
  requirements.txt
```
---

## Getting started

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/vector-insight-engine.git
cd vector-insight-engine
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

i. Copy the example file:
```bash
cp .env.example .env
```

ii. Edit `.env` and fill in your own keys:
- `GEMINI_API_KEY` from Google AI Studio or Vertex AI
- `QDRANT_URL` from Qdrant Cloud
- `QDRANT_API_KEY` from Qdrant Cloud

**5. Run smoke tests**
```bash
python smoke_test.py
```
You should see:
- A short Gemini response
- Successful connection to Qdrant

**6. Run the application**
```bash
streamlit run app.py
```
   Visit:
```bash
http://localhost:8501
```

---

## How it works

1. Ingestion
- User pastes text or uploads a TXT/PDF
- `ingest.py` extracts text and splits into chunks
- Each chunk → `text-embedding-004`
- Stored in Qdrant with keyword indexes on:
   - `project`
   - `document_name`

2. Retrieval
- `query_engine.py`:
   - Embeds the question
   - Searches Qdrant for top K similar chunks
   - Builds numbered context for citations

3. Answer generation
- Gemini receives:
   - The question
   - Retrieved context
   - Mandatory citation instructions
Gemini returns a grounded answer with citations like `[1]`

4. UI
- Streamlit displays:
   - Answer
   - Context with similarity scores
   - Document summaries
   - Per project history
   - Uploaded documents list

---

## Example workflow
1. Select a project label (client-a)

2. Upload a PDF of research notes

3. Click Ingest into Qdrant

4. View the auto summary

5. Ask:
What are the main risks identified?

6. Get an answer with citations

7. Inspect the raw context

This workflow mirrors real enterprise RAG systems.

---

## Future extensions

- Add embeddings for images, tables, and web pages

- Multi user authentication

- Per project vector collections

- Relevance re ranking with Gemini

- Exportable knowledge graphs

- PDF OCR support for scanned documents

- Switch frontend to React/Next.js

---

## License
MIT License.
