# AI Research Intelligence Platform

LIVE_App : https://agenticresearchragmodel-znuiwncezqmhtdbczvz3im.streamlit.app/


Enterprise‑style demo project implementing an **Agentic RAG system**.

Features
- Upload multiple PDFs
- Ask questions about documents
- Agentic routing (document vs web)
- Groq Llama LLM
- Chroma vector database
- Streamlit enterprise dashboard

## Run locally

1. Create virtual environment

python -m venv venv

Activate

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. Add API keys

Rename `.env.example` to `.env`

4. Run

streamlit run app.py