import html
import os
from io import BytesIO

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

import chromadb
import streamlit as st
import tqdm.auto as tqdm_auto
import tqdm.std as tqdm_std
from chromadb.config import Settings
from groq import Groq
from huggingface_hub.utils import disable_progress_bars
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from transformers.utils import logging as transformers_logging

import config
from services.ingestion import ingest_pdf
from services.llm_service import generate_answer
from services.retrieval import search_vector_db
from services.routing import route_query
from services.web_search import web_search


st.set_page_config(page_title="AI Research Assistant", page_icon="📘", layout="wide")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "you", "your",
}


def _disable_progress_output():
    disable_progress_bars()
    transformers_logging.disable_progress_bar()

    original_init = tqdm_std.tqdm.__init__

    def silent_init(self, *args, **kwargs):
        kwargs["disable"] = True
        return original_init(self, *args, **kwargs)

    tqdm_std.tqdm.__init__ = silent_init
    tqdm_auto.tqdm.__init__ = silent_init


_disable_progress_output()

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background: radial-gradient(circle at top, rgba(77, 163, 255, 0.08), transparent 32%);
    }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 0.75rem 0.9rem;
        background: rgba(255, 255, 255, 0.02);
    }
    .source-text {
        font-size: 0.96rem;
        line-height: 1.65;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .history-box {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 0.35rem;
        background: rgba(255, 255, 255, 0.02);
    }
    .history-item {
        padding: 0.55rem 0.65rem;
        border-radius: 10px;
        margin-bottom: 0.3rem;
        background: rgba(255, 255, 255, 0.03);
        font-size: 0.92rem;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system():
    embed_model = SentenceTransformer(config.EMBED_MODEL)
    chroma_client = chromadb.Client(
        Settings(persist_directory=config.VECTOR_DB_PATH, anonymized_telemetry=False)
    )
    collection = chroma_client.get_or_create_collection(name="research_paper")
    tavily = TavilyClient(api_key=config.TAVILY_API_KEY) if config.TAVILY_API_KEY else None
    llm = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None
    return embed_model, collection, tavily, llm


embed_model, collection, tavily, llm = load_system()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


class UploadedPDF(BytesIO):
    def __init__(self, uploaded_file):
        super().__init__(uploaded_file.getvalue())
        self.name = uploaded_file.name
        self.size = uploaded_file.size



def _tokenize(text):
    return {
        token
        for token in text.lower().replace("?", " ").replace(",", " ").split()
        if token not in STOPWORDS and len(token) > 2
    }



def _document_context_is_relevant(query, context):
    if not context:
        return False

    query_terms = _tokenize(query)
    if not query_terms:
        return False

    best_overlap = 0
    best_distance = None
    for item in context:
        content_terms = _tokenize(item.get("content", ""))
        overlap = len(query_terms & content_terms)
        best_overlap = max(best_overlap, overlap)
        distance = item.get("distance")
        if distance is not None and (best_distance is None or distance < best_distance):
            best_distance = distance

    if best_overlap >= 2:
        return True
    if best_overlap >= 1 and best_distance is not None and best_distance < 1.0:
        return True
    return False



def _clear_all_documents():
    try:
        existing = collection.get()
        ids = existing.get("ids", []) if existing else []
        if ids:
            collection.delete(ids=ids)
    except Exception:
        pass



def render_sources(sources):
    if not sources:
        st.info("No sources available for this answer.")
        return

    for index, source in enumerate(sources, start=1):
        metadata = source.get("metadata", {})
        title = metadata.get("file_name") or metadata.get("title") or f"Source {index}"
        page = metadata.get("page")
        url = metadata.get("url")
        meta_parts = []
        if page is not None:
            meta_parts.append(f"Page {page + 1}")
        if url:
            meta_parts.append(url)
        meta = " | ".join(meta_parts)
        content = html.escape(source.get("content", "").strip())
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if meta:
                st.caption(meta)
            st.markdown(f"<div class='source-text'>{content}</div>", unsafe_allow_html=True)



def render_chat_history_sidebar():
    user_messages = [
        message["content"]
        for message in st.session_state.messages
        if message.get("role") == "user" and message.get("content")
    ]
    sidebar.subheader("Recent Chats")
    if not user_messages:
        sidebar.caption("Your recent questions will appear here.")
        return

    recent_messages = list(reversed(user_messages[-8:]))
    sidebar.markdown("<div class='history-box'>", unsafe_allow_html=True)
    for message in recent_messages:
        preview = html.escape(message[:60] + ("..." if len(message) > 60 else ""))
        sidebar.markdown(f"<div class='history-item'>{preview}</div>", unsafe_allow_html=True)
    sidebar.markdown("</div>", unsafe_allow_html=True)


st.title("AI Research Assistant")
st.caption("Upload PDFs, ask questions over your indexed documents, and fall back to web search when needed.")

sidebar = st.sidebar
sidebar.header("Workspace")
sidebar.caption("Manage uploaded files and session state.")

uploaded_files = sidebar.file_uploader(
    "Upload Research PDFs",
    type="pdf",
    accept_multiple_files=True,
    key=f"uploaded_pdfs_{st.session_state.uploader_key}",
)

if sidebar.button("Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if sidebar.button("Forget Uploaded Files", use_container_width=True):
    st.session_state.indexed_files = set()
    st.session_state.messages = []
    st.session_state.uploader_key += 1
    _clear_all_documents()
    st.rerun()

newly_indexed = 0
index_errors = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if file_key in st.session_state.indexed_files:
            continue
        try:
            pdf_file = UploadedPDF(uploaded_file)
            with sidebar:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    ingest_pdf(pdf_file, embed_model, collection)
            st.session_state.indexed_files.add(file_key)
            newly_indexed += 1
        except Exception as exc:
            index_errors.append(f"{uploaded_file.name}: {exc}")

if newly_indexed:
    sidebar.success(f"Indexed {newly_indexed} file(s).")
if uploaded_files and not newly_indexed and not index_errors:
    sidebar.info("These uploaded files are already indexed in this session.")
if index_errors:
    for item in index_errors:
        sidebar.error(item)

sidebar.divider()
render_chat_history_sidebar()

info_col, status_col = st.columns([1.6, 1.0], gap="large")

with status_col:
    st.subheader("Status")
    st.write(f"Documents available: {'Yes' if collection.count() > 0 else 'No'}")
    st.write(f"Indexed chunks: {collection.count()}")
    st.write(f"Embedding model: {config.EMBED_MODEL}")
    st.write(f"Groq API key loaded: {'Yes' if config.GROQ_API_KEY else 'No'}")
    st.write(f"Tavily API key loaded: {'Yes' if config.TAVILY_API_KEY else 'No'}")
    st.info("The app auto-selects document or web sources. Time-sensitive questions go to web. Otherwise it uses PDFs when the retrieved match is relevant.")

with info_col:
    st.subheader("Chat")
    st.caption("Ask about uploaded documents or general research questions.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    render_sources(message["sources"])

query = st.chat_input("Ask a research question")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    has_documents = collection.count() > 0
    route = route_query(query, has_documents=has_documents)

    try:
        with st.spinner("Working on your question..."):
            context = []

            if route in {"AUTO", "DOCUMENT"}:
                context = search_vector_db(query, embed_model, collection)
                if route == "DOCUMENT":
                    if not _document_context_is_relevant(query, context) and tavily:
                        route = "WEB"
                        context = web_search(query, tavily)
                elif route == "AUTO":
                    if _document_context_is_relevant(query, context):
                        route = "DOCUMENT"
                    elif tavily:
                        route = "WEB"
                        context = web_search(query, tavily)
                    else:
                        route = "DOCUMENT"
            elif route == "WEB" and tavily:
                context = web_search(query, tavily)

            answer = generate_answer(context, query, llm, config.LLM_MODEL)

        assistant_message = {
            "role": "assistant",
            "content": f"{answer}\n\nSource mode: {route.title()}",
            "sources": context,
        }
        st.session_state.messages.append(assistant_message)

        with st.chat_message("assistant"):
            st.markdown(assistant_message["content"])
            with st.expander("Sources"):
                render_sources(context)
    except Exception as exc:
        error_message = f"Request failed: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})
        with st.chat_message("assistant"):
            st.error(error_message)
