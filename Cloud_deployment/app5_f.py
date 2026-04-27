import streamlit as st

import os
import numpy as np
import faiss

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from rag_evaluator import RAGEvaluator

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Automotive Crash Documentation Study",
    layout="wide"
)

# =====================================================
# HEADER STYLE
# =====================================================
st.markdown("""
<style>
.main-title{
background: linear-gradient(90deg,#f6a623,#f39c12);
padding:25px;
border-radius:10px;
text-align:center;
color:white;
font-size:34px;
font-weight:600;
}
.sub-title{
text-align:center;
color:gray;
margin-top:-10px;
margin-bottom:30px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 Automotive Crash Documentation Study</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">RAG-based Intelligent PDF Question Answering System</div>', unsafe_allow_html=True)

# =====================================================
# SETTINGS
# =====================================================
top_k = 8
chunk_size = 800
chunk_overlap = 100
temperature = 0.2

# =====================================================
# API KEY
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Please set GROQ_API_KEY environment variable")
    st.stop()

# =====================================================
# GROQ CLIENT
# =====================================================
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

# =====================================================
# EMBEDDING MODEL
# =====================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# =====================================================
# PDF FUNCTIONS
# =====================================================
def extract_text(pdf):

    reader = PdfReader(pdf)

    text = ""

    for i, page in enumerate(reader.pages):
        text += f"\n--- PAGE {i+1} ---\n"
        text += page.extract_text() or ""

    return text


def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_text(text)


def build_faiss(chunks):

    embeddings = embedder.encode(chunks, normalize_embeddings=True)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))

    return index


@st.cache_resource
def process_pdf(pdf):

    text = extract_text(pdf)

    chunks = split_text(text)

    index = build_faiss(chunks)

    return chunks, index


# =====================================================
# RETRIEVE
# =====================================================
def retrieve(query, chunks, index):

    q_emb = embedder.encode([query], normalize_embeddings=True)

    _, idx = index.search(np.array(q_emb), top_k)

    return [chunks[i] for i in idx[0]]


# =====================================================
# LLM
# =====================================================
def ask_groq(context, question):

    client = get_groq_client()

    prompt = f"""
You are an expert automotive technical assistant.

Rules:
- Answer only from provided context
- Do not use external knowledge
- If not found say: The answer is not available in the document
- Be concise

Context:
{context}

Question:
{question}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    return completion.choices[0].message.content


# =====================================================
# APPLICATION CONTROLS
# =====================================================
st.subheader("🔷 Application Controls")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("ℹ️ About"):
        st.session_state.page = "about"

with col2:
    if st.button("⚙️ How It Works"):
        st.session_state.page = "how"

with col3:
    if st.button("📤 Upload PDF"):
        st.session_state.page = "upload"

with col4:
    if st.button("🚀 Process PDF"):
        st.session_state.page = "process"

with col5:
    if st.button("📊 RAG Evaluation"):
        st.session_state.page = "rag"

if "page" not in st.session_state:
    st.session_state.page = "upload"


# =====================================================
# ABOUT
# =====================================================
if st.session_state.page == "about":

    st.info("""
This system is a **Retrieval-Augmented Generation (RAG) chatbot**
for **Automotive Crash Documentation Analysis**.

Features:

• Upload engineering PDFs  
• Convert text into embeddings  
• Retrieve relevant technical information  
• Generate answers using Groq LLM  
• Evaluate responses using RAG metrics
""")


# =====================================================
# HOW IT WORKS
# =====================================================
if st.session_state.page == "how":

    st.markdown("""
### System Workflow

1️⃣ Upload Automotive PDF  
2️⃣ Extract text using PyPDF2  
3️⃣ Split text into chunks  
4️⃣ Convert chunks into embeddings  
5️⃣ Store embeddings in FAISS  
6️⃣ Retrieve relevant chunks  
7️⃣ Generate answer using Groq LLM  
8️⃣ Evaluate answer quality
""")


# =====================================================
# UPLOAD PDF
# =====================================================
pdf = st.file_uploader("Select a PDF file", type="pdf")


# =====================================================
# PROCESS PDF
# =====================================================
if pdf and st.button("Process Document"):

    with st.spinner("Processing PDF..."):

        chunks, index = process_pdf(pdf)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.ready = True

    st.success("✅ PDF indexed successfully!")


# =====================================================
# CHATBOT
# =====================================================
if st.session_state.get("ready"):

    st.subheader("💬 Ask Questions From Document")

    # show chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # chat input
    query = st.chat_input("Type your question here...")

    if query:

        # show user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Generating answer..."):

            retrieved = retrieve(
                query,
                st.session_state.chunks,
                st.session_state.index
            )

            context = "\n".join(retrieved)

            answer = ask_groq(context, query)

            evaluator = RAGEvaluator()

            scores = evaluator.evaluate(query, answer, context)

            st.session_state.rag_scores = scores

        # show assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)


# =====================================================
# RAG EVALUATION PAGE
# =====================================================
if st.session_state.page == "rag":

    st.title("📊 RAG Evaluation Metrics")

    if "rag_scores" in st.session_state:

        st.json(st.session_state.rag_scores)

    else:

        st.warning("Ask a question first to generate RAG metrics.")


# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Streamlit • FAISS • SentenceTransformers • Groq • RAG Evaluation")