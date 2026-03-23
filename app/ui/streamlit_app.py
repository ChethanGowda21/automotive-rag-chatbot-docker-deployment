import os
import streamlit as st
from app.rag.rag_pipeline import process_pdf, retrieve, ask_groq
from app.evaluation.rag_evaluator import RAGEvaluator
from dotenv import load_dotenv

load_dotenv()

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
# API KEY CHECK
# =====================================================
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ Please set GROQ_API_KEY environment variable")
    st.stop()

# =====================================================
# SETTINGS
# =====================================================
top_k = 8
chunk_size = 800
chunk_overlap = 100
temperature = 0.2

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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Type your question here...")

    if query:

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