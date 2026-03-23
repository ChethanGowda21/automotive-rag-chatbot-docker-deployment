import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import os

# Settings
top_k = 8
chunk_size = 800
chunk_overlap = 100
temperature = 0.2

# Load embedding model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text(pdf):

    reader = PdfReader(pdf)
    text = ""

    for i, page in enumerate(reader.pages):
        text += f"\n--- PAGE {i+1} ---\n"
        text += page.extract_text() or ""

    return text


# -----------------------------
# TEXT SPLITTING
# -----------------------------
def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_text(text)


# -----------------------------
# FAISS INDEX
# -----------------------------
def build_faiss(chunks):

    embeddings = embedder.encode(chunks, normalize_embeddings=True)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index


# -----------------------------
# FULL PIPELINE
# -----------------------------
def process_pdf(pdf):

    text = extract_text(pdf)
    chunks = split_text(text)
    index = build_faiss(chunks)

    return chunks, index


# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve(query, chunks, index):

    q_emb = embedder.encode([query], normalize_embeddings=True)

    _, idx = index.search(np.array(q_emb), top_k)

    return [chunks[i] for i in idx[0]]


# -----------------------------
# LLM CALL (GROQ + LLAMA)
# -----------------------------
def ask_groq(context, question):

    api_key = os.getenv("GROQ_API_KEY")

    client = Groq(api_key=api_key)

    prompt = f"""
You are an expert automotive technical assistant.

Rules:
- Answer only from provided context
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