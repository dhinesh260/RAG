import streamlit as st
import os
import asyncio
import json
import torch
import shutil
import tempfile
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from autocorrect import Speller
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import ollama
import time

# ✅ Fix: Ensure PyTorch doesn’t fuse operations (Prevents torch::class_ error)
torch._C._jit_override_can_fuse_on_cpu(False)  
torch.set_num_threads(1)  # Prevents multithreading issues

# ✅ Set Ollama API to specific IP & Port
os.environ["OLLAMA_API_HOST"] = "http://<YOUR_IP>:<YOUR_PORT>"

# ✅ Initialize ChromaDB & Embedding Model
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="docs", embedding_function=embedding_function)

# ✅ Spell Checker for Auto-Correction
spell = Speller(lang='en')

# ✅ Load Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ✅ Ensure Uploads Directory Exists
UPLOAD_DIR = "uploaded_docs"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

# ✅ Chat History File for Multi-User Support
CHAT_HISTORY_FILE = "chat_history.json"
if not os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump({}, f)

# ✅ Function to Auto-Correct User Queries
def autocorrect_query(query):
    return " ".join([spell(word) for word in query.split()])

# ✅ Function to Extract and Store Document Data
def index_documents():
    collection.delete()  # Clear previous index
    for file in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, file)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        # ✅ Auto-Summarize the Document
        summary = summarizer(text[:1024], max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        
        # ✅ Store Summary & Content in ChromaDB
        collection.add(
            ids=[file],
            documents=[summary + " " + text],  # Summary + Full Content
            metadatas=[{"filename": file}]
        )

# ✅ Periodic Reindexing (Auto-Refresh Index When Files Change)
def periodic_reindexing():
    while True:
        index_documents()
        time.sleep(300)  # Run every 5 minutes

# ✅ Function to Generate Answer using Ollama
def generate_answer_ollama(question, documents, history):
    try:
        corrected_question = autocorrect_query(question)
        context = "\n\n".join(documents)
        chat_context = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history[-5:]])

        prompt = f"Previous Q&A:\n{chat_context}\n\nContext:\n{context}\n\nQuestion: {corrected_question}\nAnswer:"
        
        async def call_ollama():
            return await ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                api_base=os.getenv("OLLAMA_API_HOST")
            )
        
        response = asyncio.run(call_ollama())  # ✅ Fix: Handle async properly
        return response["message"]["content"].strip()

    except Exception as e:
        st.error(f"Error in generating the answer: {str(e)}")
        return "Answer unavailable"

# ✅ Function to Retrieve Relevant Documents
def retrieve_documents(query, top_k=5):
    corrected_query = autocorrect_query(query)
    results = collection.query(query_texts=[corrected_query], n_results=top_k)
    return [doc for doc in results.get("documents", [[]])[0]]

# ✅ Function to Load & Save Chat History
def get_chat_history(user_id):
    with open(CHAT_HISTORY_FILE, "r") as f:
        history = json.load(f)
    return history.get(user_id, [])

def save_chat_history(user_id, question, answer):
    with open(CHAT_HISTORY_FILE, "r") as f:
        history = json.load(f)
    
    if user_id not in history:
        history[user_id] = []
    
    history[user_id].append({"question": question, "answer": answer})
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f)

# ✅ Function to Delete a File
def delete_file(filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        collection.delete(ids=[filename])
        st.success(f"Deleted {filename}")

# ✅ Streamlit UI
st.title("📄 AI-Powered Document Q&A System")

tab1, tab2, tab3 = st.tabs(["Ask a Question", "Upload & Manage Files", "Analytics"])

with tab1:
    user_id = st.text_input("User ID:", "default_user")
    question = st.text_area("Ask a Question:")
    
    if st.button("Get Answer"):
        relevant_docs = retrieve_documents(question)
        history = get_chat_history(user_id)
        answer = generate_answer_ollama(question, relevant_docs, history)
        save_chat_history(user_id, question, answer)
        st.write(f"**Answer:** {answer}")

    st.subheader("Chat History")
    history = get_chat_history(user_id)
    for chat in history:
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")

with tab2:
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=["pdf", "txt", "docx"])
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"Uploaded {uploaded_file.name}")
        index_documents()  # Re-index after uploading files

    st.subheader("Uploaded Files")
    existing_files = os.listdir(UPLOAD_DIR)
    for file in existing_files:
        col1, col2 = st.columns([4, 1])
        col1.write(file)
        if col2.button("❌", key=file):
            delete_file(file)

with tab3:
    st.subheader("Analytics Dashboard")
    st.write(f"**Total Documents:** {len(os.listdir(UPLOAD_DIR))}")
    
    with open(CHAT_HISTORY_FILE, "r") as f:
        history = json.load(f)
    
    total_queries = sum(len(h) for h in history.values())
    st.write(f"**Total Queries:** {total_queries}")

# ✅ Run Periodic Reindexing in Background
import threading
threading.Thread(target=periodic_reindexing, daemon=True).start()
