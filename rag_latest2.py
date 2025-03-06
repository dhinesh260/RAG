import streamlit as st
import ollama
import chromadb
import pdfplumber
import docx
import tempfile
import os
import json
import datetime
import time
import schedule
from sentence_transformers import SentenceTransformer
from autocorrect import Speller
from collections import Counter

# Initialize Components
spell = Speller(lang='en')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent history & analytics tracking
def get_history_file(username):
    return f"{username}_history.json"

def load_history(username):
    if os.path.exists(get_history_file(username)):
        with open(get_history_file(username), "r") as file:
            return json.load(file)
    return []

def save_history(username, history):
    with open(get_history_file(username), "w") as file:
        json.dump(history, file)

def load_analytics():
    if os.path.exists("analytics.json"):
        with open("analytics.json", "r") as file:
            return json.load(file)
    return {"queries": [], "document_usage": {}}

def save_analytics(data):
    with open("analytics.json", "w") as file:
        json.dump(data, file)

# Extract text from uploaded documents
def extract_text(file_path):
    try:
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
    except Exception:
        st.error("Error reading the document. Something went wrong.")
        return None

# Auto-summarization
def summarize_document(text):
    try:
        prompt = f"Summarize this document:\n{text[:2000]}"
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}], api_base=os.getenv("OLLAMA_API_HOST"))
        return response["message"]["content"].strip()
    except Exception:
        st.error("Error in document summarization. Something went wrong.")
        return "Summary unavailable"

# Index a document
def index_document(file, file_path):
    try:
        text = extract_text(file_path)
        if text:
            summary = summarize_document(text)
            embedding = embedding_model.encode(summary).tolist()
            metadata = {"filename": file.name, "content": summary}
            collection.add(ids=[file.name], embeddings=[embedding], metadatas=[metadata])
            return f"Indexed: {file.name} (Summarized)"
        return f"Failed to extract text from {file.name}"
    except Exception:
        st.error("Error while indexing the document. Something went wrong.")
        return None

# Delete document
def delete_document(file_name):
    try:
        collection.delete(ids=[file_name])
        return f"Deleted: {file_name}"
    except Exception:
        st.error("Error while deleting the document. Something went wrong.")
        return None

# Get uploaded files
def get_uploaded_files():
    try:
        return collection.get(include=["metadatas"])["metadatas"]
    except Exception:
        st.error("Error retrieving uploaded files. Something went wrong.")
        return []

# Periodic reindexing: Detect new/deleted files
def reindex_documents():
    try:
        existing_files = {doc["filename"] for doc in get_uploaded_files()}
        current_files = {file for file in os.listdir("./documents")}

        new_files = current_files - existing_files
        deleted_files = existing_files - current_files

        for file in new_files:
            file_path = os.path.join("./documents", file)
            index_document(file, file_path)

        for file in deleted_files:
            delete_document(file)
    except Exception:
        st.error("Error during periodic reindexing. Something went wrong.")

schedule.every(10).minutes.do(reindex_documents)

# Auto-correct user query
def correct_query(query):
    try:
        return spell(query)
    except Exception:
        st.error("Error in query auto-correction. Something went wrong.")
        return query

# Generate answer with chat history
def generate_answer_ollama(question, documents, history):
    try:
        context = "\n\n".join(documents)
        chat_context = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history[-5:]])
        prompt = f"Previous Q&A:\n{chat_context}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}], api_base=os.getenv("OLLAMA_API_HOST"))
        return response["message"]["content"].strip()
    except Exception:
        st.error("Error in generating the answer. Something went wrong.")
        return "Answer unavailable"

# Search function
def search_and_answer_ollama(question, username, top_k=5, file_filter=None):
    try:
        question = correct_query(question)
        history = load_history(username)
        analytics = load_analytics()

        query_embedding = embedding_model.encode(question).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        documents = []
        for doc in results["metadatas"][0]:
            if file_filter and file_filter.lower() not in doc["filename"].lower():
                continue
            documents.append(doc["content"])
            analytics["document_usage"][doc["filename"]] = analytics["document_usage"].get(doc["filename"], 0) + 1

        analytics["queries"].append(question)
        save_analytics(analytics)

        if documents:
            answer = generate_answer_ollama(question, documents, history)
            history.append({"question": question, "answer": answer})
            save_history(username, history)
            return answer
        return "No relevant documents found."
    except Exception:
        st.error("Error in search and retrieval. Something went wrong.")
        return "Search failed"

# Streamlit UI
st.set_page_config(page_title="Advanced Local Q&A", layout="wide")
st.title("📄 Advanced Local Document Q&A")

# Sidebar
st.sidebar.header("👤 User")
username = st.sidebar.text_input("Enter your username", value="guest")

st.sidebar.header("📂 Upload & Manage Documents")
uploaded_files = st.sidebar.file_uploader("Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
                temp_file.write(file.read())
                file_path = temp_file.name
            st.sidebar.success(index_document(file, file_path))
            os.remove(file_path)
        except Exception:
            st.sidebar.error("Error uploading the file. Something went wrong.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔎 Ask a Question", "📜 Search History", "📊 Analytics"])

with tab1:
    st.subheader("💬 Ask a Question")
    question = st.text_input("Enter your question:")
    top_k = st.slider("Select number of documents to retrieve", 1, 10, 5)
    file_filter = st.text_input("Filter by filename (optional)")

    if st.button("Get Answer") and question:
        answer = search_and_answer_ollama(question, username, top_k, file_filter)
        st.write("### 🤖 Answer:")
        st.write(answer)

with tab2:
    st.subheader("📜 Question & Answer History")
    history = load_history(username)
    for entry in history[::-1]:
        with st.expander(f"**Q:** {entry['question']}"):
            st.write(f"**A:** {entry['answer']}")

with tab3:
    st.subheader("📊 Analytics Dashboard")
    analytics = load_analytics()
    st.write("📌 **Most Asked Questions:**", Counter(analytics["queries"]).most_common(5))
    st.write("📂 **Most Used Documents:**", sorted(analytics["document_usage"].items(), key=lambda x: x[1], reverse=True))
