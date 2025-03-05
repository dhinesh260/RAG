import streamlit as st
import ollama
import chromadb
import pdfplumber
import docx
import tempfile
import os
import json
import datetime
from sentence_transformers import SentenceTransformer
from autocorrect import Speller

# Initialize components
spell = Speller(lang='en')  # Auto-correct model
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent history per user
def get_history_file(username):
    return f"{username}_history.json"

def load_history(username):
    history_file = get_history_file(username)
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            return json.load(file)
    return []

def save_history(username, history):
    history_file = get_history_file(username)
    with open(history_file, "w") as file:
        json.dump(history, file)

# Extract text from uploaded documents
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return None

# Auto-summarize documents before indexing
def summarize_document(text):
    prompt = f"Summarize this document:\n{text[:2000]}"  # Limit to 2000 chars for performance
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# Index documents in ChromaDB
def index_document(file, file_path):
    text = extract_text(file_path)
    if text:
        summary = summarize_document(text)  # Auto-summarization
        embedding = embedding_model.encode(summary).tolist()
        metadata = {
            "filename": file.name,
            "content": summary,
            "date_added": str(datetime.datetime.now())
        }
        collection.add(ids=[file.name], embeddings=[embedding], metadatas=[metadata])
        return f"Indexed: {file.name} (Summarized)"
    return f"Failed to extract text from {file.name}"

# Delete document from ChromaDB
def delete_document(file_name):
    collection.delete(ids=[file_name])
    return f"Deleted: {file_name}"

# Get uploaded files
def get_uploaded_files():
    return collection.get(include=["metadatas"])["metadatas"]

# Retrieve documents based on keyword match
def keyword_search(query):
    results = get_uploaded_files()
    return [doc["content"] for doc in results if query.lower() in doc["content"].lower()]

# Retrieve document snippets
def get_snippet(text, query, window=50):
    idx = text.lower().find(query.lower())
    if idx == -1:
        return text[:window]
    start, end = max(0, idx - window), min(len(text), idx + len(query) + window)
    return "..." + text[start:end] + "..."

# Query correction
def correct_query(query):
    return spell(query)

# Generate answer using Ollama with chat history
def generate_answer_ollama(question, documents, history):
    context = "\n\n".join(documents)
    chat_context = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history[-5:]])  # Last 5 Q&As
    prompt = f"Previous Q&A:\n{chat_context}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# Search function (multi-file query, keyword + vector search)
def search_and_answer_ollama(question, username, top_k=5, use_keywords=False, file_filter=None, date_filter=None):
    question = correct_query(question)
    history = load_history(username)

    documents = []
    if use_keywords:
        documents += keyword_search(question)

    query_embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if results["metadatas"]:
        for doc in results["metadatas"][0]:
            if file_filter and file_filter.lower() not in doc["filename"].lower():
                continue  # Skip if file filter does not match
            
            if date_filter and date_filter not in doc["date_added"]:
                continue  # Skip if date filter does not match

            documents.append(doc["content"])

    if documents:
        answer = generate_answer_ollama(question, documents, history)
        history.append({"question": question, "answer": answer})
        save_history(username, history)
        return answer, results["distances"][0] if results["distances"] else None
    else:
        return "No relevant documents found.", None

# Streamlit UI
st.set_page_config(page_title="Advanced Local Q&A", layout="wide")
st.title("📄 Advanced Local Document Q&A (ChromaDB + Ollama)")

# Sidebar for user login
st.sidebar.header("👤 User")
username = st.sidebar.text_input("Enter your username", value="guest")
history = load_history(username)

# Sidebar for file upload
st.sidebar.header("📂 Upload & Manage Documents")
uploaded_files = st.sidebar.file_uploader("Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            temp_file.write(file.read())
            file_path = temp_file.name
        result = index_document(file, file_path)
        st.sidebar.success(result)
        os.remove(file_path)

# Sidebar: Show uploaded files and delete option
st.sidebar.subheader("🗂 Uploaded Files")
uploaded_files_list = get_uploaded_files()
if uploaded_files_list:
    file_to_delete = st.sidebar.selectbox("Select file to delete", [doc["filename"] for doc in uploaded_files_list])
    if st.sidebar.button("Delete File"):
        delete_msg = delete_document(file_to_delete)
        st.sidebar.warning(delete_msg)

# Tabs for Q&A and history
tab1, tab2 = st.tabs(["🔎 Ask a Question", "📜 Search History"])

with tab1:
    st.subheader("💬 Ask a Question")
    question = st.text_input("Enter your question:")
    top_k = st.slider("Select number of top documents to retrieve", min_value=1, max_value=10, value=5)
    use_keywords = st.checkbox("Enable keyword-based search")
    file_filter = st.text_input("Filter by filename (optional)")
    date_filter = st.date_input("Filter by date (optional)")

    if st.button("Get Answer") and question:
        answer, distances = search_and_answer_ollama(question, username, top_k, use_keywords, file_filter, date_filter)
        st.write("### 🤖 Answer:")
        st.write(answer)

with tab2:
    st.subheader("📜 Question & Answer History")
    for entry in history[::-1]:
        with st.expander(f"**Q:** {entry['question']}"):
            st.write(f"**A:** {entry['answer']}")
