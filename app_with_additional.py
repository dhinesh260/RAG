import streamlit as st
import ollama
import chromadb
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import tempfile
import os
import json

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent history file per user
def get_history_file(username):
    return f"{username}_history.json"

# Load user history from JSON file
def load_history(username):
    history_file = get_history_file(username)
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            return json.load(file)
    return []

# Save user history to JSON file
def save_history(username, history):
    history_file = get_history_file(username)
    with open(history_file, "w") as file:
        json.dump(history, file)

# Function to extract text from uploaded documents
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

# Function to index document into ChromaDB
def index_document(file, file_path):
    text = extract_text(file_path)
    if text:
        embedding = embedding_model.encode(text).tolist()
        collection.add(ids=[file.name], embeddings=[embedding], metadatas=[{"filename": file.name, "content": text}])
        return f"Indexed: {file.name}"
    return f"Failed to extract text from {file.name}"

# Function to delete a document from ChromaDB
def delete_document(file_name):
    collection.delete(ids=[file_name])
    return f"Deleted: {file_name}"

# Fetch uploaded files from ChromaDB
def get_uploaded_files():
    return [doc["filename"] for doc in collection.get(include=["metadatas"])["metadatas"]]

# Ollama function for answer generation
def generate_answer_ollama(question, documents):
    context = "\n\n".join(documents)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# Search function with top-k retrieval
def search_and_answer_ollama(question, username, top_k=5):
    query_embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = [doc["content"] for doc in results["metadatas"][0]] if results["metadatas"] else []
    if documents:
        answer = generate_answer_ollama(question, documents)
        history.append({"question": question, "answer": answer})
        save_history(username, history)  # Save after every Q&A
        return answer, results["distances"][0]
    else:
        return "No relevant documents found.", None

# Streamlit UI
st.set_page_config(page_title="Local Document Q&A", layout="wide")
st.title("📄 Local Document Q&A (ChromaDB + Ollama)")

# Sidebar for user login
st.sidebar.header("👤 User")
username = st.sidebar.text_input("Enter your username", value="guest")
history = load_history(username)  # Load user's history

# Sidebar for file upload
st.sidebar.header("📂 Upload & Manage Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    with st.sidebar:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
                temp_file.write(file.read())
                file_path = temp_file.name
            result = index_document(file, file_path)
            st.success(result)
            os.remove(file_path)  # Clean up temp file

# Sidebar: Show uploaded files and delete option
st.sidebar.subheader("🗂 Uploaded Files")
uploaded_files_list = get_uploaded_files()
if uploaded_files_list:
    file_to_delete = st.sidebar.selectbox("Select file to delete", uploaded_files_list)
    if st.sidebar.button("Delete File"):
        delete_msg = delete_document(file_to_delete)
        st.sidebar.warning(delete_msg)

# Tabs for Q&A and history
tab1, tab2 = st.tabs(["🔎 Ask a Question", "📜 Search History"])

with tab1:
    st.subheader("💬 Ask a Question")
    question = st.text_input("Enter your question:")
    top_k = st.slider("Select number of top documents to retrieve", min_value=1, max_value=10, value=5)

    if st.button("Get Answer") and question:
        with st.spinner("Searching & generating answer..."):
            answer, distances = search_and_answer_ollama(question, username, top_k)
            st.write("### 🤖 Answer:")
            st.write(answer)

            if distances:
                st.write("### 🔍 Document Relevance:")
                for doc, score in zip(uploaded_files_list, distances):
                    st.write(f"📄 **Document:** {doc} (Score: {score:.2f})")

with tab2:
    st.subheader("📜 Question & Answer History")
    if history:
        for entry in history[::-1]:  # Show latest first
            with st.expander(f"**Q:** {entry['question']}"):
                st.write(f"**A:** {entry['answer']}")
    else:
        st.info("No history available.")
