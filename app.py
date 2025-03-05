import streamlit as st
import ollama
import chromadb
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import tempfile
import os

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from uploaded documents
def extract_text(file_path):
    """Extract text from PDF, DOCX, and TXT files"""
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
    """Index document text and embeddings into ChromaDB"""
    text = extract_text(file_path)
    if text:
        embedding = embedding_model.encode(text).tolist()
        collection.add(ids=[file.name], embeddings=[embedding], metadatas=[{"filename": file.name, "content": text}])
        return f"Indexed: {file.name}"
    return f"Failed to extract text from {file.name}"

# Function to delete a document from ChromaDB
def delete_document(file_name):
    """Delete document by filename from ChromaDB"""
    collection.delete(ids=[file_name])
    return f"Deleted: {file_name}"

# Fetch uploaded files from ChromaDB
def get_uploaded_files():
    """Retrieve list of uploaded file names"""
    return [doc["filename"] for doc in collection.get(include=["metadatas"])["metadatas"]]

# Ollama function for answer generation
def generate_answer_ollama(question, documents):
    """Use Ollama (local LLM) to generate an answer"""
    context = "\n\n".join(documents)
    
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"].strip()

# Search function
def search_and_answer_ollama(question):
    """Search ChromaDB and generate an answer using Ollama"""
    query_embedding = embedding_model.encode(question).tolist()
    
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    documents = [doc["content"] for doc in results["metadatas"][0]] if results["metadatas"] else []
    
    if documents:
        answer = generate_answer_ollama(question, documents)
        history.append({"question": question, "answer": answer})
        return answer
    else:
        return "No relevant documents found."

# Streamlit UI
st.set_page_config(page_title="Local Document Q&A", layout="wide")
st.title("📄 Local Document Q&A (ChromaDB + Ollama)")

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

# Q&A section
st.subheader("💬 Ask a Question")
question = st.text_input("Enter your question:")

if "history" not in st.session_state:
    st.session_state.history = []  # Initialize history

history = st.session_state.history  # Keep history persistent

if st.button("Get Answer") and question:
    with st.spinner("Searching & generating answer..."):
        answer = search_and_answer_ollama(question)
        st.write("### 🤖 Answer:")
        st.write(answer)

# Display history
st.subheader("📜 Question & Answer History")
if history:
    for entry in history[::-1]:  # Show latest first
        with st.expander(f"**Q:** {entry['question']}"):
            st.write(f"**A:** {entry['answer']}")
else:
    st.info("No history available.")
