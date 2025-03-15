import streamlit as st
import os
import html
import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# Ensure torch.classes loads correctly
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Set up Streamlit page
st.set_page_config(layout="wide")

# Initialize ChromaDB directory
DB_DIR = "chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

# Load embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error loading embedding model: {e}")
    embedding_model = None

# Ensure ChromaDB is initialized properly
try:
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
except Exception as e:
    st.error(f"Error initializing ChromaDB: {e}")
    vector_db = None

# Function to load and process uploaded files
def process_file(uploaded_file):
    try:
        file_extension = uploaded_file.name.split(".")[-1]
        loader = None
        temp_path = f"temp_uploaded_file.{file_extension}"

        # Save file locally before loading
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if file_extension == "pdf":
            loader = PyPDFLoader(temp_path)
        elif file_extension == "txt":
            loader = TextLoader(temp_path)
        elif file_extension in ["docx", "doc"]:
            loader = Docx2txtLoader(temp_path)
        else:
            st.error("Unsupported file format.")
            return

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        if vector_db:
            vector_db.add_documents(split_docs)
            st.success(f"Processed and stored {len(split_docs)} chunks!")
        else:
            st.error("ChromaDB is not initialized properly.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Sidebar: File Upload
with st.sidebar:
    st.title("Upload File")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])
    if uploaded_file:
        process_file(uploaded_file)

# Main Chat Interface
st.title("Chat with Documents")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [("Bot", "Hi, how can I assist you today?")]

# Display Chat History
for user, message in st.session_state["messages"]:
    align = "flex-end" if user == "User" else "flex-start"
    icon = "👤 " if user == "User" else "🤖"
    bg_color = "#1E3A8A" if user == "User" else "#374151"

    st.markdown(
        f"""
        <div style="display: flex; justify-content: {align}; margin: 5px 0; word-wrap: break-word; align-items: center;">
            <span style='margin: 0 8px;'>{icon}</span>
            <div style="background-color: {bg_color}; color: white; padding: 10px; border-radius: 10px; max-width: 60%; text-align: left;">
                {html.escape(message)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Chat Input
user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state["messages"].append(("User", user_input))
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; margin: 5px 0; word-wrap: break-word; align-items: center;">
            <span style='margin-left: 8px;'>👤</span>
            <div style="background-color: #1E3A8A; color: white; padding: 10px; border-radius: 10px; max-width: 60%; text-align: left;">
                {html.escape(user_input)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    bot_placeholder = st.empty()

    # Show initial "Thinking..." message
    bot_placeholder.markdown(
        f"""
        <div style="display: flex; justify-content: flex-start; margin: 5px 0; word-wrap: break-word; align-items: center;">
            <span style='margin-right: 8px;'>🤖</span>
            <div style="background-color: #374151; color: white; padding: 10px; border-radius: 10px; max-width: 60%; text-align: left;">
                <em>Thinking...🤔</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    response = ""
    llm = OllamaLLM(model="mistral")

    try:
        if vector_db and vector_db._collection.count() > 0:
            print('Vector is not empty')

            retriever = vector_db.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(user_input)

            context = [doc.page_content for doc in retrieved_docs] if retrieved_docs else None

            context_text = "\n".join(context)

            prompt = """You are an AI assistant answering questions based on the following retrieved context
                    If the retrieved context is relevant, use it to answer concisely, 
                    otherwise ignore it and provide your own response\n"""
            
            prompt += f"Context: {context_text}\n\nUser Question: {user_input}" if context_text else user_input

            response_generator = llm.stream(input=prompt)

            print('||||||||', response_generator, type(response_generator))
        else:
            print('Vector is empty')
            response_generator = llm.stream(input=user_input)

        i = 0  # ✅ Re-added index tracking for first response chunk

        # **Streaming Response**
        for chunk in response_generator:  
            text_chunk = chunk if isinstance(chunk, str) else chunk.get("text", "")  # ✅ Normalized chunk handling
            response += text_chunk

            # ✅ Remove "Thinking..." after first chunk
            if i == 0:
                bot_placeholder.empty()

            i += 1  # ✅ Increment the index

            bot_placeholder.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin: 5px 0; word-wrap: break-word; align-items: center;">
                    <span style='margin-right: 8px;'>🤖</span>
                    <div style="background-color: #374151; color: white; padding: 10px; border-radius: 10px; max-width: 60%; text-align: left;">
                        {html.escape(response)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Append to Chat History
        st.session_state["messages"].append(("Bot", response))

    except Exception as e:
        st.error(f"Error during retrieval or response generation: {e}")
        raise
