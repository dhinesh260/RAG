import streamlit as st
import chromadb
import os
import fitz  # PyMuPDF for PDFs
import docx
from sentence_transformers import SentenceTransformer
from ollama import generate

# Set up Streamlit page
st.set_page_config(layout="wide")

# Initialize ChromaDB
DB_DIR = "chroma_db"
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# Load MiniLM embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name="rag_collection")

# Function to extract text from different file types
def extract_text_from_file(uploaded_file):
    try:
        file_type = uploaded_file.type
        text = ""

        if file_type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join([page.get_text("text") for page in doc])
        elif file_type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError("Unsupported file format.")

        print("[DEBUG] Extracted text length:", len(text))
        return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        print("[ERROR] File extraction error:", e)
        return None

# Function to process text: store full document in ChromaDB
def process_and_store_text(text):
    try:
        embedding = embedding_model.encode([text])[0].tolist()

        collection.add(
            ids=[str(len(collection.get()["documents"]))],
            embeddings=[embedding],
            documents=[text]
        )

        print("[DEBUG] Stored 1 document in ChromaDB")
        stored_count = len(collection.get()["documents"])
        print(f"[DEBUG] Total stored documents in ChromaDB: {stored_count}")
        st.success("File processed and stored in ChromaDB!")
    except Exception as e:
        st.error(f"Error processing text: {e}")
        print("[ERROR] ChromaDB storage error:", e)

# Sidebar: File Upload
with st.sidebar:
    st.title("Upload File")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])

    if uploaded_file:
        st.info("Processing file...")
        text_data = extract_text_from_file(uploaded_file)
        if text_data:
            process_and_store_text(text_data)

# Main Chat Interface
st.title("Chat Interface")
st.markdown("### Chat History")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [("Bot", "How can I assist you today?")]

# Display chat history
for user, message in st.session_state["messages"]:
    align = "flex-end" if user == "User" else "flex-start"
    bg_color = "#1E3A8A" if user == "User" else "#374151"

    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: {align};
            margin: 5px 0;">
            <div style="
                background-color: {bg_color}; 
                color: white;
                padding: 10px;
                border-radius: 10px;
                max-width: 60%;
                text-align: left;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Chat input box at the bottom
user_input = st.chat_input("Ask anything...")

# RAG-based response generation
if user_input:
    st.session_state["messages"].append(("User", user_input))
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: flex-end;
            margin: 5px 0;">
            <div style="
                background-color: #1E3A8A; 
                color: white;
                padding: 10px;
                border-radius: 10px;
                max-width: 60%;
                text-align: left;">
                {user_input}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show "thinking" message in UI
    bot_placeholder = st.empty()
    bot_placeholder.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: flex-start;
            margin: 5px 0;">
            <div style="
                background-color: #374151; 
                color: white;
                padding: 10px;
                border-radius: 10px;
                max-width: 60%;
                text-align: left;">
                Bot is thinking...
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        with st.spinner("Thinking..."):  # Show a spinner while processing
            # Get user query embedding
            query_embedding = embedding_model.encode([user_input])[0].tolist()

            # Retrieve relevant documents using cosine similarity
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )

            retrieved_docs = results["documents"][0] if results["documents"] else []
            print(f"[DEBUG] Retrieved {len(retrieved_docs)} relevant docs from ChromaDB")

            if retrieved_docs:
                context = "\n".join(retrieved_docs)
                system_prompt = (
                    "You are an AI assistant answering questions based on the following retrieved context:\n\n"
                    "If the retrieved context is relevant, use it to answer concisely.\n"
                    "If it is not relevant, ignore it and provide your own response.\n\n"
                    f"Context:\n{context}\n\n"
                    "Now, answer the following question:\n\n"
                    f"{user_input}"
                )
            else:
                print("[DEBUG] No relevant context found. Sending raw query to LLM.")
                system_prompt = user_input

            # Stream bot response properly
            bot_placeholder.empty()  # Remove "thinking" message
            bot_response = ""
            message_placeholder = st.empty()

            for chunk_response in generate(model="mistral", prompt=system_prompt, stream=True):
                chunk = chunk_response['response']
                bot_response += chunk

                # Update UI with streamed response
                message_placeholder.markdown(
                    f"""
                    <div style="
                        display: flex;
                        justify-content: flex-start;
                        margin: 5px 0;">
                        <div style="
                            background-color: #374151; 
                            color: white;
                            padding: 10px;
                            border-radius: 10px;
                            max-width: 60%;
                            text-align: left;">
                            {bot_response}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        print("[DEBUG] Final bot response:", bot_response)

        st.session_state["messages"].append(("Bot", bot_response))

    except Exception as e:
        st.error(f"Error generating response: {e}")
        print("[ERROR] LLM processing error:", e)
