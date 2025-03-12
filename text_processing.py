import pypdf
import yake
import torch
import ollama
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.corpus import wordnet

# Load models
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = SentenceTransformer("BAAI/bge-reranker-large")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "pdf_documents"

# Create Elasticsearch Index
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "text": {"type": "text"},  
                "keywords": {"type": "keyword"},  
                "embedding": {"type": "dense_vector", "dims": 384}  
            }
        }
    })

# Extract text & keywords from PDF
def extract_text_and_keywords(pdf_path):
    pdf_reader = pypdf.PdfReader(pdf_path)
    text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=10)
    keywords = [kw[0] for kw in kw_extractor.extract_keywords(text)]
    return text, keywords

# Chunk text for better retrieval
def chunk_text(text, chunk_size=256, overlap=128):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

# Store PDF content in Elasticsearch
def index_pdf(doc_id, text, keywords):
    embedding = embed_model.encode(text).tolist()
    es.index(index=index_name, id=doc_id, body={"text": text, "keywords": keywords, "embedding": embedding})

# Hybrid search (BM25 + Vector + Keyword Filtering)
def hybrid_search(query, filter_keywords=None, top_k=10):
    query_vector = embed_model.encode(query).tolist()
    hybrid_query = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"text": query}},  
                    {
                        "script_score": {  
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    }
                ]
            }
        }
    }
    if filter_keywords:
        hybrid_query["query"]["bool"]["filter"] = [{"terms": {"keywords": filter_keywords}}]

    response = es.search(index=index_name, body={"size": top_k, "query": hybrid_query})
    return [hit["_source"]["text"] for hit in response["hits"]["hits"]]

# Re-rank results with BGE
def rerank_results(query, retrieved_docs, top_k=3):
    rerank_scores = reranker.encode([[query, doc] for doc in retrieved_docs], convert_to_tensor=True)
    reranked_indices = torch.argsort(rerank_scores, descending=True)[:top_k]
    return [retrieved_docs[i] for i in reranked_indices]

# Summarize context before sending to RAG
def summarize_context(chunks):
    combined_text = " ".join(chunks)
    summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Expand query with synonyms (Query Expansion)
def expand_query(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    expanded_query = query + " " + " ".join(list(synonyms))
    return expanded_query

# Final RAG function
def chat_with_mistral(user_query):
    expanded_query = expand_query(user_query)
    retrieved_docs = hybrid_search(expanded_query)
    reranked_docs = rerank_results(expanded_query, retrieved_docs)
    context_summary = summarize_context(reranked_docs)

    prompt = f"Use the following context to answer the question:\n\n{context_summary}\n\nQuestion: {user_query}\nAnswer:"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Example Usage
if __name__ == "__main__":
    # Step 1: Process PDF and store in Elasticsearch
    pdf_text, pdf_keywords = extract_text_and_keywords("example.pdf")
    for i, chunk in enumerate(chunk_text(pdf_text)):
        index_pdf(i, chunk, pdf_keywords)

    # Step 2: Ask a question
    print(chat_with_mistral("How does AI work in finance?"))
