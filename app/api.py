import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.utils import build_pdf_vectorstore, load_local_index
from app.chat import deepseek_chat

# Define paths
PDF_DIRECTORY = "app/resources/pdf"
INDEX_PATH = "app/resources/vectorspace"

# Initialize FastAPI app
app = FastAPI()

# Allow frontend requests (CORS settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load or build FAISS index
if not os.path.exists(INDEX_PATH) or not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
    print(f"Vector store not found. Building from PDFs in {PDF_DIRECTORY}...")
    os.makedirs(INDEX_PATH, exist_ok=True)
    vector_store = build_pdf_vectorstore(PDF_DIRECTORY, index_save_path=INDEX_PATH)
    print(f"Vector store saved at {INDEX_PATH}")
else:
    print(f"Loading existing vector store from {INDEX_PATH}...")
    vector_store = load_local_index(INDEX_PATH)


class QueryRequest(BaseModel):
    query: str
    k: int = 3  # Default top-k retrieved documents


@app.get("/")
def home():
    return {"message": "RAG Agent API is running!"}


@app.post("/query/")
def query_rag(request: QueryRequest):
    """
    Handles user queries by retrieving context from FAISS
    and sending it to the DeepSeek model.
    """
    query_text = request.query.strip()
    if not query_text:
        return {"error": "Query cannot be empty"}

    response = deepseek_chat(vector_store, query_text, k=request.k)
    return {
        "answer": response["answer"],
        "references": response["references"],
    }


# Run API using: uvicorn backend.api:app --reload
