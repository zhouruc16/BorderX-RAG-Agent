import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.utils import build_pdf_vectorstore, load_local_index
from app.chat import deepseek_chat

# Define paths (relative to the project root)
PDF_DIRECTORY = "app/resources/pdf"
INDEX_PATH = "app/resources/vectorspace"

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (adjust allow_origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build or load the FAISS vector store
if not os.path.exists(INDEX_PATH) or not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
    print(f"Vector store not found. Building from PDFs in {PDF_DIRECTORY}...")
    os.makedirs(INDEX_PATH, exist_ok=True)
    vector_store = build_pdf_vectorstore(PDF_DIRECTORY, index_save_path=INDEX_PATH)
    print(f"Vector store saved at {INDEX_PATH}")
else:
    print(f"Loading existing vector store from {INDEX_PATH}...")
    vector_store = load_local_index(INDEX_PATH)

# Request model for incoming query
class QueryRequest(BaseModel):
    query: str
    k: int = 5
    model: str = "deepseek-r1:14b"

@app.get("/")
def home():
    return {"message": "RAG Agent API is running!"}

@app.post("/query/")
def query_rag(request: QueryRequest):
    query_text = request.query.strip()
    if not query_text:
        return {"error": "Query cannot be empty"}
    
    response = deepseek_chat(vector_store, query_text, k=request.k)
    return {
        "answer": response["answer"],
        "references": response["references"],
    }

# To run: from the project root, run
# uvicorn app.api:app --reload
