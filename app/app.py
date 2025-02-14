import os
from app.utils import build_pdf_vectorstore, load_local_index
from app.chat import interactive_session

PDF_DIRECTORY = "resources/pdf"
INDEX_PATH = "resources/vectorspace"

def main():
    """Ensure FAISS index exists before starting interactive chat."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        print(f"Vector store not found. Building from PDFs in {PDF_DIRECTORY}...")
        os.makedirs(INDEX_PATH, exist_ok=True)  # Ensure directory exists
        vector_store = build_pdf_vectorstore(PDF_DIRECTORY, index_save_path=INDEX_PATH)
        print(f"Vector store saved at {INDEX_PATH}")
    else:
        print(f"Loading existing vector store from {INDEX_PATH}...")
        vector_store = load_local_index(INDEX_PATH)

    print("\n✅ Vector store is ready! You can now start asking questions.\n")
    interactive_session(vector_store)

if __name__ == "__main__":
    main()
