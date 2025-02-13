from utils import build_pdf_vectorstore, answer_query

def main():
    # Set the directory containing your PDF files.
    pdf_directory = "resources/pdf"
    
    # Build the vector store directly from PDFs.
    print("Building vector store...")
    vector_store = build_pdf_vectorstore(pdf_directory,
                                         tesseract_cmd=None,  # Adjust if needed
                                         min_text_chars=30,
                                         dpi=300,
                                         chunk_size=500,
                                         chunk_overlap=100)
    
    print("Vector store successfully built. The following document chunks are stored:")
    for doc in vector_store.docstore._dict.values():
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        print(f"Document from {source} - Page {page}")
    
    # Get the user query.
    query = input("Enter your query: ")
    
    try:
        result = answer_query(vector_store, query, k=3)
        print("\n--- Final Answer ---")
        print(result["answer"])
        print("\n--- References ---")
        print(result["references"])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
