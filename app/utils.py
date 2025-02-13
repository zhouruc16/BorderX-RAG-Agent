import os
from typing import List
import subprocess

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Disable parallelism warnings from HuggingFace tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def extract_documents_from_pdf(pdf_path: str,
                               tesseract_cmd: str = None,
                               min_text_chars: int = 30,
                               dpi: int = 300) -> List[Document]:
    """
    Extracts text from a PDF file page by page.
    Uses direct extraction via pdfplumber; if a page returns too little text,
    it falls back to OCR via pytesseract.
    
    Args:
        pdf_path (str): Path to the PDF file.
        tesseract_cmd (str, optional): Full path to Tesseract executable if not on PATH.
        min_text_chars (int, optional): Minimum number of characters to consider the page valid.
        dpi (int, optional): Resolution for converting PDF pages to images for OCR.
    
    Returns:
        List[Document]: A list of Document objects (one per page) with metadata.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_number in range(total_pages):
            page = pdf.pages[page_number]
            page_text = page.extract_text() or ""
            page_text = page_text.strip()

            # If extracted text is too short, fallback to OCR
            if len(page_text) < min_text_chars:
                images = convert_from_path(pdf_path, dpi=dpi,
                                           first_page=page_number + 1,
                                           last_page=page_number + 1)
                if images:
                    ocr_text = pytesseract.image_to_string(images[0]) or ""
                    ocr_text = ocr_text.strip()
                    page_text = ocr_text

            # If still empty, note it (or you could choose to skip this page)
            if not page_text:
                page_text = "[No text found]"

            metadata = {"source": os.path.basename(pdf_path), "page": page_number + 1}
            documents.append(Document(page_content=page_text, metadata=metadata))
    return documents


def build_pdf_vectorstore(pdf_directory: str,
                          index_save_path: str = None,
                          tesseract_cmd: str = None,
                          min_text_chars: int = 30,
                          dpi: int = 300,
                          chunk_size: int = 500,
                          chunk_overlap: int = 100) -> FAISS:
    """
    Processes all PDF files in a given directory by extracting text from each page
    (with an OCR fallback) and builds a FAISS vector store from text chunks.
    Optionally, saves the FAISS index to a local file.

    Args:
        pdf_directory (str): Folder containing PDF files.
        index_save_path (str, optional): Path to save the FAISS index.
        tesseract_cmd (str, optional): Path to Tesseract executable (if needed).
        min_text_chars (int, optional): Minimum characters required for direct text extraction.
        dpi (int, optional): DPI for image conversion for OCR.
        chunk_size (int, optional): Maximum number of characters per chunk.
        chunk_overlap (int, optional): Overlap between chunks.

    Returns:
        FAISS: A FAISS vector store containing the embedded document chunks.
    """
    all_documents = []
    # Process each PDF in the directory
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            docs = extract_documents_from_pdf(pdf_path,
                                              tesseract_cmd=tesseract_cmd,
                                              min_text_chars=min_text_chars,
                                              dpi=dpi)
            all_documents.extend(docs)
    
    # Split the extracted documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    docs_chunks = text_splitter.split_documents(all_documents)
    # Filter out any empty chunks
    docs_chunks = [doc for doc in docs_chunks if doc.page_content.strip()]
    
    if not docs_chunks:
        raise ValueError("No valid text chunks found to build the vector store. "
                         "Check your OCR output or text splitter settings.")
    
    # Compute embeddings using a free, small model and build the FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs_chunks, embeddings)
    
    # Save the vector store to a local file if a path is provided
    if index_save_path:
        vectorstore.save_local(index_save_path)
    
    return vectorstore


def retrieve_context(vectorstore, query: str, k: int = 3) -> str:
    """
    Retrieve the top k similar document chunks from the vector store,
    assembling them into a single context string that includes
    source filenames and page numbers.
    
    Args:
        vectorstore: The FAISS vector store.
        query (str): The user's query.
        k (int): Number of top matching chunks to retrieve.
    
    Returns:
        str: An assembled context string.
    """
    # The similarity search automatically encodes the query.
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    
    context_parts = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        text = doc.page_content.strip()
        context_parts.append(f"Source: {source} - Page: {page}\n{text}")
    
    # Separate chunks clearly
    context = "\n\n".join(context_parts)
    return context

def load_local_index(index_save_path: str, 
                     embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Loads a locally saved FAISS vector store from the given index path.
    
    Args:
        index_save_path (str): Path to the saved FAISS index.
        embeddings_model_name (str, optional): The HuggingFace model to use for embeddings.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".
    
    Returns:
        FAISS: The loaded FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vectorstore = FAISS.load_local(index_save_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore


def call_deepseek_local(prompt: str, model: str = "deepseek-r1:1.5b") -> str:
    """
    Calls the locally installed DeepSeek R1:8B model via the Ollama CLI.
    
    Note: No unsupported flags are used.
    
    Args:
        prompt (str): The prompt to feed into the model.
        model (str): The model identifier.
    
    Returns:
        str: The raw output from the model.
    """
    # Build the command without any unsupported flags.
    command = ["ollama", "run", model]
    
    result = subprocess.run(command, input=prompt, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception("Local model execution failed: " + result.stderr)
    
    return result.stdout.strip()


def answer_query(vectorstore, query: str, k: int = 3) -> dict:
    """
    Given a user query, retrieve context from the vector store, construct
    a prompt, and generate an answer with the local DeepSeek R1:8B model.
    The model is instructed to include both the answer and the references.
    
    Args:
        vectorstore: The FAISS vector store.
        query (str): The user query.
        k (int): Number of retrieved chunks to use as context.
    
    Returns:
        dict: A dictionary with two keys:
              "answer": The answer text.
              "references": The document names and pages referenced.
    """
    # Retrieve context (includes source file names and page numbers).
    context = retrieve_context(vectorstore, query, k=k)
    
    # Construct the prompt with explicit instructions.
    prompt = (
        "You are a helpful assistant. Answer the query only using the following context. "
        "Do not include any information not present in the context. Include references to the source "
        "(i.e., file name and page number) in your answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Query: {query}\n\n"
        "Please respond using the following format exactly:\n"
        "Answer: <your answer here>\n"
        "References: <list of source file names and page numbers, separated by commas>\n"
    )
    
    # Call the local DeepSeek model.
    output = call_deepseek_local(prompt)
    
    # Parse the output expecting the following format:
    # Answer: <text>
    # References: <text>
    answer_lines = []
    references_lines = []
    current_section = None
    for line in output.splitlines():
        if line.startswith("Answer:"):
            current_section = "answer"
            answer_lines.append(line[len("Answer:"):].strip())
        elif line.startswith("References:"):
            current_section = "references"
            references_lines.append(line[len("References:"):].strip())
        else:
            if current_section == "answer":
                answer_lines.append(line.strip())
            elif current_section == "references":
                references_lines.append(line.strip())
    
    answer_text = "\n".join(answer_lines).strip()
    references_text = "\n".join(references_lines).strip()
    
    return {"answer": answer_text, "references": references_text}
