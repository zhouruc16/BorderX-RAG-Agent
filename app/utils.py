import os
import torch
import time
from pdfminer.pdfparser import PDFSyntaxError
import subprocess
import faiss
from typing import List
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Use updated community imports:
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Disable parallelism warnings from HuggingFace tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Determine the device for GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device.upper()}")

# Initialize EasyOCR instead of Tesseract
try:
    import easyocr
    import numpy as np
    ocr_gpu = True if device == "cuda" else False
    reader = easyocr.Reader(['en'], gpu=ocr_gpu)
    print("✅ EasyOCR initialized successfully.")
except Exception as e:
    print("⚠️ Failed to initialize EasyOCR:", e)
    reader = None

def extract_documents_from_pdf(pdf_path: str,
                               min_text_chars: int = 30,
                               dpi: int = 300) -> List[Document]:
    """
    Extracts text from a PDF file page by page using pdfplumber.
    If a page returns too little text, it falls back to OCR using EasyOCR.
    """
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_number in range(total_pages):
                page = pdf.pages[page_number]
                page_text = page.extract_text() or ""
                page_text = page_text.strip()

                # Fallback to OCR if extracted text is too short
                if len(page_text) < min_text_chars and reader is not None:
                    images = convert_from_path(pdf_path, dpi=dpi,
                                               first_page=page_number + 1,
                                               last_page=page_number + 1)
                    if images:
                        # Convert the PIL image to a numpy array
                        image_np = np.array(images[0])
                        # Use EasyOCR to read text (detail=0 returns only text)
                        result = reader.readtext(image_np, detail=0, paragraph=True)
                        ocr_text = " ".join(result) if result else ""
                        ocr_text = ocr_text.strip()
                        page_text = ocr_text

                if not page_text:
                    page_text = "[No text found]"

                metadata = {"source": os.path.basename(pdf_path), "page": page_number + 1}
                documents.append(Document(page_content=page_text, metadata=metadata))
    except PDFSyntaxError as e:
        print(f"⚠️ Skipping file {pdf_path} due to PDFSyntaxError: {e}")
    except Exception as e:
        print(f"⚠️ An unexpected error occurred while processing {pdf_path}: {e}")
    return documents

def build_pdf_vectorstore(pdf_directory: str,
                          index_save_path: str = None,
                          min_text_chars: int = 30,
                          dpi: int = 300,
                          chunk_size: int = 500,
                          chunk_overlap: int = 100) -> FAISS:
    all_documents = []
    # Gather PDF files and sort by file size (smallest to largest)
    file_list = [filename for filename in os.listdir(pdf_directory) if filename.lower().endswith(".pdf")]
    file_list = sorted(file_list, key=lambda f: os.path.getsize(os.path.join(pdf_directory, f)))
    total_files = len(file_list)
    print(f"Found {total_files} PDF files to process (sorted by file size).")

    for idx, filename in enumerate(file_list, start=1):
        pdf_path = os.path.join(pdf_directory, filename)
        print(f"📄 Processing file {idx}/{total_files}: {filename}...")
        start_time = time.time()
        docs = extract_documents_from_pdf(pdf_path,
                                          min_text_chars=min_text_chars,
                                          dpi=dpi)
        all_documents.extend(docs)
        elapsed = time.time() - start_time
        estimated_remaining = elapsed * (total_files - idx)
        print(f"Processed {filename} in {elapsed:.2f} seconds. Estimated remaining time: {estimated_remaining:.2f} seconds.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    docs_chunks = text_splitter.split_documents(all_documents)
    docs_chunks = [doc for doc in docs_chunks if doc.page_content.strip()]

    if not docs_chunks:
        raise ValueError("No valid text chunks found to build the vector store. "
                         "Check your OCR output or text splitter settings.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    print(f"🚀 HuggingFace Embeddings running on: {embeddings.client.device}")

    vectorstore = FAISS.from_documents(docs_chunks, embeddings)

    num_gpus = faiss.get_num_gpus()
    if num_gpus > 0:
        print(f"🚀 Converting FAISS index to GPU (GPUs available: {num_gpus})")
        res = faiss.StandardGpuResources()
        cpu_index = vectorstore.index
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        vectorstore.index = gpu_index
    else:
        print("⚡ No GPU detected for FAISS. Using CPU index.")

    if index_save_path:
        vectorstore.save_local(index_save_path)
        print(f"✅ Vector store saved at: {index_save_path}")

    return vectorstore

def rerank_documents(docs: List[Document], query: str, embeddings: HuggingFaceEmbeddings, top_k: int) -> List[Document]:
    """
    Reranks a list of Document objects based on their cosine similarity with the query.
    Uses the provided embeddings to compute the similarity.
    """
    # Embed the query
    query_embedding = embeddings.embed_query(query)
    # Compute embeddings for each document chunk
    doc_texts = [doc.page_content for doc in docs]
    doc_embeddings = embeddings.embed_documents(doc_texts)
    
    similarities = []
    for emb in doc_embeddings:
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)
        similarities.append(sim)
    
    # Sort documents by similarity in descending order and return top_k
    ranked = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]

def retrieve_context(vectorstore, query: str, k: int = 3) -> str:
    """
    Retrieves candidate document chunks from the vector store,
    reranks them for precision, and returns a concatenated context string.
    """
    # Retrieve more candidates (k*5 instead of k*3)
    candidates = vectorstore.similarity_search(query, k=k * 5)
    
    # Initialize an embeddings instance for reranking
    rerank_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    
    # Rerank the candidate documents based on cosine similarity
    ranked_docs = rerank_documents(candidates, query, rerank_embeddings, k)
    
    context_parts = []
    for doc in ranked_docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        text = doc.page_content.strip()
        context_parts.append(f"Source: {source} - Page: {page}\n{text}")
    return "\n\n".join(context_parts)

def load_local_index(index_save_path: str,
                     embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name,
        model_kwargs={"device": device}
    )
    vectorstore = FAISS.load_local(index_save_path, embeddings, allow_dangerous_deserialization=True)
    num_gpus = faiss.get_num_gpus()
    if num_gpus > 0:
        print(f"🚀 Converting loaded FAISS index to GPU (GPUs available: {num_gpus})")
        res = faiss.StandardGpuResources()
        cpu_index = vectorstore.index
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        vectorstore.index = gpu_index
    else:
        print("⚡ Loaded FAISS index using CPU.")
    return vectorstore
