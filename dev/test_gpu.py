import torch
import faiss
import ollama

# Check PyTorch
print("PyTorch Device:", "GPU" if torch.cuda.is_available() else "CPU")

# Check FAISS
print("FAISS GPUs:", faiss.get_num_gpus())

# Check DeepSeek (Ollama)
response = ollama.chat(model="deepseek-r1:8b", messages=[{"role": "system", "content": "Are you using GPU?"}])
print("Ollama GPU Check:", response['message'])


try:
    import easyocr
    import numpy as np
    ocr_gpu = True if device == "cuda" else False
    reader = easyocr.Reader(['en'], gpu=ocr_gpu)
    print("✅ EasyOCR initialized successfully.")
except Exception as e:
    print("⚠️ Failed to initialize EasyOCR:", e)
    reader = None