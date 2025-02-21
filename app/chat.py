from ollama import chat
from langchain_community.vectorstores import FAISS
from app.utils import retrieve_context, load_local_index
import torch

# Log GPU availability (DeepSeek auto-detects GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DeepSeek: Running on device {device.upper()}")

def deepseek_chat(vectorstore: FAISS, query: str, k: int = 3, model: str = "deepseek-r1:14b") -> dict:
    """
    Retrieves context from the vector store and sends a query to a larger DeepSeek model via Ollama.
    Tailored for legal/business documents, aiming for higher accuracy and more precise references.
    """
    # Retrieve top-k documents (with rerank in utils)
    context = retrieve_context(vectorstore, query, k=k)

    # Enhanced prompt for legal/business context
    system_message = (
        "You are a specialized legal consultant, highly trained in corporate, "
        "business, and contractual law. You have access only to the provided context. "
        "Your goal is to provide clear, concise, and precise answers, "
        "citing relevant sections and pages from the context. "
        "If the question cannot be answered from the context, respond with: "
        "'Please provide more details so I can give an accurate answer.'"
    )

    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"USER QUERY: {query}\n\n"
        "INSTRUCTIONS:\n"
        "1. Respond as a legal consultant, offering expert advice based strictly on the provided context.\n"
        "2. Do not include any information not found in the context.\n"
        "3. Cite sources at the end using the format:\n"
        "   References: <filename - page>, <filename - page>, ...\n"
        "4. If there is not enough context, state: 'Insufficient context to provide a definitive answer.'\n"
        "\n"
        "FORMAT:\n"
        "Answer: <your legal/business answer here>\n"
        "References: <list of source file names and page numbers>"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # (Optional) Check GPU usage by the model
    # gpu_check = chat(model=model, messages=[{"role": "system", "content": "Are you using GPU?"}])
    # print("DeepSeek GPU Check:", gpu_check.message.content)

    # Call the larger DeepSeek model
    response = chat(model=model, messages=messages)

    # Parse out the 'Answer:' and 'References:' sections
    answer_lines = []
    references_lines = []
    current_section = None
    for line in response.message.content.splitlines():
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

def interactive_session(vectorstore: FAISS, model: str = "deepseek-r1:14b", k: int = 5) -> None:
    """
    Runs an interactive command-line session for user queries using a larger DeepSeek model.
    """
    print("Interactive DeepSeek Legal/Business Consultant Session Started. Type 'exit' or 'quit' to end.")
    while True:
        user_query = input("Enter your legal/business query: ").strip()
        if user_query.lower() in ("exit", "quit"):
            print("Exiting interactive session.")
            break
        
        result = deepseek_chat(vectorstore, user_query, k=k, model=model)
        print("\nResponse:")
        print(result["answer"])
        print("References:", result["references"])
        print("-" * 50)

def main() -> None:
    """
    Main function that loads the local FAISS vector store and starts an interactive session
    with the bigger DeepSeek model.
    """
    vectorstore = load_local_index("app/resources/vectorspace")
    print("Loaded local FAISS vector store from 'app/resources/vectorspace'.")
    interactive_session(vectorstore)
