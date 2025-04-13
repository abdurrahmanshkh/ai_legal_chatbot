# ask_questions.py

import os
import logging

from dotenv import load_dotenv
import google.generativeai as genai  # Google Gen AI SDK :contentReference[oaicite:0]{index=0}

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logger()
    load_dotenv()  # loads GEMINI_API_KEY

    # --- Step 1: Configure Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not set in environment.")
        return
    genai.configure(api_key=gemini_api_key)  # :contentReference[oaicite:1]{index=1}
    logger.info("Gemini API configured.")

    # --- Step 2: Load embedding model
    logger.info("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("Embedding model loaded.")

    # --- Step 3: Load FAISS index
    index_dir = "embeddings/faiss_index_2"
    logger.info(f"Loading FAISS index from {index_dir}...")
    vector_store = FAISS.load_local(
        index_dir,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    logger.info("FAISS index loaded.")

    # --- Step 4: Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    logger.info("Retriever ready (k=3).")

    # --- Step 5: Prompt template
    prompt_template = """
You are a legal assistant chatbot specialized in Indian law.
Use the following context to answer the question. If the context does not provide sufficient information, reply "I do not have enough information to answer that."

Context:
{context}

Question:
{input}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=prompt_template
    )
    logger.info("Prompt template set up.")

    # --- Step 6: Initialize Gemini model
    model_name = "gemini-2.0-flash"
    logger.info(f"Initializing Gemini model: {model_name}...")
    model = genai.GenerativeModel(model_name)
    logger.info("Gemini model initialized.")

    def get_gemini_response(context: str, query: str) -> str:
        full_prompt = prompt.format(context=context, input=query)
        response = model.generate_content(full_prompt)
        return response.text or "I do not have enough information to answer that."

    # --- REPL loop
    logger.info("Starting interactive Q&A session. Type 'exit' or 'quit' to stop.")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"exit", "quit"}:
            logger.info("Exiting session.")
            break
        if not query:
            print("Please enter a valid question.")
            continue

        logger.info(f"Retrieving context for: {query!r}")
        docs = retriever.invoke(query)
        context = "\n".join(d.page_content for d in docs)
        logger.info("Context retrieved; querying Gemini...")

        answer = get_gemini_response(context, query)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
