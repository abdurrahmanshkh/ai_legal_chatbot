# process_data.py

import json
import logging
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

def load_jsonl(path: Path, logger) -> list[dict]:
    logger.info(f"Loading JSONL data from {path!s}...")
    data = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            data.append(obj)
            if i % 5000 == 0:
                logger.info(f"  Loaded {i} records so far...")
    logger.info(f"Total records loaded: {len(data)}")
    return data

def create_documents(records: list[dict], logger) -> list[Document]:
    logger.info("Converting records to LangChain Document objects...")
    docs = []
    for rec in records:
        text = f"Question: {rec['question']}\nAnswer: {rec['answer']}"
        metadata = {
            "id": rec.get("id"),
            "type": rec.get("type"),
            "category": rec.get("category"),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    logger.info(f"Created {len(docs)} Document objects.")
    return docs

def main():
    logger = setup_logger()

    # --- Step 1: Load embedding model
    logger.info("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("Embedding model loaded successfully.")

    # --- Step 2: Load JSONL data
    jsonl_path = Path("data.jsonl")
    records = load_jsonl(jsonl_path, logger)

    # --- Step 3: Convert to Documents
    docs = create_documents(records, logger)

    # --- Step 4: Split into chunks
    logger.info("Splitting documents into chunks (1,000 chars + 100 overlap)...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_docs = splitter.split_documents(docs)
    logger.info(f"Total chunks created: {len(chunked_docs)}")

    # --- Step 5: Build FAISS index
    logger.info("Building FAISS index from chunked documents...")
    vector_store = FAISS.from_documents(chunked_docs, embedding_model)
    logger.info("FAISS index created.")

    # --- Step 6: Save index to disk
    save_dir = Path("embeddings/faiss_index_2")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving FAISS index to {save_dir!s}...")
    vector_store.save_local(str(save_dir))
    logger.info("Index saved. Processing complete.")

if __name__ == "__main__":
    main()
