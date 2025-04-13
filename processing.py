"""
Program 1: Preprocess documents into a FAISS index and save it to disk.
"""

import os
import glob

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

def main():
    # Step 1: Load the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embedding model loaded.")

    # Step 2: Load all text documents from cleaned_data/
    all_documents = []
    for filepath in glob.glob("cleaned_data/*.txt"):
        loader = TextLoader(filepath, encoding="utf-8")
        all_documents.extend(loader.load())
    print(f"✅ Loaded {len(all_documents)} documents.")

    # Step 3: Split into chunks of 1000 characters (100 overlap)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_documents)
    print(f"✅ Split into {len(docs)} chunks.")

    # Step 4: Build the FAISS index from documents
    print("⏳ Building FAISS index…")
    vector_store = FAISS.from_documents(docs, embedding_model)
    print("✅ FAISS index built.")

    # Step 5: Save the FAISS index to disk
    index_dir = "embeddings/faiss_index_directory_2"
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    print(f"✅ FAISS index saved to '{index_dir}'.")

if __name__ == "__main__":
    main()
