# Step 1: Load the embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# Now you can initialize your embedding model:
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Model loaded successfully!")

import faiss

# Load the pre-built FAISS index
loaded_index = faiss.read_index("embeddings/faiss_index.idx")
print("Raw FAISS index loaded from disk.")

from langchain_community.vectorstores import FAISS

# Reconstruct the vector store with your embedding function and the loaded index.
# Note: Some implementations require also passing in the original documents or a docstore mapping.
vector_store = FAISS(embedding_model.embed_query, loaded_index)
print("FAISS vector store reconstructed using the loaded index.")

# Step 7: Create a retriever object
print("Creating retriever object...")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Retriever object created successfully.")

# Step 8: Create a prompt template
print("Defining prompt template...")
from langchain_core.prompts import PromptTemplate

# Define a template with placeholders for context and user query.
prompt_template = """
You are a legal assistant chatbot specialized in Indian law.
Use the following context to answer the question. If the context does not provide sufficient information, reply "I do not have enough information to answer that."
    
Context:
{context}

Question:
{question}

Answer:
"""

# Create a PromptTemplate object.
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)
print("Prompt template created successfully.")

# Step 9: Build the RAG chain
print("Loading environment variables...")
import os
from dotenv import load_dotenv

load_dotenv()
print("Environment variables loaded.")

# Retrieve the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    print("OpenAI API key retrieved successfully.")
else:
    print("Warning: OpenAI API key not found. Make sure to set it in your .env file.")

print("Initializing the language model...")
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Initialize your language model (adjust model name if necessary)
llm = ChatOpenAI(model_name="llama2-chat", temperature=0.3, openai_api_key=openai_api_key)
print("Language model initialized successfully.")

# Build the RAG chain
print("Building the RAG chain...")
rag_chain = RetrievalQA(
    llm=llm,
    retriever=retriever,
    prompt=prompt,
    return_source_documents=True  # Optional: to return which documents were used
)
print("RAG chain built successfully.")

# Define a sample legal query.
sample_query = "What is the difference between a petition and a plaint in Indian law?"
print(f"Processing query: {sample_query}")

# Use the RAG chain to get an answer.
result = rag_chain.run(sample_query)
print("Query processed successfully.")

# Print the generated answer.
print("Answer:", result)

# (Optional) Print source documents if return_source_documents is enabled.
if isinstance(result, dict) and "source_documents" in result:
    print("Source documents retrieved:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"Source Document {i+1}: {doc.page_content[:200]}...")  # Print the first 200 characters
else:
    print("No source documents returned.")
