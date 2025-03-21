# saved_doc_with_no_ssl_verify.py

import os
# Disable SSL certificate verification for Hugging Face Hub downloads.
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = "1"

# Step 1: Load the embedding model
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully!")

# Step 2: Load the saved FAISS index from disk with dangerous deserialization allowed
from langchain_community.vectorstores import FAISS
print("Loading FAISS index from disk...")
vector_store = FAISS.load_local(
    "embeddings/faiss_index_directory", 
    embedding_model, 
    allow_dangerous_deserialization=True
)
print("FAISS index loaded successfully.")

# Step 3: Create a retriever object
print("Creating retriever object...")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Retriever object created successfully.")

# Step 4: Define the prompt template for your legal assistant
from langchain_core.prompts import PromptTemplate

prompt_template = """
You are a legal assistant chatbot specialized in Indian law.
Use the following context to answer the question. If the context does not provide sufficient information, reply "I do not have enough information to answer that."
    
Context:
{context}

Question:
{input}

Answer:
"""

prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)
print("Prompt template created successfully.")

# Step 5: Initialize the language model and build the retrieval chain using the new API
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

load_dotenv()  # Load environment variables from .env file

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    print("OpenAI API key retrieved successfully.")
else:
    print("Warning: OpenAI API key not found. Make sure to set it in your .env file.")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)
print("Language model initialized successfully.")

# Create a combine chain using the "stuff" approach with your prompt
combine_chain = create_stuff_documents_chain(llm, prompt)

# Build the retrieval chain using your retriever and the combine chain
rag_chain = create_retrieval_chain(retriever, combine_chain)
print("RAG chain built successfully using the new API.")

# Step 6: Define and process a sample legal query
sample_query = "What is the difference between a petition and a plaint in Indian law?"
print(f"Processing query: {sample_query}")

# The retrieval chain expects the question under the key "input"
result = rag_chain.invoke({"input": sample_query})
print("Query processed successfully.")
print("Answer:", result)
