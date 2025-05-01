# Step 1: Load the embedding model
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully!")

# Step 2: Load the documents
from langchain_community.document_loaders import TextLoader
import glob

all_documents = []
for filepath in glob.glob("cleaned_data/*.txt"):
    # Specify the encoding as UTF-8
    loader = TextLoader(filepath, encoding="utf-8")
    all_documents.extend(loader.load())

print(f"Number of loaded documents: {len(all_documents)}")

# Step 3: Split the documents into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split into chunks of 1000 characters with an overlap of 100 characters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=2000,
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_documents(all_documents)
print(f"Number of chunks: {len(docs)}")

# (Optional) Step 4: Generate embeddings for each chunk (for debugging)
embeddings_list = []
total = len(docs)
for i, doc in enumerate(docs, start=1):
    embedding = embedding_model.embed_query(doc.page_content)
    embeddings_list.append(embedding)
    print(f"Processed chunk {i}/{total}")
print("Generated embeddings for", len(embeddings_list), "chunks.")

# Step 5: Create the FAISS index from documents and embeddings
from langchain_community.vectorstores import FAISS

print("Creating FAISS index from documents and embeddings...")
vector_store = FAISS.from_documents(docs, embedding_model)
print("FAISS index created.")

# Step 6: Create a retriever object
print("Creating retriever object...")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Retriever object created successfully.")

# Step 7: Save the FAISS index to disk
vector_store.save_local("embeddings/faiss_index_directory_6")
print("FAISS index saved.")

# Step 7: Define the prompt template for your legal assistant
# Note: We change the input variable from "question" to "input" to match the chain's input.
print("Defining prompt template...")
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

# Update the input variables to ["context", "input"] to match the keys passed in by the chain.
prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template)
print("Prompt template created successfully.")

# Step 8: Initialize the language model and build the RAG chain using the new API
print("Initializing the language model and building the retrieval chain...")
import os
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

# Initialize your language model (adjust model_name if needed)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)
print("Language model initialized successfully.")

# Create a combine chain using the "stuff" approach with your prompt
combine_chain = create_stuff_documents_chain(llm, prompt)

# Build the retrieval chain using your retriever and the combine chain
rag_chain = create_retrieval_chain(retriever, combine_chain)
print("RAG chain built successfully using the new API.")

# Step 9: Define and process a sample legal query
sample_query = "What is the difference between a petition and a plaint in Indian law?"
print(f"Processing query: {sample_query}")

# Note: The retrieval chain expects the question under the key "input".
result = rag_chain.invoke({"input": sample_query})
print("Query processed successfully.")
print("Answer:", result)
