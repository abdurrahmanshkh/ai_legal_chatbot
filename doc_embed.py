# Step 1: Load the embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# Now you can initialize your embedding model:
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
from langchain.text_splitter import CharacterTextSplitter

# Initialize the text splitter: for instance, split into chunks of 2000 characters with a little overlap
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(all_documents)

# docs now contains smaller Document objects
print(f"Number of chunks: {len(docs)}")

# Step 4: Generate embeddings for the chunks
embeddings_list = []
total = len(docs)
for i, doc in enumerate(docs, start=1):
    embedding = embedding_model.embed_query(doc.page_content)
    embeddings_list.append(embedding)
    print(f"Processed chunk {i}/{total}")
print("Generated embeddings for", len(embeddings_list), "chunks.")

# Step 5: Store the embeddings
from langchain_community.vectorstores import FAISS

print("Creating FAISS index from documents and embeddings...")
vector_store = FAISS.from_documents(docs, embedding_model)
print("FAISS index created.")

# Step  6: Verifying the FAISS Index
# Define a sample query that you might ask the chatbot
sample_query = "What is the difference between a petition and a plaint in Indian law?"

# Retrieve the top matching document chunks using the FAISS index
similar_docs = vector_store.similarity_search(sample_query, k=3)

# Print out the content of the most similar document chunk
for idx, doc in enumerate(similar_docs):
    print(f"Match {idx + 1}:\n{doc.page_content}\n")

# Save the index to disk
import faiss

# Save the index to disk
faiss.write_index(vector_store.index, "embeddings/faiss_index.idx")
print("FAISS index saved to disk.")

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
