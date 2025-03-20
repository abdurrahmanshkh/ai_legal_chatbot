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
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
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

# Step 7: Create a retriever object
# Assuming your FAISS vector store is stored in the variable `vector_store`
# Create a retriever object that uses FAISS's similarity search.
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 8: Create a prompt template
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

# Step 9: Build the RAG chain
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Initialize your language model (this example uses a hypothetical LLM, e.g., from OpenAI or any LLM you prefer)
llm = ChatOpenAI(model_name="llama2-chat", temperature=0.3, openai_api_key=openai_api_key)

# Build the RAG chain using RetrievalQA.
# The chain automatically retrieves context using the retriever and then injects it into the prompt.
rag_chain = RetrievalQA(
    llm=llm,
    retriever=retriever,
    prompt=prompt,
    return_source_documents=True  # Optional: to return which documents were used
)

# Define a sample legal query.
sample_query = "What is the difference between a petition and a plaint in Indian law?"

# Use the RAG chain to get an answer.
result = rag_chain.run(sample_query)

# Print the generated answer.
print("Answer:", result)

# (Optional) If return_source_documents is enabled, you can also print the sources:
if isinstance(result, dict) and "source_documents" in result:
    for i, doc in enumerate(result["source_documents"]):
        print(f"Source Document {i+1}: {doc.page_content[:200]}...")  # Print the first 200 characters
