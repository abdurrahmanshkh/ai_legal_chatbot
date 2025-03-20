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

# Create a FAISS index from your documents and embeddings
vector_store = FAISS.from_documents(docs, embedding_model)

# Step  6: Verifying the FAISS Index
# Define a sample query that you might ask the chatbot
sample_query = "What is the difference between a petition and a plaint in Indian law?"

# Retrieve the top matching document chunks using the FAISS index
similar_docs = vector_store.similarity_search(sample_query, k=3)

# Print out the content of the most similar document chunk
for idx, doc in enumerate(similar_docs):
    print(f"Match {idx + 1}:\n{doc.page_content}\n")
