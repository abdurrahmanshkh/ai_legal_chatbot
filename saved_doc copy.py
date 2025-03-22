import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai  # Import the Google Generative AI library

# Disable SSL certificate verification for Hugging Face Hub downloads.
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = "1"

# Step 1: Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully!")

# Step 2: Load the saved FAISS index from disk with dangerous deserialization allowed
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
load_dotenv()  # Load environment variables from .env file

gemini_api_key = os.getenv("GEMINI_API_KEY")  # Changed to GEMINI_API_KEY
if gemini_api_key:
    print("Gemini API key retrieved successfully.")
    genai.configure(api_key=gemini_api_key) # Configure Gemini with the API key
else:
    print("Warning: Gemini API key not found. Make sure to set it in your .env file.")
    exit() # Exit if API key is not found

model_name = "gemini-2.0-flash"  # Choose the Gemini model
model = genai.GenerativeModel(model_name) # Initialize the Gemini model
print("Gemini model initialized successfully.")

def get_gemini_response(context, query):
    """
    Retrieves a response from the Gemini model, incorporating the given context.

    Args:
        context (str): The relevant information retrieved from the vector store.
        query (str): The user's query.

    Returns:
        str: The Gemini model's response.
    """
    prompt_with_context = prompt.format(context=context, input=query) # Use the prompt template
    response = model.generate_content(prompt_with_context)
    return response.text if response.text else "I do not have enough information to answer that." #Consistent return

# Step 6: Define and process a sample legal query
sample_query = "What is a petition and is it different from a plaint?"
print(f"Processing query: {sample_query}")

# Get relevant documents from the vector store
docs = retriever.invoke(sample_query)
context = "\n".join([doc.page_content for doc in docs])

# Get the response from Gemini
answer = get_gemini_response(context, sample_query)
print("Query processed successfully.")
print("Answer:", answer)
