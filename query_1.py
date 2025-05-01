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
    "embeddings/faiss_index_directory_6",
    embedding_model,
    allow_dangerous_deserialization=True
)
print("FAISS index loaded successfully.")

# Step 3: Create a retriever object
print("Creating retriever object...")
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
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

def process_query(query):
    """
    Process a single query and return the answer.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: The model's response
    """
    print(f"\nProcessing query: {query}")
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    answer = get_gemini_response(context, query)
    return answer

def main():
    print("\nWelcome to the Indian Legal Assistant!")
    print("Enter your legal questions (type 'quit' or 'exit' to end the session)")
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit']:
            print("Thank you for using the Indian Legal Assistant. Goodbye!")
            break
            
        if not query:
            print("Please enter a valid question.")
            continue
            
        try:
            answer = process_query(query)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
