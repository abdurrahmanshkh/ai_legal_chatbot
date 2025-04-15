import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Step 1: Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded successfully.")

# Step 2: Load FAISS index
vector_store = FAISS.load_local(
    "embeddings/faiss_index_directory_3",
    embedding_model,
    allow_dangerous_deserialization=True
)
print("FAISS index loaded successfully.")

# Step 3: Create a retriever object
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 4: Define prompt template
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

# Step 5: Load API keys and initialize both models
load_dotenv()

# --- Gemini Setup ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("Error: GEMINI_API_KEY not found in .env.")
    exit()
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# --- OpenAI Setup ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Error: OPENAI_API_KEY not found in .env.")
    exit()
openai_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)
combine_chain = create_stuff_documents_chain(openai_llm, prompt)
rag_chain_openai = create_retrieval_chain(retriever, combine_chain)

# Step 6: Functions for querying each model
def get_gemini_response(context, query):
    prompt_with_context = prompt.format(context=context, input=query)
    response = gemini_model.generate_content(prompt_with_context)
    return response.text if response.text else "I do not have enough information to answer that."

def get_openai_response(query):
    result = rag_chain_openai.invoke({"input": query})
    return result['answer'] if 'answer' in result else "I do not have enough information to answer that."

# Step 7: Handle user input
def main():
    print("\nWelcome to the Indian Legal Assistant!")
    print("You can use either Gemini or OpenAI to answer your questions.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        print("\nWhich model do you want to use?")
        print("1. Gemini")
        print("2. OpenAI")
        model_choice = input("Enter 1 or 2: ").strip()

        if model_choice not in ['1', '2']:
            print("Invalid choice. Please select 1 (Gemini) or 2 (OpenAI).")
            continue

        query = input("\nYour legal question: ").strip()

        if query.lower() in ['exit', 'quit']:
            print("Thank you for using the Indian Legal Assistant. Goodbye!")
            break
        if not query:
            print("Please enter a valid question.")
            continue

        try:
            if model_choice == '1':
                docs = retriever.invoke(query)
                context = "\n".join([doc.page_content for doc in docs])
                answer = get_gemini_response(context, query)
            else:
                answer = get_openai_response(query)

            print("\nAnswer:", answer)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
