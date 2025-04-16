import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain and AI imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Indian Legal Assistant API",
    description="Ask legal questions using either Gemini or OpenAI",
    version="1.0.0"
)

# Step 1: Load embedding model and FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(
    "embeddings/faiss_index_directory_3",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 2: Prompt template
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

# Step 3: Initialize Gemini model
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment")
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Step 4: Initialize OpenAI model and RAG chain
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment")
openai_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)
combine_chain = create_stuff_documents_chain(openai_llm, prompt)
rag_chain_openai = create_retrieval_chain(retriever, combine_chain)

# Request and Response schemas
class AskRequest(BaseModel):
    question: str
    model: str  # "gemini" or "openai"

class AskResponse(BaseModel):
    answer: str

# Helper functions

def get_gemini_response(context: str, query: str) -> str:
    prompt_with_context = prompt.format(context=context, input=query)
    response = gemini_model.generate_content(prompt_with_context)
    return response.text or "I do not have enough information to answer that."


def get_openai_response(query: str) -> str:
    result = rag_chain_openai.invoke({"input": query})
    # The chain returns a dict with 'answer'
    return result.get('answer', "I do not have enough information to answer that.")

# API Endpoints

@app.get("/health", summary="Health check")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse, summary="Ask a legal question")
def ask(request: AskRequest):
    question = request.question.strip()
    model_choice = request.model.lower()

    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        if model_choice == "gemini":
            docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in docs])
            answer = get_gemini_response(context, question)
        elif model_choice == "openai":
            answer = get_openai_response(question)
        else:
            raise HTTPException(status_code=400, detail="Model must be 'gemini' or 'openai'.")

        return AskResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
