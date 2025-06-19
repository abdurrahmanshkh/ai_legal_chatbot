import os
import streamlit as st
from dotenv import load_dotenv

# LangChain and AI imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Load environment variables from .env
load_dotenv()

# Title and description
st.set_page_config(page_title="Indian Legal Assistant", layout="wide")
st.title("🇮🇳 Indian Legal Assistant Chatbot")
st.write("Ask legal questions using Gemini or OpenAI GPT-4o-mini")

# Sidebar configuration
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ["gemini", "openai"])

# Initialize embeddings and vector store (cache across reruns)


@st.cache_resource
def init_retriever():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        "embeddings/faiss_index_directory_6",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})


retriever = init_retriever()

# Prompt template
prompt_template = """
You are a legal assistant chatbot specialized in Indian law.
Use the following context to answer the question. If the context does not provide sufficient information, reply "I do not have enough information to answer that."

Context:
{context}

Question:
{input}

Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "input"], template=prompt_template)

# Initialize Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your environment.")
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize OpenAI RAG chain


def init_openai_chain():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY not found. Please set it in your environment.")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3,
                     openai_api_key=openai_api_key)
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, stuff_chain)


if model_choice == "openai":
    rag_chain_openai = init_openai_chain()

# Helper functions


def get_gemini_response(context: str, query: str) -> str:
    formatted = prompt.format(context=context, input=query)
    resp = gemini_model.generate_content(formatted)
    return resp.text or "I do not have enough information to answer that."


def get_openai_response(query: str) -> str:
    result = rag_chain_openai.invoke({"input": query})
    return result.get('answer', "I do not have enough information to answer that.")

# Chat input and response


def main():
    question = st.text_area("Your Question", height=100)
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
            return
        with st.spinner("Thinking..."):
            if model_choice == "gemini":
                docs = retriever.invoke(question)
                context = "\n".join([doc.page_content for doc in docs])
                answer = get_gemini_response(context, question)
            else:
                answer = get_openai_response(question)
        st.subheader("Answer:")
        st.write(answer)


if __name__ == "__main__":
    main()
