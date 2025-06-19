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

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="iLegalBot",
    page_icon="⚖️",
    layout="wide"
)

# Apply custom CSS for modern look
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
        color: #333333;
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stTextInput>div>div>input {
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 8px;
    }
    .chat-box {
        max-height: 60vh;
        overflow-y: auto;
        padding: 16px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align:center;'>🤖 iLegalBot</h1>",
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox("Choose Model", ["Gemini", "OpenAI"])

# Initialize retriever as cache


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
Use the following context to answer the question. If the context does not provide sufficient information, reply \"I do not have enough information to answer that.\".

Context:
{context}

Question:
{input}

Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "input"], template=prompt_template)

# Initialize Gemini
genai_key = os.getenv("GEMINI_API_KEY")
if genai_key:
    genai.configure(api_key=genai_key)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
else:
    st.sidebar.error("GEMINI_API_KEY missing")

# Initialize OpenAI RAG chain


def init_openai_chain():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        llm = ChatOpenAI(model_name="gpt-4o-mini",
                         temperature=0.3, openai_api_key=key)
        chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, chain)
    else:
        st.sidebar.error("OPENAI_API_KEY missing")
        return None


rag_chain_openai = init_openai_chain() if model_choice == "OpenAI" else None

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Helper functions


def get_gemini_response(context, user_query):
    formatted = prompt.format(context=context, input=user_query)
    resp = gemini_model.generate_content(formatted)
    return resp.text or "I do not have enough information to answer that."


def get_openai_response(user_query):
    result = rag_chain_openai.invoke({"input": user_query})
    return result.get("answer", "I do not have enough information to answer that.")


# Chat interface
with st.container():
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for i, (q, a) in enumerate(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**iLegalBot:** {a}\n---")
    st.markdown("</div>", unsafe_allow_html=True)

user_input = st.text_input("Enter your legal question:")
if st.button("Send"):
    if user_input.strip():
        with st.spinner("iLegalBot is thinking..."):
            if model_choice == "Gemini":
                docs = retriever.invoke(user_input)
                context = "\n".join([d.page_content for d in docs])
                answer = get_gemini_response(context, user_input)
            else:
                answer = get_openai_response(user_input)
        st.session_state.history.append((user_input, answer))
        st.experimental_rerun()
