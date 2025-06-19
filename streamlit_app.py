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

# Custom CSS for modern design
st.markdown(
    """
    <style>
    .stApp { background-color: #f5f7fa; }
    header { visibility: hidden; }
    .css-1d391kg { padding: 0; }
    .chat-box {
        max-height: 60vh;
        overflow-y: auto;
        padding: 16px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .user-msg { color: #0b486b; margin-bottom: 8px; }
    .bot-msg { color: #3b3e66; margin-bottom: 16px; }
    .stButton>button {
        background-color: #119da4;
        color: white;
        border-radius: 8px;
        padding: 8px 24px;
    }
    .stForm>div { display: flex; align-items: center; gap: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("<h1 style='text-align:center; font-family:sans-serif;'>🤖 iLegalBot</h1>",
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox("Choose Model", ["Gemini", "OpenAI"])

# Initialize retriever once
tmp = None


@st.cache_resource
def init_retriever():
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.load_local(
        "embeddings/faiss_index_directory_6",
        emb,
        allow_dangerous_deserialization=True
    )
    return store.as_retriever(search_kwargs={"k": 3})


retriever = init_retriever()

# Prompt
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

# Initialize models
genai_key = os.getenv("GEMINI_API_KEY")
if genai_key:
    genai.configure(api_key=genai_key)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
else:
    gemini_model = None


@st.cache_resource
def init_openai_chain():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    llm = ChatOpenAI(model_name="gpt-4o-mini",
                     temperature=0.3, openai_api_key=key)
    chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, chain)


openai_chain = init_openai_chain()

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Helper functions


def get_response(user_query):
    if model_choice == "Gemini":
        if not gemini_model:
            return "Error: GEMINI_API_KEY not set."
        docs = retriever.invoke(user_query)
        context = "\n".join([d.page_content for d in docs])
        formatted = prompt.format(context=context, input=user_query)
        resp = gemini_model.generate_content(formatted)
        return resp.text or "I do not have enough information to answer that."
    else:
        if not openai_chain:
            return "Error: OPENAI_API_KEY not set."
        result = openai_chain.invoke({"input": user_query})
        return result.get("answer", "I do not have enough information to answer that.")


# Display chat
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
for q, a in st.session_state.history:
    st.markdown(
        f"<div class='user-msg'><strong>You:</strong> {q}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='bot-msg'><strong>iLegalBot:</strong> {a}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your legal question", key="input")
    submit = st.form_submit_button("Send")
    if submit and user_input.strip():
        with st.spinner("iLegalBot is thinking..."):
            answer = get_response(user_input.strip())
        st.session_state.history.append((user_input, answer))

# Footer space
st.markdown("\n")
