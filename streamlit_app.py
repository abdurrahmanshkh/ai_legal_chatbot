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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling with blue/gray theme
st.markdown("""
<style>
:root {
    --primary: #2563eb;
    --primary-light: #3b82f6;
    --primary-dark: #1d4ed8;
    --secondary: #f8fafc;
    --text: #1e293b;
    --text-light: #64748b;
    --user-bg: #dbeafe;
    --bot-bg: #f1f5f9;
    --border: #e2e8f0;
    --sidebar-bg: #b7d7f7;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}

* {
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.stApp {
    background-color: #f8fafc;
}

/* Header styling */
.st-emotion-cache-1avcm0n {
    background-color: var(--primary-dark) !important;
}

h1 {
    color: var(--primary-dark);
    text-align: center;
    font-weight: 700;
    margin-bottom: 0.25rem !important;
    font-size: 2.25rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
    border-right: 1px solid var(--border);
    padding: 1.5rem 1rem;
}

[data-testid="stSidebar"] h2 {
    color: var(--primary-dark) !important;
    margin-bottom: 1.5rem !important;
    border-bottom: 2px solid var(--primary-light);
    padding-bottom: 0.5rem;
}

[data-testid="stSidebar"] .stSelectbox label {
    color: var(--text) !important;
    font-weight: 500;
}

[data-testid="stSelectbox"] {
    background-color: white;
    border-radius: 8px;
    padding: 0.5rem;
}

/* Chat container */
.chat-container {
    display: flex;
    flex-direction: column;
    max-height: calc(100vh - 180px);
    padding: 1rem 0;
}

.chat-box {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    background-color: white;
    border-radius: 12px;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
    border: 1px solid var(--border);
}

/* Message styling */
.message {
    padding: 1.25rem;
    border-radius: 12px;
    margin-bottom: 1.25rem;
    max-width: 85%;
    position: relative;
    line-height: 1.6;
    animation: fadeIn 0.3s ease;
    box-shadow: var(--shadow);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-message {
    background-color: var(--user-bg);
    color: var(--text);
    margin-left: auto;
    border-bottom-right-radius: 4px;
}

.bot-message {
    background-color: var(--bot-bg);
    color: var(--text);
    margin-right: auto;
    border-bottom-left-radius: 4px;
}

.message-header {
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.05rem;
}

.user-header {
    color: var(--primary-dark);
}

.bot-header {
    color: var(--primary-dark);
}

/* Input area */
.input-container {
    background: white;
    padding: 1.25rem;
    border-radius: 12px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}

.stTextInput input {
    padding: 1rem 1.25rem !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    font-size: 1rem !important;
}

.stButton button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 1.75rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
    height: 100%;
}

.stButton button:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-1px);
}

.stButton button:active {
    transform: translateY(0);
}

/* Scrollbar styling */
.chat-box::-webkit-scrollbar {
    width: 8px;
}

.chat-box::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

.chat-box::-webkit-scrollbar-thumb {
    background: #bfdbfe;
    border-radius: 4px;
}

.chat-box::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* Footer */
.footer {
    text-align: center;
    padding: 1rem;
    color: var(--text-light);
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* Welcome message */
.welcome-message {
    padding: 1.5rem;
    background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%);
    color: white;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}

.welcome-message h3 {
    color: white !important;
    margin-bottom: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# App title and header
st.markdown("<h1 style='color: #0B1D51;'>⚖️ iLegalBot</h1>",
            unsafe_allow_html=True)
st.caption("<p style='text-align:center; color:#64748b; margin-bottom:1.5rem; font-size:1.1rem;'>AI Legal Assistant for Indian Law</p>",
           unsafe_allow_html=True)

# Sidebar configuration
sidebar_state = st.session_state.get("sidebar_state", True)
with st.sidebar:
    st.markdown("<h2>Configuration</h2>", unsafe_allow_html=True)
    model_choice = st.selectbox("Choose AI Model", ["Gemini", "OpenAI"])
    st.markdown("---")
    st.markdown("""
    <div style='margin-top:1.5rem;'>
        <p style='font-size:0.95rem; color: var(--text);'>
            This specialized AI assistant provides legal information related to Indian law. 
            <span style='color: var(--primary-dark); font-weight: 600;'>
                Always consult a qualified legal professional for official advice.
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Initialize retriever once


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


# Chat container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown("""
    <div class='welcome-message'>
        <h3>Hello! I'm your specialized legal assistant for Indian law</h3>
        <p>How can I help you today? Ask me anything about Indian legal matters.</p>
    </div>
    """, unsafe_allow_html=True)

for q, a in st.session_state.history:
    st.markdown(
        f"""
        <div class='user-message message'>
            <div class='message-header user-header'>
                <span>You</span>
            </div>
            <div>{q}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class='bot-message message'>
            <div class='message-header bot-header'>
                <span>iLegalBot</span>
            </div>
            <div>{a}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)  # Close chat-box

# Input area
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        user_input = st.text_input(
            "Enter your legal question",
            key="input",
            placeholder="Type your legal question here...",
            label_visibility="collapsed"
        )
    with cols[1]:
        submit = st.form_submit_button("Send →")

    if submit and user_input.strip():
        with st.spinner("Analyzing legal context..."):
            answer = get_response(user_input.strip())
        st.session_state.history.append((user_input, answer))
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)  # Close input-container
st.markdown("</div>", unsafe_allow_html=True)  # Close chat-container

# Footer
st.markdown("<div class='footer'>iLegalBot • AI Legal Assistant • Specialized in Indian Law</div>",
            unsafe_allow_html=True)
