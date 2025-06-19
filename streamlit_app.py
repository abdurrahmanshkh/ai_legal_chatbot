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

# Modern CSS styling
st.markdown("""
<style>
:root {
    --primary: #4f46e5;
    --primary-dark: #3730a3;
    --secondary: #f9fafb;
    --text: #1f2937;
    --text-light: #6b7280;
    --user-bg: #e0e7ff;
    --bot-bg: #f3f4f6;
    --border: #e5e7eb;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

* {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.stApp {
    background-color: #f9fafb;
}

header {
    visibility: hidden;
}

h1 {
    color: var(--primary-dark);
    text-align: center;
    font-weight: 700;
    margin-bottom: 0.5rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    padding: 1.5rem 1rem;
}

[data-testid="stSidebar"] h1 {
    color: white !important;
    font-size: 1.5rem;
    margin-bottom: 2rem !important;
}

[data-testid="stSidebar"] .stSelectbox label {
    color: white !important;
    font-weight: 500;
}

[data-testid="stSelectbox"] {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 0.5rem;
}

/* Chat container */
.chat-container {
    display: flex;
    flex-direction: column;
    max-height: calc(100vh - 200px);
    padding: 1rem 0;
}

.chat-box {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    background-color: white;
    border-radius: 16px;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
    border: 1px solid var(--border);
}

/* Message styling */
.message {
    padding: 1rem 1.25rem;
    border-radius: 18px;
    margin-bottom: 1.25rem;
    max-width: 80%;
    position: relative;
    line-height: 1.5;
    animation: fadeIn 0.3s ease;
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
    font-weight: 600;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.user-header {
    color: var(--primary-dark);
}

.bot-header {
    color: var(--primary);
}

/* Input area */
.input-container {
    background: white;
    padding: 1.25rem;
    border-radius: 16px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}

.stTextInput input {
    padding: 0.85rem 1rem !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

.stButton button {
    background: linear-gradient(45deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 14px rgba(79, 70, 229, 0.25) !important;
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
    background: #c7d2fe;
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
    margin-top: auto;
}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1>⚖️ iLegalBot - AI Legal Assistant</h1>",
            unsafe_allow_html=True)
st.caption("<p style='text-align:center; color:#6b7280; margin-bottom:2rem;'>Your specialized assistant for Indian legal queries</p>",
           unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("<h1>Configuration</h1>", unsafe_allow_html=True)
    model_choice = st.selectbox("Choose AI Model", ["Gemini", "OpenAI"])
    st.markdown("---")
    st.markdown("""
    <div style='margin-top:2rem;'>
        <p style='font-size:0.9rem;'>This specialized AI assistant provides legal information related to Indian law. Always consult a qualified legal professional for official advice.</p>
    </div>
    """, unsafe_allow_html=True)

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


# Chat container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown("""
    <div class='bot-message message'>
        <div class='message-header bot-header'>
            <span>iLegalBot</span>
        </div>
        <div>Hello! I'm your specialized legal assistant for Indian law. How can I help you today?</div>
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
