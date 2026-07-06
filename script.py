import streamlit as st
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from docx import Document as DocxDocument
import requests
import json
from supabase import create_client, Client
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="CollegeGPT", layout="wide")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
USER_AGENT = os.environ.get("USER_AGENT", "mujtaba/1.0")

# Load Word Document
def load_word_document(doc_path):
    try:
        if "github.com" in doc_path:
            doc_path = doc_path.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        
        response = requests.get(doc_path)
        response.raise_for_status()
        with open("temp.docx", "wb") as f:
            f.write(response.content)
        doc = DocxDocument("temp.docx")
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Failed to load document from URL: {e}")
        return ""

# Supabase credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.warning("Supabase credentials not found. Chat history will not be saved.")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_chat(user_message, bot_response):
    if supabase is None:
        return
    try:
        data = {
            "user_message": user_message,
            "boot_response": bot_response,
            "timestamp": datetime.utcnow().isoformat()
        }
        response = supabase.table("gpt").insert(data).execute()
    except Exception as e:
        st.error(f"Failed to save chat: {e}")

# Initialize Session
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Default": []}
    st.session_state.active_session = "Default"

# Delete Session
def delete_session(session_name):
    if session_name in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[session_name]
        remaining = list(st.session_state.chat_sessions.keys())
        st.session_state.active_session = remaining[0] if remaining else "Default"
        if not remaining:
            st.session_state.chat_sessions["Default"] = []

# ✅ SIDEBAR - Chat History and Footer
st.sidebar.markdown("### Chat History")
for session in list(st.session_state.chat_sessions.keys()):
    active = " (Active)" if session == st.session_state.active_session else ""
    with st.sidebar.container():
        col1, col2 = st.columns([0.75, 0.25])
        with col1:
            if st.button(session + active, key=f"select_{session}"):
                st.session_state.active_session = session

        with col2:
            with st.expander("⋮", expanded=False):
                if st.button("🗑 Delete", key=f"delete_{session}"):
                    delete_session(session)
                    st.rerun()

# New Chat
def create_new_session():
    new_title = f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[new_title] = []
    st.session_state.active_session = new_title

st.sidebar.markdown("---")
st.sidebar.button("➕ New Chat", on_click=create_new_session)

# ✅ ADD THIS: Footer at the very bottom of sidebar (one line, black text)
st.sidebar.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px 0; background-color: transparent;">
        <span style="color: black; font-size: 0.8rem;">Powered by Cohere AI & LangChain</span>
    </div>
""", unsafe_allow_html=True)

# Initialize LLM and QA System
@st.cache_resource
def initialize_qa_system():
    try:
        # Load document
        DOC_PATH = "https://raw.githubusercontent.com/Mujtaba916/college-gpt/main/data/sample.docx"
        doc_text = load_word_document(DOC_PATH)
        
        if not doc_text:
            st.error("Failed to load document. Please check the file path.")
            return None
        
        # Initialize embeddings
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=COHERE_API_KEY
        )
        
        store_filename = "word_doc_faiss.index"
        
        # Load or create vectorstore
        if os.path.exists(store_filename):
            vectorstore = FAISS.load_local(
                store_filename, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            chunks = [doc_text[i:i+1000] for i in range(0, len(doc_text), 1000)]
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            vectorstore.save_local(store_filename)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # ✅ CORRECT MODEL - Use this exact name
        llm = ChatCohere(
            model="command-r-08-2024",  # ✅ THIS WORKS
            temperature=0.7,
            cohere_api_key=COHERE_API_KEY
        )
        
        # Create the prompt template
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for college students. Answer the question based on the following context:

        Context: {context}

        Question: {question}

        Answer (be concise, accurate, and helpful):""")
        
        # Create the chain
        qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Failed to initialize QA system: {e}")
        return None

# Load QA system
qa_chain = initialize_qa_system()

# Send Query
def send_query():
    user_input = st.session_state.get("user_input", "").strip()
    current_session = st.session_state.active_session
    
    if not user_input:
        return
        
    if qa_chain is None:
        st.error("QA system is not initialized. Please check your setup.")
        return
    
    # Auto-rename on first message
    if (current_session.startswith("Chat") or current_session.startswith("Imported")) and len(st.session_state.chat_sessions[current_session]) == 0:
        new_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
        if new_title not in st.session_state.chat_sessions:
            st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(current_session)
            st.session_state.active_session = new_title
            current_session = new_title
    
    try:
        answer = qa_chain.invoke(user_input)
        
        # Save to Supabase
        save_chat(user_input, answer)
        
        # Append conversation to session
        st.session_state.chat_sessions[current_session].append({"role": "user", "content": user_input})
        st.session_state.chat_sessions[current_session].append({"role": "assistant", "content": answer})
        st.session_state.user_input = ""
        
    except Exception as e:
        st.error(f"Error getting response: {e}")
        import traceback
        st.error(traceback.format_exc())

# Chat Styling
st.markdown("""
    <style>
        body { background-color: black; color: white; }
        .stApp { background-color: black; }
        .chat-box { background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .user { text-align: right; color: cyan; }
        .assistant { text-align: left; color: white; }
        .stTextInput > div > div > input {
            background-color: #1e1e1e;
            color: white;
            border: 1px solid #333;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Logo
col1, col2 = st.columns([0.1, 0.9])
with col1:
    try:
        st.image("https://raw.githubusercontent.com/Mujtaba916/college-gpt/main/logo.jpg", width=80)
    except:
        st.image("https://via.placeholder.com/80", width=80)
with col2:
    st.title("CollegeGPT")

# Chat Display
current_session = st.session_state.active_session
for chat in st.session_state.chat_sessions.get(current_session, []):
    role_class = "user" if chat["role"] == "user" else "assistant"
    st.markdown(f"<div class='chat-box {role_class}'><b>{chat['role'].capitalize()}:</b> {chat['content']}</div>", unsafe_allow_html=True)

# Input Field
st.text_input("Ask another question:", key="user_input", on_change=send_query, placeholder="Type your question here...")

# ❌ REMOVE THIS - No longer needed at bottom
# st.markdown("---")
# st.markdown("💡 *Powered by Cohere AI and LangChain*")
