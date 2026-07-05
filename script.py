import streamlit as st
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain.chains import RetrievalQA
import os
from docx import Document
import requests
import base64
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
        response = requests.get(doc_path)
        response.raise_for_status()
        with open("temp.docx", "wb") as f:
            f.write(response.content)
        doc = Document("temp.docx")
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Failed to load document from URL: {e}")
        return ""

# Supabase credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)



def save_chat(user_message, boot_response):
    from datetime import datetime
    data = {
        "user_message": user_message,
        "boot_response": boot_response,
        "timestamp": datetime.utcnow().isoformat()
    }

    response = supabase.table("gpt").insert(data).execute()



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

# Sidebar Chat History
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

# Load FAISS Index
DOC_PATH = "https://raw.githubusercontent.com/5298479/college-GPT/main/data/sample.docx"
doc_text = load_word_document(DOC_PATH)

embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY, user_agent=USER_AGENT)
store_filename = "word_doc_faiss.index"

if os.path.exists(store_filename):
    vectorstore = FAISS.load_local(store_filename, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_texts([doc_text], embedding=embeddings)
    vectorstore.save_local(store_filename)

retriever = vectorstore.as_retriever()
llm = Cohere(model="command", temperature=0.7, cohere_api_key=COHERE_API_KEY, user_agent=USER_AGENT)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Send Query
def send_query():
    user_input = st.session_state.get("user_input", "").strip()
    current_session = st.session_state.active_session

    if user_input and qa_chain:
        # Auto-rename on first message
        if (current_session.startswith("Chat") or current_session.startswith("Imported")) and len(st.session_state.chat_sessions[current_session]) == 0:
            new_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
            if new_title not in st.session_state.chat_sessions:
                st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(current_session)
                st.session_state.active_session = new_title
                current_session = new_title  # Update reference to avoid KeyError

        answer = qa_chain.run(user_input)
        
        # Save to Firestore after getting the response
        save_chat(user_input, answer)
        
        # Append conversation to session
        st.session_state.chat_sessions[current_session].append({"role": "user", "content": user_input})
        st.session_state.chat_sessions[current_session].append({"role": "assistant", "content": answer})
        st.session_state.user_input = ""

# Chat Styling
st.markdown("""
    <style>
        body { background-color: black; color: white; }
        .stApp { background-color: black; }
        .chat-box { background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .user { text-align: right; color: cyan; }
        .assistant { text-align: left; color: white; }
    </style>
""", unsafe_allow_html=True)

# Title and Logo
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://raw.githubusercontent.com/5298479/college-GPT/main/logo.jpg", width=80)
with col2:
    st.title("CollegeGPT")

# Chat Display
current_session = st.session_state.active_session
for chat in st.session_state.chat_sessions.get(current_session, []):
    role_class = "user" if chat["role"] == "user" else "assistant"
    st.markdown(f"<div class='chat-box {role_class}'><b>{chat['role'].capitalize()}:</b> {chat['content']}</div>", unsafe_allow_html=True)

# Input Field
st.text_input("Ask another question:", key="user_input", on_change=send_query)
