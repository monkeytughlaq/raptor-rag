"""
Enhanced Streamlit TDS Expert with Authentication, Feedback, and Learning
"""

import streamlit as st
from bfs_backtrack_enhanced import bfs_search  # Use enhanced version
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import io
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz  # PyMuPDF
import docx
import tempfile
import json
import re
from datetime import datetime
import numpy as np
import cv2
import time

# Import our new modules
from auth import (
    authenticate_user, register_user, init_auth_state,
    login_user_to_session, logout_user_from_session
)
from database import get_db_session, Session as DBSession
from feedback import log_query, log_response, log_retrieved_contexts, log_feedback
from learning_engine import LearningEngine

# Import config
from config import (
    ENABLE_LEARNING, ENABLE_FEEDBACK_UI, ENABLE_ADMIN_DASHBOARD,
    DEFAULT_BFS_THRESHOLD, DEFAULT_MAX_DOCS
)

# Load environment variables
load_dotenv()
required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT_NAME']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize learning engine
learning_engine = LearningEngine() if ENABLE_LEARNING else None

# Azure OpenAI config
llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2024-10-21"),
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.2,
)
# [Previous OCR and extraction functions remain the same]
# Copy from tds_app1.py lines 28-510

# Streamlit Page Config
st.set_page_config(page_title="TDS Expert - Self-Learning RAG", layout="wide", initial_sidebar_state="expanded")

# Initialize authentication state
init_auth_state()

# --- AUTHENTICATION UI ---
def show_login_page():
    """Display login/register page"""
    st.title("🔐 TDS Expert - Self-Learning Assistant")
    st.markdown("### Welcome! Please login or register to continue")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_btn"):
            if not login_username or not login_password:
                st.error("Please enter both username and password")
            else:
                user = authenticate_user(login_username, login_password)
                if user:
                    login_user_to_session(user)
                    st.success(f"Welcome back, {user.username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Register New Account")
        reg_username = st.text_input("Choose Username", key="reg_username")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Choose Password", type="password", key="reg_password")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
        reg_full_name = st.text_input("Full Name (Optional)", key="reg_full_name")
        
        if st.button("Register", key="reg_btn"):
            if not reg_username or not reg_email or not reg_password:
                st.error("Please fill in all required fields")
            elif reg_password != reg_password_confirm:
                st.error("Passwords do not match")
            elif len(reg_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                user_id = register_user(
                    reg_username,
                    reg_email,
                    reg_password,
                    reg_full_name if reg_full_name else None
                )
                if user_id:
                    st.success(f"Account created successfully! Please login.")
                else:
                    st.error("Username or email already exists")


def show_feedback_ui(query_id: int):
    """Show feedback UI for the last response"""
    st.markdown("---")
    st.markdown("### 💭 How was this response?")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("👍 Helpful", key=f"helpful_{query_id}"):
            log_feedback(
                query_id=query_id,
                user_id=st.session_state.user['id'],
                rating=5,
                is_helpful=True
            )
            st.success("Thanks for your feedback!")
            st.session_state.feedback_given = True
    
    with col2:
        if st.button("👎 Not Helpful", key=f"not_helpful_{query_id}"):
            log_feedback(
                query_id=query_id,
                user_id=st.session_state.user['id'],
                rating=1,
                is_helpful=False
            )
            st.info("Thanks! We'll improve.")
            st.session_state.feedback_given = True
    
    # Star rating
    with col3:
        st.markdown("**Or rate (1-5 stars):**")
        rating_cols = st.columns(5)
        for i, col in enumerate(rating_cols, 1):
            with col:
                if st.button(f"⭐" * i, key=f"star_{i}_{query_id}"):
                    log_feedback(
                        query_id=query_id,
                        user_id=st.session_state.user['id'],
                        rating=i,
                        is_helpful=i >= 3
                    )
                    st.success(f"{i} stars submitted!")
                    st.session_state.feedback_given = True


def get_context_for_query_enhanced(query, user_id, session_id_str, max_docs=None, threshold=None):
    """Enhanced context retrieval with learning and logging"""
    
    # Use learned optimal parameters if available
    if max_docs is None:
        max_docs = DEFAULT_MAX_DOCS
    if threshold is None:
        threshold = DEFAULT_BFS_THRESHOLD
    
    # Check for similar queries if learning is enabled
    similar_contexts = None
    if ENABLE_LEARNING and learning_engine:
        similar_queries = learning_engine.find_similar_queries(query, top_k=3)
        if similar_queries:
            # Try to get optimal contexts from similar high-rated query
            similar_contexts = learning_engine.get_optimal_contexts_for_similar_query(
                query, similarity_threshold=0.85
            )
    
    # Use cached contexts if available
    if similar_contexts:
        docs = similar_contexts
        st.info(f"♻️ Reusing successful context from similar query (faster response!)")
    else:
        # Call enhanced BFS with learning integration
        docs = bfs_search(
            query,
            threshold=threshold,
            max_docs=max_docs,
            user_id=user_id,
            session_id_str=session_id_str
        )
    
    # Format context
    context = []
    for doc in docs:
        meta = doc.metadata
        ref = f"{meta.get('source_file','unknown')} (chunk {meta.get('chunk_index',meta.get('cluster','-'))})"
        context.append(f"[{ref}]: {doc.page_content[:500]}")
    
    return "\\n\\n".join(context), docs


def run_enhanced_tds_analysis(query, conversation_history, invoice_data=None, user_id=None, session_id_str=None):
    """Enhanced TDS analysis with learning and feedback logging"""
    start_time = time.time()
    
    # Get enhanced context with learning
    context, retrieved_docs = get_context_for_query_enhanced(
        query, user_id, session_id_str
    )
    
    # Format invoice info (reuse from original)
    invoice_info = format_invoice_info(invoice_data) if invoice_data else "No invoice information available."
    
    # Run LLM analysis (use existing logic from tds_app1.py)
    # ... [Copy the LLM prompting logic from run_tds_analysis lines 465-510]
    
    # Simplified version here:
    prompt = f"""
    Role: TDS expert under Indian Income Tax Act.
    
    User Query: {query}
    Context: {context}
    Invoice Info: {invoice_info}
    
    Provide a clear, helpful response about TDS compliance.
    """
    
    response = (ChatPromptTemplate.from_template("{prompt}") 
                | llm | StrOutputParser()).invoke({"prompt": prompt})
    
    processing_time = (time.time() - start_time) * 1000  # ms
    
    # Log to database
    if user_id and session_id_str:
        # Get DB session ID
        with get_db_session() as db_session:
            session_obj = db_session.query(DBSession).filter(
                DBSession.session_id == session_id_str
            ).first()
            db_session_id = session_obj.id if session_obj else None
        
        if db_session_id:
            # Log query
            query_id = log_query(
                session_id=db_session_id,
                user_id=user_id,
                query_text=query,
                query_type="tds_query",
                processing_time_ms=processing_time
            )
            
            # Log response
            if query_id:
                log_response(
                    query_id=query_id,
                    response_text=response,
                    response_type="conversational",
                    model_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
                )
                
                # Log contexts
                contexts = [
                    {
                        "text": doc.page_content,
                        "type": doc.metadata.get("type", "chunk"),
                        "level": doc.metadata.get("level", 0),
                        "cluster_id": doc.metadata.get("cluster", 0),
                        "similarity_score": 0.0,  # Would need to compute
                        "rank": i+1
                    }
                    for i, doc in enumerate(retrieved_docs[:10])
                ]
                log_retrieved_contexts(query_id, contexts)
                
                # Store query_id for feedback
                st.session_state.last_query_id = query_id
                st.session_state.feedback_given = False
    
    return response


# --- MAIN APP LOGIC ---
if not st.session_state.get('authenticated', False):
    show_login_page()
    st.stop()

# User is authenticated - show main app
st.title(f"🤖 TDS Expert - Welcome {st.session_state.user['username']}!")
st.markdown("*Self-learning assistant powered by RAPTOR-RAG*")

# Sidebar
with st.sidebar:
    st.header(f"👤 {st.session_state.user.get('full_name', st.session_state.user['username'])}")
    
    if st.button("🚪 Logout"):
        logout_user_from_session()
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("📊 Session Info")
    st.write(f"**Session:** {st.session_state.session_id[:8]}...")
    st.write(f"**Messages:** {len(st.session_state.get('messages', []))}")
    
    # Admin dashboard link
    if ENABLE_ADMIN_DASHBOARD and st.session_state.user.get('is_admin', False):
        st.markdown("---")
        if st.button("📈 Admin Dashboard"):
            st.session_state.show_admin = True
            st.rerun()
    
    st.markdown("---")
    st.subheader("🔄 Controls")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "invoice_data" not in st.session_state:
    st.session_state.invoice_data = None

# File upload section (similar to original)
# ... [Copy from tds_app1.py lines 527-600]

# Chat interface
chat_container = st.container()

with chat_container:
    # Display messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show feedback UI for last assistant message if enabled
            if (ENABLE_FEEDBACK_UI and 
                msg["role"] == "assistant" and 
                i == len(st.session_state.messages) - 1 and
                st.session_state.get('last_query_id') and
                not st.session_state.get('feedback_given', False)):
                show_feedback_ui(st.session_state.last_query_id)
    
    # Chat input
    if prompt := st.chat_input("Ask your TDS question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing with self-learning RAG..."):
                response = run_enhanced_tds_analysis(
                    prompt,
                    st.session_state.messages,
                    st.session_state.invoice_data,
                    user_id=st.session_state.user['id'],
                    session_id_str=st.session_state.session_id
                )
            st.markdown(response)
            
            # Show feedback UI immediately
            if ENABLE_FEEDBACK_UI and st.session_state.get('last_query_id'):
                show_feedback_ui(st.session_state.last_query_id)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

# Show quick start if no messages
if not st.session_state.messages:
    st.markdown("### 🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 TDS Rates"):
            st.session_state.messages.append({
                "role": "user", 
                "content": "What are the different TDS rates?"
            })
            st.rerun()
    
    with col2:
        if st.button("📋 Section 194J"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Explain Section 194J"
            })
            st.rerun()
    
    with col3:
        if st.button("🔍 Professional Fees"):
            st.session_state.messages.append({
                "role": "user",
                "content": "TDS on professional fees?"
            })
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "🧠 Self-Learning TDS Expert | Powered by RAPTOR-RAG + Learning Engine"
    "</div>",
    unsafe_allow_html=True
)
