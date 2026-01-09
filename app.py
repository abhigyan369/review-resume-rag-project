import streamlit as st
import os
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# Import our functional modules
from src.pdf_loader import load_pdf_documents
from src.vector_store import create_vector_store
from src.rag_chain import get_rag_chain

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Resume Advisor RAG", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom Styling ---
st.markdown("""
    <style>
        /* Main background gradient */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Enhanced button styling */
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            color: white;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: white;
        }
        
        /* Chat message styling */
        [data-testid="stChatMessage"] {
            background-color: #2d3748;
            border-radius: 15px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            color: #e2e8f0;
        }
        
        [data-testid="stChatMessage"][data-testid*="user"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Assistant message text */
        [data-testid="stChatMessage"]:not([data-testid*="user"]) p,
        [data-testid="stChatMessage"]:not([data-testid*="user"]) li,
        [data-testid="stChatMessage"]:not([data-testid*="user"]) span {
            color: #e2e8f0 !important;
        }
        
        /* Success/Warning boxes */
        .element-container div[data-testid="stMarkdownContainer"] > div[data-testid="stMarkdown"] {
            border-radius: 10px;
        }
        
        /* Input field styling */
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 2px solid #e2e8f0;
            padding: 0.5rem;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: rgba(255,255,255,0.1);
            border-radius: 10px;
            font-weight: 600;
        }
        
        /* Chat input */
        .stChatInput>div {
            border-radius: 25px;
        }
        
        /* Custom header */
        .custom-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Stats card */
        .stats-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None

if "processed_filename" not in st.session_state:
    st.session_state["processed_filename"] = None

if "processing_time" not in st.session_state:
    st.session_state["processing_time"] = None

if "document_stats" not in st.session_state:
    st.session_state["document_stats"] = {"chunks": 0, "pages": 0}

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    # 1. API Key Section
    with st.expander("🔑 Authentication", expanded=True):
        api_key = st.text_input(
            "HuggingFace API Key", 
            type="password", 
            help="Enter your HF token to enable the LLM.",
            placeholder="hf_..."
        )
        if api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
            st.success("✓ API Key configured", icon="✅")
    
    st.markdown("---")
    
    # 2. File Upload Section
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF)", 
        type="pdf", 
        label_visibility="collapsed",
        help="Upload a PDF resume to analyze"
    )
    
    if uploaded_file:
        # Show file info
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        st.info(f"📎 **{uploaded_file.name}** ({file_size:.1f} KB)")
        
        # Check if it's a new file
        if uploaded_file.name != st.session_state["processed_filename"]:
            if st.button("🚀 Process & Index", use_container_width=True):
                if not api_key:
                    st.error("⚠️ Please enter an API Key first.")
                else:
                    start_time = datetime.now()
                    with st.status("Analyzing Document...", expanded=True) as status:
                        try:
                            # Save and Load
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            st.write("📖 Extracting text...")
                            documents = load_pdf_documents(tmp_file_path)
                            os.remove(tmp_file_path)
                            
                            if documents:
                                st.write("🔨 Building knowledge base...")
                                vector_store = create_vector_store(documents)
                                
                                st.write("🤖 Initializing RAG chain...")
                                qa_chain = get_rag_chain(vector_store)
                                
                                # Store stats
                                processing_time = (datetime.now() - start_time).total_seconds()
                                st.session_state["rag_chain"] = qa_chain
                                st.session_state["processed_filename"] = uploaded_file.name
                                st.session_state["processing_time"] = processing_time
                                st.session_state["document_stats"] = {
                                    "chunks": len(documents),
                                    "pages": len(set([doc.metadata.get('page', 0) for doc in documents]))
                                }
                                
                                status.update(
                                    label=f"✅ Complete! ({processing_time:.1f}s)", 
                                    state="complete", 
                                    expanded=False
                                )
                                st.rerun()
                            else:
                                st.error("❌ Text extraction failed.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        else:
            st.success("✓ Document already processed")
    
    st.markdown("---")
    
    # 3. Document Stats
    if st.session_state["rag_chain"]:
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state["messages"]))
        with col2:
            st.metric("Chunks", st.session_state["document_stats"]["chunks"])
        
        if st.session_state["processing_time"]:
            st.caption(f"⏱️ Processing: {st.session_state['processing_time']:.1f}s")
    
    st.markdown("---")
    
    # 4. Quick Actions
    st.markdown("### 🎯 Quick Actions")
    
    if st.session_state["rag_chain"]:
        # Suggested questions
        st.markdown("**Suggested Questions:**")
        suggestions = [
            "What are the key skills?",
            "Summarize work experience",
            "What certifications are listed?",
            "List the education background"
        ]
        
        for suggestion in suggestions:
            if st.button(f"💡 {suggestion}", key=suggestion, use_container_width=True):
                st.session_state["messages"].append({"role": "user", "content": suggestion})
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state["rag_chain"].invoke({"query": suggestion})
                        answer = response["result"]
                        st.session_state["messages"].append({"role": "assistant", "content": answer})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # 5. Clear Chat Button
    if st.session_state["messages"]:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("💡 **Tip:** Upload a PDF resume and start asking questions!")
    st.caption("Built with ❤️ using Streamlit")

# --- Main Interface ---
# Custom header
st.markdown("""
    <div class="custom-header">
        <h1 style="margin:0;">📄 Resume Intelligence Assistant</h1>
        <p style="margin:0.5rem 0 0 0; opacity: 0.9;">Extract insights from resumes using advanced RAG technology</p>
    </div>
""", unsafe_allow_html=True)

# Connection Status Indicator
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if st.session_state["rag_chain"]:
        st.success(f"✅ Active Session: **{st.session_state['processed_filename']}**")
    else:
        st.warning("⚠️ No document processed. Upload a PDF in the sidebar to begin.")

with col2:
    if st.session_state["rag_chain"]:
        st.metric("Pages", st.session_state["document_stats"]["pages"])

with col3:
    if st.session_state["messages"]:
        st.metric("Q&A", len([m for m in st.session_state["messages"] if m["role"] == "user"]))

st.markdown("---")

# Chat Container
chat_container = st.container()

with chat_container:
    if not st.session_state["messages"]:
        # Welcome message
        st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #64748b;">
                <h3>👋 Welcome to Resume Intelligence!</h3>
                <p>Upload a resume PDF and start asking questions to extract valuable insights.</p>
                <p style="margin-top: 1rem;">
                    <strong>Try asking:</strong><br>
                    "What programming languages does the candidate know?"<br>
                    "Summarize their work experience"<br>
                    "What are their top 3 skills?"
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Display chat history
        for idx, message in enumerate(st.session_state["messages"]):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# User Input Logic
if prompt := st.chat_input("💬 Ask something about the resume...", disabled=not st.session_state["rag_chain"]):
    if not st.session_state["rag_chain"]:
        st.error("⚠️ Please process a document first.")
    else:
        # User message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing context..."):
                try:
                    response = st.session_state["rag_chain"].invoke({"query": prompt})
                    answer = response["result"]
                    st.markdown(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"❌ Response Error: {e}")
        
        st.rerun()