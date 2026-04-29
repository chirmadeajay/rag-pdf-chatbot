import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
import tempfile
import datetime  # ← NEW: used to timestamp the downloaded file

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
    <style>
        .stApp { background-color: #0f1117; }

        [data-testid="stSidebar"] {
            background-color: #1a1c24;
            padding: 20px;
        }
        [data-testid="stChatInput"] textarea {
            background-color: #1e2030;
            color: white;
            border-radius: 12px;
        }
        [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-user"]) {
            background-color: #1e2030;
            border-radius: 12px;
            padding: 10px;
        }
        [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-assistant"]) {
            background-color: #161822;
            border-radius: 12px;
            padding: 10px;
        }

        /* Style the download button */
        .stDownloadButton button {
            width: 100%;
            background-color: #1e3a5f;
            color: white;
            border-radius: 8px;
            border: 1px solid #2d5a8e;
        }
        .stDownloadButton button:hover {
            background-color: #2d5a8e;
        }

        /* Style reset button */
        .stButton button {
            width: 100%;
            border-radius: 8px;
        }

        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE — initialize all variables
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ── NEW: Track uploaded file names ──
# So we can show them in the sidebar file manager
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

# ── NEW: Track question count ──
if "question_count" not in st.session_state:
    st.session_state.question_count = 0


# ─────────────────────────────────────────
# NEW HELPER FUNCTION: Generate download text
# A function is a reusable block of code
# This one converts chat history to a string
# ─────────────────────────────────────────
def generate_chat_export():
    """
    Converts chat messages into a neat text file.
    datetime.now() adds the current time as a header.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 50,
        f"RAG PDF Chatbot - Chat Export",
        f"Exported: {now}",
        f"PDFs used: {', '.join(st.session_state.uploaded_file_names)}",
        "=" * 50,
        ""
    ]

    for msg in st.session_state.messages:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        lines.append(f"{role}:")
        lines.append(msg["content"])
        lines.append("-" * 40)

    # "\n".join() connects all lines with a newline character
    return "\n".join(lines)


# ─────────────────────────────────────────
# NEW HELPER FUNCTION: Reset everything
# Called when user wants to start fresh
# ─────────────────────────────────────────
def reset_app():
    """
    Clears all session state so user can
    upload new PDFs and start a fresh chat.
    st.rerun() refreshes the page after reset.
    """
    st.session_state.messages = []
    st.session_state.retriever = None
    st.session_state.uploaded_file_names = []
    st.session_state.question_count = 0
    st.rerun()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/pdf-2.png", width=60)
    st.title("📁 PDF Manager")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once"
    )

    st.markdown("---")

    # ── Show loaded files or prompt to upload ──
    if st.session_state.retriever:
        st.success("✅ PDFs ready to chat!")

        # ── NEW: File list ──
        # Shows each uploaded file name with an icon
        st.markdown("**📋 Loaded Files:**")
        for fname in st.session_state.uploaded_file_names:
            st.markdown(f"- 📄 `{fname}`")

        st.markdown("---")

        # ── NEW: Stats row ──
        # col1, col2 splits the sidebar into 2 columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="💬 Questions",
                value=st.session_state.question_count
            )
        with col2:
            st.metric(
                label="📄 Files",
                value=len(st.session_state.uploaded_file_names)
            )

        st.markdown("---")

        # ── NEW: Download chat button ──
        # Only shows if there are messages to download
        if st.session_state.messages:
            chat_text = generate_chat_export()

            # st.download_button creates a download link
            # The file is generated in memory, no saving needed
            st.download_button(
                label="⬇️ Download Chat",
                data=chat_text,
                file_name=f"chat_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

        # ── NEW: Clear chat only (keep PDFs loaded) ──
        if st.button("🗑️ Clear Chat Only"):
            st.session_state.messages = []
            st.session_state.question_count = 0
            st.rerun()

        # ── NEW: Full reset (clears PDFs too) ──
        if st.button("🔄 Upload New PDFs"):
            reset_app()

    else:
        st.info("👆 Upload PDFs to get started")

    st.markdown("---")
    st.caption("Built with Streamlit + Groq + BM25")


# ─────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────
st.markdown("""
    <h1 style='text-align: center; color: #7eb8f7;'>
        🚀 RAG PDF Chatbot
    </h1>
    <p style='text-align: center; color: #888;'>
        Upload PDFs in the sidebar → Ask questions below
    </p>
    <hr style='border-color: #333;'>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# PROCESS UPLOADED PDFs
# ─────────────────────────────────────────
if uploaded_files and st.session_state.retriever is None:
    with st.spinner("📄 Reading and indexing your PDFs..."):
        all_docs = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                file_path = tmp.name

            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = file.name

            all_docs.extend(loaded_docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        docs = splitter.split_documents(all_docs)

        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 6

        st.session_state.retriever = retriever

        # ── NEW: Save file names to session state ──
        st.session_state.uploaded_file_names = [
            f.name for f in uploaded_files
        ]

        st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) indexed!")


# ─────────────────────────────────────────
# NEW: WELCOME SCREEN
# Shows when no PDFs are uploaded yet
# Instead of a blank/confusing empty page
# ─────────────────────────────────────────
if not st.session_state.retriever:
    st.markdown("""
        <div style='text-align: center; padding: 60px 20px;'>
            <div style='font-size: 80px;'>📄</div>
            <h2 style='color: #7eb8f7;'>No PDFs loaded yet</h2>
            <p style='color: #888; font-size: 16px;'>
                Upload one or more PDFs using the sidebar on the left<br>
                Then ask any question about their content!
            </p>
            <br>
            <p style='color: #555; font-size: 14px;'>
                ✅ Supports multiple PDFs &nbsp;|&nbsp;
                ✅ Remembers conversation &nbsp;|&nbsp;
                ✅ Shows sources
            </p>
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# GROQ LLM
# ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0.1
)


# ─────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ─────────────────────────────────────────
# CHAT INPUT + RESPONSE
# ─────────────────────────────────────────
if query := st.chat_input("💬 Ask anything about your documents..."):

    if st.session_state.retriever is None:
        st.warning("⚠️ Please upload at least one PDF using the sidebar.")

    else:
        st.chat_message("user").write(query)
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # ── NEW: Increment question counter ──
        st.session_state.question_count += 1

        retrieved_docs = st.session_state.retriever.invoke(query)

        context = ""
        sources = []

        for doc in retrieved_docs:
            context += doc.page_content + "\n\n"
            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "Unknown file")
            sources.append(f"📄 {source} — Page {page}")

        recent_history = st.session_state.messages[-5:]
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in recent_history
        ])

        prompt = f"""You are a precise document assistant.
Your job is to answer questions using ONLY the document context below.

STRICT RULES:
1. Only use information from the CONTEXT section
2. If the answer is not in the context, say exactly:
   "I could not find this in the uploaded documents."
3. Keep answers clear and well structured
4. If quoting from document, mention the source name
5. Never make up or assume information

RECENT CONVERSATION:
{history_text}

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {query}

YOUR ANSWER:"""

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_answer = ""

            with st.spinner("🤔 Thinking..."):
                stream = llm.stream(prompt)

            for chunk in stream:
                full_answer += chunk.content
                response_placeholder.markdown(full_answer + "▌")

            response_placeholder.markdown(full_answer)

            unique_sources = sorted(list(set(sources)))
            with st.expander("📚 View Sources"):
                for src in unique_sources:
                    st.markdown(f"- {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer
        })
