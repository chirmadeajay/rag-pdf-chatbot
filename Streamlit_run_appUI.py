import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
import tempfile

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
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/pdf-2.png", width=60)
    st.title("Upload PDFs")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once"
    )

    st.markdown("---")

    if st.session_state.get("retriever"):
        st.success("PDFs ready to chat!")

        # ── NEW: Clear chat button in sidebar ──
        # Lets user reset without refreshing page
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Upload PDFs to get started")

    st.markdown("---")
    st.caption("Built with Streamlit + Groq + BM25")


# ─────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────
st.markdown("""
    <h1 style='text-align: center; color: #7eb8f7;'>
        RAG PDF Chatbot
    </h1>
    <p style='text-align: center; color: #888;'>
        Upload PDFs in the sidebar → Ask questions below
    </p>
    <hr style='border-color: #333;'>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ─────────────────────────────────────────
# PROCESS UPLOADED PDFs
# CHANGE: chunk_size 500→800 so sentences
# are not cut off in the middle
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
            chunk_size=800,       # ← was 500, bigger = more complete ideas
            chunk_overlap=100     # ← was 50, more overlap = less info lost
        )

        docs = splitter.split_documents(all_docs)

        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 6           # ← was 4, more chunks = richer context

        st.session_state.retriever = retriever
        st.sidebar.success(f" {len(uploaded_files)} PDF(s) indexed!")


# ─────────────────────────────────────────
# GROQ LLM
# CHANGE: temperature 0.3→0.1
# Lower = more focused, less random answers
# ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0.1               # ← was 0.3, more precise now
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
        st.warning("Please upload at least one PDF using the sidebar.")

    else:
        st.chat_message("user").write(query)
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # Retrieve relevant chunks
        retrieved_docs = st.session_state.retriever.invoke(query)

        context = ""
        sources = []

        for doc in retrieved_docs:
            context += doc.page_content + "\n\n"
            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "Unknown file")
            sources.append(f"📄 {source} — Page {page}")

        # ── NEW: Only pass last 5 messages as history ──
        # Why: Sending ALL history makes prompt too long
        # Last 5 is enough to remember recent context
        recent_history = st.session_state.messages[-5:]
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in recent_history
        ])

        # ── NEW: Much better prompt structure ──
        # Clear rules = AI follows them better
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

        # ── NEW: Streaming response ──
        # Instead of waiting for full answer,
        # words appear one by one like ChatGPT
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_answer = ""

            # stream=True makes it stream word by word
            with st.spinner("Thinking..."):
                stream = llm.stream(prompt)

            for chunk in stream:
                # Each chunk is a small piece of the answer
                full_answer += chunk.content
                response_placeholder.markdown(full_answer + "▌")

            # Remove the blinking cursor at the end
            response_placeholder.markdown(full_answer)

            # Show sources in expander
            unique_sources = sorted(list(set(sources)))
            with st.expander("View Sources"):
                for src in unique_sources:
                    st.markdown(f"- {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer
        })
