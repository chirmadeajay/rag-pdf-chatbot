import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
import tempfile

# ─────────────────────────────────────────
# PAGE CONFIG — must be first streamlit call
# Sets browser tab title, icon, and wide layout
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS — makes the app look polished
# This is just styling, like CSS in a website
# ─────────────────────────────────────────
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0f1117;
        }

        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #1a1c24;
            padding: 20px;
        }

        /* Chat input box */
        [data-testid="stChatInput"] textarea {
            background-color: #1e2030;
            color: white;
            border-radius: 12px;
        }

        /* User message bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background-color: #1e2030;
            border-radius: 12px;
            padding: 10px;
        }

        /* Assistant message bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background-color: #161822;
            border-radius: 12px;
            padding: 10px;
        }

        /* Success message */
        .stSuccess {
            background-color: #1a3a2a;
            border-radius: 8px;
        }

        /* Hide default streamlit footer */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR — PDF upload lives here
# Keeps the main chat screen clean
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/pdf-2.png", width=60)
    st.title("📁 Upload PDFs")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once"
    )

    st.markdown("---")

    # Show how many files are loaded
    if st.session_state.get("retriever"):
        st.success(f"✅ PDFs ready to chat!")
    else:
        st.info("👆 Upload PDFs to get started")

    st.markdown("---")
    st.caption("Built with Streamlit + Groq + BM25")


# ─────────────────────────────────────────
# MAIN AREA HEADER
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
# SESSION STATE — stores data between reruns
# Think of it like short-term memory
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ─────────────────────────────────────────
# PROCESS UPLOADED PDFs
# Runs only when new files are uploaded
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
            chunk_size=500,
            chunk_overlap=50
        )

        docs = splitter.split_documents(all_docs)

        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 4

        st.session_state.retriever = retriever
        st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) indexed!")


# ─────────────────────────────────────────
# GROQ LLM SETUP
# ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0.3
)


# ─────────────────────────────────────────
# DISPLAY CHAT HISTORY
# Shows all previous messages when page reruns
# ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ─────────────────────────────────────────
# CHAT INPUT + RESPONSE LOGIC
# ─────────────────────────────────────────
if query := st.chat_input("💬 Ask anything about your documents..."):

    if st.session_state.retriever is None:
        st.warning("⚠️ Please upload at least one PDF first using the sidebar.")

    else:
        # Show user message
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

        history = "\n".join([
            m["content"] for m in st.session_state.messages
        ])

        prompt = f"""
You are an expert assistant.

Use ONLY the provided context.
If the answer is not in the context, say:
"I could not find this in the uploaded documents."

Chat History:
{history}

Context:
{context}

Question:
{query}
"""

        # Show assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = llm.invoke(prompt)
                answer = response.content

            st.write(answer)

            # Show sources in a neat expander
            # Expander = collapsed section user can click to open
            unique_sources = sorted(list(set(sources)))
            with st.expander("📚 View Sources"):
                for src in unique_sources:
                    st.markdown(f"- {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
