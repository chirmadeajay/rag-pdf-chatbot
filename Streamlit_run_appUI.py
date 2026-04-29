import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
import tempfile
import datetime
import time      # ← NEW: used to measure how long things take

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
        .stDownloadButton button {
            width: 100%;
            background-color: #1e3a5f;
            color: white;
            border-radius: 8px;
            border: 1px solid #2d5a8e;
        }
        .stDownloadButton button:hover { background-color: #2d5a8e; }
        .stButton button {
            width: 100%;
            border-radius: 8px;
        }

        /* NEW: Style for chunk preview cards */
        .chunk-card {
            background-color: #1a1f2e;
            border-left: 3px solid #7eb8f7;
            border-radius: 6px;
            padding: 10px 14px;
            margin: 6px 0;
            font-size: 13px;
            color: #ccc;
        }

        /* NEW: Style for activity log entries */
        .log-entry {
            font-size: 12px;
            color: #888;
            padding: 2px 0;
        }

        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# ── NEW: Activity log ──
# A list that stores recent actions like a diary
# We append to it and show the last 5 entries
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []


# ─────────────────────────────────────────
# NEW HELPER: Add to activity log
# Adds a timestamped entry to the log list
# ─────────────────────────────────────────
def log_activity(message):
    """
    Adds a message with current time to the log.
    We keep only the last 8 entries to avoid clutter.
    """
    now = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"[{now}] {message}"
    st.session_state.activity_log.append(entry)
    # Keep only last 8 entries
    st.session_state.activity_log = st.session_state.activity_log[-8:]


# ─────────────────────────────────────────
# HELPER: Generate chat export
# ─────────────────────────────────────────
def generate_chat_export():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 50,
        "RAG PDF Chatbot - Chat Export",
        f"Exported: {now}",
        f"PDFs used: {', '.join(st.session_state.uploaded_file_names)}",
        "=" * 50,
        ""
    ]
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}:")
        lines.append(msg["content"])
        lines.append("-" * 40)
    return "\n".join(lines)


# ─────────────────────────────────────────
# HELPER: Reset app
# ─────────────────────────────────────────
def reset_app():
    st.session_state.messages = []
    st.session_state.retriever = None
    st.session_state.uploaded_file_names = []
    st.session_state.question_count = 0
    st.session_state.activity_log = []
    st.rerun()


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/pdf-2.png", width=60)
    st.title("PDF Manager")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once"
    )

    st.markdown("---")

    if st.session_state.retriever:
        st.success("PDFs ready to chat!")

        st.markdown("**Loaded Files:**")
        for fname in st.session_state.uploaded_file_names:
            st.markdown(f"- `{fname}`")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.question_count)
        with col2:
            st.metric("Files", len(st.session_state.uploaded_file_names))

        st.markdown("---")

        # ── NEW: Activity log section ──
        # Shows a live feed of what the app has done
        if st.session_state.activity_log:
            st.markdown("**Activity Log:**")
            for entry in reversed(st.session_state.activity_log):
                st.markdown(
                    f"<div class='log-entry'>{entry}</div>",
                    unsafe_allow_html=True
                )
            st.markdown("---")

        if st.session_state.messages:
            chat_text = generate_chat_export()
            st.download_button(
                label="Download Chat",
                data=chat_text,
                file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

        if st.button("Clear Chat Only"):
            st.session_state.messages = []
            st.session_state.question_count = 0
            log_activity("Chat cleared")
            st.rerun()

        if st.button("Upload New PDFs"):
            reset_app()

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
# PROCESS UPLOADED PDFs
# NEW: Added step-by-step status messages
# so user knows what is happening
# ─────────────────────────────────────────
if uploaded_files and st.session_state.retriever is None:

    # st.status() is a live updating box that shows steps
    # Each "st.write()" inside it adds a new step line
    with st.status("📄 Processing your PDFs...", expanded=True) as status:
        all_docs = []

        st.write("Reading PDF files...")
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                file_path = tmp.name

            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = file.name

            all_docs.extend(loaded_docs)
            st.write(f" Read: `{file.name}` ({len(loaded_docs)} pages)")

        st.write("Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        docs = splitter.split_documents(all_docs)
        st.write(f" Created {len(docs)} text chunks")

        st.write("Building search index...")
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 6
        st.write("BM25 index ready")

        st.session_state.retriever = retriever
        st.session_state.uploaded_file_names = [f.name for f in uploaded_files]

        # Mark the status box as complete — turns green
        status.update(
            label=f"{len(uploaded_files)} PDF(s) indexed successfully!",
            state="complete",
            expanded=False
        )

        log_activity(f"Indexed {len(uploaded_files)} PDF(s), {len(docs)} chunks")


# ─────────────────────────────────────────
# GROQ LLM
# ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0.1
)


# ─────────────────────────────────────────
# WELCOME SCREEN
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
                Supports multiple PDFs &nbsp;|&nbsp;
                Remembers conversation &nbsp;|&nbsp;
                Shows sources
            </p>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ─────────────────────────────────────────
# CHAT INPUT + RESPONSE
# ─────────────────────────────────────────
if query := st.chat_input("Ask anything about your documents..."):

    if st.session_state.retriever is None:
        st.warning("Please upload at least one PDF using the sidebar.")

    else:
        st.chat_message("user").write(query)
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        st.session_state.question_count += 1
        log_activity(f"Question #{st.session_state.question_count} asked")

        # ── NEW: Step indicators while processing ──
        # These placeholders update in real time
        # Think of them as temporary text boxes
        step_placeholder = st.empty()

        # STEP 1: Show searching message
        step_placeholder.info("Step 1/3 — Searching your documents...")
        time.sleep(0.3)   # tiny pause so user can read it

        retrieved_docs = st.session_state.retriever.invoke(query)
        log_activity(f"Retrieved {len(retrieved_docs)} chunks")

        context = ""
        sources = []
        chunk_previews = []   # ← NEW: store text previews of found chunks

        for i, doc in enumerate(retrieved_docs):
            context += doc.page_content + "\n\n"
            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "Unknown file")
            sources.append(f"{source} — Page {page}")

            # Save first 200 characters of each chunk for preview
            # This is what gets shown in the "What was found" section
            chunk_previews.append({
                "source": source,
                "page": page,
                "preview": doc.page_content[:200] + "..."
            })

        # STEP 2: Building prompt message
        step_placeholder.info("Step 2/3 — Building context for AI...")
        time.sleep(0.3)

        # ── NEW: Estimate token count ──
        # Tokens are roughly 4 characters each in English
        # This is an estimate, not exact
        estimated_tokens = len(context) // 4
        log_activity(f"Context: ~{estimated_tokens} tokens")

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

        # STEP 3: Generating answer
        step_placeholder.info("Step 3/3 — Generating answer...")

        # Record start time to measure speed
        start_time = time.time()

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_answer = ""

            stream = llm.stream(prompt)

            for chunk in stream:
                full_answer += chunk.content
                response_placeholder.markdown(full_answer + "▌")

            response_placeholder.markdown(full_answer)

            # Calculate how long it took
            elapsed = round(time.time() - start_time, 1)
            log_activity(f"Answer generated in {elapsed}s")

            # ── NEW: Info bar below answer ──
            # Shows token estimate and time taken
            st.markdown(
                f"<div style='font-size:12px; color:#555; margin-top:8px;'>"
                f"{elapsed}s &nbsp;|&nbsp; "
                f"~{estimated_tokens} tokens used &nbsp;|&nbsp; "
                f"{len(retrieved_docs)} chunks retrieved"
                f"</div>",
                unsafe_allow_html=True
            )

            # ── NEW: Chunk preview expander ──
            # Shows the actual text that was found and used
            with st.expander("What was found in your PDFs"):
                for i, chunk in enumerate(chunk_previews):
                    st.markdown(
                        f"<div class='chunk-card'>"
                        f"<strong>Chunk {i+1}</strong> — "
                        f"{chunk['source']} | Page {chunk['page']}<br><br>"
                        f"{chunk['preview']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            # Sources expander
            unique_sources = sorted(list(set(sources)))
            with st.expander("View Sources"):
                for src in unique_sources:
                    st.markdown(f"- {src}")

        # Clear the step indicator now that we're done
        step_placeholder.empty()

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer
        })
