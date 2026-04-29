import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import tempfile
import os

st.title("🚀 Production RAG App (Multi-PDF + Memory + Sources)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

PERSIST_DIR = "chroma_db"

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Process uploaded PDFs
if uploaded_files and st.session_state.db is None:
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
    st.write("App reached here ✅")
    db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    st.session_state.db = db
    st.success("✅ PDFs indexed and saved!")

# Load existing DB if available
if st.session_state.db is None and os.path.exists(PERSIST_DIR):
    st.session_state.db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

# OpenAI LLM
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://your-streamlit-app-url.streamlit.app",
        "X-Title": "RAG PDF Chatbot"
    }
)

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if query := st.chat_input("Ask anything about your documents"):

    if st.session_state.db is None:
        st.warning("⚠️ Upload PDFs first")

    else:
        st.chat_message("user").write(query)

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        retriever = st.session_state.db.as_retriever(
            search_kwargs={"k": 4}
        )

        retrieved_docs = retriever.invoke(query)

        context = ""
        sources = []

        for doc in retrieved_docs:
            context += doc.page_content + "\n\n"

            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "Unknown file")

            sources.append(f"{source} - Page {page}")

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

        response = llm.invoke(prompt)
        answer = response.content

        with st.chat_message("assistant"):
            st.write(answer)

            unique_sources = sorted(list(set(sources)))

            st.markdown("### 📊 Sources:")
            for src in unique_sources:
                st.write("-", src)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
