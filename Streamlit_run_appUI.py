import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
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

if uploaded_files:
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)

    if st.session_state.db is None:
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

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        db = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )

        st.session_state.db = db
        st.success("✅ PDFs indexed and saved!")

if st.session_state.db is None and os.path.exists(PERSIST_DIR):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(model="gpt-4o-mini")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

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

        docs = retriever.invoke(query)

        context = ""
        sources = []

        for doc in docs:
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

        message_placeholder = st.empty()
        full_response = ""

        for chunk in response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        unique_sources = list(set(sources))

        st.markdown("### 📊 Sources:")
        for src in unique_sources:
            st.write("-", src)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })