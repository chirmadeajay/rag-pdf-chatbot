import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
import tempfile

st.title("🚀 RAG PDF Chatbot (100% Free)")

# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# --- Process Uploaded PDFs ---
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
        st.success("✅ PDFs indexed successfully!")

# --- Free Groq LLM (no quota issues) ---
llm = ChatGroq(
    model="llama3-8b-8192",                      # ← free, fast, no quota issues
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0.3
)

# --- Show Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
if query := st.chat_input("Ask anything about your documents"):

    if st.session_state.retriever is None:
        st.warning("⚠️ Please upload at least one PDF first.")

    else:
        st.chat_message("user").write(query)

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        retrieved_docs = st.session_state.retriever.invoke(query)

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
