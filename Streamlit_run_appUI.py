import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore   # ← no faiss needed
import tempfile

st.title("🚀 RAG PDF Chatbot (100% Free)")

# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# --- Process Uploaded PDFs ---
if uploaded_files and st.session_state.db is None:
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

        # Pure Python in-memory store — no C++ deps, no install issues
        db = InMemoryVectorStore.from_documents(
            docs,
            embedding=st.session_state.embeddings
        )

        st.session_state.db = db
        st.success("✅ PDFs indexed successfully!")

# --- Free Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.3
)

# --- Show Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
if query := st.chat_input("Ask anything about your documents"):

    if st.session_state.db is None:
        st.warning("⚠️ Please upload at least one PDF first.")

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
