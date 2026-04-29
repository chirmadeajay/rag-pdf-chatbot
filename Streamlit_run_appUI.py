import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
import tempfile

st.title("🚀 RAG PDF Chatbot - OpenRouter Free Version")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "sources" not in st.session_state:
    st.session_state.sources = []

def split_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.chunks:
    for file in uploaded_files:
        reader = PdfReader(file)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            page_chunks = split_text(text)

            for chunk in page_chunks:
                if chunk.strip():
                    st.session_state.chunks.append(chunk)
                    st.session_state.sources.append(
                        f"{file.name} - Page {page_num + 1}"
                    )

    st.success("✅ PDFs processed successfully!")

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask anything about your PDFs"):

    if not st.session_state.chunks:
        st.warning("⚠️ Please upload PDFs first")

    else:
        st.chat_message("user").write(query)
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(st.session_state.chunks + [query])

        similarities = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        ).flatten()

        top_indexes = similarities.argsort()[-4:][::-1]

        context = ""
        used_sources = []

        for idx in top_indexes:
            context += st.session_state.chunks[idx] + "\n\n"
            used_sources.append(st.session_state.sources[idx])

        history = "\n".join([
            m["content"] for m in st.session_state.messages
        ])

        prompt = f"""
You are a helpful RAG assistant.

Use ONLY the context below.
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

            st.markdown("### 📊 Sources")
            for src in sorted(set(used_sources)):
                st.write("-", src)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
