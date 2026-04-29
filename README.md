# RAG Document QA System

An intelligent **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDFs and ask questions with accurate, source-grounded answers.

Built using **LangChain, OpenRouter (LLMs), ChromaDB (vector database), and Streamlit UI**.

---

# Features

-  Multi-PDF document support  
-  Semantic search using embeddings + vector database  
-  Context-aware answers using LLMs (OpenRouter)  
-  Real-time streaming responses  
-  Conversation memory (multi-turn chat)  
-  Source attribution (see which document + page answer came from)  
-  Chunk preview (transparency into retrieved context)  
-  Activity logging (retrieval + response tracking)  
-  Token usage & response time tracking  

---

# How It Works (Architecture)
User Query
↓
Retriever (Chroma Vector DB + Embeddings)
↓
Top-K Relevant Chunks
↓
Prompt Builder (Context + Chat History)
↓
LLM (OpenRouter - LLaMA / Mistral)
↓
Final Answer (with sources)

# Key Highlights
Designed a production-style RAG pipeline
Reduced hallucinations using context-only prompting
Implemented semantic retrieval with vector database
Built explainable AI system with source visibility
Optimized UX with streaming + activity logs
Future Improvements
Hybrid search (BM25 + Vector DB)
Reranking with cross-encoders
FastAPI backend for production deployment
Docker containerization
Cloud deployment (AWS / GCP / Render)

 # Future Improvements

- **Hybrid Retrieval (BM25 + Vector Search)**  
  Combine keyword-based (BM25) and semantic (embedding) search to improve recall and answer accuracy

- **Reranking with Cross-Encoders**  
  Apply a second-stage reranker to reorder retrieved chunks based on relevance to the query

- **FastAPI Backend (Production API)**  
  Expose the RAG pipeline as REST endpoints for scalable integration with other applications

- **Dockerization**  
  Containerize the application for consistent environments and easy deployment

- **Cloud Deployment**  
  Deploy on AWS / GCP / Render with persistent vector storage and scalable inference

- **RAG Evaluation Pipeline**  
  Measure retrieval quality, response accuracy, and hallucination rates using test datasets

- **Caching & Performance Optimization**  
  Add response caching and optimize latency for faster query handling

- **Authentication & Multi-User Support**  
  Enable secure access and user-specific document sessions

# Connect With Me
  Ajay Chirmade
  Pune, India
  LinkedIn: linkedin.com/in/ajay-chirmade-470b64105
  GitHub: https://github.com/chirmadeajay

⭐ If you like this project

Give it a ⭐ on GitHub!
