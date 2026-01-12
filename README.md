🏆 Competitive Programming Assistant (RAG Agent)
A specialized AI assistant designed to help users master competitive programming concepts. Built with LangChain, Google Gemini, and Streamlit, this application uses Advanced RAG (Parent-Document Retrieval) to provide accurate, context-aware answers from the "Competitive Programming Handbook."


🚀 View Live Demo: https://cp-rag-assistant.streamlit.app/


📖 Project Overview
Standard RAG (Retrieval-Augmented Generation) often fails on complex technical topics because splitting text into small chunks removes necessary context.

This project solves that using Parent-Document Retrieval. It indexes small, granular chunks to find the most relevant algorithms (e.g., "Dynamic Programming optimizations") but retrieves the entire parent section to generate the answer. This ensures the LLM has full context (proofs, code examples, edge cases) before answering.
Additionally, the system features a Feedback Loop: every interaction and user rating (👍/👎) is streamed to Google BigQuery for performance monitoring.


✨ Key Features
🧠 Advanced RAG Architecture: Utilizes ParentDocumentRetriever to decouple indexing chunks (small) from generation chunks (large).

✅ Hallucination Guardrails: Implements a "Grader Chain" using Pydantic structured output to verify if retrieved documents are actually relevant before answering.

🗣️ Context-Aware Chat: Rewrites follow-up questions (e.g., "Can you show me the code for that?") into standalone queries using conversation history.

📊 Analytics Pipeline: Automatically logs questions, answers, cited sources, and user feedback to a Google BigQuery data warehouse.

📂 Transparent Citations: Displays the exact pages and source text used to generate every answer.


🛠️ Tech Stack
LLM & Embeddings: Google Gemini Pro (gemini-3-flash-preview), Google GenAI Embeddings.

Framework: LangChain (Python), LangChain

Vector Store: FAISS (Facebook AI Similarity Search).

Frontend: Streamlit.

Data Warehouse: Google BigQuery.

Deployment: Streamlit Community Cloud.


🏗️ Architecture
Ingestion: The PDF is split into "Parent" documents, which are further split into "Child" chunks. Child chunks are embedded and stored in FAISS; Parents are stored in an InMemoryStore.

Retrieval: The system searches for Child chunks similar to the query, then fetches their corresponding Parent documents.

Grading: A lightweight LLM call grades the documents. Irrelevant ones are discarded.

Generation: Relevant Parent documents are passed to Gemini Pro to generate the final answer.

Logging: The interaction is asynchronously logged to BigQuery.
