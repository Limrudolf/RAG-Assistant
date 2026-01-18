import os
import pickle
import datetime
from typing import List, Literal, Optional
from operator import itemgetter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google.cloud import bigquery

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.retrievers import ParentDocumentRetriever 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Global State ---
ml_models = {}

# --- Pydantic Schemas (The Contract) ---
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: List[Message] = []

class SourceDoc(BaseModel):
    page_content: str
    metadata: dict

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    sources: List[SourceDoc]
    feedback: str # "Positive" or "Negative"

class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="'yes' or 'no'")

# --- Helper Functions ---
def log_to_bigquery(question, answer, sources, feedback=None):
    """Logs interactions to BigQuery (Runs in Background)"""
    try:
        # Assumes GOOGLE_APPLICATION_CREDENTIALS is set in .env
        client = bigquery.Client()
        
        # Convert source objects to string
        source_str = "None"
        if sources:
            source_str = "; ".join([f"Page {d.metadata.get('page', '?')}" for d in sources])

        rows = [{
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "sources": source_str
        }]

        # Change this ID to your actual dataset.table
        table_id = f"{client.project}.rag_logs.Interactions"
        
        errors = client.insert_rows_json(table_id, rows)
        if errors:
            print(f"BigQuery Error: {errors}")
            
    except Exception as e:
        print(f"Logging Failed: {e}")

# --- Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initialising model")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=GOOGLE_API_KEY, temperature=0)

    # 1. Grader Chain
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "Grade relevance 'yes' or 'no'."),
        ("human", "Doc: {context} \n\n Question: {question}"),
    ])
    grader_chain = grader_prompt | structured_llm_grader

    # 2. Retriever
    try:
        child_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        with open("docstore.pkl", "rb") as f:
            store = pickle.load(f)
            
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        retriever = ParentDocumentRetriever(
            vectorstore=child_vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            search_kwargs={"k": 5}
        )
    except Exception as e:
        print(f"Failed to load Index: {e}")
        yield
        return

    # 3. Search Query Generator (History Contextualizer)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformulate the question based on history if needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    def get_search_query(input_dict):
        if input_dict.get("chat_history"):
            return contextualize_q_chain.invoke(input_dict)
        return input_dict["question"]

    # 4. Filter Logic
    def filter_documents(inputs):
        question = inputs["question"]
        docs = inputs["context"]
        filtered = []
        for d in docs:
            score = grader_chain.invoke({"question": question, "context": d.page_content})
            if score.binary_score == "yes":
                filtered.append(d)
        return {"context": filtered, "question": question}

    # 5. Final Assembly
    def format_docs(docs):
        if not docs: return "NO RELEVANT CONTEXT FOUND."
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Answer based ONLY on context: {context} \n Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        RunnableParallel({
            "context": RunnableLambda(get_search_query) | retriever,
            "question": RunnableLambda(get_search_query),
            "chat_history": itemgetter("chat_history")
        })
        | RunnableLambda(filter_documents)
    )

    final_chain = retrieval_chain | RunnableParallel({
        "answer": (RunnablePassthrough.assign(context=lambda x: format_docs(x["context"])) | prompt | llm | StrOutputParser()),
        "sources": itemgetter("context")
    })

    ml_models["chain"] = final_chain
    print("--- ✅ Model Ready ---")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    if "chain" not in ml_models:
        raise HTTPException(status_code=503, detail="Model Loading")
    
    # Convert string history to LangChain Message objects
    history_objs = []
    for msg in request.chat_history:
        if msg.role == "user":
            history_objs.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            history_objs.append(AIMessage(content=msg.content))

    # Invoke
    res = ml_models["chain"].invoke({
        "question": request.question, 
        "chat_history": history_objs
    })

    # Prepare Sources for JSON response
    # We must convert LangChain Documents to Pydantic SourceDocs
    valid_sources = [
        SourceDoc(page_content=d.page_content, metadata=d.metadata) 
        for d in res["sources"]
    ]

    # Log to BQ in background
    background_tasks.add_task(
        log_to_bigquery,
        request.question,
        res["answer"],
        valid_sources,
        None
    )

    return ChatResponse(answer=res["answer"], sources=valid_sources)

@app.post("/feedback")
async def feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        log_to_bigquery,
        request.question,
        request.answer,
        request.sources,
        request.feedback
    )
    return {"status": "ok"}