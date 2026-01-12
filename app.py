import streamlit as st
import os
import datetime
from operator import itemgetter
from pydantic import BaseModel, Field
from typing import Literal, List
import pickle
from google.cloud import bigquery
from google.oauth2 import service_account

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.retrievers import ParentDocumentRetriever 
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

MY_API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Competitive Programming Handbook", layout="wide")
st.title("Competitive Programming Handbook")

def save_feedback(index):
    """Triggered when a user clicks thumbs up/down"""
    # Get the message and the new feedback value
    msg = st.session_state.messages[index]
    feedback_score = st.session_state[f"feedback_{index}"]
    
    if feedback_score:
        # Map score to string if needed, or just log the raw dictionary/score
        sentiment_map = {1 : "Positive", 0: "Negative"}
        sentiment = sentiment_map.get(feedback_score, "Unknown")
        
        # Log to BigQuery (Updating the row with feedback)
        # We re-log the entry but this time WITH feedback
        log_to_bigquery(
            question=msg["question"], 
            answer=msg["content"], 
            sources=msg["sources"], 
            feedback=sentiment
        )
        st.toast(f"Feedback {sentiment} recorded!", icon="📝")

def log_to_bigquery(question, answer, sources, feedback=None):
    """Logs the interaction to BigQuery."""
    try:
        if "gcp_service_account" not in st.secrets:
            return

        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)

        source_str = "; ".join([f"Page {d.metadata.get('page', '?')}" for d in sources]) if sources else "None"
        
        rows_to_insert = [{
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "sources": source_str
        }]

        table_id = f"{credentials.project_id}.rag_logs.Interactions"
        errors = client.insert_rows_json(table_id, rows_to_insert)
        
        if errors:
            st.error(f"BigQuery Error: {errors}")
            
    except Exception as e:
        st.error(f"Logging Failed: {e}")

#Structured Output Schema (Modern Pydantic v2)
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Are the documents relevant to the question? 'yes' or 'no'"
    )

@st.cache_resource
def load_chain():

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=MY_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=MY_API_KEY, temperature=0)
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    grader_system = """Grade the relevance of the retrieved document to the user's question. 
    If the document contains information related to the query, grade it as 'yes'. Otherwise, 'no'."""
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", grader_system),
        ("human", "Document: {context} \n\n Question: {question}"),
    ])
    grader_chain = grader_prompt | structured_llm_grader

    child_vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    #Load the Parent Content
    with open("docstore.pkl", "rb") as f:
        store = pickle.load(f)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": 5}
    )

    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    def get_search_query(input_dict):
        if input_dict.get("chat_history"):
            return contextualize_q_chain.invoke(input_dict)
        else:
            return input_dict["question"]
    search_query_chain = RunnableLambda(get_search_query)

    def filter_documents(inputs):
        """Filters out chunks that the grader deems irrelevant."""
        question = inputs["question"]
        docs = inputs["context"]
        
        filtered_docs = []
        for d in docs:
            score = grader_chain.invoke({"question": question, "context": d.page_content})
            if score.binary_score == "yes":
                filtered_docs.append(d)
        
        return {"context": filtered_docs, "question": question}

    def format_docs(docs):
        if not docs:
            return "NO RELEVANT CONTEXT FOUND. Tell the user you couldn't find a specific rule."
        return "\n\n".join(doc.page_content for doc in docs)

    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        RunnableParallel({
            "context": search_query_chain | retriever,
            "question": search_query_chain
        })
        | RunnableLambda(filter_documents)
    )

    generation_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    final_pipeline = retrieval_chain | RunnableParallel({
        "answer": generation_chain,
        "sources": itemgetter("context")
    })

    return final_pipeline

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Display Chat History (Now with Feedback Buttons!)
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Only show extra controls for the Assistant
        if message["role"] == "assistant":
            # A. Show Sources (Persisted)
            if message.get("sources"):
                with st.expander(f"View {len(message['sources'])} Verified Sources"):
                    for idx, doc in enumerate(message["sources"]):
                        st.info(f"**Source {idx+1} (Page {doc.metadata.get('page')}):**\n\n{doc.page_content}")
            
            # B. Show Feedback Button
            # We use 'key' to bind this specific button to this specific message index
            st.feedback(
                "thumbs", 
                key=f"feedback_{i}", 
                on_change=save_feedback,
                args=[i]  # Pass the index to the function
            )

# 2. Handle New Input
if user_input := st.chat_input("Ask about Competitive Programming"):
    # A. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # B. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and Grading Evidence..."):
            # Prepare context for the chain
            chat_history = []
            for msg in st.session_state.messages[:-1]: 
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            # Run Chain
            response = chain.invoke({
                "question": user_input,
                "chat_history": chat_history
            })
            
            answer_text = response["answer"]
            filtered_docs = response["sources"]

            # Display Answer immediately
            st.markdown(answer_text)
            
            # Display Sources immediately
            if filtered_docs:
                with st.expander(f"View {len(filtered_docs)} Verified Sources"):
                    for i, doc in enumerate(filtered_docs):
                        st.info(f"**Source {i+1} (Page {doc.metadata.get('page')}):**\n\n{doc.page_content}")

            # C. Log the "Base" Interaction (Feedback is None for now)
            log_to_bigquery(user_input, answer_text, filtered_docs, feedback=None)

    # D. Save to History (With extra metadata for the UI loop)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer_text,
        "sources": filtered_docs, # Save sources so they persist!
        "question": user_input    # Save question so we can log it with feedback later
    })
    
    # Rerun to show the feedback button immediately
    st.rerun()