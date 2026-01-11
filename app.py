import streamlit as st
import os
from operator import itemgetter
from pydantic import BaseModel, Field
from typing import Literal, List
import pickle

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about Competitive Programming"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and Grading Evidence..."):
            chat_history = []
            for msg in st.session_state.messages[:-1]: # Skip the latest msg
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            response = chain.invoke({
                "question": user_input,
                "chat_history": chat_history
            })
            
            answer_text = response["answer"]
            filtered_docs = response["sources"]

            st.markdown(answer_text)
            
            if filtered_docs:
                with st.expander(f"View {len(filtered_docs)} Verified Sources"):
                    for i, doc in enumerate(filtered_docs):
                        st.info(f"**Source {i+1} (Page {doc.metadata.get('page')}):**\n\n{doc.page_content}")
            else:
                st.warning("No relevant sources found.")

    st.session_state.messages.append({"role": "assistant", "content": answer_text})