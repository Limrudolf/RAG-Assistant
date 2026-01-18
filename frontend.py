import streamlit as st
import requests

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="CP Handbook (FastAPI)", layout="wide")
st.title("Competitive Programming Handbook")

if "messages" not in st.session_state:
    st.session_state.messages = []

def send_feedback(index, question, answer, sources, feedback_score):
    """Calls the Feedback API Endpoint"""
    sentiment_map = {1: "Positive", 0: "Negative"}
    sentiment = sentiment_map.get(feedback_score, "Unknown")
    
    try:
        payload = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "feedback": sentiment
        }
        requests.post(f"{API_URL}/feedback", json=payload)
        st.toast(f"Feedback {sentiment} recorded!", icon="📝")
    except Exception as e:
        st.error(f"Feedback Error: {e}")

# 1. Display History
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            # Show Sources
            if message.get("sources"):
                with st.expander(f"View {len(message['sources'])} Verified Sources"):
                    for idx, doc in enumerate(message["sources"]):
                        st.info(f"**Source {idx+1} (Page {doc['metadata'].get('page')}):**\n\n{doc['page_content']}")
            
            # Show Feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                on_change=lambda idx=i: send_feedback(
                    idx, 
                    message["question"], 
                    message["content"], 
                    message["sources"], 
                    st.session_state[f"feedback_{idx}"]
                )
            )

# 2. Handle Input
if user_input := st.chat_input("Ask a question..."):
    # Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Connecting to API..."):
            try:
                # Prepare History (Just list of strings for simplicity)
                history = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages 
                    # Optional: Filter out the very last message if it's the current prompt, 
                    # but usually you want previous history excluding current prompt.
                ][:-1]
                
                response = requests.post(
                    f"{API_URL}/chat", 
                    json={"question": user_input, "chat_history": history}
                )
                response.raise_for_status()
                data = response.json()
                
                answer_text = data["answer"]
                sources = data["sources"]

                st.markdown(answer_text)
                
                if sources:
                    with st.expander("Sources"):
                         for s in sources:
                             st.write(f"Page {s['metadata'].get('page')}")

                # Append to State
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text, 
                    "sources": sources,
                    "question": user_input
                })

                st.rerun()

            except Exception as e:
                st.error(f"API Connection Failed: {e}")