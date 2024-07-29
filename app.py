
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_huggingface import ChatHuggingFace

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
huggingfacehub_api_token = os.getenv("huggingfacehub_api_token")

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
#repo_id = "meta-llama/Meta-Llama-3-70B-Instruct"
llm = HuggingFaceEndpoint(repo_id=repo_id,
                        max_new_tokens=512,
                        temperature=0.1,
                        token=huggingfacehub_api_token)

chat_model = ChatHuggingFace(llm=llm)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LLM: Simple Conversation Chat App", page_icon=":robot:")
st.header("Hey, I'm your Chat GPT")

if "sessionMessages" not in st.session_state:
     st.session_state.sessionMessages = [
        SystemMessage(content="You are a helpful assistant.")
    ]

def load_answer(question):
    st.session_state.sessionMessages.append(HumanMessage(content=question))
    assistant_answer  = chat_model(st.session_state.sessionMessages)
    st.session_state.sessionMessages.append(AIMessage(content=assistant_answer.content))
    return assistant_answer.content

def get_text():
    input_text = st.text_input("You question: ", key= input)
    return input_text

user_input=get_text()
submit = st.button('Generate')  

if submit:
    
    response = load_answer(user_input)
    st.subheader("Answer:")

    st.write(response,key= 1)