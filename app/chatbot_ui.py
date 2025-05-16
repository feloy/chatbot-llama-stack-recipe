from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage

import streamlit as st
import requests
import time
import os 

model_service_base = os.getenv("MODEL_ENDPOINT",
                               "http://localhost:8001")
model_service = f"{model_service_base}/v1"
request_kwargs = {}

@st.cache_resource(show_spinner=False)
def checking_model_service():
    start = time.time()
    print("Checking Model Service Availability...")
    ready = False
    while not ready:
        try:
            request_cpp = requests.get(f'{model_service}/models', **request_kwargs)
            if request_cpp.status_code == 200:
                j = request_cpp.json()
                model = j["data"][0]
                id = model["identifier"]
                print('modelId', id)
                ready = True
        except Exception as inst:
            print(inst)
            pass
        time.sleep(1)
    print(f"{id} Model Service Available")
    print(f"{time.time()-start} seconds")
    return id

with st.spinner("Checking Model Service Availability..."):
    modelId = checking_model_service()

def enableInput():
    st.session_state["input_disabled"] = False

def disableInput():
    st.session_state["input_disabled"] = True

st.title("💬 Chatbot")  
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", 
                                     "content": "How can I help you?"}]
if "input_disabled" not in st.session_state:
    enableInput()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(disabled=st.session_state["input_disabled"],on_submit=disableInput):
    message_placeholder = st.empty()
    full_response = ""

    st.session_state.messages.append({"role": "user", "content": prompt})
    client = LlamaStackClient(
        base_url=model_service_base,
    )

    response = client.inference.chat_completion(
        messages=[
            UserMessage(
                content=prompt,
                role="user",
            ),
        ],
        model_id=modelId,
        stream=True
    )
    for chunk in response:
        if chunk.event is not None and chunk.event.event_type == "progress":
            full_response += chunk.event.delta.text
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    enableInput()
    st.rerun()
    