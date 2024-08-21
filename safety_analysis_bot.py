

import streamlit as st
import random
import time
import numpy as np

import torch
from transformers import RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import gdown


@st.cache_resource
def load_model_from_drive(url):
    output = 'model.pt'
    gdown.download(url, output, quiet=False)
    model = torch.load(output)
    model.eval()
    return model



def response_generator():
  response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
  for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!')

url = 'https://drive.google.com/file/d/1-0cIDQrII4JaRL3Vvnz07Ad-Yccnc_nV/view?usp=sharing'
model = load_model_from_drive(url)

st.write("Model loaded successfully!")

#initialise history storage
if 'messages' not in st.session_state:
  st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter the description of the incident..."):
    # Display user message in chat message container
    with st.chat_message("Analyst", avatar='👤'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "Analyst", "content": prompt})
    with st.chat_message("assistant"):
      response = st.write_stream(response_generator())
      #st.markdown(response)
      st.session_state.messages.append({"role": "assistant", "content": response})
      # Add assistant response to chat history
