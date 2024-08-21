

import streamlit as st
import random
import time
import numpy as np
import pandas as pd

import torch
from transformers import RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


#Loading the data
acc_data = pd.read_csv("IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv")
#acc_data = acc_data.drop("Unnamed: 0", axis=1, inplace=True)
acc_data



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
st.write(acc_data)
  


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
