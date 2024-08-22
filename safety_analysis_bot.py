import streamlit as st
import random
import time
import numpy as np
import pandas as pd

import torch
from transformers import RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import tensorflow as tf


from transformers import AutoModel

model = AutoModel.from_pretrained("sudraj/acc_state_dic", use_auth_token="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")


st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!!')
st.write("Model loaded successfully")

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



