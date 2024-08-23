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


import streamlit as st
from transformers import pipeline

st.title('⛑️ Safety Bot ⛑️')

# Load the language model (replace with your desired model)
#model = pipeline("text-generation", model="gpt2")

st.write('Hello world!!')
st.write("Model loaded successfully")

# Initialize history storage
if 'messages' not in st.session_state:
    st.session_state.messages = []

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

    # Generate response from the language model
    response = model.generate(prompt, max_length=100, num_beams=4)
    response_text = response[0]['text']

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_text)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})





