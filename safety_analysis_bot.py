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


from transformers import AutoModel, RobertaTokenizer 

model = AutoModel.from_pretrained("sudraj/acc_state_dic", use_auth_token="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")
tokenizer_r = RobertaTokenizer.from_pretrained("roberta-base")

#NLP Text Cleanup
def nlp_text_prep(text):
    # Lowercase conversion
    text = text.lower()
    # Punctuation, Special Charcters removal (optional)
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Adjust for desired punctuation handling
    #Stopwords and numeric characters removal
    #stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if not word.isdigit()]
    return ' '.join(words)

#RoBERTa Tokenizer
from transformers import RobertaTokenizer
tokenizer_r = RobertaTokenizer.from_pretrained("roberta-base")
def roberta_text_prep(text):
  # Max length of 256 ensures a larger yet more standard acceptance of text input size
  tokens = tokenizer_r.encode_plus(text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length')
  input_ids = tokens['input_ids']
  attention_mask = tokens['attention_mask']
  return input_ids, attention_mask

#Predict accident level
def predict_accident_roberta(input_ids, attention_mask):
    with torch.no_grad():
        encoded_input = torch.tensor([input_ids], dtype=torch.long)  # Convert input_ids to Long tensor
        logits = model(encoded_input, attention_mask=torch.tensor([attention_mask])).logits
        predicted_label = torch.argmax(logits, dim=1).item()
        mapped_label = predicted_label + 1  # Map 0 to 1, 1 to 2, etc.
    return mapped_label


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
    #response = model.generate(prompt, max_length=100, num_beams=4)
    #response_text = response[0]['text']

    # Preprocess text
    cleaned_text = nlp_text_prep(prompt)

    # Tokenize text using Roberta tokenizer
    input_ids, attention_mask = roberta_text_prep(cleaned_text)

    # Make prediction
    predicted_label = predict_accident_roberta(input_ids, attention_mask)

    # Display prediction result (you can customize this)
    st.write("Predicted Label:", predicted_label)

    # Display predicted label in chat message container
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown(f"Predicted Label: {predicted_label}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})





