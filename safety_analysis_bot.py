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

from transformers import AutoModelForSequenceClassification, RobertaTokenizer 

# Assuming your model is indeed a sequence classification model
model = AutoModelForSequenceClassification.from_pretrained("sudraj/acc_state_dic", use_auth_token="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") 


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


def predict_accident_roberta(text):
  with torch.no_grad():
      text = nlp_text_prep(text)
      input_ids, attention_mask = roberta_text_prep(text)
      # Access logits from the correct attribute
      outputs = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
      logits = outputs.logits 
      predicted_label = torch.argmax(logits, dim=1).item()
      mapped_label = predicted_label + 1  # Map 0 to 1, 1 to 2, etc.
  return mapped_label
    
prompt = "The floor supervisor called on the foreman to lift the iron bar lying next to the work table. The foreman had to take the ladder to go up to the first level. He tripped a rung and sprained his upper ankle and foot. Advised to take rest for one week."

st.write(predict_accident_roberta(prompt))

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

    # Make prediction
    accident_level = predict_accident_roberta(prompt)

    # Display prediction result (you can customize this)
    #st.write()

    # Display predicted label in chat message container
    with st.chat_message("assistant"):
        response = "Based on the description, we can ascertain this incident to be an Accident Level:", accident_level
        #st.write("Based on the description, we can ascertain this incident to be an Accident Level:", accident_level)
        st.markdown(f"Based on the description, we can ascertain this incident to be an Accident Level: {accident_level}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})





