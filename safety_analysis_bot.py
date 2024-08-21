

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


#Loading the data
acc_data = pd.read_csv("IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv")
#col = acc_data.columns[0]
#acc_data = acc_data.drop(acc_data.columns[acc_data.columns.str.contains('Unnamed')], axis=1)
columns =['Accident Level', 'Potential Accident Level', 'Description']
acc_data = acc_data[columns]
acc_data


#Required functions
#NLP Basic Prep
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

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


#Training the model
#Function to create a base RoBERTa model
def train_model_roberta(model,uf_layers, X_train, X_test, num_classes, learning_rate,epochs, optimizer,scheduler):
    #Unfreezing last x layers
    for param in model.roberta.encoder.layer[uf_layers:].parameters():
      param.requires_grad = True

    start_time = time.time()

    # Prepare training and validation data
    X_train_ids, X_train_masks = [], []
    for text in X_train:
      ids, mask = roberta_text_prep(text)
      X_train_ids.append(ids)
      X_train_masks.append(mask)

    X_test_ids, X_test_masks = [], []
    for text in X_test:
      ids, mask = roberta_text_prep(text)
      X_test_ids.append(ids)
      X_test_masks.append(mask)

    train_data = TensorDataset(torch.tensor(X_train_ids),
                          torch.tensor(X_train_masks),
                         y_train) # Convert to NumPy array first

    test_data = TensorDataset(torch.tensor(X_test_ids),
                            torch.tensor(X_test_masks),
                            y_test) # Convert to NumPy array first

    for epoch in range(epochs):
      train_dataloader = DataLoader(train_data, batch_size=8)  # Adjust batch size based on memory constraints
      model.train()

      # Training loop
      for batch in tqdm(train_dataloader):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()

      # Validation loop
      model.eval()
      with torch.no_grad():
        train_loss = 0
        train_preds = []
        train_labels = []
        for batch in DataLoader(train_data, batch_size=8):
          input_ids, attention_mask, labels = batch
          outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # Get model outputs
          train_loss += outputs.loss.item() # Accumulate validation loss
          train_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())  # Get predicted labels from logits
          train_labels.extend(labels.cpu().numpy()) # Accumulate true labels
        print(f"Epoch: {epoch+1}, Training Loss: {train_loss/len(train_data)}")

        with torch.no_grad():
          test_loss = 0
          test_preds = []
          test_labels = []
          for batch in DataLoader(test_data, batch_size=8):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # Get model outputs
            test_loss += outputs.loss.item() # Accumulate validation loss
            test_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())  # Get predicted labels from logits
            test_labels.extend(labels.cpu().numpy()) # Accumulate true labels
          print(f"Epoch: {epoch+1}, Test Loss: {test_loss/len(test_data)}")

    end_time = time.time()  # Record end time
    training_time_r_ul = end_time - start_time
    print(f"Training completed in {training_time_r_ul:.2f} seconds")

#Predict Accident Level
def predict_accident_roberta(text):
  with torch.no_grad():
      text = nlp_text_prep(text)
      input_ids, attention_mask = roberta_text_prep(text)
      logits = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask])).logits
      predicted_label = torch.argmax(logits, dim=1).item()
      mapped_label = predicted_label + 1  # Map 0 to 1, 1 to 2, etc.
  return mapped_label

##Response Generator
def response_generator(prompt):
    prompt = nlp_text_prep(prompt)
    prompt = roberta_text_prep(prompt)
    acc_pred = predict_accident_roberta(prompt)
    response = f"The predicted accident level is {acc_pred}"
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

#Applying functions
##Basic NLP Cleanup
acc_data['Description_cleaned'] = acc_data['Description'].apply(nlp_text_prep)
#acc_data['Description_cleaned'].head() 



##Train-test split
X = acc_data['Description_cleaned']
y = acc_data['Accident Level']

from sklearn.model_selection import train_test_split
#Stratified split is required because of a heavily imbalanced dataframe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

##Training the model
from transformers import AdamW, get_linear_schedule_with_warmup
# Define model hyperparameters
model_name = "roberta-base"
num_classes = 5
model_roberta_ft = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
learning_rate = 1e-3
epochs = 5
uf_layers = -2
# Define optimizer and scheduler
weight_decay = 0.001
optimizer = torch.optim.AdamW(model_roberta_ft.parameters(), lr=learning_rate, weight_decay=weight_decay )
total_steps = len(X_train) * epochs
warmup_steps = 0.001 * total_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

train_model_roberta(model_roberta_ft,uf_layers, X_train, X_test, num_classes, learning_rate,epochs, optimizer,scheduler)
print("Model successfully trained")


st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!')
st.write(acc_data.head())
  


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
