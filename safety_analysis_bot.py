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

#Applying nlp prep
acc_data['Description_cleaned'] = acc_data['Description'].apply(nlp_text_prep)
#acc_data['Description_cleaned'].head() 

# Applying label encoding for target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
acc_data['Accident Level'] = le.fit_transform(acc_data['Accident Level'])
acc_data['Potential Accident Level'] = le.fit_transform(acc_data['Potential Accident Level'])

##Train-test split
X = acc_data['Description_cleaned']
y = acc_data['Accident Level']

from sklearn.model_selection import train_test_split
#Stratified split is required because of a heavily imbalanced dataframe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


#RoBERTa Tokenizer
from transformers import RobertaTokenizer
tokenizer_r = RobertaTokenizer.from_pretrained("roberta-base")
def roberta_text_prep(text):
  # Max length of 256 ensures a larger yet more standard acceptance of text input size
  tokens = tokenizer_r.encode_plus(text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length')
  input_ids = tokens['input_ids']
  attention_mask = tokens['attention_mask']
  return input_ids, attention_mask

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
                         torch.tensor(y_train.to_numpy())) # Convert to NumPy array first

test_data = TensorDataset(torch.tensor(X_test_ids),
                         torch.tensor(X_test_masks),
                         torch.tensor(y_test.to_numpy())) # Convert to NumPy array first

##Training the model
from transformers import AdamW, get_linear_schedule_with_warmup
# Define model hyperparameters
model_name = "roberta-base"
num_classes = 5
#_roberta_ft 
model= RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
learning_rate = 1e-3
epochs = 5
uf_layers = -2
# Define optimizer and scheduler
weight_decay = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay )
total_steps = len(X_train) * epochs
warmup_steps = 0.001 * total_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Training loop
for epoch in range(epochs):
    train_dataloader = DataLoader(train_data, batch_size=8)
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
        test_loss = 0
        test_preds = []
        test_labels = []
        for batch in DataLoader(test_data, batch_size=8):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            test_loss += outputs.loss.item()
            test_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

        print(f"Epoch: {epoch+1}, Training Loss: {train_loss/len(train_data)}, Test Loss: {test_loss/len(test_data)}")

st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!')
st.write(acc_data.head())
st.write(X.head())
st.write(y.head())
st.write("Shape of X_train:",X_train.shape)
st.write("Shape of X_test:",X_test.shape)
st.write("Shape of y_train:",y_train.shape)
st.write("Shape of y_test:",y_test.shape)

'''
st.write("X_train_ids:", X_train_ids)
st.write("X_train_masks:",X_train_masks)
st.write("X_test_ids:",X_test_ids)
st.write("X_test_masks:",X_test_masks)
st.write("Train Data:", train_data[0])
st.write("Test Data:", test_data[0])
'''


