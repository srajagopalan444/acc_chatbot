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

st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!')
st.write(acc_data.head())
st.write(X.head())
st.write(y.head())
st.write("Shape of X_train:",X_train.shape)
st.write("Shape of X_test:",X_test.shape)
st.write("Shape of y_train:",y_train.shape)
st.write("Shape of y_test:",y_test.shape)

st.write("X_train_ids:", type(X_train_ids[0][0])))
st.write("X_train_masks:",type(X_train_masks[0][0])))
st.write("X_test_ids:",type(X_test_ids[0][0])))
st.write("X_test_masks:",type(X_test_masks[0][0])))


