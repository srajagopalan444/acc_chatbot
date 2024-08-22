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


st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!')
st.write(acc_data.head())



