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

'''
model_name = "https://huggingface.co/sudraj/acc_state_dic/tree/main"
model = RobertaForSequenceClassification.from_pretrained(model_name)
'''

st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!!')
st.write("Model loaded successfully")




