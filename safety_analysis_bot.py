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


model_name = "https://huggingface.co/sudraj/acc_state_dic"
model = RobertaForSequenceClassification.from_pretrained(model_name,token = "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")


st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!!')
st.write("Model loaded successfully")




