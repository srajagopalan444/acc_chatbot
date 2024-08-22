

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


