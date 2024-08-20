import streamlit as st

st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!')

with st.chat_message(name="Assistant"):
  st.write("How many I help you today?")
with st.chat_message(name="Analyst", avatar='👤'):
  st.write("Test")
