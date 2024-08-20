import streamlit as st

st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!')

with st.chat_message(name="Analyst", avatar='👤'):
  st.write("Hi")
with st.chat_message(name="Assistant"):
  st.write("How many I help you today?")

if 'messages' not in st.session_state:
  st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter the description of the incident..."):
    # Display user message in chat message container
    with st.chat_message("Analyst"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "Analyst", "content": prompt})
