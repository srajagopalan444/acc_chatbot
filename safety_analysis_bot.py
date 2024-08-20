import streamlit as st

st.title('⛑️ Safety Bot ⛑️')

st.write('Hello world!')

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

response = "How may I help you today?"
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})
