# app.py
import streamlit as st
from agent import SearchAgent
from tools import (
    get_page_content,
    get_wikipedia_page,
    search_duck_duck_go,
    search_wikipedia,
)
import streamlit.components.v1 as components
import json

# Initialize the agent
agent = SearchAgent(
    tools=[
        get_wikipedia_page,
        search_wikipedia,
        search_duck_duck_go,
        get_page_content,
    ]
)

# Function to initialize conversation history in session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

# Function to save the conversation to a JSON file (simulating a database)
def save_conversation(filename="conversation_history.json"):
    with open(filename, "w") as f:
        json.dump(st.session_state.get('chat_history', []), f)

# Function to load the conversation from a JSON file
def load_conversation(filename="conversation_history.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Streamlit UI
st.title("AI-Powered Search Agent")

# Initialize session state
initialize_session_state()

# Load previous conversation history
st.session_state['chat_history'] = load_conversation()

# Display the conversation history
if st.session_state['chat_history']:
    st.subheader("Conversation History")
    for entry in st.session_state['chat_history']:
        st.markdown(f"**User:** {entry['user']}")
        st.markdown(f"**Agent:** {entry['agent']}")
        st.markdown("---")

# User input
query = st.text_input("Enter your query:")

if query:
    with st.spinner("Searching..."):
        try:
            response = agent.invoke(query)  # Call the invoke method
            st.success("Results:")
            st.write(response)

            # Update chat history
            st.session_state['chat_history'].append({"user": query, "agent": response})

            # Save the updated conversation
            save_conversation()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")