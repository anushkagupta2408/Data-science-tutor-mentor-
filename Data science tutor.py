
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Load API Key from environment variables
load_dotenv("apidot.env")
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize AI Model with memory
chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.7, google_api_key=api_key)
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)

# Streamlit UI Configuration
st.set_page_config(page_title='Data Science Mentor', page_icon="📊", layout='wide')

# Sidebar - Learning Level Selection
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg", width=100)


st.sidebar.title("📚 Learning Hub")
st.sidebar.write("Select your experience level and start your journey in Data Science!")
learning_level = st.sidebar.radio("Choose your expertise level:", ["Beginner", "Intermediate", "Expert"])

# Sidebar - Expandable Help Section
with st.sidebar.expander("❓ How to Use"):
    st.write("1. Type your Data Science question in the chat box below.\n"
             "2. Get AI-powered responses tailored to your expertise level.\n"
             "3. Click 'Reset Chat' to start a new conversation.")

# System Message Setup
system_message = SystemMessage(
    content=f"You are an AI Data Science tutor. Answer only Data Science-related queries. "
            f"Provide responses based on the user's level: {learning_level}."
)

# Main Page Title & Description
st.title("🧑‍💻 Data Science Mentor")
st.subheader("Your AI-powered assistant for learning Data Science")
st.markdown(
    "---\n"
    "💡 **Ask any question related to Machine Learning, Deep Learning, Statistics, Python, and more!**\n"
    "---"
)

# Display Chat History
for message in st.session_state.chat_memory.chat_memory.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Handle User Query
user_query = st.chat_input("Ask your Data Science question...")
if user_query:
    conversation = [system_message] + st.session_state.chat_memory.chat_memory.messages + [HumanMessage(content=user_query)]
    ai_response = chat_model.invoke(conversation)
    st.session_state.chat_memory.chat_memory.add_user_message(user_query)
    st.session_state.chat_memory.chat_memory.add_ai_message(ai_response.content)
    
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.chat_message("assistant"):
        st.markdown(ai_response.content)

# Reset Conversation Button
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.write("🔄 **Click below to reset the chat.**")
with col2:
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.chat_memory.clear()
        st.rerun()



