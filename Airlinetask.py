import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

sys_template = '''
You are a helpful assistant who functions as a live chat agent of a leading airline.
You need to generate replies for the users based on the {userqn} and also the chat history provided as {chats}.
Your task involves satisfying the below conditions:
If user has not given source or destination or travel class, prompt the user to get all the three.
If user has given only source, prompt the user to give destination and travel class.
If user has given only destination, prompt the user to give source and travel class.
If user has given source and destination alone, prompt the user to give travel class.
Then after collecting these information ask the user questions based on {userqn}
'''

userqn = """
- Baggage Allowance:
  - Domestic Flights:
    - Economy: Comfort - 15 kg/33 lb, Comfort Plus - 15 kg/33 lb, Flex - 25 kg/55.1 lb
    - Premium Economy: Comfort Plus - 15 kg/33 lb, Flex - 25 kg/55.1 lb
    - Business: Comfort Plus - 25 kg/55.1 lb, Flex - 35 kg/77.1 lb
    - First: First - 40 kg/88.1 lb
  - International Flights:
    - India to Sri Lanka: Economy - 30 kg/66.1 lb, Comfort Plus - 30 kg/66.1 lb, Flex - 40 kg/88.1 lb
    - India to Bangladesh: Economy - 30 kg/66.1 lb, Comfort Plus - 30 kg/66.1 lb, Flex - 35 kg/77.1 lb
    ...
- Points to remember:
  - The allowances apply only to Air India-operated flights.
  - Baggage allowance for international sectors applies to domestic flights on the same ticket.
  - Free Baggage Allowance (FBA) available for connecting flights within 24 hours.
  - Star Alliance Gold members can carry an additional 20 kg/44 lb in economy class.
  - Infants are entitled to one collapsible stroller/carrycot/infant car seat.
  - Maximum weight for a single piece of baggage is 32 kg/71 lb.
  - Assistive devices can be carried free of charge as per DGCA guidelines.
  
- Visa, Documents, and Travel Tips:
  - Ensure you have your flight tickets, valid visa, and passport.
  - Specific travel guidelines for countries like the US, Canada, and Gulf countries.
  - Australian travel guidelines: Report any BNI over AUD 10,000.

- Special Assistance:
  - Medical needs and wheelchair services.
  - Guidelines for traveling with service dogs and pets.
"""

Prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_template),
        ("user", "{item}")
    ]
)

out = StrOutputParser()

chain = Prompt | llm | out

st.markdown("<h1 style='text-align: center; color: #1A73E8;'>Airline Chatbot</h1>", unsafe_allow_html=True)

st.markdown("<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'>", unsafe_allow_html=True)

if 'chats' not in st.session_state:
    st.session_state['chats'] = []

def update_conversation_history():
    conversation_history = ""
    for msg in st.session_state['chats']:
        if isinstance(msg, HumanMessage):
            conversation_history += f"<div style='background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 5px; display: flex; align-items: center;'><i class='fas fa-user' style='margin-right: 10px; color: #1A73E8;'></i><strong>Customer:</strong> {msg.content}</div>"
        else:
            conversation_history += f"<div style='background-color: #E8E8E8; padding: 10px; border-radius: 10px; margin-bottom: 5px; display: flex; align-items: center;'><i class='fas fa-robot' style='margin-right: 10px; color: #1A73E8;'></i><strong>Chatbot:</strong> {msg.content}</div>"
    st.markdown(f"<div style='height: 400px; overflow-y: scroll; padding: 10px; border: 1px solid #CCC; border-radius: 10px;'>{conversation_history}</div>", unsafe_allow_html=True)

update_conversation_history()

user_input = st.text_input("Type your message here...", placeholder="Ask your travel-related queries...", key="user_input_field")

if st.button("Send"):
    if user_input:
        st.session_state['chats'].append(HumanMessage(content=user_input))
        response = chain.invoke({"chats": st.session_state['chats'], "userqn": userqn, "item": user_input})
        st.session_state['chats'].append(AIMessage(content=response))
        st.experimental_rerun()  
