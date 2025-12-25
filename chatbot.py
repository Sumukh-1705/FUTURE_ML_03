import pandas as pd
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Future Support AI", page_icon="🤖", layout="wide")

@st.cache_resource
def build_brain():
    try:
        df = pd.read_csv('customer_support.csv')
    except:
        data = {
            'text': ['hi', 'hello', 'password', 'login', 'shipping', 'order', 'refund', 'return', 'bye'],
            'category': ['greeting', 'greeting', 'reset_password', 'reset_password', 'shipping', 'shipping', 'refund', 'refund', 'goodbye']
        }
        df = pd.DataFrame(data)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    
    clf = LogisticRegression()
    clf.fit(X, df['category'])
    
    return vectorizer, clf

vectorizer, clf = build_brain()

response_bank = {
    "greeting": ["Hello! Welcome to Future Support AI.", "Hi there! How can I help you today?"],
    "reset_password": ["To reset your password, go to Settings > Security > Reset Password.", "Click 'Forgot Password' on the login page."],
    "shipping": ["You can track your order using your Tracking ID in the 'Orders' tab.", "Shipping usually takes 3-5 business days."],
    "refund": ["Refunds are available within 30 days of purchase.", "Please visit our Returns Center to start a refund."],
    "goodbye": ["Goodbye! Have a great day.", "Happy to help!"],
    "unknown": ["I'm not sure I understand. Could you ask about orders, password, or shipping?"]
}

def get_response(user_input):
    input_vec = vectorizer.transform([user_input])
    confidence = max(clf.predict_proba(input_vec)[0])
    if confidence < 0.3:
        return random.choice(response_bank["unknown"])
    predicted_category = clf.predict(input_vec)[0]
    return random.choice(response_bank.get(predicted_category, response_bank["unknown"]))

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    [data-testid="stChatMessageAvatar"] {
        display: none !important;
    }

    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
    }

    [data-testid="stChatMessageContent"] {
        background: #ffffff;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }

    div[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stChatMessageContent"] {
        background: #2563EB !important;
        color: white !important;
        border: none;
    }

    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 70%;
        z-index: 999;
    }

    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #1e293b;
        text-align: center;
        letter-spacing: -1px;
    }
    
    .subtitle {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ System Control")
    if st.button("🗑️ Clear Session"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("Server Status: **Active**")
    st.caption("Engine: **Scikit-Learn (Local)**")

st.markdown('<div class="main-title">Future Support AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enterprise Grade Automated Assistance</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ai_reply = get_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})