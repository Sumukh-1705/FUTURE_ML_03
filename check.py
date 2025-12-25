import pandas as pd
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- 1. SETUP ---
st.set_page_config(page_title="Future Support AI", page_icon="🤖", layout="wide")

# --- 2. LOCAL BRAIN ---
@st.cache_resource
def build_brain():
    try:
        df = pd.read_csv('customer_support.csv')
    except:
        # Fallback data
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

# --- 3. UI CSS (THE NUCLEAR FIX) ---
st.markdown("""
    <style>
    /* 1. BACKGROUND */
    .stApp {
        background-color: #f7f9fc;
    }

    /* 2. NUCLEAR REMOVAL OF ICONS */
    /* Target the specific class for avatars */
    [data-testid="stChatMessageAvatar"] {
        display: none !important;
        visibility: hidden !important;
        width: 0px !important;
    }
    
    /* Target the first child of the message container (Fallback) */
    div[data-testid="stChatMessage"] > div:first-child {
        display: none !important;
        width: 0px !important;
    }

    /* 3. USER BUBBLE (Blue, Right Aligned) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        flex-direction: row-reverse;
        text-align: right;
    }
    
    [data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background-color: #2563EB;
        color: white;
        border-radius: 20px 20px 5px 20px;
        padding: 10px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: inline-block;
        max-width: 80%;
    }

    /* 4. BOT BUBBLE (White, Left Aligned) */
    [data-testid="stChatMessage"]:nth-child(even) {
        flex-direction: row;
        text-align: left;
    }

    [data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        color: #1f2937;
        border: 1px solid #e5e7eb;
        border-radius: 20px 20px 20px 5px;
        padding: 10px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        display: inline-block;
        max-width: 80%;
    }

    /* 5. REMOVE DEFAULT PADDING */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 10px 0px;
    }

    /* 6. INPUT BAR (Fixed Bottom) */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        width: 70%;
        max-width: 800px;
        z-index: 1000;
    }

    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-top: 20px;
    }
    .subtitle {
        font-family: 'Arial', sans-serif;
        font-size: 1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.write("**System:** Local ML")
    st.write("**Status:** Active 🟢")

# --- 5. MAIN APP ---
st.markdown('<div class="main-title">Future Support AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Intelligent Automated Assistance</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("How can we help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ai_reply = get_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})