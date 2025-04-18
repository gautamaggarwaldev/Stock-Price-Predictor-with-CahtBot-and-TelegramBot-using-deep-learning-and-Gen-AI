import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from keras.models import load_model #type: ignore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from stock_alert import send_stock_alert
import mplfinance as mpf

# Add Gemini API imports
import google.generativeai as genai
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import logging

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import sqlite3
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- User Authentication Setup --------------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  name TEXT, 
                  mobile TEXT, 
                  password TEXT, 
                  telegram_chat_id TEXT)''')
    conn.commit()
    conn.close()

def add_user(username, name, mobile, password, telegram_chat_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
              (username, name, mobile, hashed_password, telegram_chat_id))
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", 
              (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result is not None

def get_user_chat_id(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT telegram_chat_id FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_user_name(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT name FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else "User"

def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["chat_history"] = []
    st.rerun()

# Initialize database
init_db()

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="📈 Stock Price Predictor + 🤖 Assistant", layout="wide")

# -------------------- Initialize Gemini AI Model --------------------
# This function will be used to initialize the model only once when needed
@st.cache_resource
def initialize_gemini(max_retries=3, retry_delay=5):
    """Initialize the Gemini model with retry mechanism"""
    retries = 0
    while retries < max_retries:
        try:
            # Get API key from environment variable or Streamlit secrets
            api_key = os.getenv("GEMINI_API_KEY")
            
            # If not in environment, try to get from Streamlit secrets
            if not api_key and hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
                
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment or secrets")
                return None, None
                
            genai.configure(api_key=api_key)
            
            # Configure safety settings
            safety_settings = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE}
            ]
            
            # Initialize model
            model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini model initialized successfully")
            return model, safety_settings
            
        except Exception as e:
            retries += 1
            logger.error(f"Attempt {retries} failed to initialize Gemini: {str(e)}")
            if retries < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    logger.error(f"Failed to initialize Gemini after {max_retries} attempts")
    return None, None

# DoraFinance system prompt
SYSTEM_PROMPT = """You are DoraFinance, an expert AI stock market advisor with 50+ years of experience in financial markets.
Your role is to provide insightful, accurate, and responsible guidance on stock market questions.

Guidelines:
1. Provide clear, concise explanations suitable for both beginners and experienced investors
2. When discussing specific stocks or investment strategies, ALWAYS include a disclaimer that this is not financial advice
3. Include relevant metrics when analyzing companies (P/E ratio, market cap, revenue growth, debt-to-equity, etc.)
4. Explain technical terms in accessible language, adding brief definitions for specialized terminology
5. For price predictions, discuss multiple factors that could influence the stock's movement rather than giving specific price targets
6. Consistently emphasize the importance of diversification, risk management, and investing with a long-term horizon
7. When information is limited or outdated, acknowledge limitations and avoid speculation
8. Suggest reliable sources for further research when appropriate
9. Consider macroeconomic factors and industry trends in your analysis
10. Be mindful of cognitive biases that affect investment decisions
11. Also explain the potential risks and rewards of any investment strategy you discuss
12. Avoid making absolute statements; instead, use phrases like "may", "could", or "might" to indicate uncertainty
13. If a question is outside your expertise, politely decline to answer and suggest consulting a financial advisor
15. Explain market technical terms and concepts in a way that is easy to understand for someone without a finance background
16. Avoid overly technical jargon unless necessary, and provide definitions when you do use it
17. Be aware of the potential for market manipulation and the ethical implications of your responses
18. Avoid discussing or promoting any illegal activities related to the stock market
19. Be cautious about discussing specific stocks or investment strategies that could be considered insider trading or market manipulation
20. If user ask any financial question or query please provide a solution not telling any illogical advice and knowledge.
21. At last also add a some random useful tips about share market and stock market investment in every answer.

Remember: Your guidance could influence financial decisions. Be thorough, balanced, and responsible in your responses.
"""

# -------------------- Custom CSS Styling --------------------
st.markdown("""
<style>
    /* Main background and text colors */
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    
    /* Enhanced Chat Section */
    .chat-container {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #333;
        max-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #2a2a2a;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin-bottom: 12px;
        max-width: 80%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        border: 1px solid #444;
    }
    .bot-message {
        background-color: #252525;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin-bottom: 12px;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        border: 1px solid #444;
    }
    .message-sender {
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .user-sender {
        color: #a0a0a0;
    }
    .bot-sender {
        color: #4a8bfc;
    }
    
    /* Enhanced Future Prediction Section */
    .prediction-graph-container {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #333;
    }
    .prediction-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .stat-card {
        background-color: #252525;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        border: 1px solid #444;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4a8bfc;
        margin-bottom: 0.25rem;
    }
    .stat-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .trend-up {
        color: #4CAF50;
    }
    .trend-down {
        color: #F44336;
    }
    
    /* Stock Metric Cards */
    .metric-card {
        background-color: #252525;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        border: 1px solid #444;
        height: 100%;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4a8bfc;
        margin-bottom: 0.25rem;
    }
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Stock Cards */
    .stock-card {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #333;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #252525 !important;
        color: #e0e0e0 !important;
        border: 1px solid #444 !important;
    }
    
    /* Slider */
    .stSlider>div>div>div>div {
        background-color: #4a8bfc !important;
    }
    
    /* Button */
    .stButton>button {
        background-color: #4a8bfc !important;
        color: white !important;
        border: none !important;
    }
    
    /* Expander */
    .stExpander {
        background-color: #1e1e1e !important;
        border: 1px solid #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Authentication UI --------------------
def show_auth_page():
    st.title("🔐 DoraFinance Authentication")
    
    auth_tab, help_tab = st.tabs(["Login/Signup", "How to get Telegram Chat ID"])
    
    with auth_tab:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    if verify_user(username, password):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        with tab2:
            with st.form("signup_form"):
                username = st.text_input("Username", key="signup_username")
                name = st.text_input("Full Name", key="signup_name")
                mobile = st.text_input("Mobile Number", key="signup_mobile")
                password = st.text_input("Password", type="password", key="signup_password")
                telegram_chat_id = st.text_input("Telegram Chat ID", key="signup_chat_id", 
                                               help="See 'How to get Telegram Chat ID' tab for instructions")
                submitted = st.form_submit_button("Sign Up")
                if submitted:
                    if username and name and mobile and password and telegram_chat_id:
                        try:
                            add_user(username, name, mobile, password, telegram_chat_id)
                            st.success("Account created successfully! Please login.")
                        except sqlite3.IntegrityError:
                            st.error("Username already exists")
                    else:
                        st.error("Please fill all fields")
    
    with help_tab:
        st.markdown("""
        ### How to Get Your Telegram Chat ID
        
        1. Start a conversation with the Telegram bot you want to receive alerts
        2. Click on this URL "https://t.me/dorafinancebot" send /start message
        3. Open Telegram click on search icon and search for "RawDataBot" and send /start message
        4. The bot will reply with your chat ID.
        5. Copy the chat ID and paste it in the "Telegram Chat ID" field during signup.
        """)

# -------------------- Main App Access Control --------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    show_auth_page()
    st.stop()

# Set matplotlib style to dark
user_name = get_user_name(st.session_state["username"])
plt.style.use('dark_background')
header_col1, header_col2, header_col3 = st.columns([3, 5, 2])
with header_col1:
    st.markdown(f"<div style='color: #4a8bfc; font-size: 14px;'>👤 Welcome, {user_name}</div>", unsafe_allow_html=True)
with header_col3:
    if st.button("🚪 Logout", key="logout_button"):
        logout()
st.markdown("<h1 style='text-align: center; color: #4a8bfc;'>📈 Stock Price Predictor & 🤖 DoraFinance Assistant</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; margin-bottom: 2rem; color: #a0a0a0;'>Your AI-powered financial analysis platform</div>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 2rem 0; height: 2px; background-color: #333; border: none;'>", unsafe_allow_html=True)

# Initialize session state variables
if "voice_input" not in st.session_state:
    st.session_state["voice_input"] = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_initialized" not in st.session_state:
    st.session_state.model_initialized = False
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
if "safety_settings" not in st.session_state:
    st.session_state.safety_settings = None

# -------------------- Stock Selection --------------------
col1, col2 = st.columns([3, 1])
with col1:
    stock = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA)", value="TSLA").upper()

# -------------------- Telegram Alert --------------------
with col2:
    if st.button("📱 Get Stock Alerts on Telegram"):
        chat_id = get_user_chat_id(st.session_state["username"])
        if chat_id:
            send_stock_alert(stock, chat_id)
            st.success("Alert sent to your Telegram account!")
        else:
            st.error("No Telegram Chat ID found in your account. Please update your profile.")

model_file = "Latest_bit_coin_model.keras"  # Static model (update if needed)
try:
    model = load_model(model_file)
except Exception as e:
    st.error(f"Model not found or error loading model: {e}")
    st.stop()

# -------------------- Fetch Stock Data --------------------
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
data = yf.download(stock, start=start, end=end)

if data.empty:
    st.error(f"No data found for symbol '{stock}'.")
    st.stop()

# Reset index to get 'Date' column for plotting
data.reset_index(inplace=True)

# -------------------- Key Metrics --------------------
st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
st.subheader(f"📊 {stock} Key Metrics")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

# Convert pandas Series values to Python float/int for proper formatting
current_price = float(data['Close'].iloc[-1])
prev_price = float(data['Close'].iloc[-2])
volume = float(data['Volume'].iloc[-1])
avg_price = float(data['Close'].mean())

with metric_col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>${current_price:.2f}</div>
        <div class='metric-label'>Current Price</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col2:
    change = ((current_price - prev_price) / prev_price) * 100
    color = "#4CAF50" if change >= 0 else "#F44336"
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color: {color};'>{change:.2f}%</div>
        <div class='metric-label'>24h Change</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>${volume:,.0f}</div>
        <div class='metric-label'>Volume</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>${avg_price:.2f}</div>
        <div class='metric-label'>Avg Price (10Y)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Display Raw Data --------------------
with st.expander("📜 View Latest 100 Days of Stock Data"):
    st.dataframe(data.tail(100), use_container_width=True)


# -------------------- Candlestick Chart --------------------
st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
st.subheader("🕯 Candlestick Chart")

# ✅ Fix for MultiIndex columns (e.g., ('Open', 'TSLA'))
if isinstance(data.columns, pd.MultiIndex):
    # Flatten by selecting level 0 (e.g., 'Open', 'High', etc.)
    data.columns = data.columns.get_level_values(0)

# ✅ Ensure 'Date' column is datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
candlestick_data = data.set_index('Date')[['Open', 'High', 'Low', 'Close', 'Volume']]

# 🎨 Custom style
mc = mpf.make_marketcolors(up='#4CAF50', down='#F44336', inherit=True)
s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)

# 📈 Plot candlestick chart
fig_candle, _ = mpf.plot(
    candlestick_data.tail(100),  # Last 100 days
    type='candle',
    style=s,
    volume=True,
    title=f'{stock} Candlestick Chart (Last 100 Days)',
    returnfig=True,
    figsize=(15, 5),
)

st.pyplot(fig_candle)
st.markdown("</div>", unsafe_allow_html=True)


# -------------------- Close Price Line Chart --------------------
st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
st.subheader("📈 Close Price Over Time")
fig_line = plt.figure(figsize=(15, 5))
plt.plot(data['Date'], data['Close'], label='Close Price', color='#4a8bfc', linewidth=2)
plt.xlabel("Date", color='#e0e0e0')
plt.ylabel("Price ($)", color='#e0e0e0')
plt.title(f"{stock} Close Price", color='#e0e0e0')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig_line)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Data Preparation --------------------
splitting_len = int(len(data) * 0.9)
x_test = pd.DataFrame(data[['Close']][splitting_len:])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']].values)

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])
x_data, y_data = np.array(x_data), np.array(y_data)

# -------------------- Predictions --------------------
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data.reshape(-1, 1))

plot_df = pd.DataFrame({
    'Original': inv_y_test.flatten(),
    'Predicted': inv_pre.flatten()
}, index=data['Date'][splitting_len + 100:])

# -------------------- Plot Predictions --------------------
st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
st.subheader("📉 Predicted vs Actual")
fig_pred = plt.figure(figsize=(15, 5))
plt.plot(data['Date'][:splitting_len+100], data['Close'][:splitting_len+100], label="Historical", color="#a0a0a0", linewidth=1.5)
plt.plot(plot_df.index, plot_df['Original'], label="Actual", color="#4CAF50", linewidth=2)
plt.plot(plot_df.index, plot_df['Predicted'], label="Predicted", color="#F44336", linewidth=2, linestyle='--')
plt.title(f"{stock} - Prediction vs Actual", color='#e0e0e0')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig_pred)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Future Forecasting --------------------
def predict_future(n_days, prev):
    future = []
    for _ in range(n_days):
        prev = np.array(prev).reshape(1, 100, 1)
        next_day = model.predict(prev)
        future.append(scaler.inverse_transform(next_day)[0][0])
        prev = np.append(prev[:, 1:, :], next_day.reshape(1, 1, 1), axis=1)
    return future

st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
st.subheader("🔮 Predict Future Prices")
col1, col2 = st.columns([3, 1])
with col1:
    n_days = st.slider("Number of days to predict", 1, 100, 10, key='n_days_slider')
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    
# Move the prediction logic outside the button to update automatically
last_100 = scaler.fit_transform(data[['Close']].tail(100).values.reshape(-1, 1))
future_prices = predict_future(n_days, last_100.tolist())

# Create the prediction graph with larger size
st.markdown("<div class='prediction-graph-container'>", unsafe_allow_html=True)
future_fig = plt.figure(figsize=(18, 8))  # Increased figure size
plt.plot(future_prices, marker='o', linestyle='--', color='#4a8bfc', linewidth=2.5, markersize=8)
plt.title(f"{stock} Price Forecast for Next {n_days} Days", fontsize=16, pad=20, color='#e0e0e0')
plt.xlabel("Days from Today", color='#e0e0e0')
plt.ylabel("Price ($)", color='#e0e0e0')
plt.grid(True, alpha=0.2)
plt.tight_layout()
st.pyplot(future_fig)
st.markdown("</div>", unsafe_allow_html=True)

# Calculate statistics
avg_future = sum(future_prices) / len(future_prices)
max_future = max(future_prices)
min_future = min(future_prices)
trend = future_prices[-1] > future_prices[0]
trend_class = "trend-up" if trend else "trend-down"
trend_text = "Upward ↗" if trend else "Downward ↘"

# Display forecast statistics with enhanced styling
st.markdown("<div class='prediction-stats'>", unsafe_allow_html=True)
st.markdown(f"""
<div class='stat-card'>
    <div class='stat-value'>${avg_future:.2f}</div>
    <div class='stat-label'>Average Forecast</div>
</div>
<div class='stat-card'>
    <div class='stat-value'>${max_future:.2f}</div>
    <div class='stat-label'>Maximum Price</div>
</div>
<div class='stat-card'>
    <div class='stat-value'>${min_future:.2f}</div>
    <div class='stat-label'>Minimum Price</div>
</div>
<div class='stat-card'>
    <div class='stat-value {trend_class}'>{trend_text}</div>
    <div class='stat-label'>Price Trend</div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Voice Chat Assistant --------------------
st.markdown("<hr style='margin: 2rem 0; height: 2px; background-color: #333; border: none;'>", unsafe_allow_html=True)
st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
st.subheader("🤖 DoraFinance Chat Assistant")
st.caption("💬 Ask me anything about stocks, trading, or financial analysis")

# Function to get Gemini response
def get_gemini_response(question, stock_info=None):
    """Get response from the Gemini model using the financial advisor prompt"""
    try:
        # Ensure the model is initialized
        if not st.session_state.model_initialized:
            model, safety_settings = initialize_gemini()
            if model is None:
                return "I'm having trouble connecting to my knowledge base. Please try again later."
            st.session_state.gemini_model = model
            st.session_state.safety_settings = safety_settings
            st.session_state.model_initialized = True
        
        # Add stock context to the prompt if available
        stock_context = ""
        if stock_info:
            stock_context = f"""
            Current analysis is for: {stock_info['symbol']}
            Current Price: ${stock_info['current_price']:.2f}
            24hr Change: {stock_info['change']:.2f}%
            Price Trend: {stock_info['trend']}
            """
        
        current_date = datetime.now().strftime("%B %d, %Y")
        full_prompt = SYSTEM_PROMPT.replace("{current_date}", current_date)
        
        if stock_context:
            full_prompt += f"\n\nCurrent Stock Context:\n{stock_context}"
            
        full_prompt += f"\n\nQuestion: {question}"
        
        response = st.session_state.gemini_model.generate_content(
            full_prompt,
            safety_settings=st.session_state.safety_settings
        )
        
        if not hasattr(response, 'text'):
            logger.error("Response from Gemini has no 'text' attribute")
            return "I couldn't generate a proper response. Please try asking in a different way."
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error getting Gemini response: {str(e)}")
        return f"I encountered an error while processing your request. Please try again or ask a different question."

# Input and Send button
col1, col2 = st.columns([5, 1])
with col1:
    text_input = st.text_input(
        "Your Question",
        key="text_input_field",
        placeholder="Ask me about stocks, market trends, or technical analysis..."
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Send 📤", key="send_button"):
        if text_input.strip():
            # Add user message to chat history
            st.session_state.chat_history.append(("You", text_input))
            
            # Create stock info dict to provide context
            stock_info = {
                "symbol": stock,
                "current_price": current_price,
                "change": change,
                "trend": "upward" if change > 0 else "downward"
            }
            
            # Show a spinner while getting the response
            with st.spinner("DoraFinance is thinking..."):
                # Get response from Gemini
                response = get_gemini_response(text_input, stock_info)
                st.session_state.chat_history.append(("DoraFinance", response))
            
            # Rerun to update the chat display
            st.rerun()
        else:
            st.warning("Please enter a question.")

# Display chat history
if st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"""
            <div class='user-message'>
                <div class='message-sender user-sender'>You</div>
                {message}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='bot-message'>
                <div class='message-sender bot-sender'>DoraFinance</div>
                {message}
            </div>
            """, unsafe_allow_html=True)
    # st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Start a conversation with DoraFinance Assistant by typing your question above!")

# st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align: center; margin-top: 2rem; color: #a0a0a0;'>© 2025 DoraFinance - AI-Powered Stock Analysis Made By Piyush & GG 💵💵</div>", unsafe_allow_html=True)