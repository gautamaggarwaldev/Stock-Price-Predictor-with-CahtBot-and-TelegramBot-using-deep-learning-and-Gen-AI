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

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="📈 Stock Price Predictor + 🤖 Assistant", layout="wide")

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

# Set matplotlib style to dark
plt.style.use('dark_background')

st.markdown("<h1 style='text-align: center; color: #4a8bfc;'>📈 Stock Price Predictor & 🤖 DoraFinance Assistant</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; margin-bottom: 2rem; color: #a0a0a0;'>Your AI-powered financial analysis platform</div>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 2rem 0; height: 2px; background-color: #333; border: none;'>", unsafe_allow_html=True)

# Initialize session state variables
if "voice_input" not in st.session_state:
    st.session_state["voice_input"] = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- Stock Selection --------------------
col1, col2 = st.columns([3, 1])
with col1:
    stock = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA)", value="TSLA").upper()

# -------------------- Telegram Alert --------------------
with col2:
    if st.button("📱 Get Stock Alerts on Telegram"):
        send_stock_alert(stock)
        st.success("Sent Telegram alert based on recent stock activity.")

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
    n_days = st.slider("Number of days to predict", 1, 100, 10, key='n_days_slider')  # Added key
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

# Initialize session state for chat
if "text_question" not in st.session_state:
    st.session_state.text_question = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input and Send button
col1, col2 = st.columns([5, 1])
with col1:
    text_input = st.text_input(
        "Your Question",
        value=st.session_state.text_question,
        key="text_input_field",
        placeholder="Ask me about stocks, market trends, or technical analysis..."
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Send 📤", key="send_button"):
        if text_input.strip():
            try:
                # Request to local chatbot backend
                response = requests.post(
                    "http://localhost:5000/ask",
                    json={"question": text_input}
                )
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer returned.")
                    st.session_state.chat_history.append(("You", text_input))
                    st.session_state.chat_history.append(("DoraFinance", answer))
                else:
                    st.error(f"Chatbot error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                # Fallback if server isn't running
                st.session_state.chat_history.append(("You", text_input))
                # Get last 30 days data safely
                last_idx = min(30, len(data)-1)
                current = float(data['Close'].iloc[-1])
                past = float(data['Close'].iloc[-last_idx]) if last_idx > 0 else current
                trend = "upward" if current > past else "downward"
                
                fallback_response = (
                    f"Based on the data shown in the charts, {stock} stock has shown "
                    f"{trend} trend based on recent performance."
                )
                st.session_state.chat_history.append(("DoraFinance", fallback_response))
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

            # Clear input after sending
            st.session_state.text_question = ""
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
else:
    st.info("Start a conversation with DoraFinance Assistant by typing your question above!")


# Footer
st.markdown("<div style='text-align: center; margin-top: 2rem; color: #a0a0a0;'>© 2025 DoraFinance - AI-Powered Stock Analysis Made By Piyush & GG 💵💵</div>", unsafe_allow_html=True)