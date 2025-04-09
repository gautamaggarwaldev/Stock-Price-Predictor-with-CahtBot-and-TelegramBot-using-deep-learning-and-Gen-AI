import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def send_stock_alert(STOCK_NAME):
    STOCK_ENDPOINT = "https://www.alphavantage.co/query"
    NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

    # Get API keys from environment variables
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

    parameters = {
        "function": "TIME_SERIES_DAILY",
        "symbol": STOCK_NAME,
        "apikey": api_key
    }

    try:
        response = requests.get(STOCK_ENDPOINT, params=parameters)
        response.raise_for_status()
        data = response.json()

        time_series = data.get("Time Series (Daily)")
        if not time_series:
            telegram_bot_sendtext(f"⚠️ Could not retrieve stock data for {STOCK_NAME}. Please check the symbol.")
            return

        latest_date = sorted(time_series.keys())[0]
        stock_info = time_series[latest_date]

        message = (
            f"📊 *{STOCK_NAME} Stock Info - {latest_date}*\n"
            f"Open: {stock_info['1. open']}\n"
            f"High: {stock_info['2. high']}\n"
            f"Low: {stock_info['3. low']}\n"
            f"Close: {stock_info['4. close']}\n"
            f"Volume: {stock_info['5. volume']}\n"
        )

        telegram_bot_sendtext(message)
        print(message)


        # Optionally send related news
        params_news = {
            "q": STOCK_NAME,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt"
        }

        response_news = requests.get(NEWS_ENDPOINT, params=params_news)
        response_news.raise_for_status()
        articles = response_news.json().get("articles", [])[:3]

        if articles:
            for article in articles:
                news_message = (
                    f"📰 *{article['title']}*\n"
                    f"{article['description'] or 'No description available.'}\n"
                    f"[Read more]({article['url']})"
                )
                telegram_bot_sendtext(news_message)
        else:
            telegram_bot_sendtext("📭 No recent news articles found.")

    except Exception as e:
        telegram_bot_sendtext(f"❌ Error retrieving stock info for {STOCK_NAME}: {e}")

def telegram_bot_sendtext(bot_message):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    bot_chatID = os.getenv("TELEGRAM_CHAT_ID")
    send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    response = requests.post(send_text, data={
        'chat_id': bot_chatID,
        'text': bot_message,
        'parse_mode': 'Markdown'
    })
    print("Telegram API response:", response.json())  # Debugging line
    return response.json()

if __name__ == "__main__":
    stock_symbol = input("Enter a stock symbol (e.g., TSLA): ").strip().upper()
    send_stock_alert(stock_symbol)
    print(f"Stock alert sent for {stock_symbol}!")