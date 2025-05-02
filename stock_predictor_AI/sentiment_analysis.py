import yfinance as yf
from textblob import TextBlob
import requests
import datetime
import pandas as pd

# Set your NewsData.io or other API key here
NEWS_API_KEY = 'e7deceffb50a4a38b9412d14d4aebd81'

def fetch_news(stock_name):
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_KEY}&q={stock_name}&language=en"
    response = requests.get(url)
    articles = response.json().get("results", [])
    return [article["title"] for article in articles if "title" in article]

def get_average_sentiment(titles):
    scores = []
    for title in titles:
        blob = TextBlob(title)
        scores.append(blob.sentiment.polarity)
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score

if __name__ == "__main__":
    stock_name = "AAPL"  # Example
    headlines = fetch_news(stock_name)
    avg_sentiment = get_average_sentiment(headlines)
    print(f"Average sentiment score for {stock_name}: {avg_sentiment}")

    # Save to CSV for use in LSTM
    today = datetime.date.today()
    df = pd.DataFrame({'Date': [today], 'Sentiment': [avg_sentiment]})
    df.to_csv("sentiment.csv", index=False)
