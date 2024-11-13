from newsapi import NewsApiClient
import openai
import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from time import sleep
from tqdm import tqdm
import streamlit as st
import json
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request

# Load environment variables
load_dotenv()

# Initialize API keys
newsapi = NewsApiClient(api_key=os.getenv("IEEE_NEWS_API"))
openai.api_key = os.getenv("APIKEY")
root = "https://finviz.com/quote.ashx?t="

# Streamlit UI
st.title("Sentiment-Based Stock Portfolio Allocator and Backtest")
st.sidebar.header("Stock Selection")

# User input for tickers
tickers_input = st.sidebar.text_input("Enter stock symbols (comma-separated, e.g., 'AAPL, MSFT'):", "NVDA, MSFT")
start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 20))
end_date = st.sidebar.date_input("End Date", datetime(2024, 11, 10))
max_headlines = st.sidebar.slider("Maximum number of news headlines per stock:", 1, 20, 10)
process_button = st.sidebar.button("Run Analysis")

# Parse tickers from user input
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Function to scrape news headlines from Finviz
def scrape_news(tickers, max_headlines):
    news_tables = {}
    for ticker in tickers:
        url = root + ticker
        req = Request(url=url, headers={"user-agent": 'my-app'})
        response = urlopen(req)
        html = BeautifulSoup(response, "html.parser")
        news_table = html.find(id="news-table")
        news_tables[ticker] = news_table
    return news_tables

# Parse the scraped news data with a limit on the number of headlines
def parse_news(news_tables, max_headlines):
    parsed_data = []
    for ticker, news_table in news_tables.items():
        if news_table:
            count = 0
            for row in news_table.findAll('tr'):
                if row.a:
                    title = row.a.text
                    date_data = row.td.text.split(' ')
                    if len(date_data) == 1:  # Only time is available
                        time = date_data[0]
                        date = "Today"
                    else:
                        date = date_data[0]
                        time = date_data[1]
                    parsed_data.append([ticker, date, time, title])
                    count += 1
                    if count >= max_headlines:
                        break
    return parsed_data

# Function to analyze sentiment using OpenAI
def analyze_sentiment(headlines):
    sentiment_scores = []
    for i, headline in enumerate(headlines):
        st.session_state["progress_bar"].progress((i + 1) / len(headlines))
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Rate the sentiment of this headline on a scale from -1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive: '{headline}'"
                }],
                temperature=0.2,
                max_tokens=60
            )
            sentiment = response.choices[0].message['content'].strip()
            try:
                score = float(sentiment)
                score = max(-1, min(1, score))
            except ValueError:
                score = 0
            sentiment_scores.append(score)
        except Exception as e:
            sentiment_scores.append(0)
    return sentiment_scores

# Portfolio allocation based on sentiment
def allocate_portfolio(ticker_sentiments):
    positive_weights = {ticker: max(np.mean(sentiments), 0) for ticker, sentiments in ticker_sentiments.items()}
    total_weight = sum(positive_weights.values())
    if total_weight > 0:
        return {ticker: weight / total_weight for ticker, weight in positive_weights.items()}
    else:
        return {ticker: 1 / len(ticker_sentiments) for ticker in ticker_sentiments}

# Fetch price data for backtesting
def fetch_price_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

# Run backtesting with progress bar
def backtest_sentiment_strategy(tickers, start_date, end_date, ticker_sentiments):
    price_data = fetch_price_data(tickers, start_date, end_date)
    daily_returns = price_data.pct_change().dropna()
    portfolio_values = [1]
    allocations = {}

    for date in daily_returns.index:
        date_sentiments = {ticker: np.mean(ticker_sentiments.get(ticker, [0])) for ticker in tickers}
        allocations[date] = allocate_portfolio(date_sentiments)
        daily_portfolio_return = sum(
            allocations[date].get(ticker, 0) * daily_returns.loc[date, ticker]
            for ticker in tickers if ticker in daily_returns.columns
        )
        portfolio_values.append(portfolio_values[-1] * (1 + daily_portfolio_return))

    portfolio_index = pd.to_datetime([start_date] + list(daily_returns.index)).normalize()
    portfolio_df = pd.DataFrame(portfolio_values, index=portfolio_index, columns=["Portfolio Value"])

    # Ensure cumulative_return is a scalar value
    cumulative_return = (portfolio_df.iloc[-1]["Portfolio Value"] / portfolio_df.iloc[0]["Portfolio Value"]) - 1
    sharpe_ratio = (daily_returns.mean().mean() / daily_returns.std().mean()) * np.sqrt(252)
    max_drawdown = (portfolio_df / portfolio_df.cummax() - 1).min()[0]

    # Display results
    st.write("Sentiment-Driven Portfolio Performance:")
    st.write(f"Cumulative Return: {cumulative_return:.2%}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    st.write(f"Max Drawdown: {max_drawdown:.2%}")
    st.line_chart(portfolio_df, use_container_width=True)

    return portfolio_df, cumulative_return, sharpe_ratio, max_drawdown

# Run analysis when the button is clicked
if process_button and tickers:
    st.write(f"Fetching news for: {', '.join(tickers)}")
    news_tables = scrape_news(tickers, max_headlines)
    news_data = parse_news(news_tables, max_headlines)
    df = pd.DataFrame(news_data, columns=['ticker', 'date', 'time', 'title'])
    st.write("Scraped News Headlines:")
    st.dataframe(df)

    st.session_state["progress_bar"] = st.progress(0)
    ticker_sentiments = {}
    for ticker in tickers:
        headlines = df[df['ticker'] == ticker]['title'].tolist()
        sentiments = analyze_sentiment(headlines)
        ticker_sentiments[ticker] = sentiments

    portfolio_allocation = allocate_portfolio(ticker_sentiments)
    st.session_state["progress_bar"].empty()

    st.write("Recommended Portfolio Allocation Based on Sentiment:")
    st.json(json.dumps(portfolio_allocation, indent=4))

    st.write("Running Backtest on Sentiment-Based Strategy...")
    portfolio_df, cumulative_return, sharpe_ratio, max_drawdown = backtest_sentiment_strategy(tickers, start_date, end_date, ticker_sentiments)
else:
    st.write("Please enter stock symbols and click 'Run Analysis'.")
