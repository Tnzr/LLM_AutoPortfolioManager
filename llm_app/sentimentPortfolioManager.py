import openai
from urllib.request import urlopen, Request
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import streamlit as st
import json
import os
from dotenv import load_dotenv
load_dotenv()

# Set up OpenAI API Key
openai.api_key = os.getenv("APIKEY")

# Set up the root URL for scraping
root = "https://finviz.com/quote.ashx?t="

# Streamlit UI
st.title("Sentiment-Based Stock Portfolio Allocator")
st.sidebar.header("Stock Selection")

# User input for tickers
tickers_input = st.sidebar.text_input("Enter stock symbols (comma-separated, e.g., 'AAPL, MSFT, TSLA'):", "AMZN, AAPL")
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

# Optimized function to analyze sentiment of all headlines for each ticker at once
def analyze_sentiment(ticker_headlines):
    print("Analyzing sentiment of headlines for each ticker...")
    ticker_sentiments = {}

    for ticker, headlines in ticker_headlines.items():
        try:
            # Join all headlines for the ticker in a single request
            combined_headlines = " | ".join(headlines)
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": f"Rate the sentiment of each headline in the following list on a scale from -1 (very negative) to 1 (very positive). "
                                   f"Separate each score with a space. Here are the headlines: '{combined_headlines}'"
                    }
                ],
                temperature=0.2,
                max_tokens=150
            )

            # Extract scores from the response
            sentiment_text = response.choices[0].message['content'].strip()
            scores = [float(score) for score in sentiment_text.split() if score.replace('.', '', 1).replace('-', '', 1).isdigit()]

            # Ensure the number of scores matches the number of headlines
            if len(scores) == len(headlines):
                ticker_sentiments[ticker] = scores
            else:
                # Fallback to zero if parsing fails
                ticker_sentiments[ticker] = [0] * len(headlines)

        except Exception as e:
            print(f"An error occurred during sentiment analysis for {ticker}: {e}")
            ticker_sentiments[ticker] = [0] * len(headlines)  # Neutral score if there's an error

    print("Sentiment analysis completed.\n")
    return ticker_sentiments

# Portfolio allocation based on sentiment
def allocate_portfolio(ticker_sentiments):
    portfolio_weights = {}

    # Calculate the average sentiment for each stock
    for ticker, sentiments in ticker_sentiments.items():
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        portfolio_weights[ticker] = avg_sentiment  # Allow both positive and negative weights

    # If all sentiments are zero, set equal weights
    total_weight = sum(abs(weight) for weight in portfolio_weights.values())

    if total_weight > 0:
        # Normalize weights by the sum of absolute values
        portfolio_weights = {ticker: weight / total_weight for ticker, weight in portfolio_weights.items()}
    else:
        # Equal weighting if all are neutral
        portfolio_weights = {ticker: 1 / len(ticker_sentiments) for ticker in ticker_sentiments}

    # Convert weights to positive values if needed
    portfolio_weights = {ticker: max(weight, 0) for ticker, weight in portfolio_weights.items()}
    return portfolio_weights

# Run analysis when the button is clicked
if process_button and tickers:
    st.write(f"Fetching news for: {', '.join(tickers)}")

    # Scrape and parse news
    news_tables = scrape_news(tickers, max_headlines)
    news_data = parse_news(news_tables, max_headlines)
    df = pd.DataFrame(news_data, columns=['ticker', 'date', 'time', 'title'])
    st.write("Scraped News Headlines:")
    st.dataframe(df)

    # Organize headlines by ticker
    ticker_headlines = {ticker: df[df['ticker'] == ticker]['title'].tolist() for ticker in tickers}

    # Perform sentiment analysis
    ticker_sentiments = analyze_sentiment(ticker_headlines)

    # Calculate portfolio allocation
    portfolio_allocation = allocate_portfolio(ticker_sentiments)

    # Display portfolio allocation
    st.write("Recommended Portfolio Allocation Based on Sentiment:")
    st.json(json.dumps(portfolio_allocation, indent=4))
