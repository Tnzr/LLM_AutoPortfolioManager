import openai
from urllib.request import urlopen, Request
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import streamlit as st
import json
import os
from time import sleep  # For simulating progress

# Set up OpenAI API Key
openai.api_key = os.getenv("APIKEY")

# Set up the root URL for scraping
root = "https://finviz.com/quote.ashx?t="

# Streamlit UI
st.title("Sentiment-Based Stock Portfolio Allocator")
st.sidebar.header("Stock Selection")

# User inputs
tickers = st.sidebar.multiselect("Select stocks to analyze:", ["AMZN", "AAPL", "MSFT", "AMD", "NVDA"],
                                 default=["AMZN", "AAPL"])
max_headlines = st.sidebar.slider("Maximum number of news headlines per stock:", 1, 20, 10)
process_button = st.sidebar.button("Run Analysis")


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
        # Update the progress bar
        st.session_state["progress_bar"].progress((i + 1) / len(headlines))
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Analyze the sentiment of this headline: '{headline}'"}],
                temperature=0.2,
                max_tokens=60
            )
            sentiment = response.choices[0].message['content'].strip()

            # Assign a simple sentiment score
            if "positive" in sentiment.lower():
                score = 1
            elif "negative" in sentiment.lower():
                score = -1
            else:
                score = 0
            sentiment_scores.append(score)
        except Exception as e:
            sentiment_scores.append(0)  # Neutral score if there's an error
    return sentiment_scores


# Portfolio allocation based on sentiment
def allocate_portfolio(ticker_sentiments):
    portfolio_weights = {}
    for ticker, sentiments in ticker_sentiments.items():
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        portfolio_weights[ticker] = max(avg_sentiment, 0)  # Only positive weights

    total_weight = sum(portfolio_weights.values())
    if total_weight > 0:
        for ticker in portfolio_weights:
            portfolio_weights[ticker] /= total_weight
    else:
        portfolio_weights = {ticker: 1 / len(ticker_sentiments) for ticker in
                             ticker_sentiments}  # Equal weighting if all neutral

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

    # Initialize progress bar for sentiment analysis
    st.session_state["progress_bar"] = st.progress(0)

    # Perform sentiment analysis and allocation
    ticker_sentiments = {}
    for ticker in tickers:
        headlines = df[df['ticker'] == ticker]['title'].tolist()
        sentiments = analyze_sentiment(headlines)
        ticker_sentiments[ticker] = sentiments

    portfolio_allocation = allocate_portfolio(ticker_sentiments)

    # Remove the progress bar after processing
    st.session_state["progress_bar"].empty()

    # Display portfolio allocation
    st.write("Recommended Portfolio Allocation Based on Sentiment:")
    st.json(json.dumps(portfolio_allocation, indent=4))

else:
    st.write("Please select at least one stock to analyze and click 'Run Analysis'.")
