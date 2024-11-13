from newsapi import NewsApiClient
import openai
import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from time import sleep
from tqdm import tqdm  # Import tqdm for progress bars

# Load environment variables
load_dotenv()

# Initialize NewsAPI and OpenAI clients
newsapi = NewsApiClient(api_key=os.getenv("IEEE_NEWS_API"))
openai.api_key = os.getenv("APIKEY")

# Parameters
tickers = ["NVDA", "MSFT"]
start_date = datetime(2024, 10, 20)
end_date = datetime(2024, 11, 10)

# Fetch historical headlines for a ticker with progress bar
def fetch_historical_news(ticker, start_date, end_date):
    all_headlines = []
    date = start_date

    with tqdm(total=(end_date - start_date).days, desc=f"Fetching news for {ticker}") as pbar:
        while date <= end_date:
            articles = newsapi.get_everything(
                q=ticker,
                from_param=date.strftime('%Y-%m-%d'),
                to=(date + timedelta(days=1)).strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=100
            )

            for article in articles['articles']:
                headline = article['title']
                published_date = article['publishedAt'][:10]
                all_headlines.append({'ticker': ticker, 'date': published_date, 'headline': headline})

            date += timedelta(days=1)
            sleep(1)  # To avoid hitting API rate limits
            pbar.update(1)  # Update progress bar

    return pd.DataFrame(all_headlines)

# Analyze sentiment of headlines for a given date
def analyze_sentiment(headlines):
    try:
        combined_headlines = " | ".join(headlines)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Analyze the sentiment of each headline in the following list on a scale from -1 (very negative) to 1 (very positive): '{combined_headlines}'"}
            ],
            temperature=0.2,
            max_tokens=150
        )
        sentiment_text = response.choices[0].message['content'].strip()
        scores = [float(score) for score in sentiment_text.split() if score.replace('.', '', 1).replace('-', '', 1).isdigit()]
        return np.mean(scores) if scores else 0
    except Exception as e:
        print(f"An error occurred during sentiment analysis: {e}")
        return 0  # Default to neutral sentiment

# Fetch price data for backtesting
def fetch_price_data(tickers, start_date, end_date):
    price_data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    return price_data

# Run backtesting with progress bar
# Run backtesting with progress bar
def backtest_sentiment_strategy(tickers, start_date, end_date):
    price_data = fetch_price_data(tickers, start_date, end_date)
    daily_returns = price_data.pct_change().dropna()

    # Prepare to store sentiment-driven allocations and portfolio values
    portfolio_values = [1]  # Starting with an initial portfolio value of 1
    allocations = {}

    # Fetch and analyze headlines for each ticker on each day
    with tqdm(total=len(daily_returns.index), desc="Running Backtest") as pbar:
        for date in daily_returns.index:
            ticker_sentiments = {}

            for ticker in tickers:
                # Fetch and analyze sentiment for ticker on the given date
                headlines_df = fetch_historical_news(ticker, date, date)
                headlines = headlines_df['headline'].tolist()
                ticker_sentiments[ticker] = analyze_sentiment(headlines) if headlines else 0

            # Calculate daily allocations based on sentiment scores
            allocations[date] = allocate_portfolio(ticker_sentiments)

            # Calculate daily portfolio return
            daily_portfolio_return = sum(
                allocations[date].get(ticker, 0) * daily_returns.loc[date, ticker]
                for ticker in tickers if ticker in daily_returns.columns
            )
            portfolio_values.append(portfolio_values[-1] * (1 + daily_portfolio_return))

            pbar.update(1)  # Update progress bar

    # Convert start_date and daily_returns.index to timezone-naive
    start_date = pd.to_datetime(start_date).tz_localize(None)
    daily_returns.index = pd.to_datetime(daily_returns.index).tz_localize(None)

    # Create a timezone-naive portfolio index
    portfolio_index = pd.to_datetime([start_date] + list(daily_returns.index))

    # Normalize the dates
    portfolio_index = portfolio_index.normalize()

    # Create the portfolio DataFrame
    portfolio_df = pd.DataFrame(portfolio_values, index=portfolio_index, columns=["Portfolio Value"])

    # Calculate performance metrics
    cumulative_return = portfolio_df.iloc[-1] / portfolio_df.iloc[0] - 1
    sharpe_ratio = (daily_returns.mean().mean() / daily_returns.std().mean()) * np.sqrt(252)
    max_drawdown = (portfolio_df / portfolio_df.cummax() - 1).min()[0]

    print("Sentiment-Driven Portfolio Performance:")
    print("Cumulative Return:", cumulative_return)
    print("Sharpe Ratio:", sharpe_ratio)
    print("Max Drawdown:", max_drawdown)

    portfolio_df.plot(title="Portfolio Value Over Time")

    return portfolio_df, cumulative_return, sharpe_ratio, max_drawdown

# Portfolio allocation based on sentiment
def allocate_portfolio(ticker_sentiments):
    positive_weights = {ticker: max(sentiment, 0) for ticker, sentiment in ticker_sentiments.items()}
    total_weight = sum(positive_weights.values())
    if total_weight > 0:
        return {ticker: weight / total_weight for ticker, weight in positive_weights.items()}
    else:
        return {ticker: 1 / len(ticker_sentiments) for ticker in ticker_sentiments}

# Run backtest
portfolio_df, cumulative_return, sharpe_ratio, max_drawdown = backtest_sentiment_strategy(tickers, start_date, end_date)

print(portfolio_df, cumulative_return, sharpe_ratio, max_drawdown)