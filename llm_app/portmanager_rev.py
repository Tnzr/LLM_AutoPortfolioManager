import openai
import pandas as pd
import numpy as np
import yfinance as yf
import os
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI API Key
openai.api_key = os.getenv("APIKEY")

# List of tickers and date range for backtesting
tickers = ["AMZN", "AAPL"]
start_date = "2023-01-01"
end_date = "2023-12-31"

# Function to fetch historical price data
def fetch_price_data(tickers, start_date, end_date):
    price_data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    return price_data

# Function to calculate daily returns
def calculate_daily_returns(price_data):
    return price_data.pct_change().dropna()

# Function to perform sentiment analysis
def analyze_sentiment(headlines):
    try:
        # Join headlines into a single prompt
        combined_headlines = " | ".join(headlines)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Analyze the sentiment of each headline in the list on a scale from -1 (very negative) to 1 (very positive). "
                                            f"Here are the headlines: '{combined_headlines}'"}
            ],
            temperature=0.2,
            max_tokens=100
        )
        # Parse response to get sentiment scores
        sentiment_text = response.choices[0].message['content'].strip()
        scores = [float(score) for score in sentiment_text.split() if score.replace('.', '', 1).replace('-', '', 1).isdigit()]
        return np.mean(scores) if scores else 0
    except Exception as e:
        print(f"An error occurred during sentiment analysis: {e}")
        return 0  # Default to neutral sentiment

# Function to simulate portfolio allocation based on sentiment
def allocate_portfolio(sentiments):
    # Convert sentiments to weights
    positive_weights = {ticker: max(sentiment, 0) for ticker, sentiment in sentiments.items()}
    total_weight = sum(positive_weights.values())
    if total_weight > 0:
        return {ticker: weight / total_weight for ticker, weight in positive_weights.items()}
    else:
        return {ticker: 1 / len(sentiments) for ticker in sentiments}

# Backtest function
def backtest(price_data, daily_returns, tickers, headlines_data):
    portfolio_value = [1]  # Start with an initial portfolio value of 1
    allocations = {}  # Track daily allocations

    for date in daily_returns.index:
        # Simulate LLM-based sentiment scores for the date
        sentiments = {ticker: analyze_sentiment(headlines_data.get((ticker, date), [])) for ticker in tickers}
        allocation = allocate_portfolio(sentiments)
        allocations[date] = allocation

        # Calculate daily return based on allocation and daily returns
        daily_return = sum(allocation[ticker] * daily_returns.loc[date, ticker] for ticker in tickers)
        portfolio_value.append(portfolio_value[-1] * (1 + daily_return))

    # Convert portfolio value to DataFrame for analysis
    portfolio_df = pd.DataFrame(portfolio_value, index=daily_returns.index.insert(0, start_date), columns=["Portfolio Value"])

    # Calculate performance metrics
    cumulative_return = portfolio_df.iloc[-1] / portfolio_df.iloc[0] - 1
    sharpe_ratio = np.mean(daily_returns.mean()) / daily_returns.std() * np.sqrt(252)
    max_drawdown = (portfolio_df / portfolio_df.cummax() - 1).min()

    return portfolio_df, cumulative_return, sharpe_ratio, max_drawdown

# Main function to run backtest
def main():
    # Fetch historical price data
    price_data = fetch_price_data(tickers, start_date, end_date)
    daily_returns = calculate_daily_returns(price_data)

    # Example headlines data (replace with actual scraped headlines)
    headlines_data = {
        # Example structure: {(ticker, date): ["headline1", "headline2", ...]}
        ("AMZN", "2023-01-01"): ["AMZN rises on positive earnings", "AMZN stock shows resilience"],
        ("AAPL", "2023-01-01"): ["AAPL faces supply chain challenges", "AAPL releases new iPhone"]
    }

    # Run backtest
    portfolio_df, cumulative_return, sharpe_ratio, max_drawdown = backtest(price_data, daily_returns, tickers, headlines_data)

    # Output results
    print("Portfolio Performance:")
    print("Cumulative Return:", cumulative_return)
    print("Sharpe Ratio:", sharpe_ratio)
    print("Max Drawdown:", max_drawdown)

    # Plot portfolio value over time
    portfolio_df.plot(title="Portfolio Value Over Time")

if __name__ == "__main__":
    main()
