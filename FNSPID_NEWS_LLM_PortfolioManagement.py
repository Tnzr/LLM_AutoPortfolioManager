import os
import openai
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from textblob import TextBlob
import numpy as np
from datetime import datetime
import altair as alt
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Load environment variables
load_dotenv()
# Configure OpenAI API key
openai.api_key = os.getenv("APIKEY")  # Replace with your actual OpenAI API key

# Paths
news_data_path = "FNSPID_Financial_News_Dataset/data_processor/news_data_raw/aa.csv"
price_data_path = "FNSPID_Financial_News_Dataset/data_processor/stock_price_data_raw/aa.csv"

# Load data
news_data = pd.read_csv(news_data_path)
price_data = pd.read_csv(price_data_path)

# Custom function to parse the date column
def parse_date(date_string):
    try:
        return datetime.strptime(date_string, "%Y-%m-%d")
    except Exception as e:
        return None  # Return None if parsing fails

# Sentiment Analysis Function
def analyze_sentiment_nlp(texts):
    sentiments = [TextBlob(text).sentiment.polarity for text in texts]
    return sentiments


def analyze_sentiment_vader(texts):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        # Use the compound score to represent overall sentiment
        sentiments.append(scores['compound'])
    return sentiments


# Visualization: Price with Moving Averages and Legend
def visualize_price_with_moving_averages_and_legend(data):
    data = data.reset_index()

    # Base chart for price
    price_chart = alt.Chart(data).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Adj Close:Q', title='Price/Moving Averages'),
        tooltip=['Date:T', 'Adj Close:Q'],
        color=alt.value('blue')  # Fixed color for price
    ).properties(
        title="Price with Short and Long Moving Averages"
    )

    # Chart for Short Moving Average
    short_ma_chart = alt.Chart(data).mark_line(color='green').encode(
        x='Date:T',
        y='Short_MA:Q',
        tooltip=['Date:T', 'Short_MA:Q'],
        color=alt.value('green')  # Fixed color for short MA
    )

    # Chart for Long Moving Average
    long_ma_chart = alt.Chart(data).mark_line(color='red').encode(
        x='Date:T',
        y='Long_MA:Q',
        tooltip=['Date:T', 'Long_MA:Q'],
        color=alt.value('red')  # Fixed color for long MA
    )

    # Combine charts with a legend
    combined_chart = alt.layer(
        price_chart.encode(color=alt.value('blue')).properties(name="Price"),
        short_ma_chart.encode(color=alt.value('green')).properties(name="Short MA"),
        long_ma_chart.encode(color=alt.value('red')).properties(name="Long MA")
    ).resolve_scale(color='independent')  # Separate legend for each

    st.altair_chart(combined_chart, use_container_width=True)

# Visualization: Price Over Time
def visualize_price_and_signals(data):
    data = data.reset_index()
    chart = alt.Chart(data).mark_line().encode(
        x='Date:T',
        y='Adj Close:Q',
        tooltip=['Date:T', 'Adj Close:Q']
    ).properties(title="Price with Buy/Sell Signals")

    buy_signals = alt.Chart(data[data['Signal'] == 1]).mark_point(color='blue', size=50).encode(
        x='Date:T',
        y='Adj Close:Q',
        tooltip=['Date:T', 'Adj Close:Q']
    )

    sell_signals = alt.Chart(data[data['Signal'] == -1]).mark_point(color='red', size=50).encode(
        x='Date:T',
        y='Adj Close:Q',
        tooltip=['Date:T', 'Adj Close:Q']
    )

    st.altair_chart(chart + buy_signals + sell_signals, use_container_width=True)

# Visualization: Sentiment Over Time
def visualize_sentiment(data):
    data = data.reset_index()

    # Line chart for cumulative sentiment
    sentiment_chart = alt.Chart(data).mark_line(color='orange').encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Cumulative Sentiment:Q', title='Cumulative Sentiment'),
        tooltip=['Date:T', 'Cumulative Sentiment:Q']
    ).properties(title="Cumulative Sentiment Over Time")

    # Line chart for daily sentiment
    daily_sentiment_chart = alt.Chart(data).mark_line(color='purple').encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Sentiment:Q', title='Daily Sentiment'),
        tooltip=['Date:T', 'Sentiment:Q']
    ).properties(title="Daily Sentiment Over Time")

    combined_chart = sentiment_chart + daily_sentiment_chart
    st.altair_chart(combined_chart, use_container_width=True)

# Visualization: Price with Stop-Loss Markers
def visualize_price_with_stop_loss(data):
    data = data.reset_index()

    # Base chart for price
    price_chart = alt.Chart(data).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Adj Close:Q', title='Price'),
        tooltip=['Date:T', 'Adj Close:Q']
    ).properties(title="Price with Stop-Loss Events")

    # Stop-loss markers
    stop_loss_chart = alt.Chart(data[data['Stop_Loss_Triggered'] == True]).mark_point(color='orange', size=100).encode(
        x='Date:T',
        y='Adj Close:Q',
        tooltip=['Date:T', 'Adj Close:Q']
    )

    combined_chart = price_chart + stop_loss_chart
    st.altair_chart(combined_chart, use_container_width=True)

# Visualization: Price with Buy Orders and Stop-Loss Markers
def visualize_price_with_trades_and_stop_loss(data):
    data = data.reset_index()

    # Base chart for price
    price_chart = alt.Chart(data).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Adj Close:Q', title='Price'),
        tooltip=['Date:T', 'Adj Close:Q']
    ).properties(title="Price with Buy Orders and Stop-Loss Events")

    # Stop-loss markers
    stop_loss_chart = alt.Chart(data[data['Stop_Loss_Triggered'] == True]).mark_point(
        color='orange', size=100, shape='triangle'
    ).encode(
        x='Date:T',
        y='Adj Close:Q',
        tooltip=['Date:T', 'Adj Close:Q']
    ).properties(title="Stop-Loss Events")

    # Buy order markers
    buy_orders_chart = alt.Chart(data[(data['Signal'] == 1) & (data['Position'] > 0)]).mark_point(
        color='green', size=100, shape='circle'
    ).encode(
        x='Date:T',
        y='Adj Close:Q',
        tooltip=['Date:T', 'Adj Close:Q']
    ).properties(title="Buy Orders")

    # Combine the charts
    combined_chart = price_chart + stop_loss_chart + buy_orders_chart
    st.altair_chart(combined_chart, use_container_width=True)


def analyze_sentiment_finbert(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            sentiments = sentiment_pipeline(batch)
            results.extend(
                sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
                for sentiment in sentiments
            )
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            results.extend([0.0] * len(batch))  # Default to neutral sentiment on error
    return results

# ChatGPT Sentiment Analysis
def analyze_sentiment_chatgpt(texts):
    results = []
    for text in texts:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant."},
                {"role": "user", "content": f"Analyze the sentiment of the following text: '{text}'. Return 'Positive', 'Negative', or 'Neutral'."}
            ]
        )
        sentiment = response['choices'][0]['message']['content'].strip().lower()
        if "positive" in sentiment:
            results.append(0.9)
        elif "negative" in sentiment:
            results.append(-0.9)
        else:
            results.append(0.0)
    return results

# TextBlob Sentiment Analysis
def analyze_sentiment_textblob(texts):
    return [TextBlob(text).sentiment.polarity for text in texts]


# Benchmark All Methods
def benchmark_sentiment_methods(texts):
    benchmarks = pd.DataFrame({
        "TextBlob": analyze_sentiment_textblob(texts),
        "FinBERT": analyze_sentiment_finbert(texts),
        "ChatGPT": analyze_sentiment_chatgpt(texts),
        "VADER": analyze_sentiment_vader(texts)
    })
    return benchmarks


# Backtesting Metrics Calculation
def calculate_backtesting_metrics(data, risk_free_rate=0.01):
    """
    Calculate backtesting metrics, including Sharpe Ratio.
    :param data: DataFrame containing portfolio values and daily returns.
    :param risk_free_rate: Annualized risk-free rate (default: 1%).
    :return: Dictionary of calculated metrics.
    """
    # Ensure 'Daily_Return' is calculated
    if 'Daily_Return' not in data:
        data['Daily_Return'] = data['Portfolio_Value'].pct_change().fillna(0)

    # Risk-free rate for daily
    daily_risk_free_rate = risk_free_rate / 252

    # Calculate metrics
    final_portfolio_value = data['Portfolio_Value'].iloc[-1]
    initial_portfolio_value = data['Portfolio_Value'].iloc[0]
    total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
    annualized_return = ((1 + total_return) ** (1 / (len(data) / 252))) - 1
    max_drawdown = ((data['Portfolio_Value'] / data['Portfolio_Value'].cummax()) - 1).min()

    # Sharpe Ratio
    excess_daily_return = data['Daily_Return'] - daily_risk_free_rate
    sharpe_ratio = (
        excess_daily_return.mean() / excess_daily_return.std()
        if excess_daily_return.std() != 0 else np.nan
    ) * np.sqrt(252)  # Annualize Sharpe Ratio

    metrics = {
        "Total Return (%)": round(total_return * 100, 2),
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2) if not np.isnan(sharpe_ratio) else "N/A"
    }
    return metrics


# Apply date parsing and sentiment analysis
news_data['Date'] = pd.to_datetime(news_data['Date'], errors='coerce')
news_data = news_data.dropna(subset=['Date'])  # Drop rows with invalid dates
print("NewsData: ", news_data['Text'].head(10))
print("Type: ", type(news_data['Text'] ))


tokenized_sentence = tokenizer.encode(news_data['Text'].tolist(), padding=True, truncation=True,max_length=50, add_special_tokens = True)
news_data['Sentiment'] = analyze_sentiment_finbert(news_data['Text'].tolist())
print(news_data.head())

# sentiments = sentiment_pipeline(news_data['Text'])
print(news_data["Sentiment"].head())
news_data['Cumulative Sentiment'] = news_data['Sentiment'].cumsum()
print(news_data["Cumulative Sentiment"].head())

# Streamlit App
st.title("Sentiment-Aided Momentum Trading Strategy")

# Sidebar Inputs
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))
short_window = st.sidebar.slider("Short-term Moving Average Window", 5, 20, 5)
long_window = st.sidebar.slider("Long-term Moving Average Window", 8, 50, 8)
stop_loss_threshold = st.sidebar.slider("Stop-Loss Threshold (%)", min_value=1, max_value=50, value=10) / 100
initial_portfolio_value = st.sidebar.number_input("Initial Portfolio Value", min_value=100.0, value=1000.0, step=100.0)

# Convert start_date and end_date to datetime64[ns]
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

# Filter data by date range
price_data['Date'] = pd.to_datetime(price_data['Date'], errors='coerce')
price_data = price_data.dropna(subset=['Date'])
price_data = price_data[(price_data['Date'] >= start_date) & (price_data['Date'] <= end_date)]

news_data = news_data[(news_data['Date'] >= start_date) & (news_data['Date'] <= end_date)]

# Calculate Moving Averages and their Derivatives
price_data['Adj Close'] = price_data['Adj Close'].rolling(window=2).mean()
price_data['Short_MA'] = price_data['Adj Close'].ewm(span=short_window).mean()
price_data['Long_MA'] = price_data['Adj Close'].ewm(span=long_window).mean()

# Calculate Derivatives (Slopes) of Moving Averages
price_data['Short_MA_Derivative'] = price_data['Short_MA'].diff()
price_data['Long_MA_Derivative'] = price_data['Long_MA'].diff()

# Generate Buy/Sell Signals based on Moving Averages and Derivatives
ma_difference = (price_data['Short_MA'] - price_data['Long_MA']) / price_data['Long_MA']
threshold = 0.01  # Example: 1% threshold
price_data['Signal'] = 0

# Buy Signal: Short MA > Long MA, Positive Slopes, and Difference > Threshold
price_data.loc[
    (ma_difference > threshold) &
    (price_data['Short_MA_Derivative'] > 0) &
    (price_data['Long_MA_Derivative'] > 0),
    'Signal'
] = 1

# Sell Signal: Short MA < Long MA or Negative Slope on Short MA
price_data.loc[
    (ma_difference < -threshold) |
    (price_data['Short_MA_Derivative'] < 0),
    'Signal'
] = -1

# Sidebar Selector for NLP Method
st.sidebar.title("Sentiment Analysis Method")
selected_method = st.sidebar.selectbox(
    "Choose NLP Method:",
    ["TextBlob", "FinBERT", "ChatGPT", "VADER", "Benchmark All"]
)


# Apply Sentiment Analysis Based on Selected Method
if selected_method == "TextBlob":
    news_data['Sentiment'] = analyze_sentiment_textblob(news_data['Text'].tolist())
elif selected_method == "FinBERT":
    news_data['Sentiment'] = analyze_sentiment_finbert(news_data['Text'].tolist())
elif selected_method == "ChatGPT":
    news_data['Sentiment'] = analyze_sentiment_chatgpt(news_data['Text'].tolist())
elif selected_method == "VADER":
    news_data['Sentiment'] = analyze_sentiment_vader(news_data['Text'].tolist())
elif selected_method == "Benchmark All":
    benchmarks = benchmark_sentiment_methods(news_data['Text'].tolist())
    st.write("Benchmark Results (First 5 rows):")
    st.write(benchmarks.head())



# Calculate Cumulative Sentiment
news_data['Cumulative Sentiment'] = news_data['Sentiment'].rolling(window=5, min_periods=1).sum()


# Combine price and news data
price_data = price_data.set_index('Date')
news_data = news_data.set_index('Date')
combined_data = price_data.join(news_data[['Cumulative Sentiment']], how='left').fillna(0)  # Fill missing sentiment with 0

# Add a checkbox to toggle sentiment analysis
use_sentiment = st.sidebar.checkbox("Use Sentiment Analysis", value=True)


# Backtesting with Sentiment Confirmation and Stop-Loss
current_portfolio_value = initial_portfolio_value
combined_data['Portfolio_Value'] = initial_portfolio_value
combined_data['Position'] = 0  # Track the number of shares held
combined_data['Cash'] = initial_portfolio_value  # Track the cash position
combined_data['Entry_Price'] = 0  # Track the entry price for stop-loss calculation
combined_data['Daily_Return'] = 0.0
combined_data['Stop_Loss_Triggered'] = False  # Track stop-loss events


for i in range(1, len(combined_data)):
    date = combined_data.index[i]
    row = combined_data.iloc[i]
    prev_row = combined_data.iloc[i - 1]

    signal = row['Signal']
    adj_close = row['Adj Close']
    sentiment = row['Cumulative Sentiment']
    print(sentiment)
    # Use previous values for cash, position, and entry price
    cash = prev_row['Cash']
    position = prev_row['Position']
    entry_price = prev_row['Entry_Price']

    # Check stop-loss
    stop_loss_triggered = False
    if position > 0:  # Stop-loss applies only if holding a position
        # Calculate the price drop percentage
        loss_percentage = (adj_close - entry_price) / entry_price
        if loss_percentage < -stop_loss_threshold:
            stop_loss_triggered = True
            cash += position * adj_close  # Sell all shares
            position = 0  # Reset position
            entry_price = 0  # Reset entry price


    # Apply trading logic
    if signal == 1 and not stop_loss_triggered:  # Buy signal
        if position == 0 and sentiment >= 0:  # Buy condition depends on sentiment toggle
            shares_to_buy = cash // adj_close  # Calculate how many shares to buy
            if shares_to_buy > 0:
                cash -= shares_to_buy * adj_close
                position += shares_to_buy
                entry_price = adj_close  # Set the entry price for stop-loss calculation

    elif signal == -1 or stop_loss_triggered:  # Sell signal or stop-loss
        if position > 0:
            cash += position * adj_close  # Sell all shares
            position = 0  # Reset position
            entry_price = 0  # Reset entry price


    # Update portfolio value
    portfolio_value = cash + position * adj_close
    daily_return = (portfolio_value - prev_row['Portfolio_Value']) / prev_row['Portfolio_Value']
    combined_data.loc[date, 'Portfolio_Value'] = portfolio_value
    combined_data.loc[date, 'Position'] = position
    combined_data.loc[date, 'Cash'] = cash
    combined_data.loc[date, 'Entry_Price'] = entry_price
    combined_data.loc[date, 'Daily_Return'] = daily_return
    combined_data.loc[date, 'Stop_Loss_Triggered'] = stop_loss_triggered


    # Debugging: Print portfolio status for each iteration
    print(f"Date: {date}, Signal: {signal}, Sentiment: {sentiment}, Cash: {cash}, "
          f"Position: {position}, Entry_Price: {entry_price}, Portfolio_Value: {portfolio_value}, "
          f"Stop_Loss_Triggered: {stop_loss_triggered}")


# Display Backtesting Metrics
if 'Portfolio_Value' in combined_data:
    metrics = calculate_backtesting_metrics(combined_data)
    st.subheader(f"Backtesting Metrics using {selected_method}")
    for metric, value in metrics.items():
        st.metric(label=metric, value=value)


# Visualization: Portfolio Performance
st.subheader("Portfolio Performance")
st.line_chart(combined_data['Portfolio_Value'])

# Call the function for moving averages visualization
st.subheader("Price and Moving Averages with Legend")
visualize_price_with_moving_averages_and_legend(price_data)

# Call the updated visualization function
st.subheader("Price with Buy Orders and Stop-Loss Events")
visualize_price_with_trades_and_stop_loss(combined_data)

st.subheader("Sentiment Analysis")
visualize_sentiment(news_data)

# Notify Completion
st.success("Sentiment Analysis and Backtesting Complete!")

