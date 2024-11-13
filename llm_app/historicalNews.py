from newsapi import NewsApiClient
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv



load_dotenv()

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key="1af76f2f99e94bb7ae3f61cffaad0af3")

# Fetch historical headlines
def fetch_historical_news(ticker, start_date, end_date):
    all_headlines = []
    date = start_date

    while date <= end_date:
        # NewsAPI query for each day to get headlines
        articles = newsapi.get_everything(
            q=ticker,
            from_param=date.strftime('%Y-%m-%d'),
            to=(date + timedelta(days=1)).strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy',
            page_size=100
        )

        # Extract relevant information from each article
        for article in articles['articles']:
            headline = article['title']
            published_date = article['publishedAt'][:10]  # Format: YYYY-MM-DD
            all_headlines.append({'ticker': ticker, 'date': published_date, 'headline': headline})

        date += timedelta(days=1)

    # Return as a DataFrame
    return pd.DataFrame(all_headlines)

# Example usage
start_date = datetime(2024, 10, 10)
end_date = datetime(2024, 11, 10)
ticker = "AAPL"
headlines_df = fetch_historical_news(ticker, start_date, end_date)
print(headlines_df)
