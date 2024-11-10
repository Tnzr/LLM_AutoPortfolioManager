from urllib.request import urlopen, Request

import pandas as pd
from bs4 import BeautifulSoup


root = "https://finviz.com/quote.ashx?t="

tickers = ["AMZN", "AAPL", "MSFT", "AMD", "NVDA"]
news_tables = {}
for ticker in tickers:
    url = root + ticker
    req = Request(url=url, headers={"user-agent": 'my-app'})
    response = urlopen(req)

    # Specify the parser to avoid warning
    html = BeautifulSoup(response, "html.parser")
    news_table = html.find(id="news-table")
    news_tables[ticker] = news_table

# print(news_tables)
amzn_data = news_tables["AMZN"]
amzn_rows = amzn_data.findAll('tr')

# for index, row in enumerate(amzn_rows):
#     title = row.a.text
#     timestamp = row.td.text
#     print(title, timestamp)

parsed_data = []

for ticker, news_table in news_tables.items():
    if news_table:  # Check if news_table was found
        for row in news_table.findAll('tr'):
            # Check if row.a exists to avoid NoneType error
            if row.a:
                title = row.a.text
                date_data = row.td.text.split(' ')
                print(row.td.text)
                if len(date_data) == 1:  # Only time is available
                    time = date_data[0]
                    date = "Today"  # Set date as None if not present
                else:
                    date = date_data[0]
                    time = date_data[1]

                parsed_data.append([ticker, date, time, title])

print(parsed_data)

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
print(df)