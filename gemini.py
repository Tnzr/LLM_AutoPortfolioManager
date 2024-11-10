import google.generativeai as genai
import os


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="You are a Portfolio Manager",
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "What sectors should I allocate my portfolio into november",
      ],
    },
    {
      "role": "model",
      "parts": [
        "I cannot provide specific investment advice, as I am an AI and not a financial professional. \n\nHowever, I can offer some general insights on sector allocation for November, along with important considerations:\n\n**General Market Considerations:**\n\n* **Economic Outlook:** Understand the current state of the economy (e.g., inflation, interest rates, growth expectations) and how it might impact different sectors.\n* **Geopolitical Events:**  Factors like international tensions, energy prices, and global supply chains can significantly affect market performance.\n* **Interest Rate Expectations:**  The Federal Reserve's actions on interest rates can influence the valuation of different assets and sectors.\n\n**Potential Sector Opportunities in November:**\n\n* **Energy:** Rising energy prices and global demand may benefit energy producers (oil and gas, renewable energy).\n* **Healthcare:** Healthcare stocks can often be seen as defensive, meaning they tend to hold value during economic downturns.\n* **Consumer Staples:** Companies that provide essential goods (e.g., food, household products) may perform well due to consistent demand.\n* **Technology:** The technology sector, particularly in areas like cloud computing, cybersecurity, and artificial intelligence, can offer growth potential.\n\n**Sectors to Consider Avoiding:**\n\n* **Financials:** Rising interest rates can impact banks' profitability.\n* **Discretionary Consumer:** Companies that sell non-essential goods may struggle if consumer spending slows.\n* **Materials:** The performance of this sector can be tied to global economic growth and commodity prices.\n\n**Important Considerations:**\n\n* **Your Investment Goals and Risk Tolerance:** How much risk are you comfortable with? What are your financial goals?\n* **Your Investment Horizon:** How long do you plan to hold your investments?\n* **Your Portfolio Diversification:** Diversify your investments across different sectors and asset classes.\n\n**Recommendations:**\n\n* **Consult with a Financial Advisor:** A financial advisor can help you create a personalized investment strategy based on your individual circumstances.\n* **Do your Research:** Stay informed about market trends and company performance. \n* **Be Patient:** Investing is a long-term game. Don't get caught up in short-term fluctuations.\n\n**Remember:** This is not financial advice. Always consult a financial professional before making any investment decisions. \n",
      ],
    },
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

print(response.text)