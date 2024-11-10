import os

import openai

# Replace 'your_api_key' with your actual API key from OpenAI
openai.api_key = os.getenv("APIKEY")

def chat_with_gpt_conversation(model="gpt-4", temperature=0.7, max_tokens=150):
    conversation_history = []

    print("You can start chatting with the AI. Type 'exit' to end the conversation.")
    while True:
        # Get user input
        user_input = input("You: ")

        # Exit the loop if the user wants to end the chat
        if user_input.lower() == "exit":
            print("Ending conversation. Goodbye!")
            break

        # Add user input to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        try:
            # Get response from ChatGPT
            response = openai.ChatCompletion.create(
                model=model,
                messages=conversation_history,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract the assistant's message and add it to the history
            assistant_message = response.choices[0].message['content'].strip()
            conversation_history.append({"role": "assistant", "content": assistant_message})

            # Print the assistant's response
            print("ChatGPT:", assistant_message)

        except Exception as e:
            print(f"An error occurred: {e}")
            break


# Run the continuous chat
chat_with_gpt_conversation()