import openai

api_key = "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"
openai.api_key = api_key


def chat_with_gpt(prompt, max_tokens=50):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
    )
    print(response)
    return response.choices[0].text.strip()


# Start a conversation with the chatbot
conversation_history = "User: Hi, how can you help me?\nChatbot:"
while True:
    user_input = input("You: ")
    conversation_history += f"\nUser: {user_input}\nChatbot:"
    chatbot_response = chat_with_gpt(conversation_history)
    conversation_history += f" {chatbot_response}"
    print(f"Chatbot: {chatbot_response}")
