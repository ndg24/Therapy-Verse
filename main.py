from models.chatbot_transformer import simpleIntentChatbot, predict

def main():
    print("Welcome to the Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = simpleIntentChatbot(user_input)
        print(f"Chatbot: {response}")
        predict(user_input)

if __name__ == "__main__":
    main()
