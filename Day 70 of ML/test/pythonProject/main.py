# chatbot_llama.py

from langchain_community.llms import Ollama


class OllamaChatBot:
    def __init__(self, model_name="llama3:latest"):
        # Initialize the Ollama LLM with the correct model identifier
        self.llm = Ollama(model=model_name)

    def chat(self, prompt):
        # Use invoke instead of __call__ for generating a response
        response = self.llm.invoke(prompt)
        return response


if __name__ == "__main__":
    chatbot = OllamaChatBot()

    print("Ollama Chatbot is ready. Type 'exit' to end the chat.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Timing the response to measure performance
        import time

        start_time = time.time()

        response = chatbot.chat(user_input)

        end_time = time.time()
        print(f"Bot: {response}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
