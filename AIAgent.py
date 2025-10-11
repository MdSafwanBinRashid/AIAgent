from langchain_core.messages import HumanMessage  # high level framework that allows us to build AI applications
from langchain_google_genai import ChatGoogleGenerativeAI  # allows us to use Google Gemini within LangChain and LangGraph
from langchain.tools import tool  # register tools that our AI can use
from langgraph.prebuilt import create_react_agent  # prebuilt AI agents
from dotenv import load_dotenv  # environment variables are kept, such as GOOGLE_API_KEY

import os

load_dotenv()  # load environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyAAaYAWZJqIoj0j6z8UO1hoH9vd2kMeU3c" # what the SDK actually reads

# Tools
@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"

@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    return f"Hello {name}, I hope you are well today"

# Pick a working Gemini model (2.x names)
PREFERRED_MODELS = [
    "gemini-2.5-flash",   # best price/perf; widely available
    "gemini-2.5-pro",     # strongest general model
    "gemini-2.0-flash",   # extra fallback
    "gemini-2.0-pro",
]

def get_model():
    last_error = None
    for mid in PREFERRED_MODELS:
        try:
            return ChatGoogleGenerativeAI(model=mid, temperature=0)  # higher temp == more random the model will be
        except Exception as e:
            last_error = e
    raise RuntimeError(
        f"None of the preferred models worked: {PREFERRED_MODELS}\n"
        f"Last error: {last_error}\n"
        "Tip: update packages and list your accessible models."
    )

def main():  # chatbot and ai agent
    model = get_model()
    tools = [calculator, say_hello]
    agent_executor = create_react_agent(model, tools)

    print("\nWelcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()  # func of strip: " hello " -> "hello"

        if user_input.lower() == "quit":
            break

        print("\nAssistant: ", end="")  # to stay in the same line

        # Safely stream the response from the LLM
        try:
            for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    for message in chunk["agent"]["messages"]:
                        print(message.content, end="")  # streaming the response (like typing)
        except Exception as e:
            print(f"\n[Error] {e}")
            print("It looks like your API key might be invalid or out of quota. Please check your account.")

        print()


if __name__ == "__main__":  # only call main if we execute this python file directly
    main()