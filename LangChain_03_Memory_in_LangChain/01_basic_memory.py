"""
01_basic_memory.py (Modern LangChain Version)
---------------------------------------------
Concept: Interactive Chatbot WITH Memory (Runnable API)

Purpose:
Demonstrate a live terminal chatbot that remembers previous conversation turns,
using LangChain’s new Runnable + MessageHistory system.

LangChain Concept: RunnableWithMessageHistory
"""

import os
import json
from dotenv import load_dotenv

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
cfg = config[provider]

if provider == "gemini":
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in the .env file before running."
        )

# Initialize LLM
if provider == "openai":
    llm = ChatOpenAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.6),
        max_tokens=cfg.get("max_tokens", 250),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.6),
        max_output_tokens=cfg.get("max_output_tokens", 250),
        google_api_key=google_api_key,
    )


# ==========================================
# 2️⃣ Define Prompt + Chain
# ==========================================
prompt = ChatPromptTemplate.from_template(
    "You are a friendly customer support assistant.\n"
    "Previous messages: {history}\n"
    "User: {input}\n"
    "Assistant:"
)

base_chain = prompt | llm | StrOutputParser()


# ==========================================
# 3️⃣ Configure Memory Store
# ==========================================
# Create a simple in-memory store for sessions
store = {}

def get_session_history(session_id: str):
    """Returns or creates message history for a session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory() #creates a new, empty chat history object that stores all messages (user and assistant) in memory (RAM).
    return store[session_id]


# Wrap the chain with memory
chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
# RunnableWithMessageHistory is a LangChain utility that wraps a "chain" (here, base_chain) and automatically manages chat history (memory) for you.
# base_chain is your prompt + LLM + output parser pipeline.
# get_session_history is a function that, given a session ID, returns the correct InMemoryChatMessageHistory object (from the store dictionary).
# input_messages_key="input" tells LangChain which key in your input dictionary contains the user’s message.
# history_messages_key="history" tells LangChain which key should be used to pass the chat history to the prompt.

# ==========================================
# 4️⃣ Interactive CLI Chat
# ==========================================
print("\n💬 CUSTOMER SUPPORT CHAT — MODERN MEMORY")
print("Type 'exit' or 'quit' to end.\n")
print("----------------------------------------")

session_id = "demo_session"

while True:
    user_input = input("👤 You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Chat ended. Thanks for testing!")
        break

    if not user_input:
        continue

    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"🤖 Assistant: {response}\n")


# ==========================================
# 5️⃣ Display Stored Messages
# ==========================================
print("----------------------------------------")
print("🧠 Chat Memory Contents (Modern API):\n")

history = get_session_history(session_id)
for msg in history.messages:
    role = "USER" if msg.type == "human" else "ASSISTANT"
    print(f"{role}: {msg.content}")

print("""
----------------------------------------
📘 Key Takeaways:
1️⃣ RunnableWithMessageHistory replaces ConversationChain.
2️⃣ Memory is session-based and flexible (InMemory, Redis, DB, etc.).
3️⃣ 100% compatible with LangChain 0.3+ — no deprecation warnings!
      
Summary: InMemoryChatMessageHistory is great for demos, testing, or small-scale use, but not for production or multi-user systems where persistence and scalability are needed.      
""")
