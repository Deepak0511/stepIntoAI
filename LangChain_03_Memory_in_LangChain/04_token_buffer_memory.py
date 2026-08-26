"""
04_token_buffer_memory.py (Modern LangChain Version)
----------------------------------------------------
Concept: Chatbot with Token-Limited Memory

Purpose:
Show how to maintain conversation memory within a token limit
using LangChain’s modern Runnable + MessageHistory architecture.

LangChain Concept:
RunnableWithMessageHistory + token counting (manual control).
"""

import os
import json
from dotenv import load_dotenv
from typing import List

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from tiktoken import get_encoding


# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

# Load configuration
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

# Initialize model
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
# 2️⃣ Define Prompt and Base Chain
# ==========================================
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant.\n"
    "Conversation so far (trimmed to fit token limit): {history}\n"
    "User: {input}\n"
    "Assistant:"
)

base_chain = prompt | llm | StrOutputParser()


# ==========================================
# 3️⃣ Token-Aware Memory Store
# ==========================================
TOKEN_LIMIT = 250  # approximate context size to retain
encoding = get_encoding("cl100k_base")  # for GPT-like tokenization
store = {}


def num_tokens_from_messages(messages: List[str]) -> int:
    """Roughly estimate token count from text messages."""
    return sum(len(encoding.encode(m)) for m in messages)


def get_session_history(session_id: str):
    """Return or create a session, trimming history by token budget."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    history = store[session_id]

    # Estimate token usage and trim if necessary
    total_tokens = num_tokens_from_messages([m.content for m in history.messages])
    while total_tokens > TOKEN_LIMIT and len(history.messages) > 2:
        # Remove the oldest two messages (1 user + 1 assistant pair)
        history.messages = history.messages[2:]
        total_tokens = num_tokens_from_messages([m.content for m in history.messages])

    return history


# ==========================================
# 4️⃣ Wrap Chain with RunnableWithMessageHistory
# ==========================================
chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ==========================================
# 5️⃣ Interactive Chat
# ==========================================
print("\n💬 CUSTOMER SUPPORT CHAT — TOKEN-LIMITED MEMORY (Modern API)")
print(f"(Keeps memory within ~{TOKEN_LIMIT} tokens)")
print("Type 'exit' or 'quit' to end.\n")
print("----------------------------------------")

session_id = "token_session"

while True:
    user_input = input("👤 You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Chat ended. Thanks for testing!")
        break

    if not user_input:
        continue

    response = chat_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    print(f"🤖 Assistant: {response}\n")

    history = get_session_history(session_id)
    total_tokens = num_tokens_from_messages([m.content for m in history.messages])
    print(f"🧠 [Approx. {total_tokens} tokens stored in memory]\n")


# ==========================================
# 6️⃣ Final Memory Dump
# ==========================================
print("----------------------------------------")
print("🧠 Final Token-Limited Memory:\n")

for msg in get_session_history(session_id).messages:
    role = "USER" if msg.type == "human" else "ASSISTANT"
    print(f"{role}: {msg.content}")

print(f"""
----------------------------------------
📘 Key Takeaways:
1️⃣ Token-based trimming ensures chat fits within model context.
2️⃣ Prevents overflow errors for large models (like GPT-4, Gemini).
3️⃣ This approach mimics ConversationTokenBufferMemory using the modern API.
""")
