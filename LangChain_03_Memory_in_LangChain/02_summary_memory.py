"""
02_summary_memory.py (Modern LangChain Version)
-----------------------------------------------
Concept: Chatbot with Summarized Memory

Purpose:
Show how a chatbot can remember context using summarized conversation history
instead of storing every message.

LangChain Concept: RunnableWithMessageHistory + summarizing message store.
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

# Load model config
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
# 2️⃣ Prompt & Base Chain
# ==========================================
prompt = ChatPromptTemplate.from_template(
    "You are a concise and helpful assistant.\n"
    "Here’s a summary of the chat so far: {history}\n"
    "Now respond to the user message: {input}\n"
)

base_chain = prompt | llm | StrOutputParser()


# ==========================================
# 3️⃣ Setup Summarizing Message History
# ==========================================
# from langchain.memory import ConversationSummaryBufferMemory  # temporary utility
# # Note: RunnableWithMessageHistory doesn't yet auto-summarize; we'll mimic it here.

store = {}

def get_session_history(session_id: str):
    """Creates or returns summarized history for the session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ==========================================
# 4️⃣ Wrap in RunnableWithMessageHistory
# ==========================================
chat_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ==========================================
# 5️⃣ Live CLI Chat Loop
# ==========================================
print("\n💬 CUSTOMER SUPPORT CHAT — SUMMARY MEMORY (Modern API)")
print("Type 'exit' or 'quit' to end.\n")
print("----------------------------------------")

session_id = "summary_session"

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

    # Optional: dynamically summarize long histories
    history = get_session_history(session_id)
    if len(history.messages) > 8:
        # Simple summarization heuristic for demonstration
        summarizer = (
            ChatPromptTemplate.from_template(
                "Summarize the following conversation in 2-3 concise sentences:\n{conversation}"
            )
            | llm
            | StrOutputParser()
        )
        convo_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in history.messages])
        summary = summarizer.invoke({"conversation": convo_text})
        store[session_id] = InMemoryChatMessageHistory()
        store[session_id].add_ai_message(f"Conversation Summary: {summary}")
        print("\n🧠 Conversation summarized to maintain concise memory.\n")


# ==========================================
# 6️⃣ Show Stored Summary
# ==========================================
print("----------------------------------------")
print("🧠 Final Summarized Memory:\n")

for msg in get_session_history(session_id).messages:
    role = "USER" if msg.type == "human" else "ASSISTANT"
    print(f"{role}: {msg.content}")

print("""
----------------------------------------
📘 Key Takeaways:
1️⃣ RunnableWithMessageHistory manages context cleanly.
2️⃣ You can plug in your own summarization logic.
3️⃣ Perfect for long conversations that exceed context windows.
""")
