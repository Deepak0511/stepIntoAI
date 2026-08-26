"""
07_runnable_branch.py (Modern LangChain Version)
------------------------------------------------
Concept: RunnableBranch — Conditional Logic Flow

Purpose:
Show how to use RunnableBranch to create intelligent, conditional
pipelines where the execution path depends on input.

Example:
Given a user query, the system decides:
  1️⃣ If it's a question about data science → use the "expert" response chain
  2️⃣ If it's a casual greeting → respond politely
  3️⃣ Otherwise → use a generic fallback chain

This replaces the legacy RouterChain concept.
"""

import os
import json
from dotenv import load_dotenv

# Silence logs
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda


# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
cfg = config[provider]

# Initialize model
if provider == "openai":
    llm = ChatOpenAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 250),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.7),
        max_output_tokens=cfg.get("max_output_tokens", 250),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ==========================================
# 2️⃣ Define Branch-Specific Chains
# ==========================================

# Branch 1: Data Science Expert Response
ds_prompt = ChatPromptTemplate.from_template(
    "You are a data science expert. Answer this question clearly and briefly:\n\n{query}"
)
ds_chain = ds_prompt | llm | StrOutputParser()

# Branch 2: Greeting Response
greet_prompt = ChatPromptTemplate.from_template(
    "Respond cheerfully to this greeting:\n\n{query}"
)
greet_chain = greet_prompt | llm | StrOutputParser()

# Branch 3: Generic Fallback
generic_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Respond to this input:\n\n{query}"
)
generic_chain = generic_prompt | llm | StrOutputParser()


# ==========================================
# 3️⃣ Define Conditional Logic Function
# ==========================================
def route_input(inputs: dict):
    """Route input based on query type."""
    q = inputs["query"].lower()
    if any(word in q for word in ["data", "model", "machine learning", "ai"]):
        return "data_science"
    elif any(word in q for word in ["hello", "hi", "hey", "good morning"]):
        return "greeting"
    else:
        return "generic"


# ==========================================
# 4️⃣ Combine with RunnableBranch
# ==========================================
router_chain = RunnableBranch(
    # Condition → Chain pairs
    (lambda x: route_input(x) == "data_science", ds_chain),
    (lambda x: route_input(x) == "greeting", greet_chain),
    # Default branch
    generic_chain,
)


# ==========================================
# 5️⃣ Interactive CLI Demo
# ==========================================
print("\n🔀 RUNNABLE BRANCH DEMO — Conditional Logic Flow")
print("Routes user queries intelligently to different chains.\n")
print("----------------------------------------")

while True:
    user_input = input("💬 Enter your message ('exit' to quit): ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\n👋 Chat ended. Thanks for testing!")
        break

    result = router_chain.invoke({"query": user_input})
    print(f"🤖 Assistant: {result}\n")

print("""
----------------------------------------
📘 Key Takeaways:
1️⃣ RunnableBranch allows conditional routing between chains.
2️⃣ Each branch can represent a different logic or LLM prompt.
3️⃣ It's the modern replacement for RouterChain (cleaner & flexible).
""")
