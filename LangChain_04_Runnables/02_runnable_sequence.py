"""
02_runnable_pipeline.py (Modern LangChain Version)
--------------------------------------------------
Concept: RunnableSequence (Modern)

Purpose:
Rewrite the SAME pipeline using LangChain Runnables.

Pipeline:
    1️⃣ Generate a blog title from user interest
    2️⃣ Expand it into a short outline

Goal:
Show how Runnables simplify chaining, improve readability,
and eliminate boilerplate from SequentialChain.
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
from langchain_core.runnables import RunnablePassthrough

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
cfg = config[provider]

# Initialize LLM
if provider == "openai":
    llm = ChatOpenAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 200),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.7),
        max_output_tokens=cfg.get("max_output_tokens", 200),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ==========================================
# 2️⃣ Define Runnable Steps
# ==========================================

# Step 1: Generate title
title_chain = (
    ChatPromptTemplate.from_template(
        "Suggest a creative blog title idea related to the topic: {interest}"
    )
    | llm
    | StrOutputParser()
)

# Step 2: Generate outline
outline_chain = (
    ChatPromptTemplate.from_template(
        "Write a short outline for the blog titled: {title}"
    )
    | llm
    | StrOutputParser()
)


# ==========================================
# 3️⃣ Build Runnable Pipeline
# ==========================================

blog_pipeline = (
    RunnablePassthrough()
    .assign(title=title_chain)   # adds "title" to the flow
    .assign(outline=outline_chain)  # uses title automatically
)


# ==========================================
# 4️⃣ Run Modern Pipeline
# ==========================================

print("\n⚡ MODERN PIPELINE — RunnableSequence")
print("Generates a blog title and outline using Runnables.\n")
print("----------------------------------------")

user_interest = input("✏️ Enter your area of interest: ").strip()

result = blog_pipeline.invoke({"interest": user_interest})

print("\n----------------------------------------")
print(f"📝 Blog Title: {result['title']}\n")
print(f"🧠 Outline:\n{result['outline']}")
print("----------------------------------------")

print("""
📘 Key Takeaways:
1️⃣ Runnables use simple pipe (|) composition — no heavy chain objects.
2️⃣ RunnablePassthrough lets you incrementally build structured outputs.
3️⃣ No need to manually define input/output variables like SequentialChain.
4️⃣ More flexible, readable, and production-ready.
""")