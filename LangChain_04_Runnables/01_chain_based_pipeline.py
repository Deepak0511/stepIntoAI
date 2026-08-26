"""
01_chain_based_pipeline.py (Classic LangChain Version)
------------------------------------------------------
Concept: SequentialChain (Legacy)

Purpose:
Demonstrate how LangChain pipelines used to be written before
the modern Runnable API — step-by-step chain execution.

We'll build a simple "blog idea generator" pipeline:
    1️⃣ Step 1 — Generate a blog topic from a user interest
    2️⃣ Step 2 — Expand it into a short outline

Next file will rewrite this same logic using Runnables.
"""

import os
import json
from dotenv import load_dotenv

# Silence unnecessary logs
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

import warnings
warnings.filterwarnings('ignore')


# ==========================================
# 1️⃣ Setup
# ==========================================
load_dotenv()

# Load provider configuration
with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
cfg = config[provider]

# Initialize model
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
# 2️⃣ Define Individual Steps
# ==========================================

# --- Step 1: Generate a blog topic ---
topic_prompt = PromptTemplate(
    input_variables=["interest"],
    template="Suggest a creative blog title idea related to the topic: {interest}",
)

topic_chain = LLMChain(llm=llm, prompt=topic_prompt, output_key="title")

# --- Step 2: Expand the topic into an outline ---
outline_prompt = PromptTemplate(
    input_variables=["title"],
    template="Write a short outline for the blog titled: {title}",
)

outline_chain = LLMChain(llm=llm, prompt=outline_prompt, output_key="outline")


# ==========================================
# 3️⃣ Combine with SequentialChain
# ==========================================
blog_pipeline = SequentialChain(
    chains=[topic_chain, outline_chain],
    input_variables=["interest"],
    output_variables=["title", "outline"],
    verbose=True,  # shows internal flow
)


# ==========================================
# 4️⃣ Run the Old-School Chain
# ==========================================
print("\n🧩 OLD-SCHOOL PIPELINE — SequentialChain (Legacy)")
print("Generates a blog topic and an outline based on your interest.\n")
print("----------------------------------------")

user_interest = input("✏️ Enter your area of interest: ").strip()

result = blog_pipeline.invoke({"interest": user_interest})

print("\n----------------------------------------")
print(f"📝 Blog Title: {result['title']}\n")
print(f"🧠 Outline:\n{result['outline']}")
print("----------------------------------------")

print("""
📘 Key Takeaways:
1️⃣ SequentialChain executes multiple LLM steps in sequence.
2️⃣ Each step feeds its output to the next step.
3️⃣ Before Runnables, this was the standard pipeline approach.
""")
