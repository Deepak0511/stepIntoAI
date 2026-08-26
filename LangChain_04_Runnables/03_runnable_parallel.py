"""
05_runnable_parallel.py (Modern LangChain Version)
--------------------------------------------------
Concept: RunnableParallel (Concurrent Multi-Branch Execution)

Purpose:
Demonstrate how to run multiple branches (tasks) in parallel.
Each branch can use a different prompt or even logic type.

Example use case:
Given a topic, generate:
  1️⃣ A catchy title
  2️⃣ A summary paragraph
  3️⃣ A list of keywords

This shows how to perform multi-output reasoning efficiently.
"""

import os
import json
from dotenv import load_dotenv

# Silence logs before imports
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

import warnings
warnings.filterwarnings('ignore')


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
# 2️⃣ Define Each Branch Runnable
# ==========================================

# Branch 1: Generate Title
title_prompt = ChatPromptTemplate.from_template(
    "Create a catchy blog title about: {topic}"
)
title_chain = title_prompt | llm | StrOutputParser()

# Branch 2: Generate Summary
summary_prompt = ChatPromptTemplate.from_template(
    "Write a 2-sentence summary about: {topic}"
)
summary_chain = summary_prompt | llm | StrOutputParser()

# Branch 3: Generate Keywords
keywords_prompt = ChatPromptTemplate.from_template(
    "List 5 relevant keywords for: {topic}"
)
keywords_chain = keywords_prompt | llm | StrOutputParser()


# ==========================================
# 3️⃣ Combine with RunnableParallel
# ==========================================
# Each key becomes an output key in the final dictionary
parallel_chain = RunnableParallel(
    title=title_chain,
    summary=summary_chain,
    keywords=keywords_chain,
)


# ==========================================
# 4️⃣ Run the Parallel Pipeline
# ==========================================
print("\n⚡ RUNNABLE PARALLEL DEMO — Multi-Branch LLM Execution")
print("Generates title, summary, and keywords in one go.\n")
print("----------------------------------------")

topic = input("💬 Enter your blog topic: ").strip()

result = parallel_chain.invoke({"topic": topic})

print("\n----------------------------------------")
print(f"📝 Title:\n{result['title']}\n")
print(f"📘 Summary:\n{result['summary']}\n")
print(f"🔑 Keywords:\n{result['keywords']}\n")
print("----------------------------------------")

print("""
📘 Key Takeaways:
1️⃣ RunnableParallel executes multiple branches at once.
2️⃣ Each branch can have its own LLM or prompt.
3️⃣ Output is a combined dictionary of all results.
""")
