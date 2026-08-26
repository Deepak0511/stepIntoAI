"""
06_runnable_map.py (Modern LangChain Version)
---------------------------------------------
Concept: RunnableMap (Batch / Multi-Input Processing)

Purpose:
Demonstrate how to use RunnableMap to process multiple
inputs (like reviews or paragraphs) in one go.

Example workflow:
Given a list of customer reviews:
  1️⃣ Generate a short summary for each
  2️⃣ Detect overall sentiment for each
Then, return all results as a structured dictionary.
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
from langchain_core.runnables import RunnableLambda, RunnableMap

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
        temperature=cfg.get("temperature", 0.5),
        max_tokens=cfg.get("max_tokens", 200),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=cfg.get("model"),
        temperature=cfg.get("temperature", 0.5),
        max_output_tokens=cfg.get("max_output_tokens", 200),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ==========================================
# 2️⃣ Define Review Processing Chains
# ==========================================

# Summarize each review
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this customer review in one sentence:\n\n{review}"
)
summary_chain = summary_prompt | llm | StrOutputParser()

# Detect sentiment
sentiment_prompt = ChatPromptTemplate.from_template(
    "Classify the sentiment of this review as Positive, Negative, or Neutral:\n\n{review}"
)
sentiment_chain = sentiment_prompt | llm | StrOutputParser()

# Combine both tasks for a single review
single_review_processor = RunnableMap(
    summary=summary_chain,
    sentiment=sentiment_chain,
)


# ==========================================
# 3️⃣ Wrap for Batch Processing
# ==========================================
# RunnableMap applies the same logic to multiple inputs
batch_processor = RunnableLambda(
    lambda inputs: [
        single_review_processor.invoke({"review": r}) for r in inputs["reviews"]
    ]
)


# ==========================================
# 4️⃣ Interactive Demo
# ==========================================
print("\n🧩 RUNNABLE MAP DEMO — Multi-Review Processing")
print("Processes a list of reviews (summary + sentiment per review).\n")
print("----------------------------------------")

# Example reviews (can also let user enter)
sample_reviews = [
    "The product quality is amazing! I love how durable it feels.",
    "Delivery took too long, and the item was slightly damaged.",
    "Customer support was okay, but not super helpful.",
    "Fantastic experience overall — will definitely buy again!",
]

# Optionally let user input their own
choice = input("Use sample reviews? (y/n): ").strip().lower()
if choice == "n":
    sample_reviews = []
    print("Enter reviews (blank line to stop):")
    while True:
        text = input("🗣 Review: ").strip()
        if not text:
            break
        sample_reviews.append(text)

# Run the batch pipeline
results = batch_processor.invoke({"reviews": sample_reviews})

print("\n----------------------------------------")
for i, r in enumerate(results, start=1):
    print(f"🗣 Review {i}: {sample_reviews[i-1]}")
    print(f"📘 Summary: {r['summary']}")
    print(f"💬 Sentiment: {r['sentiment']}\n")
print("----------------------------------------")

print("""
📘 Key Takeaways:
1️⃣ RunnableMap processes multiple inputs in one go.
2️⃣ Each subtask (summary/sentiment) runs in parallel per input.
3️⃣ Ideal for bulk text analysis, document review, or batch summarization.
""")
