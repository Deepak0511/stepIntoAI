
---

# Customer Support QA Evaluator (GenAI + LangChain)

## Overview

This project implements a production-grade GenAI pipeline for evaluating customer support call transcripts. It leverages LangChain to build a modular system that classifies calls, dynamically routes evaluation logic, assesses agent performance across multiple dimensions, and generates structured QA reports.

The system is designed with clear separation of concerns, making it easy to extend, maintain, and deploy.

---

## Key Features

* Multi-provider LLM support (OpenAI and Gemini)
* Structured output using Pydantic
* Dynamic routing based on call type
* Modular evaluator design (tone, knowledge, resolution)
* End-to-end pipeline orchestration
* Production-ready project structure
* Reproducible and configurable workflow

---

## Project Structure

```
qa-evaluator/
│
├── config/
│   └── config.json
│
├── data/
│   ├── transcripts.csv
│   └── output.xlsx
│
├── logs/
│   └── app.log
│
├── src/
│   ├── components/
│   │   ├── classification.py
│   │   ├── evaluation.py
│   │   ├── router.py
│   │   ├── aggregation.py
│   │   └── reporting.py
│   │
│   ├── pipeline/
│   │   └── pipeline.py
│   │
│   └── utils/
│       ├── config_loader.py
│       ├── data_loader.py
│       ├── llm_loader.py
│       └── helpers.py
│
├── .env
├── requirements.txt
├── template.py
└── main.py
```

---

## Pipeline Flow

The system follows a structured, multi-stage pipeline:

1. Input Layer
   Load transcripts from dataset

2. Classification
   Predict call type using LLM with structured output

3. Routing
   Determine evaluation plan based on call type

4. Evaluation (Dynamic + Conditional)
   Execute relevant evaluators:

   * Tone & Empathy
   * Knowledge Accuracy
   * Resolution Quality

5. Aggregation
   Compute overall scores and identify strengths/weaknesses

6. Reporting
   Generate summary and recommendations using LLM

7. Output
   Save final results to Excel

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd qa-evaluator
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
LLM_PROVIDER=openai
```

---

## Configuration

Edit `config/config.json`:

```json
{
  "llm": {
    "openai_model": "gpt-4o-mini",
    "gemini_model": "gemini-1.5-flash",
    "temperature": 0.3
  },
  "evaluation": {
    "criteria": [
      "tone_empathy",
      "knowledge_accuracy",
      "resolution_quality"
    ]
  },
  "classification": {
    "labels": [
      "billing",
      "claims",
      "complaint",
      "general_query"
    ]
  }
}
```

---

## Running the Pipeline

```bash
python main.py
```

Output will be saved to:

```
data/output.xlsx
```

---

## Development Workflow (Important)

When extending or modifying the system, follow this structured approach:

### Step 1: Utilities Layer

* `utils/config_loader.py`
  Responsible for loading `.env` and `config.json`

* `utils/data_loader.py`
  Handles dataset loading

* `utils/llm_loader.py`
  Initializes LLM (OpenAI or Gemini)

---

### Step 2: Component Layer

Each logical unit is implemented independently:

* `src/components/classification.py`
  Build classification chain

* `src/components/evaluation.py`
  Implement all evaluator chains (tone, knowledge, resolution)

* `src/components/reporting.py`
  Build final report generation chain

---

### Step 3: Pipeline Integration

All execution logic must be implemented inside:

* `src/pipeline/pipeline.py`

This file:

* Calls components
* Applies logic over dataset
* Orchestrates flow

---

### Step 4: Entry Point

* `main.py`
  Wires everything together and runs the full pipeline

---

## Design Principles

* Separation of concerns (utils vs components vs pipeline)
* No business logic inside notebooks
* No hardcoded dependencies
* Reusable chain builders
* Structured outputs over raw text
* Explicit dependency passing (no global state)

---

## Future Improvements

* Add logging (replace print statements)
* Introduce retry logic for LLM calls
* Parallel execution using RunnableParallel
* Add API layer (FastAPI)
* Extend routing using LLM-based decision making
* Persist outputs to database

---

## Use Cases

* Customer support QA automation
* Agent performance monitoring
* Training and feedback systems
* GenAI pipeline architecture learning

---
