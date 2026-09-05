# AI Research Assistant using LangChain Tool Calling

## Project Overview

In this project, we will build an AI Research Assistant using LangChain Tool Calling.

The objective of this project is to understand how Large Language Models (LLMs) can interact with external tools to solve user queries more effectively.

By default, an LLM can only generate responses based on its training data and reasoning capabilities. However, many real-world tasks require external actions such as performing calculations, searching the internet, retrieving information, or interacting with other systems.

Tool Calling enables an LLM to use external tools whenever required.

This project demonstrates how to:

1. Create custom tools
2. Use pre-built LangChain tools
3. Register tools with an agent
4. Create a Tool Calling Agent
5. Execute tools automatically based on user queries
6. Perform multi-step reasoning using multiple tools

---

# What is Tool Calling?

Tool Calling is the ability of an LLM to decide when a tool is needed, select the appropriate tool, execute it, and use the result to generate a final response.

Without Tool Calling:

User Question

↓

LLM

↓

Answer

With Tool Calling:

User Question

↓

LLM Agent

↓

Tool Selection

↓

Tool Execution

↓

Observation

↓

Final Answer

The major advantage is that the LLM is no longer limited to its internal knowledge.

---

# Why Do We Need Tool Calling?

Consider the following query:

"What is 125 * 75?"

An LLM might solve this using its reasoning capabilities.

Now consider:

"Search the web for the latest trends in Agentic AI."

The LLM cannot access live internet data on its own.

To answer this query, it needs a search tool.

Similarly, if we want to:

* Search websites
* Query databases
* Call APIs
* Generate reports
* Access research papers
* Execute Python code

the LLM must use tools.

Tool Calling makes this possible.

---

# Components of This Project

This project consists of four major components.

## 1. Large Language Model (LLM)

The LLM acts as the brain of the system.

Supported Models:

* Gemini
* OpenAI

The project is designed so that switching between providers requires only a configuration change.

Example:

```yaml
provider: gemini
```

or

```yaml
provider: openai
```

---

## 2. Prompt

The prompt defines the behavior of the agent.

The system prompt tells the agent:

* What its role is
* Which tools are available
* When tools should be used
* How responses should be generated

Without a proper prompt, the agent may not use tools effectively.

---

## 3. Tools

Tools are functions that perform specific tasks.

In this project we use the following tools.

### Calculator Tool

Purpose:

Perform mathematical calculations.

Example:

Input:

```text
125 * 75
```

Output:

```text
9375
```

This tool is implemented using the LangChain @tool decorator.

---

### Search Tool

Purpose:

Search the web using DuckDuckGo.

Example:

Input:

```text
Latest trends in Agentic AI
```

Output:

Relevant search results from the internet.

This demonstrates the use of a pre-built LangChain community tool.

---

### Keyword Extractor Tool

Purpose:

Extract important keywords from a piece of text.

Example:

Input:

```text
LangGraph is a framework for building stateful multi-agent applications.
```

Output:

```text
LangGraph, stateful, multi-agent, framework
```

This tool internally uses an LLM.

---

### Research Summary Tool

Purpose:

Generate a concise summary of a topic.

Example:

Input:

```text
LangGraph
```

Output:

A short research-oriented summary.

This tool also internally uses an LLM.

---

## 4. Agent

The Agent acts as the decision-making layer.

The agent receives the user query and decides:

* Whether a tool is required
* Which tool should be used
* Whether multiple tools are needed
* When enough information has been collected

The agent then generates the final response.

---

# Understanding Agent Workflow

The workflow used in this project is:

User Query

↓

Agent

↓

Tool Selection

↓

Tool Execution

↓

Observation

↓

Agent Decision

↓

Final Response

If the agent requires additional information, it may call another tool before generating the final response.

---

# Single Tool Calling Example

Query:

```text
What is 125 * 75?
```

Execution Flow:

User Query

↓

Agent

↓

Calculator Tool

↓

9375

↓

Final Answer

Only one tool is required.

---

# Multi Tool Calling Example

Query:

```text
Extract keywords and summarize the following text.
```

Execution Flow:

User Query

↓

Keyword Extractor Tool

↓

Keywords

↓

Research Summary Tool

↓

Final Answer

In this scenario the agent uses multiple tools sequentially.

This is known as Multi-Step Tool Calling.

---

# Custom Tools vs Pre-Built Tools

There are two common approaches to creating tools.

## Custom Tools

Created by developers.

Example:

```python
@tool
def calculator(expression):
    ...
```

Advantages:

* Full control
* Easy customization
* Suitable for business-specific use cases

---

## Pre-Built Tools

Provided by LangChain or third-party integrations.

Example:

```python
DuckDuckGoSearchRun()
```

Advantages:

* Faster development
* Less code
* Ready-to-use integrations

---

# Understanding Agent Executor

The Agent Executor is responsible for running the agent.

Responsibilities:

* Send user input to the agent
* Execute tools
* Collect observations
* Return final responses

Example:

```python
response = agent_executor.invoke(
    {
        "input": "What is 125 * 75?"
    }
)
```

The Agent Executor manages the entire interaction cycle.

---

# Verbose Mode

During development, we enabled:

```python
verbose=True
```

This allows us to observe:

* Tool selection
* Tool execution
* Intermediate outputs
* Agent reasoning flow

Example:

```text
Invoking: calculator

9375

Finished chain
```

Verbose mode is extremely useful when debugging agent behavior.

---

# Key Learnings from This Project

After completing this project, you should understand:

1. What Tool Calling is
2. Why Tool Calling is important
3. How to create custom tools
4. How to use LangChain community tools
5. How to register tools
6. How to create Tool Calling Agents
7. How agents decide which tool to use
8. How agents perform multi-step reasoning
9. How to inspect tool execution using verbose logs
10. How to build practical LLM applications using LangChain

---

# Conclusion

Tool Calling is one of the most important capabilities in modern Generative AI applications.

Without tools, an LLM can only generate responses.

With tools, an LLM can interact with the outside world, perform actions, retrieve information, and solve more complex problems.

In this project, we built an AI Research Assistant that demonstrates the complete lifecycle of Tool Calling using LangChain, including tool creation, tool registration, agent construction, single-tool execution, and multi-tool orchestration.

The concepts learned in this project form the foundation for more advanced topics such as AI Agents, Agentic Workflows, LangGraph, Multi-Agent Systems, and Autonomous AI Applications.