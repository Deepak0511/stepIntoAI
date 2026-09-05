from pathlib import Path

project_files = [

    "config/config.yaml",

    "src/__init__.py",

    "src/utils/__init__.py",
    "src/utils/config_loader.py",
    "src/utils/model_loader.py",

    "src/tools/__init__.py",
    "src/tools/search_tool.py",
    "src/tools/wikipedia_tool.py",
    "src/tools/arxiv_tool.py",
    "src/tools/calculator_tool.py",

    "src/agents/__init__.py",
    "src/agents/research_agent.py",

    "src/prompts/__init__.py",
    "src/prompts/system_prompt.py",

    "notebooks/tool_calling_demo.ipynb",

    ".env",
    "requirements.txt",
    "README.md",
    "setup.py"
]

for file in project_files:

    path = Path(file)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.touch(exist_ok=True)

print("Project created successfully")