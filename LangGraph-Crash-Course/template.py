import os
from pathlib import Path
import logging

# Configure logging to see the execution status cleanly
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# 1. Define the target project structure using structural Path objects
list_of_files = [
    "config/config.yaml",
    "notebooks/01_langgraph_basics.ipynb",
    "notebooks/02_sequential_workflows.ipynb",
    "notebooks/03_parallel_workflows.ipynb",
    "notebooks/04_conditional_workflows.ipynb",
    "notebooks/05_iterative_workflows.ipynb",
    "notebooks/06_checkpointing_and_memory.ipynb",
    "src/utils/__init__.py",
    "src/utils/config_loader.py",
    "src/utils/model_loader.py",
    "src/states/__init__.py",
    "src/states/state_schema.py",
    "src/nodes/__init__.py",
    "src/nodes/planner.py",
    "src/nodes/researcher.py",
    "src/nodes/reviewer.py",
    "src/nodes/summarizer.py",
    "src/tools/__init__.py",
    "src/tools/demo_tools.py",
    ".env",
    "requirements.txt",
    "README.md"
]

# 2. Iterate and construct the directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Handle directory creation if a path segment exists
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create the file only if it doesn't exist or is currently empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # Just initializing an empty file placeholder
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists and contains data. Skipping overwrite.")

logging.info("🎉 Project file structure successfully generated!")