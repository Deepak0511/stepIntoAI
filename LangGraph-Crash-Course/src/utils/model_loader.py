import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def load_llm(provider=None):

    config = load_config()

    if provider is None:
        provider = config["llm"]["provider"]

    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add OPENAI_API_KEY=your-api-key to "
                f"{PROJECT_ROOT / '.env'} or set it in the environment."
            )

        return ChatOpenAI(
            model=config["llm"]["openai"]["model_name"],
            temperature=config["llm"]["openai"].get("temperature", 0),
            api_key=api_key
        )

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add GEMINI_API_KEY=your-api-key to "
                f"{PROJECT_ROOT / '.env'} or set it in the environment."
            )

        return ChatGoogleGenerativeAI(
            model=config["llm"]["gemini"]["model_name"],
            temperature=config["llm"]["gemini"].get("temperature", 0),
            google_api_key=api_key
        )

    else:

        raise ValueError(
            f"Unsupported provider : {provider}"
        )