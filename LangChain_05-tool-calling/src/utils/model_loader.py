import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.config_loader import load_config


load_dotenv()


def load_llm(provider=None):

    config = load_config()

    if provider is None:
        provider = config["llm"]["provider"]

    provider = provider.lower()

    if provider == "openai":

        return ChatOpenAI(
            model=config["llm"]["openai"]["model_name"],
            api_key=os.getenv("OPENAI_API_KEY")
        )

    elif provider == "gemini":

        return ChatGoogleGenerativeAI(
            model=config["llm"]["gemini"]["model_name"],
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    else:

        raise ValueError(
            f"Unsupported provider : {provider}"
        )