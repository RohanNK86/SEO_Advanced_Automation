"""
LLM Model Configuration
FlyRank Content Intelligence Platform

Fixed version: correct dotenv loading, openai client init,
and langchain-openai ChatOpenAI setup.
"""
import os
from dotenv import load_dotenv

# Load .env from project root (two levels up from Agents/)
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_env_path)

import openai

API_KEY  = os.getenv("api_keys", "")
BASE_URL = os.getenv("base_url", "")
LLM_MODEL = os.getenv("llm_model", "gpt-4o-mini")

# Raw OpenAI client (used for direct calls if needed)
_client_kwargs = {"api_key": API_KEY} if API_KEY else {}
if BASE_URL:
    _client_kwargs["base_url"] = BASE_URL

client = openai.OpenAI(**_client_kwargs) if API_KEY else None

# LangChain LLM wrapper
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        temperature=0.0,
        model=LLM_MODEL,
        openai_api_key=API_KEY or "placeholder",
        openai_api_base=BASE_URL if BASE_URL else None,
    ) if API_KEY else None
except Exception as e:
    print(f"[model] LLM unavailable: {e}")
    llm = None

LLM_AVAILABLE = llm is not None
print(f"[model] LLM available: {LLM_AVAILABLE}  model={LLM_MODEL}")