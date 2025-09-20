import os
from typing import Any, cast
from dotenv import load_dotenv

load_dotenv()


CEREBRAS_API_KEY = cast(Any, os.getenv("CEREBRAS_API_KEY"))
NVIDIA_API_KEY = cast(Any, os.getenv("NVIDIA_API_KEY"))
NVIDIA_BASE_URL = cast(Any, os.getenv("NVIDIA_API_BASE_URL"))
NVIDIA_RERANK_BASE_URL = cast(Any, os.getenv("NVIDIA_API_RERANK_BASE_URL"))
GROQ_BASE_URL = cast(Any, os.getenv("GROQ_API_BASE_URL"))
GROQ_API_KEY = cast(Any, os.getenv("GROQ_API_KEY"))


def GetCerebrasApiKey() -> str:
    return CEREBRAS_API_KEY


def GetNvidiaApiKey() -> str:
    return NVIDIA_API_KEY


def GetNvidiaBaseUrl() -> str:
    return NVIDIA_BASE_URL

def GetNvidiaRerankBaseUrl() -> str:
    return NVIDIA_RERANK_BASE_URL 




def GetGroqApiKey() -> str:
    return GROQ_API_KEY


def GetGroqBaseUrl() -> str:
    return GROQ_BASE_URL
