import os
from typing import Any, cast
from dotenv import load_dotenv

load_dotenv()


CEREBRAS_API_KEY = cast(Any, os.getenv("CEREBRAS_API_KEY"))
CEREBRAS_API_KEY1 = cast(Any, os.getenv("CEREBRAS_API_KEY1"))
CEREBRAS_API_KEY2 = cast(Any, os.getenv("CEREBRAS_API_KEY2"))
CEREBRAS_API_KEY3 = cast(Any, os.getenv("CEREBRAS_API_KEY3"))
CEREBRAS_API_KEY4 = cast(Any, os.getenv("CEREBRAS_API_KEY4"))
NVIDIA_API_KEY = cast(Any, os.getenv("NVIDIA_API_KEY"))
NVIDIA_API_KEY1 = cast(Any, os.getenv("NVIDIA_API_KEY1"))
NVIDIA_BASE_URL = cast(Any, os.getenv("NVIDIA_API_BASE_URL"))
NVIDIA_RERANK_BASE_URL = cast(Any, os.getenv("NVIDIA_API_RERANK_BASE_URL"))
GROQ_BASE_URL = cast(Any, os.getenv("GROQ_API_BASE_URL"))
GROQ_API_KEY = cast(Any, os.getenv("GROQ_API_KEY"))


def GetCerebrasApiKey() -> str:
    return CEREBRAS_API_KEY


def GetCerebrasApiKey1() -> str:
    return CEREBRAS_API_KEY1


def GetCerebrasApiKey2() -> str:
    return CEREBRAS_API_KEY2


def GetCerebrasApiKey3() -> str:
    return CEREBRAS_API_KEY3


def GetCerebrasApiKey4() -> str:
    return CEREBRAS_API_KEY4


def GetNvidiaApiKey() -> str:
    return NVIDIA_API_KEY
def GetNvidiaApiKey1() -> str:
    return NVIDIA_API_KEY1


def GetNvidiaBaseUrl() -> str:
    return NVIDIA_BASE_URL


def GetNvidiaRerankBaseUrl() -> str:
    return NVIDIA_RERANK_BASE_URL


def GetGroqApiKey() -> str:
    return GROQ_API_KEY


def GetGroqBaseUrl() -> str:
    return GROQ_BASE_URL
