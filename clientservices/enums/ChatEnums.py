from enum import Enum


class ChatMessageRoleEnum(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class ChatResponseStatusEnum(Enum):
    SUCCESS = (200, "SUCCESS")
    BAD_REQUEST = (400, "BAD_REQUEST")
    UNAUTHROZIED = (401, "UNAUTHROZIED")
    PERMISSION_DENIED = (403, "PERMISSION_DENIED")
    NOT_FOUND = (404, "NOT_FOUND")
    REQUEST_TIMEOUT = (408, "REQUEST_TIMEOUT")
    CONFLICT = (409, "CONFLICT")
    ENTITY_ERROR = (422, "ENTITY_ERROR")
    RATE_LIMIT = (429, "RATE_LIMIT")
    SERVER_ERROR = (500, "SERVER_ERROR")
    ERROR = (200, "ERROR")
    JSON_NOT_SUPPORTED = (500, "JSON_NOT_SUPPORTED")


class CerebrasChatModelEnum(Enum):
    QWEN_235B_THINKING = ("qwen-3-235b-a22b-thinking-2507", 7000, True, True)
    QWEN_235B = ("qwen-3-235b-a22b-instruct-2507", 5000, False, True)
    GPT_OSS_120B = ("gpt-oss-120b", 5000, True, True)
    LLAMA_70B = ("llama-3.3-70b", 5000, False,  True)
    # Optional Models
    QWEN_32B = ("qwen-3-32b", 5000, True, True)
    META_LLAMA_17B_MAVERICK = (
        "llama-4-maverick-17b-128e-instruct",
        3000,
        False,
        True,
    )


class OpenaiChatModelsEnum(Enum):
    LLAMA_235B_130K = ("nvidia/llama-3.1-nemotron-ultra-253b-v1", 15000, True)
    QWEN_480B_CODER_260K = ("qwen/qwen3-coder-480b-a35b-instruct", 20000, False)
    LLAMA_405B_110K = ("meta/llama-3.1-405b-instruct", 110000, 15000, False)
    SEED_OSS_32B_500K = ("bytedance/seed-oss-36b-instruct", 500000, 20000, True)


class GroqChatModelsEnum(Enum):
    GROQ_COMPOUND = "groq/compound"
