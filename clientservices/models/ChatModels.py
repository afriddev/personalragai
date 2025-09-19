from typing import Any, List, Optional
from pydantic import BaseModel

from clientservices.enums import (
    ChatMessageRoleEnum,
    ChatResponseStatusEnum,
    CerebrasChatModelEnum,
    OpenaiChatModelsEnum,
    GroqChatModelsEnum,
)


class ChatMessageModel(BaseModel):
    role: Optional[ChatMessageRoleEnum] = ChatMessageRoleEnum.USER
    content: str


class ChatRequestModel(BaseModel):
    model: CerebrasChatModelEnum | OpenaiChatModelsEnum | GroqChatModelsEnum = (
        OpenaiChatModelsEnum.LLAMA_405B_110K
    )
    messages: List[ChatMessageModel]
    maxCompletionTokens: Optional[int] = 1024
    stream: Optional[bool] = True
    temperature: Optional[float] = 0.2
    responseFormat: Optional[Any] = None
    topP: float = 0.7
    seed: int = 42
    method: str = "nvidia"


class ChatChoiceMessageModel(BaseModel):
    role: ChatMessageRoleEnum = ChatMessageRoleEnum.ASSISTANT
    content: str


class ChatChoiceModel(BaseModel):
    index: int = 0
    message: ChatChoiceMessageModel


class ChatUsageModel(BaseModel):
    promptTokens: int | None = None
    completionTokens: int | None = None
    totalTokens: int | None = None


class ChatDataModel(BaseModel):
    id: str
    choices: List[ChatChoiceModel] = []
    created: int
    model: str = "llama-3.3-70b"
    usage: ChatUsageModel


class ChatResponseModel(BaseModel):
    status: ChatResponseStatusEnum = ChatResponseStatusEnum.SUCCESS
    content: str | None = None
