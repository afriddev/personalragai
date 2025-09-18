from abc import ABC, abstractmethod
from typing import Any

from fastapi.responses import StreamingResponse

from clientservices.models import ChatRequestModel, ChatResponseModel


class ChatImpl(ABC):

    @abstractmethod
    def HandleApiStatusError(self, statusCode: int) -> ChatResponseModel:
        pass

    @abstractmethod
    async def Chat(
        self, modelParams: ChatRequestModel
    ) -> ChatResponseModel | StreamingResponse:
        pass

    @abstractmethod
    async def OpenaiChat(self, modelParams: ChatRequestModel) -> Any:
        pass

    @abstractmethod
    async def OpenaiGroqChat(self, modelParams: ChatRequestModel) -> Any:
        pass

    @abstractmethod
    async def CerebrasChat(self, modelParams: ChatRequestModel) -> Any:
        pass
