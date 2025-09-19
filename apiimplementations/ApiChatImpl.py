from abc import ABC, abstractmethod
from apimodels import ApiChatRequestModel
from fastapi.responses import StreamingResponse
from clientservices.enums import (
    OpenaiChatModelsEnum,
    CerebrasChatModelEnum,
    GroqChatModelsEnum,
)


class ApiChatImpl(ABC):

    @abstractmethod
    async def ApiChat(self, request: ApiChatRequestModel) -> StreamingResponse:
        pass

    @abstractmethod
    def GetModel(
        self, request: ApiChatRequestModel
    ) -> OpenaiChatModelsEnum | CerebrasChatModelEnum | GroqChatModelsEnum:
        pass
