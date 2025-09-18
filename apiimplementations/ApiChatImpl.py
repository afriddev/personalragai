from abc import ABC, abstractmethod
from apimodels import ApiChatRequestModel
from fastapi.responses import StreamingResponse


class ApiChatImpl(ABC):

    @abstractmethod
    async def ApiChat(self, request: ApiChatRequestModel) -> StreamingResponse:
        pass
