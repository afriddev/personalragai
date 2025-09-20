from abc import ABC, abstractmethod
from ragservices.models import ChunkInstanceModel
from clientservices.models import ChatMessageModel


class ChunkInstanceImpl(ABC):

    @abstractmethod
    async def ExtractChunkInstance(
        self, chunk: str, messages: list[ChatMessageModel], retryLimit: int
    ) -> ChunkInstanceModel:
        pass
