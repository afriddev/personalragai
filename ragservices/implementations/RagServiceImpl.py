from abc import ABC, abstractmethod
from ragservices.models import ChunkInstanceModel, AllQaResponseModel
from clientservices.models import ChatMessageModel


class ExtractInstancesFromChunkServiceImpl(ABC):

    @abstractmethod
    async def ExtracInstancesFromChunk(
        self, chunk: str, messages: list[ChatMessageModel], retryLimit: int
    ) -> ChunkInstanceModel:
        pass


class ExtractChunksFromDocServiceImpl(ABC):

    @abstractmethod
    async def ExtractChunksFromPdf(self, file: str) -> list[str]:
        pass

    @abstractmethod
    def ExtractQaChunkFromCsv(self, file: str) -> AllQaResponseModel:
        pass

    @abstractmethod
    def ExtractChunksFromYtVideo(self, file: str) -> AllQaResponseModel:
        pass
