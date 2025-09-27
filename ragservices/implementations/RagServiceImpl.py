from abc import ABC, abstractmethod
from ragservices.models import (
    ChunkInstanceModel,
    AllQaResponseModel,
    ExtractTextFromYtResponseModel,
    ExtractQuestionsFromChunkResponseModel,
)
from clientservices.models import ChatMessageModel


class ExtractInstancesFromChunkServiceImpl(ABC):

    @abstractmethod
    async def ExtractInstancesFromChunk(
        self, chunk: str, messages: list[ChatMessageModel], retryLimit: int
    ) -> ChunkInstanceModel:
        pass

    @abstractmethod
    async def ExtractNodeSummary(
        self, messages: list[ChatMessageModel], retryLimit: int
    ) -> str:
        pass

    @abstractmethod
    async def ExtractQuestionsFromChunk(
        self, messages: list[ChatMessageModel], retryLimit: int
    ) -> ExtractQuestionsFromChunkResponseModel:
        pass


class ExtractChunksFromDocServiceImpl(ABC):

    @abstractmethod
    async def ExtractChunksFromPdf(self, file: str) -> list[str]:
        pass

    @abstractmethod
    def ExtractQaChunkFromCsv(self, file: str) -> AllQaResponseModel:
        pass

    @abstractmethod
    def ExtractChunksFromYtVideo(
        self, videoId: str
    ) -> list[ExtractTextFromYtResponseModel]:
        pass


class BuildRagServiceImpl(ABC):

    @abstractmethod
    async def BuildGraphRagFromPdf(self, file: str):
        pass
    
    @abstractmethod
    async def BuildQaRagFromPdf(self, file: str) :
        pass
    
    @abstractmethod
    async def BuildQaRagFromCsv(self, file: str) :
        pass
    
    
