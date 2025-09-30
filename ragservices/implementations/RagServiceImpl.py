from abc import ABC, abstractmethod
from ragservices.models import (
    ExtractChunkInstanceResponseModel,
    AllQaResponseModel,
    ExtractQaFromChunkResponseModel,
)
from clientservices.models import ChatMessageModel


class ExtractInstancesFromChunkServiceImpl(ABC):

    @abstractmethod
    async def ExtractInstancesFromChunk(
        self, chunk: str, messages: list[ChatMessageModel], retryLimit: int
    ) -> ExtractChunkInstanceResponseModel:
        pass

    @abstractmethod
    async def ExtractNodeSummary(
        self, messages: list[ChatMessageModel], retryLimit: int
    ) -> str:
        pass

    @abstractmethod
    async def ExtractQuestionsFromChunk(
        self, messages: list[ChatMessageModel], retryLimit: int
    ) -> ExtractQaFromChunkResponseModel:
        pass

    @abstractmethod
    async def CleanYoutubeChunk(
        self, messages: list[ChatMessageModel], retryLimit: int
    ) -> str:
        pass


class ExtractChunksFromDocServiceImpl(ABC):

    @abstractmethod
    async def ExtractChunksFromPdf(self, file: str) -> list[str]:
        pass

    @abstractmethod
    def ExtractQaChunkFromCsv(self, file: str) -> AllQaResponseModel:
        pass

    @abstractmethod
    def ExtractChunksFromYtVideo(self, videoId: str, chunkSec: int) -> list[str]:
        pass


class BuildRagServiceImpl(ABC):

    @abstractmethod
    async def ExtractQaRagInstancesFromYtVideo(self, videoId: str):
        pass

    @abstractmethod
    async def ExtractQaRagInstancesFromPdf(self, file: str):
        pass

    @abstractmethod
    async def ExtractQaRagInstancesFromCsv(self, file: str):
        pass

    @abstractmethod
    async def ExtractLightRagFromPdf(self, file: str):
        pass
