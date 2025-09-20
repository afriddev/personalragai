from abc import ABC, abstractmethod
from clientservices.models import (
    EmbeddingRequestModel,
    EmbeddingResponseModel,
    RerankRequestModel,
    RerankResponseModel,
)


class EmbeddingImpl(ABC):

    @abstractmethod
    async def Embed(self, request: EmbeddingRequestModel) -> EmbeddingResponseModel:
        pass

    @abstractmethod
    async def RerankDocs(self, request: RerankRequestModel) -> RerankResponseModel:
        pass
