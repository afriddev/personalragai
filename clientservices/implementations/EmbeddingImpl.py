from abc import ABC, abstractmethod
from clientservices.models import EmbeddingRequestModel, EmbeddingResponseModel


class EmbeddingImpl(ABC):

    @abstractmethod
    async def embed(self, request: EmbeddingRequestModel) -> EmbeddingResponseModel:
        pass
