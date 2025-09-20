from abc import ABC, abstractmethod
from typing import List, Tuple
from clientservices.models import ExtarctQaResponseModel, ExtractTextFromYtResponseModel


class DocUtilsImpl(ABC):

    @abstractmethod
    def ExtractTextFromCsv(self, docPath: str) -> str:
        pass

    @abstractmethod
    def ExtractTextAndImagesFromPdf(self, docPath: str) -> Tuple[str, List[str]]:
        pass

    @abstractmethod
    def ExtractTextFromDoc(self, docPath: str) -> Tuple[str, List[str]]:
        pass


class ChunkUtilsImpl(ABC):

    @abstractmethod
    def ExtractChunksFromDoc(
        self, file: str, chunkSize: int, chunkOLSize: int | None = 0
    ) -> Tuple[list[str], list[str]]:
        pass

    @abstractmethod
    def ExtarctQaFromText(self, text: str) -> ExtarctQaResponseModel:
        pass


class YoutubeUtilsImpl(ABC):

    @abstractmethod
    def ExtractText(
        self, videoId: str, chunkSec: int = 30
    ) -> list[ExtractTextFromYtResponseModel]:
        pass
