from abc import ABC, abstractmethod
from typing import List, Tuple


class RagUtilsImpl(ABC):

    @abstractmethod
    def ExtractTextFromXlsx(self, docPath: str) -> str:
        pass

    @abstractmethod
    def ExtractTextFromCsv(self, docPath: str) -> str:
        pass

    @abstractmethod
    def ExtractTextAndImagesFromPdf(self, docPath: str) -> Tuple[str, List[str]]:
        pass

    @abstractmethod
    def ExtractTextFromDoc(self, docPath: str) -> Tuple[str, List[str]]:
        pass


class ChunkUtils(ABC):

    @abstractmethod
    def ExtractChunksFromDoc(
        self, file: str, chunkSize: int, chunkOLSize: int | None = 0
    ) -> Tuple[list[str], list[str]]:
        pass
