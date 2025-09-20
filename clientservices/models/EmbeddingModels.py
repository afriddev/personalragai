from pydantic import BaseModel


class EmbeddingRequestModel(BaseModel):
    texts: list[str]
    model: str = "baai/bge-m3"
    # model: str = "nvidia/nv-embedqa-mistral-7b-v2"


class EmbeddingDataModel(BaseModel):
    embedding: list[float]
    index: int


class EmbeddingUsageModel(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponseModel(BaseModel):
    data: list[EmbeddingDataModel] | None = None
    model: str = "baai/bge-m3"
    # model: str = "nvidia/nv-embedqa-mistral-7b-v2"
    usage: EmbeddingUsageModel | None = None


class RerankRequestModel(BaseModel):
    query: str
    docs: list[str]
    model: str = "nvidia/nv-rerankqa-mistral-4b-v3"


class RerankResultModel(BaseModel):
    score: float
    doc: str


class RerankResponseModel(BaseModel):
    results: list[RerankResultModel]


class FindTopKresultsFromVectorsRequestModel(BaseModel):
    sourceVectors: list[list[float]]
    queryVector: list[float]
    topK: int = 20


class FindTopKresultsFromVectorsResponseModel(BaseModel):
    distances: list[float] | None = None
    indeces: list[int] | None = None
