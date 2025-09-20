from pydantic import BaseModel


class EmbeddingRequestModel(BaseModel):
    texts: list[str]
    model: str = "nvidia/nv-embedqa-mistral-7b-v2"
    dimensions: int = 1536


class EmbeddingDataModel(BaseModel):
    embedding: list[float]
    index: int


class EmbeddingUsageModel(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponseModel(BaseModel):
    data: list[EmbeddingDataModel] | None = None
    model: str = "nvidia/nv-embedqa-mistral-7b-v2"
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
