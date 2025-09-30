from pydantic import BaseModel
from uuid import UUID
from ragservices.enums import RagServiceResponseEnum


class ChunkInstanceEntityModel(BaseModel):
    id: int
    entity: str
    entityDescription: str


class ChunkInstanceRelationModel(BaseModel):
    id: int
    sourceEntityId: int
    targetEntityId: int
    relation: str
    relationDescription: str


class ChunkInstanceClaimModel(BaseModel):
    id: int
    entityId: int
    claim: str
    claimDescription: str


class ChunkInstanceDataModel(BaseModel):
    entities: list[ChunkInstanceEntityModel]
    relations: list[ChunkInstanceRelationModel]
    claims: list[ChunkInstanceClaimModel]
    chunk: str


class ExtractChunkInstanceResponseModel(BaseModel):
    status: RagServiceResponseEnum = RagServiceResponseEnum.SUCCESS
    data: ChunkInstanceDataModel | None = None


class QaRagAllChunksModel(BaseModel):
    id: UUID
    text: str
    embedding: list[float] | None = None


class QaRagAllQuestionsModel(BaseModel):
    id: UUID
    chunkId: UUID
    text: str


class ExtractQaFromChunkResponseModel(BaseModel):
    questions: list[str]
    chunk: str


class AllChunksModel(BaseModel):
    id: UUID
    text: str


class AllEntitiesModel(BaseModel):
    id: UUID
    chunkId: UUID
    entity: str
    entityDescription: str


class AllRelationsModel(BaseModel):
    id: UUID
    sourceEntityId: UUID | None = None
    targetEntityId: UUID | None = None
    relation: str
    relationDescription: str


class AllClaimsModel(BaseModel):
    id: UUID
    entityId: UUID | None = None
    claimDescription: str


class LightRagResponseModel(BaseModel):
    entities: list[AllEntitiesModel]
    relations: list[AllRelationsModel]
    claims: list[AllClaimsModel]
    chunks: list[AllChunksModel]
    matchedNodeIds: list[list[UUID]]
