from pydantic import BaseModel
from uuid import UUID, uuid4


class ChunkEntityModel(BaseModel):
    id: int
    entity: str
    entityDescription: str


class ChunkRelationModel(BaseModel):
    id: int
    sourceEntityId: int
    targetEntityId: int
    relation: str
    relationDescription: str


class ChunkClaimModel(BaseModel):
    id: int
    entityId: int
    claim: str
    claimDescription: str


class ChunkInstanceModel(BaseModel):
    entities: list[ChunkEntityModel]
    relations: list[ChunkRelationModel]
    claims: list[ChunkClaimModel]
    chunk: str


class ChunkEntityNodeModel(BaseModel):
    id: UUID = uuid4()
    entityEmbedding: list[float]
    entityDescription: str
    relations: list[str]
    claims: list[str]
    chunkId: UUID
    nodeId: UUID | None = None
    chunkIndex: int


class ChunkModel(BaseModel):
    chunk: str
    id: UUID


class ChunkNodeModel(BaseModel):
    nodeId: UUID
    nodeSummary: str


class BuildRagProcessFromPdfResponseModel(BaseModel):
    allChunks: list[ChunkModel]
    allEntities: list[ChunkEntityNodeModel]
    allNodes: list[ChunkNodeModel]
