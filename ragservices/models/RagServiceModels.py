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
    entity: str
    entityEmbedding: list[float]
    entityDescription: str
    relations: list[str]
    relationEmbeddings: list[list[float]]
    claims: list[str]
    claimEmbeddings: list[list[float]]
    chunk: str
    nodeId: UUID | None = None


class ChunkNodeModel(BaseModel):
    id: UUID = uuid4()
    nodeSummary: str
    chunks: list[str]
    nodeRelations: list[str]
    nodeEntities: list[str]
    nodeClaims: list[str]
