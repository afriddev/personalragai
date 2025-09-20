from pydantic import BaseModel


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
