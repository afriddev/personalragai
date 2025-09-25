import time
from sympy import re
from ragservices.implementations import (
    ExtractInstancesFromChunkServiceImpl,
    ExtractChunksFromDocServiceImpl,
    BuildRagServiceImpl,
)
from clientservices.services import Chat, Embedding
from ragservices.models import (
    ChunkInstanceModel,
    ChunkRelationModel,
    ChunkEntityModel,
    ChunkClaimModel,
    AllQaResponseModel,
    ExtractTextFromYtResponseModel,
    ChunkEntityNodeModel,
)
from ragservices.services.RagUtils import ChunkUtils, DocUtils, YoutubeUtils
from clientservices.models import (
    ChatRequestModel,
    ChatMessageModel,
    EmbeddingRequestModel,
)
from clientservices.enums import CerebrasChatModelEnum, ChatMessageRoleEnum
from ragservices.utils import EXTARCT_INSTANCE_FROM_CHUNK_PROMPT
from typing import Any, cast
import json
import re
from uuid import uuid4

cerebrasChat = Chat()
chunkUtils = ChunkUtils()
docUtils = DocUtils()
youtubeUtils = YoutubeUtils()
embeddingService = Embedding()


class ExtractInstanceFromChunkService(ExtractInstancesFromChunkServiceImpl):

    def __init__(self):
        self.retryLimit = 3

    async def ExtractInstancesFromChunk(
        self, chunk: str, messages: list[ChatMessageModel], retryLimit: int
    ) -> ChunkInstanceModel:
        if retryLimit > self.retryLimit:
            raise Exception("Exception while extracting questions from chunk")

        cerebrasChatResponse: Any = await cerebrasChat.Chat(
            modelParams=ChatRequestModel(
                model=CerebrasChatModelEnum.QWEN_235B,
                messages=messages,
                responseFormat={
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "entity": {"type": "string"},
                                    "entityDescription": {"type": "string"},
                                },
                                "required": ["id", "entity", "entityDescription"],
                                "additionalProperties": False,
                            },
                        },
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "sourceEntityId": {"type": "integer"},
                                    "targetEntityId": {"type": "integer"},
                                    "relation": {"type": "string"},
                                    "relationDescription": {"type": "string"},
                                },
                                "required": [
                                    "id",
                                    "sourceEntityId",
                                    "targetEntityId",
                                    "relation",
                                    "relationDescription",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "entityId": {"type": "integer"},
                                    "claim": {"type": "string"},
                                    "claimDescription": {"type": "string"},
                                },
                                "required": [
                                    "id",
                                    "entityId",
                                    "claim",
                                    "claimDescription",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "chunk": {"type": "string"},
                    },
                    "required": ["entities", "relations", "claims", "chunk"],
                    "additionalProperties": False,
                },
                method="cerebras",
                stream=False,
            )
        )
        chatResponse: Any = {}
        try:

            chatResponse = json.loads(cerebrasChatResponse.content).get("response")

        except Exception as e:
            print("Error occured while extracting realtions from chunk retrying ...")
            print(e)
            messages.append(
                ChatMessageModel(
                    role=ChatMessageRoleEnum.USER,
                    content="Please generate a valid json object",
                )
            )
            time.sleep(1)

            await self.ExtractInstancesFromChunk(
                chunk=chunk,
                messages=messages,
                retryLimit=retryLimit + 1,
            )

        chunkEntities = [
            ChunkEntityModel(
                id=entity.get("id"),
                entity=entity.get("entity"),
                entityDescription=entity.get("entityDescription"),
            )
            for entity in chatResponse.get("entities", [])
        ]
        chunkRelations = [
            ChunkRelationModel(
                id=relation.get("id"),
                sourceEntityId=relation.get("sourceEntityId"),
                targetEntityId=relation.get("targetEntityId"),
                relation=relation.get("relation"),
                relationDescription=relation.get("relationDescription"),
            )
            for relation in chatResponse.get("relations", [])
        ]
        chunkClaims = [
            ChunkClaimModel(
                id=claim.get("id"),
                entityId=claim.get("entityId"),
                claim=claim.get("claim"),
                claimDescription=claim.get("claimDescription"),
            )
            for claim in chatResponse.get("claims", [])
        ]

        response = ChunkInstanceModel(
            entities=chunkEntities,
            relations=chunkRelations,
            claims=chunkClaims,
            chunk=cast(str, chatResponse.get("chunk", chunk)),
        )
        return response


class ExtractChunksFromDocService(ExtractChunksFromDocServiceImpl):

    def __init__(self):
        self.chunkUtils = chunkUtils
        self.docUtils = docUtils
        self.youtubeUtils = youtubeUtils

    async def ExtractChunksFromPdf(self, file: str) -> list[str]:
        chunks, images = self.chunkUtils.ExtractChunksFromDoc(
            file=file, chunkOLSize=100, chunkSize=1200
        )
        processedChunk: list[str] = []

        for chunk in chunks:

            matchedIndex = re.findall(r"<<[Ii][Mm][Aa][Gg][Ee]-([0-9]+)>>", chunk)
            indeces = list(map(int, matchedIndex))
            if len(indeces) == 0:
                processedChunk.append(chunk)
            else:
                chunkText = chunk
                for index in indeces:
                    imageUrl = await self.chunkUtils.UploadImageToBucket(
                        base64Str=images[index - 1],
                        extension="png",
                        folder="images",
                    )
                    token = f"<<image-{index}>>"
                    chunkText = chunkText.replace(token, f"![Image]({imageUrl})")
                processedChunk.append(chunkText)

        return processedChunk

    def ExtractQaChunkFromCsv(self, file: str) -> AllQaResponseModel:
        text, _ = self.docUtils.ExtractTextFromDoc(docPath=file)
        return self.chunkUtils.ExtractQaFromText(text=text)

    def ExtractChunksFromYtVideo(
        self, videoId: str
    ) -> list[ExtractTextFromYtResponseModel]:
        return self.youtubeUtils.ExtractText(videoId=videoId, chunkSec=200)


class BuildRagService(BuildRagServiceImpl):

    def __init__(self):
        self.extractChunksFromDocService = ExtractChunksFromDocService()
        self.extractInstanceFromChunkService = ExtractInstanceFromChunkService()
        self.embedding = embeddingService

    async def BuildRagFromPdf(self, file: str):
        orginalChunks = await self.extractChunksFromDocService.ExtractChunksFromPdf(
            file
        )
        allNodes: list[ChunkEntityNodeModel] = []
        for chunk in orginalChunks:
            time.sleep(1)
            chunkInstance: ChunkInstanceModel = (
                await self.extractInstanceFromChunkService.ExtractInstancesFromChunk(
                    chunk=chunk,
                    messages=[
                        ChatMessageModel(
                            role=ChatMessageRoleEnum.SYSTEM,
                            content=EXTARCT_INSTANCE_FROM_CHUNK_PROMPT,
                        ),
                        ChatMessageModel(
                            role=ChatMessageRoleEnum.USER,
                            content=f"""
                                {chunk}
                            """,
                        ),
                    ],
                    retryLimit=3,
                )
            )
            time.sleep(1)
            for entity in chunkInstance.entities:
                entityRelations = [
                    relation.relation
                    for relation in chunkInstance.relations
                    if relation.sourceEntityId == entity.id
                    or relation.targetEntityId == entity.id
                ]
                entityClaims = [
                    claim.claim
                    for claim in chunkInstance.claims
                    if claim.entityId == entity.id
                ]
                claims = entityClaims
                claims.append(entity.entityDescription)
                entityEmnbeddingResponse = await self.embedding.Embed(
                    request=EmbeddingRequestModel(
                        model="baai/bge-m3",
                        texts=claims,
                        type="entity",
                    )
                )
                time.sleep(1)
                relationEmnbeddingResponse: Any = []
                if len(entityRelations) > 0:
                    relationEmnbeddingResponse = await self.embedding.Embed(
                        request=EmbeddingRequestModel(
                            model="baai/bge-m3",
                            texts=entityRelations,
                            type="entity",
                        )
                    )
                time.sleep(1)
                allNodes.append(
                    ChunkEntityNodeModel(
                        id=uuid4(),
                        entity=entity.entity,
                        entityDescription=entity.entityDescription,
                        entityEmbedding=cast(Any, entityEmnbeddingResponse)
                        .data[len(claims) - 1]
                        .embedding,
                        relations=entityRelations,
                        relationEmbeddings=[
                            resp.embedding for resp in relationEmnbeddingResponse.data
                        ],
                        claims=entityClaims,
                        claimEmbeddings=[
                            resp.embedding
                            for resp in cast(Any, entityEmnbeddingResponse).data[
                                0, len(claims) - 1
                            ]
                        ],
                        chunk=chunkInstance.chunk,
                    )
                )
                time.sleep(1)
