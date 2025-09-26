import time
import re
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
    ChunkNodeModel,
    ChunkModel,
    BuildRagProcessFromPdfResponseModel,
)
from ragservices.services.RagUtils import ChunkUtils, DocUtils, YoutubeUtils
from clientservices.models import (
    ChatRequestModel,
    ChatMessageModel,
    EmbeddingRequestModel,
    FindTopKresultsFromVectorsRequestModel,
    FindTopKresultsFromVectorsResponseModel,
)
from clientservices.enums import CerebrasChatModelEnum, ChatMessageRoleEnum
from ragservices.utils import (
    EXTARCT_INSTANCE_FROM_CHUNK_PROMPT,
    EXTRACT_NODE_SUMMARY_PROMPT,
)
from typing import Any, cast
import json
from uuid import uuid4
from database import psqlDbClient

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
                topP=1.0,
                temperature=0.5,
                maxCompletionTokens=5000,
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
                entityDescription=entity.get("entity")
                + " "
                + entity.get("entityDescription"),
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

    async def ExtractNodeSummary(
        self,
        messages: list[ChatMessageModel],
        retryLimit: int,
    ) -> str:
        if retryLimit > self.retryLimit:
            raise Exception("Exception while extracting questions from chunk")

        cerebrasChatResponse: Any = await cerebrasChat.Chat(
            modelParams=ChatRequestModel(
                topP=1.0,
                temperature=0.5,
                maxCompletionTokens=2000,
                model=CerebrasChatModelEnum.QWEN_235B,
                messages=messages,
                responseFormat={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
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

            await self.ExtractNodeSummary(messages=messages, retryLimit=retryLimit + 1)

        summary: str = chatResponse.get("summary", "")

        return summary


class ExtractChunksFromDocService(ExtractChunksFromDocServiceImpl):

    def __init__(self):
        self.chunkUtils = chunkUtils
        self.docUtils = docUtils
        self.youtubeUtils = youtubeUtils

    async def ExtractChunksFromPdf(self, file: str) -> list[str]:
        chunks, images = self.chunkUtils.ExtractChunksFromDoc(
            file=file, chunkOLSize=100, chunkSize=2000
        )
        processedChunk: list[str] = []

        for chunk in chunks:
            processedChunk.append(chunk)

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

    async def BuildRagFromPdf(self, file: str) -> BuildRagProcessFromPdfResponseModel:
        orginalChunks = await self.extractChunksFromDocService.ExtractChunksFromPdf(
            file
        )
        allEntitys: list[ChunkEntityNodeModel] = []
        allNodes: list[ChunkNodeModel] = []
        allChunks: list[ChunkModel] = []
        for index, chunk in enumerate(orginalChunks):
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
            chunkId = uuid4()
            allChunks.append(ChunkModel(id=chunkId, chunk=chunkInstance.chunk))

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
                print(index)
                time.sleep(1)
                entityEmnbeddingResponse = await self.embedding.Embed(
                    request=EmbeddingRequestModel(
                        model="baai/bge-m3",
                        texts=[entity.entityDescription],
                        type="passage",
                    )
                )
                time.sleep(1)
                allEntitys.append(
                    ChunkEntityNodeModel(
                        id=uuid4(),
                        entityDescription=entity.entityDescription,
                        entityEmbedding=cast(Any, entityEmnbeddingResponse)
                        .data[0]
                        .embedding,
                        relations=entityRelations,
                        claims=entityClaims,
                        chunkId=chunkId,
                        chunkIndex=index,
                    )
                )
            if index == 2:
                break

        allEntitiesEmbeddings = [entity.entityEmbedding for entity in allEntitys]

        for index, entityEmbedding in enumerate(allEntitiesEmbeddings):

            allMergedNodes: FindTopKresultsFromVectorsResponseModel = (
                self.embedding.FindTopKResultsFromVectors(
                    request=FindTopKresultsFromVectorsRequestModel(
                        topK=10,
                        queryVector=entityEmbedding,
                        sourceVectors=[
                            e for e in allEntitiesEmbeddings if e != entityEmbedding
                        ],
                    )
                )
            )

            nodeEntities: list[str] = []
            nodeRelations: list[str] = []
            nodeClaims: list[str] = []
            mergedNodeIndeces: list[int] = []

            if (
                allMergedNodes.distances is not None
                and allMergedNodes.indeces is not None
            ):
                mergedNodeIndeces = [
                    allMergedNodes.indeces[i]
                    for i, d in enumerate(allMergedNodes.distances)
                    if d < 0.5
                ]

                if len(mergedNodeIndeces) > 0:

                    for nodeIndex in mergedNodeIndeces:

                        nodeEntities.append(allEntitys[nodeIndex].entityDescription)
                        nodeRelations = nodeRelations + allEntitys[nodeIndex].relations
                        nodeClaims = nodeClaims + allEntitys[nodeIndex].claims

            if len(nodeEntities) > 0:

                summary = await self.extractInstanceFromChunkService.ExtractNodeSummary(
                    retryLimit=3,
                    messages=[
                        ChatMessageModel(
                            role=ChatMessageRoleEnum.SYSTEM,
                            content=EXTRACT_NODE_SUMMARY_PROMPT,
                        ),
                        ChatMessageModel(
                            role=ChatMessageRoleEnum.USER,
                            content=f"""
                                Entity Descriptions: {nodeEntities}
                                Relations: {nodeRelations}
                                Claims: {nodeClaims}
                                Please generate a concise summary for the node.
                            """,
                        ),
                    ],
                )
                nodeId = uuid4()
                allNodes.append(
                    ChunkNodeModel(
                        nodeId=nodeId,
                        nodeSummary=summary,
                    )
                )
                for entityIndex in mergedNodeIndeces:
                    allEntitys[entityIndex].nodeId = nodeId

        finalChuks = [(str(chunk.id), chunk.chunk) for chunk in allChunks]

        async with psqlDbClient.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO chunks (id, text) VALUES ($1, $2)",
                finalChuks,
            )

            await conn.executemany(
                "INSERT INTO nodes (id, summary) VALUES ($1, $2)",
                [(str(node.nodeId), node.nodeSummary) for node in allNodes],
            )
            await conn.executemany(
                "INSERT INTO entities (id,  chunk_id, node_id,embedding) VALUES ($1, $2, $3, $4)",
                [
                    (
                        str(entity.id),
                        str(entity.chunkId),
                        str(entity.nodeId) if entity.nodeId else None,
                        entity.entityEmbedding,
                    )
                    for entity in allEntitys
                ],
            )

        return BuildRagProcessFromPdfResponseModel(
            allChunks=allChunks, allEntities=allEntitys, allNodes=allNodes
        )
