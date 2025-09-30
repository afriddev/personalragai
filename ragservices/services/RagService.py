from typing import Any, cast
import json
from uuid import uuid4, UUID
from database import psqlDbClient
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from clientservices.services import Chat, Embedding
from clientservices.models import (
    ChatRequestModel,
    ChatMessageModel,
    EmbeddingRequestModel,
)
from clientservices.enums import CerebrasChatModelEnum, ChatMessageRoleEnum
from ragservices.implementations import (
    ExtractInstancesFromChunkServiceImpl,
    ExtractChunksFromDocServiceImpl,
    BuildRagServiceImpl,
)
from ragservices.models import (
    ChunkInstanceDataModel,
    ChunkInstanceRelationModel,
    ChunkInstanceEntityModel,
    ChunkInstanceClaimModel,
    AllQaResponseModel,
    ExtractQaFromChunkResponseModel,
    QaRagAllQuestionsModel,
    QaRagAllChunksModel,
    AllRelationsModel,
    AllChunksModel,
    AllClaimsModel,
    AllEntitiesModel,
    ExtractChunkInstanceResponseModel,
    LightRagResponseModel,
)
from ragservices.services.RagUtils import ChunkUtils, DocUtils, YoutubeUtils
from ragservices.enums import RagServiceResponseEnum
from ragservices.utils import (
    EXTARCT_INSTANCE_FROM_CHUNK_PROMPT,
    EXTRACT_QUESTIONS_FROM_CHUNK_PROMPT,
    CLEAN_YT_CHUNK_PROMPT,
)






cerebrasChat = Chat()
chunkUtils = ChunkUtils()
docUtils = DocUtils()
youtubeUtils = YoutubeUtils()
embeddingService = Embedding()

class ExtractInstanceFromChunkService(ExtractInstancesFromChunkServiceImpl):

    def __init__(self):
        self.retryLimit = 5

    async def ExtractInstancesFromChunk(
        self, chunk: str, messages: list[ChatMessageModel], retryLimit: int
    ) -> ExtractChunkInstanceResponseModel:
        if retryLimit > self.retryLimit:
            return ExtractChunkInstanceResponseModel(
                data=None, status=RagServiceResponseEnum.SERVER_ERROR
            )

        cerebrasChatResponse: Any = await cerebrasChat.Chat(
            modelParams=ChatRequestModel(
                topP=0.9,
                temperature=0.3,
                maxCompletionTokens=8000,
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
            print(e)
            print("Error occured while extracting realtions from chunk retrying ...")
            messages.append(
                ChatMessageModel(
                    role=ChatMessageRoleEnum.USER,
                    content="Please generate a valid json object",
                )
            )

            await self.ExtractInstancesFromChunk(
                chunk=chunk,
                messages=messages,
                retryLimit=retryLimit + 1,
            )

        chunkEntities = [
            ChunkInstanceEntityModel(
                id=entity.get("id"),
                entity=entity.get("entity"),
                entityDescription=entity.get("entity")
                + " "
                + entity.get("entityDescription"),
            )
            for entity in chatResponse.get("entities", [])
        ]
        chunkRelations = [
            ChunkInstanceRelationModel(
                id=relation.get("id"),
                sourceEntityId=relation.get("sourceEntityId"),
                targetEntityId=relation.get("targetEntityId"),
                relation=relation.get("relation"),
                relationDescription=relation.get("relationDescription"),
            )
            for relation in chatResponse.get("relations", [])
        ]
        chunkClaims = [
            ChunkInstanceClaimModel(
                id=claim.get("id"),
                entityId=claim.get("entityId"),
                claim=claim.get("claim"),
                claimDescription=claim.get("claimDescription"),
            )
            for claim in chatResponse.get("claims", [])
        ]

        response = ChunkInstanceDataModel(
            entities=chunkEntities,
            relations=chunkRelations,
            claims=chunkClaims,
            chunk=cast(str, chatResponse.get("chunk", chunk)),
        )

        return ExtractChunkInstanceResponseModel(
            data=response, status=RagServiceResponseEnum.SUCCESS
        )

    async def ExtractNodeSummary(
        self,
        messages: list[ChatMessageModel],
        retryLimit: int,
    ) -> str:
        if retryLimit > self.retryLimit:
            raise Exception("Exception while extracting questions from chunk")

        cerebrasChatResponse: Any = await cerebrasChat.Chat(
            modelParams=ChatRequestModel(
                topP=0.9,
                temperature=0.3,
                maxCompletionTokens=6000,
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

        except Exception:
            print("Error occured while extracting realtions from chunk retrying ...")
            messages.append(
                ChatMessageModel(
                    role=ChatMessageRoleEnum.USER,
                    content="Please generate a valid json object",
                )
            )

            await self.ExtractNodeSummary(messages=messages, retryLimit=retryLimit + 1)

        summary: str = chatResponse.get("summary", "")

        return summary

    async def ExtractQuestionsFromChunk(
        self,
        messages: list[ChatMessageModel],
        retryLimit: int,
    ) -> ExtractQaFromChunkResponseModel:
        if retryLimit > self.retryLimit:
            raise Exception("Exception while extracting questions from chunk")

        cerebrasChatResponse: Any = await cerebrasChat.Chat(
            modelParams=ChatRequestModel(
                topP=0.9,
                temperature=0.1,
                maxCompletionTokens=2000,
                model=CerebrasChatModelEnum.QWEN_235B,
                messages=messages,
                responseFormat={
                    "type": "object",
                    "properties": {
                        "questions": {"type": "array", "items": {"type": "string"}},
                        "chunk": {"type": "string"},
                    },
                    "required": ["chunk", "questions"],
                    "additionalProperties": False,
                },
                method="cerebras",
                stream=False,
            )
        )
        chatResponse: Any = {}
        try:

            chatResponse = json.loads(cerebrasChatResponse.content).get("response")

        except Exception:
            print("Error occured while extracting realtions from chunk retrying ...")
            messages.append(
                ChatMessageModel(
                    role=ChatMessageRoleEnum.USER,
                    content="Please generate a valid json object",
                )
            )

            await self.ExtractNodeSummary(messages=messages, retryLimit=retryLimit + 1)

        return ExtractQaFromChunkResponseModel(
            chunk=chatResponse.get("chunk", ""),
            questions=chatResponse.get("questions", []),
        )

    async def CleanYoutubeChunk(
        self,
        messages: list[ChatMessageModel],
        retryLimit: int,
    ) -> str:
        if retryLimit > self.retryLimit:
            raise Exception("Exception while extracting questions from chunk")

        cerebrasChatResponse: Any = await cerebrasChat.Chat(
            modelParams=ChatRequestModel(
                topP=0.9,
                temperature=0.1,
                maxCompletionTokens=2000,
                model=CerebrasChatModelEnum.QWEN_235B,
                messages=messages,
                responseFormat={
                    "type": "object",
                    "properties": {
                        "chunk": {"type": "string"},
                    },
                    "required": ["chunk"],
                    "additionalProperties": False,
                },
                method="cerebras",
                stream=False,
            )
        )
        chatResponse: Any = {}
        try:

            chatResponse = json.loads(cerebrasChatResponse.content).get("response")

        except Exception:
            print("Error occured while extracting realtions from chunk retrying ...")
            messages.append(
                ChatMessageModel(
                    role=ChatMessageRoleEnum.USER,
                    content="Please generate a valid json object",
                )
            )

            await self.ExtractNodeSummary(messages=messages, retryLimit=retryLimit + 1)

        return chatResponse.get("chunk", "")


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
            # matchedIndex = re.findall(r"<<[Ii][Mm][Aa][Gg][Ee]-([0-9]+)>>", chunk)
            # indeces = list(map(int, matchedIndex))
            # if len(indeces) == 0:
            #     processedChunk.append(chunk)
            # else:
            #     chunkText = chunk
            #     for index in indeces:
            #         imageUrl = await self.chunkUtils.UploadImageToBucket(
            #             base64Str=images[index - 1],
            #             extension="png",
            #             folder="images",
            #         )
            #         token = f"<<image-{index}>>"
            #         print(imageUrl)
            #         chunkText = chunkText.replace(token, f"![Image]({imageUrl})")
            #     processedChunk.append(chunkText)

        return processedChunk

    def ExtractQaChunkFromCsv(self, file: str) -> AllQaResponseModel:
        text, _ = self.docUtils.ExtractTextFromDoc(docPath=file)
        return self.chunkUtils.ExtractQaFromText(text=text)

    def ExtractChunksFromYtVideo(self, videoId: str, chunkSec: int) -> list[str]:
        text = self.youtubeUtils.ExtractText(videoId=videoId, chunkSec=chunkSec)
        response: list[str] = [
            f"{item.chunkText} for this [video link]({item.chunkUrl})" for item in text
        ]
        return response


class BuildRagService(BuildRagServiceImpl):

    def __init__(self):
        self.extractChunksFromDocService = ExtractChunksFromDocService()
        self.extractInstanceFromChunkService = ExtractInstanceFromChunkService()
        self.embedding = embeddingService

    async def ExtractQaRagInstancesFromYtVideo(self, videoId: str):

        chunks = self.extractChunksFromDocService.ExtractChunksFromYtVideo(
            chunkSec=400, videoId=videoId
        )

        chunkTexts: list[QaRagAllChunksModel] = []
        chunkQuestions: list[QaRagAllQuestionsModel] = []

        for chunk in chunks:
            cleanedChunk = await self.extractInstanceFromChunkService.CleanYoutubeChunk(
                retryLimit=0,
                messages=[
                    ChatMessageModel(
                        role=ChatMessageRoleEnum.SYSTEM,
                        content=CLEAN_YT_CHUNK_PROMPT,
                    ),
                    ChatMessageModel(
                        role=ChatMessageRoleEnum.USER,
                        content=chunk,
                    ),
                ],
            )
            messages: list[ChatMessageModel] = [
                ChatMessageModel(
                    role=ChatMessageRoleEnum.SYSTEM,
                    content=EXTRACT_QUESTIONS_FROM_CHUNK_PROMPT,
                ),
                ChatMessageModel(
                    role=ChatMessageRoleEnum.USER,
                    content=cleanedChunk,
                ),
            ]

            chunkGraphRagInfo = (
                await self.extractInstanceFromChunkService.ExtractQuestionsFromChunk(
                    messages=messages, retryLimit=0
                )
            )
            print(chunkGraphRagInfo)

            chunkId = uuid4()

            thisChunkText = QaRagAllChunksModel(
                id=chunkId, text=chunkGraphRagInfo.chunk
            )

            thisChunkQuestions = [
                QaRagAllQuestionsModel(id=uuid4(), chunkId=chunkId, text=question)
                for question in chunkGraphRagInfo.questions
            ]

            texts: list[str] = [chunkGraphRagInfo.chunk]
            texts.extend(chunkGraphRagInfo.questions)

            textVectors = await self.embedding.Embed(
                request=EmbeddingRequestModel(
                    model="baai/bge-m3",
                    texts=texts,
                    type="passage",
                )
            )

            if textVectors.data is not None:
                thisChunkText.embedding = textVectors.data[0].embedding

                qLen = len(chunkGraphRagInfo.questions)
                for i, item in enumerate(textVectors.data[1 : 1 + qLen]):
                    thisChunkQuestions[i].embedding = item.embedding

            chunkTexts.append(thisChunkText)
            chunkQuestions.extend(thisChunkQuestions)

    async def ExtractQaRagInstancesFromPdf(self, file: str):
        chunks = await self.extractChunksFromDocService.ExtractChunksFromPdf(file=file)
        chunkTexts: list[QaRagAllChunksModel] = []
        chunkQuestions: list[QaRagAllQuestionsModel] = []

        for chunk in chunks:
            messages: list[ChatMessageModel] = [
                ChatMessageModel(
                    role=ChatMessageRoleEnum.SYSTEM,
                    content=EXTRACT_QUESTIONS_FROM_CHUNK_PROMPT,
                ),
                ChatMessageModel(role=ChatMessageRoleEnum.USER, content=chunk),
            ]

            chunkGraphRagInfo = (
                await self.extractInstanceFromChunkService.ExtractQuestionsFromChunk(
                    messages=messages, retryLimit=0
                )
            )
            chunkId = uuid4()
            thisChunkText = QaRagAllChunksModel(
                id=chunkId, text=chunkGraphRagInfo.chunk
            )
            thisChunkQuestions = [
                QaRagAllQuestionsModel(id=uuid4(), chunkId=chunkId, text=rel)
                for rel in chunkGraphRagInfo.questions
            ]

            texts: list[str] = []
            texts.append(chunkGraphRagInfo.chunk)
            for _, claim in enumerate(chunkGraphRagInfo.questions):
                texts.append(claim)

            textVectors = await self.embedding.Embed(
                request=EmbeddingRequestModel(
                    model="baai/bge-m3",
                    texts=texts,
                    type="passage",
                )
            )
            c = 1
            cLen = len(chunkGraphRagInfo.questions)
            if textVectors.data is not None:
                thisChunkText.embedding = textVectors.data[0].embedding
                for cIndex, item in enumerate(textVectors.data[c : c + cLen]):
                    thisChunkQuestions[cIndex].embedding = item.embedding

            chunkTexts.append(thisChunkText)
            chunkQuestions.extend(thisChunkQuestions)

        finalChuks = [(str(chunk.id), chunk.text) for chunk in chunkTexts]

        async with psqlDbClient.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO chunks (id, text) VALUES ($1, $2)",
                finalChuks,
            )

            await conn.executemany(
                "INSERT INTO claims (id,  chunk_id, node_id,embedding) VALUES ($1, $2, $3, $4)",
                [
                    (
                        str(claim.id),
                        str(claim.chunkId),
                        None,
                        claim.embedding,
                    )
                    for claim in chunkQuestions
                ],
            )

    async def ExtractQaRagInstancesFromCsv(self, file: str):
        qa = self.extractChunksFromDocService.ExtractQaChunkFromCsv(file=file)

        chunks: list[QaRagAllChunksModel] = []
        questions: list[QaRagAllQuestionsModel] = []
        qaBatchSize = 10

        for index in range(0, len(qa.questions), qaBatchSize):
            queVecRes = await self.embedding.Embed(
                request=EmbeddingRequestModel(
                    model="baai/bge-m3",
                    texts=qa.questions[index : index + qaBatchSize],
                    type="passage",
                )
            )

            if queVecRes.data is not None:
                for idx, q in enumerate(queVecRes.data):
                    chunkId = uuid4()

                    chunks.append(
                        QaRagAllChunksModel(id=chunkId, text=qa.answers[index + idx])
                    )
                    questions.append(
                        QaRagAllQuestionsModel(
                            id=uuid4(),
                            chunkId=chunkId,
                            embedding=q.embedding,
                            text=qa.questions[index + idx],
                        )
                    )

    async def ExtractLightRagFromPdf(self, file: str):
        orginalChunks = await self.extractChunksFromDocService.ExtractChunksFromPdf(
            file
        )

        allChunks: list[AllChunksModel] = []
        allClaims: list[AllClaimsModel] = []
        allEntities: list[AllEntitiesModel] = []
        allRelations: list[AllRelationsModel] = []
        matchedNodeIds: list[list[UUID]] = []

        # Extracting Entities, Relations and Claims from Chunks
        for index, chunk in enumerate(orginalChunks):
            # Extracting chunk instances
            chunkResponse = await self.extractInstanceFromChunkService.ExtractInstancesFromChunk(
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
                retryLimit=0,
            )

            if (
                chunkResponse.status != RagServiceResponseEnum.SUCCESS
                or chunkResponse.data is None
            ):
                continue

            chunkInstance: ChunkInstanceDataModel = chunkResponse.data

            # Storing the extracted instances
            tempChunkId = uuid4()
            allChunks.append(AllChunksModel(id=tempChunkId, text=chunkInstance.chunk))
            tempDictEntityIdToUuid: dict[int, UUID] = {}
            # Storing Entities, Relations and Claims
            for entity in chunkInstance.entities:
                tempEntityId = uuid4()
                tempDictEntityIdToUuid[entity.id] = tempEntityId
                allEntities.append(
                    AllEntitiesModel(
                        chunkId=tempChunkId,
                        id=tempEntityId,
                        entity=entity.entity,
                        entityDescription=entity.entityDescription,
                    )
                )

            for relation in chunkInstance.relations:
                allRelations.extend(
                    [
                        AllRelationsModel(
                            id=uuid4(),
                            relation=relation.relation,
                            relationDescription=relation.relationDescription,
                            sourceEntityId=tempDictEntityIdToUuid.get(
                                relation.sourceEntityId, None
                            ),
                            targetEntityId=tempDictEntityIdToUuid.get(
                                relation.targetEntityId, None
                            ),
                        )
                    ]
                )

            for claim in chunkInstance.claims:
                allClaims.extend(
                    [
                        AllClaimsModel(
                            id=uuid4(),
                            entityId=tempDictEntityIdToUuid.get(claim.entityId, None),
                            claimDescription=claim.claimDescription,
                        )
                    ]
                )

            if index == 5:
                break

            print(f"{index + 1} of {len(orginalChunks)}")

        # Indexing Entities using BM25
        tempEntitiesBm25Docs: list[Document] = []
        for index, entity in enumerate(allEntities):
            tempEntitiesBm25Docs.append(
                Document(
                    page_content=entity.entity,
                    metadata={"index": index},
                )
            )
        entitiesTexts = [d.page_content for d in tempEntitiesBm25Docs]
        entitiesTokenized = [t.split() for t in entitiesTexts]
        bm25 = BM25Okapi(entitiesTokenized)

        # Merging Entities using BM25
        tempEntities = allEntities.copy()
        for index, entity in enumerate(tempEntities):

            matchedNodes: list[UUID] = []
            mergedNodeIndeces: list[int] = []

            scores = cast(Any, bm25).get_scores(entity.entity.split())
            ranked = sorted(
                zip(tempEntitiesBm25Docs, scores), key=lambda x: x[1], reverse=True
            )

            for doc, score in ranked:
                if score == 0.0:
                    break
                else:

                    docIndex: int = cast(Any, doc).metadata.get("index")

                    if docIndex not in mergedNodeIndeces:
                        mergedNodeIndeces.append(docIndex)
                        matchedNodes.append(tempEntities[docIndex].id)  # type: ignore

            # Removeing the merged nodes from tempEntities
            mergedNodeIndeces = sorted(set(mergedNodeIndeces), reverse=True)
            for docIndex in mergedNodeIndeces:
                if 0 <= docIndex < len(tempEntities):
                    tempEntities.pop(docIndex)
            # Recreating the BM25 index
            tempDoc: list[Document] = []
            for index, entity in enumerate(tempEntities):
                tempDoc.append(
                    Document(
                        page_content=entity.entity,
                        metadata={"index": index},
                    )
                )
            tempTexts = [d.page_content for d in tempDoc]
            tempTokens = [t.split() for t in tempTexts]
            tempEntitiesBm25Docs = tempDoc
            bm25 = BM25Okapi(tempTokens)

            # Storing the matched node ids for each entity
            if len(matchedNodes) > 0:
                matchedNodeIds.append(matchedNodes)

            print(f"Node {index + 1} of {len(allEntities)}")

        return JSONResponse(
            content=jsonable_encoder(
                {
                    "chunks": allChunks,
                    "entities": allEntities,
                    "relations": allRelations,
                    "claims": allClaims,
                    "matchedNodeIds": matchedNodeIds,
                }
            )
        )


# summary = await self.extractInstanceFromChunkService.ExtractNodeSummary(
#     retryLimit=0,
#     messages=[
#         ChatMessageModel(
#             role=ChatMessageRoleEnum.SYSTEM,
#             content=EXTRACT_NODE_SUMMARY_PROMPT,
#         ),
#         ChatMessageModel(
#             role=ChatMessageRoleEnum.USER,
#             content=f"""
#                 Entity Descriptions: {nodeEntities}
#                 Relations: {nodeRelations}
#                 Claims: {nodeClaims}
#                 Please generate a long  summary for the node.
#             """,
#         ),
#     ],
# )
