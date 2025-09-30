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
    ChunkEntityNodeModel,
    ChunkNodeModel,
    ChunkModel,
    ExtractQuestionsFromChunkResponseModel,
    QaRagQuestionModel,
    QaRagChunkTextsModel,
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
    EXTRACT_QUESTIONS_FROM_CHUNK_PROMPT,
    CLEAN_YT_CHUNK_PROMPT,
)
from typing import Any, cast
import json
from uuid import uuid4
from database import psqlDbClient
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


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
    ) -> ChunkInstanceModel:
        if retryLimit > self.retryLimit:
            raise Exception("Exception while extracting questions from chunk")

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
                topP=0.9,
                temperature=0.1,
                maxCompletionTokens=3000,
                model=CerebrasChatModelEnum.LLAMA_70B,
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
    ) -> ExtractQuestionsFromChunkResponseModel:
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

        return ExtractQuestionsFromChunkResponseModel(
            chunk=chatResponse.get("chunk", ""),
            questions=chatResponse.get("questions", []),
        )

    async def CleanYoutubeChunks(
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

    async def BuildQaRagFromYtVideo(self, videoId: str):

        chunks = self.extractChunksFromDocService.ExtractChunksFromYtVideo(
            chunkSec=400, videoId=videoId
        )

        chunkTexts: list[QaRagChunkTextsModel] = []
        chunkQuestions: list[QaRagQuestionModel] = []

        for chunk in chunks:
            cleanedChunk = (
                await self.extractInstanceFromChunkService.CleanYoutubeChunks(
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

            thisChunkText = QaRagChunkTextsModel(
                id=chunkId, text=chunkGraphRagInfo.chunk
            )

            thisChunkQuestions = [
                QaRagQuestionModel(id=uuid4(), chunkId=chunkId, text=question)
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

    async def BuildQaRagFromPdf(self, file: str):
        chunks = await self.extractChunksFromDocService.ExtractChunksFromPdf(file=file)
        chunkTexts: list[QaRagChunkTextsModel] = []
        chunkQuestions: list[QaRagQuestionModel] = []

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
            thisChunkText = QaRagChunkTextsModel(
                id=chunkId, text=chunkGraphRagInfo.chunk
            )
            thisChunkQuestions = [
                QaRagQuestionModel(id=uuid4(), chunkId=chunkId, text=rel)
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

    async def BuildQaRagFromCsv(self, file: str):
        qa = self.extractChunksFromDocService.ExtractQaChunkFromCsv(file=file)

        chunks: list[QaRagChunkTextsModel] = []
        questions: list[QaRagQuestionModel] = []
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
                        QaRagChunkTextsModel(id=chunkId, text=qa.answers[index + idx])
                    )
                    questions.append(
                        QaRagQuestionModel(
                            id=uuid4(),
                            chunkId=chunkId,
                            embedding=q.embedding,
                            text=qa.questions[index + idx],
                        )
                    )

    async def BuildGraphRagFromPdf(self, file: str):
        orginalChunks = await self.extractChunksFromDocService.ExtractChunksFromPdf(
            file
        )
        allEntitys: list[ChunkEntityNodeModel] = []
        allNodes: list[ChunkNodeModel] = []
        allChunks: list[ChunkModel] = []
        for index, chunk in enumerate(orginalChunks):

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
                    retryLimit=0,
                )
            )
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

                allEntitys.append(
                    ChunkEntityNodeModel(
                        id=uuid4(),
                        entity=entity.entity,
                        entityDescription=entity.entityDescription,
                        relations=entityRelations,
                        claims=entityClaims,
                        chunkId=chunkId,
                        chunkIndex=index,
                    )
                )
            if index == 5:
                break
            print(f"{index + 1} of {len(orginalChunks)}")

        allEntitiesBm25Documents: list[Document] = []
        for index, entity in enumerate(allEntitys):
            allEntitiesBm25Documents.append(
                Document(
                    page_content=entity.entity,
                    metadata={"index": index},
                )
            )
        entitiesTexts = [d.page_content for d in allEntitiesBm25Documents]
        entitiesTokenized = [t.split() for t in entitiesTexts]
        print(entitiesTexts)

        bm25 = BM25Okapi(entitiesTokenized)

        for index, entity in enumerate(allEntitys):

            nodeEntities: list[str] = []
            nodeRelations: list[str] = []
            nodeClaims: list[str] = []

            mergedNodeIndeces: list[int] = []

            scores = cast(Any, bm25).get_scores(entity.entity.split())
            ranked = sorted(
                zip(allEntitiesBm25Documents, scores), key=lambda x: x[1], reverse=True
            )

            for doc, score in ranked:
                if score == 0.0:
                    break
                else:

                    docIndex: int = cast(Any, doc).metadata.get("index")

                    if docIndex not in mergedNodeIndeces:
                        mergedNodeIndeces.append(docIndex)
                        nodeEntities.append(allEntitys[docIndex].entity)
                        nodeRelations.extend(allEntitys[docIndex].relations)
                        nodeClaims.extend(allEntitys[docIndex].claims)

            if len(nodeEntities) > 0:

                summary = await self.extractInstanceFromChunkService.ExtractNodeSummary(
                    retryLimit=0,
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
