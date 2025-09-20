from ragservices.implementations import ChunkInstanceImpl
from clientservices.services import Chat
from ragservices.models import (
    ChunkInstanceModel,
    ChunkRelationModel,
    ChunkEntityModel,
    ChunkClaimModel,
)
from clientservices.models import ChatRequestModel, ChatMessageModel
from clientservices.enums import CerebrasChatModelEnum, ChatMessageRoleEnum
from typing import Any, cast
import json


cerebrasChat = Chat()


class ChunkInstanceService(ChunkInstanceImpl):

    def __init__(self):
        self.retryLimit = 3

    async def ExtractChunkInstance(
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

            await self.ExtractChunkInstance(
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
        print(response)
        return response
