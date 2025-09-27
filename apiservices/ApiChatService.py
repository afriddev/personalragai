from apiimplementations import ApiChatImpl
from apimodels import (
    ApiChatRequestModel,
)
from clientservices.services import Chat, Embedding
from clientservices.models import (
    ChatMessageModel,
    ChatRequestModel,
    EmbeddingRequestModel,
)
from clientservices.enums import (
    ChatMessageRoleEnum,
    OpenaiChatModelsEnum,
    CerebrasChatModelEnum,
    GroqChatModelsEnum,
)
from typing import Any, cast
from fastapi.responses import StreamingResponse
from database import psqlDbClient
import json

chatService = Chat()
embeddingService = Embedding()


class ApiChatService(ApiChatImpl):

    def __init__(self):
        self.embeddingService = embeddingService
        self.db = psqlDbClient

    def GetModel(
        self, request: ApiChatRequestModel
    ) -> OpenaiChatModelsEnum | CerebrasChatModelEnum | GroqChatModelsEnum:

        if request.useWebSearch:
            return GroqChatModelsEnum.GROQ_COMPOUND

        elif request.useFlash == False:
            if request.useCode and request.useDeepResearch:
                return OpenaiChatModelsEnum.LLAMA_235B_130K
            elif request.useCode and request.useDeepResearch == False:
                return OpenaiChatModelsEnum.QWEN_480B_CODER_260K
            elif request.useCode == False and request.useDeepResearch:
                return OpenaiChatModelsEnum.SEED_OSS_32B_500K
            else:
                return OpenaiChatModelsEnum.LLAMA_405B_110K

        else:
            if request.useCode and request.useDeepResearch:
                return CerebrasChatModelEnum.GPT_OSS_120B
                # return CerebrasChatModelEnum.QWEN_235B_THINKING
            elif request.useCode and request.useDeepResearch == False:
                return CerebrasChatModelEnum.QWEN_235B
            elif request.useCode == False and request.useDeepResearch:
                # return CerebrasChatModelEnum.QWEN_32B
                return CerebrasChatModelEnum.GPT_OSS_120B
            else:
                return CerebrasChatModelEnum.LLAMA_70B

    async def ApiChat(self, request: ApiChatRequestModel) -> StreamingResponse:

        queryVector = await self.embeddingService.Embed(
            request=EmbeddingRequestModel(
                model="baai/bge-m3",
                texts=[request.query],
                type="query",
            )
        )

        docs: list[str] = []
        async with self.db.pool.acquire() as conn:
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )
            rows = await conn.fetch(
                """
                    SELECT
                    c.id AS claim_id,
                    (c.embedding <-> $1) AS distance,
                    ch.text   AS chunk_text,
                    n.summary AS node_text
                    FROM claims c
                    LEFT JOIN chunks ch ON ch.id = c.chunk_id
                    LEFT JOIN nodes  n  ON n.id  = c.node_id
                    ORDER BY c.embedding <-> $1
                    LIMIT LEAST(COALESCE($2::int, 10), 10);

                """,
                cast(Any, queryVector).data[0].embedding,
                10,
            )

            for row in rows:
                claim_id = row.get("claim_id")
                node_text = row.get("node_text") or ""
                docs.append(
                    f"Claim: {claim_id}\nChunk: {node_text}\n"
                )

        PROFESSIONAL_SYSTEM_PROMPT = f"""
                Retrieved documents:\n\n" + "\n\n".join({docs})
                You are given:
                - A list `Retrieved  documents ` retrived from a knowledge base.

                Strict task (follow exactly):
              
                . If there is no answer in the retrieved docs, respond with:
                "We don't have any information about that. do you want me to search through other sources for you ?"


                Formatting constraints:
                - Output plain Markdown only. No raw HTML, no tables.
                

                """

        userMessages: list[ChatMessageModel] = [
            ChatMessageModel(
                role=ChatMessageRoleEnum.SYSTEM,
                content=PROFESSIONAL_SYSTEM_PROMPT,
            ),
        ]

        for message in request.messages:
            userMessages.append(
                ChatMessageModel(
                    role=(
                        ChatMessageRoleEnum.USER
                        if (message.role == "user")
                        else ChatMessageRoleEnum.ASSISTANT
                    ),
                    content=message.content,
                )
            )
        userMessages.append(
            ChatMessageModel(role=ChatMessageRoleEnum.USER, content=request.query)
        )

        response: Any = await chatService.Chat(
            modelParams=ChatRequestModel(
                model=self.GetModel(request=request),
                messages=userMessages,
                topP=0.9,
                temperature=0.0,
                maxCompletionTokens=3000,
                method=(
                    "groq"
                    if request.useWebSearch
                    else "cerebras" if request.useFlash else "nvidia"
                ),
            )
        )

        if response is not None:
            return response
        else:

            async def errorStream():
                yield "data: Sorry, Something went wrong !. Please Try again?\n\n"

            return StreamingResponse(errorStream(), media_type="text/event-stream")
        
