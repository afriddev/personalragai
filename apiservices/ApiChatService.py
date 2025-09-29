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
from apimodels import PreProcessUserQueryResponseModel
from ragservices.utils import PRE_PROCESS_USER__QUERY_PROMPT


chatService = Chat()
embeddingService = Embedding()


class ApiChatService(ApiChatImpl):

    def __init__(self):
        self.embeddingService = embeddingService
        self.db = psqlDbClient
        self.RetryLoopIndexLimit = 5

    async def PreProcessUserQuery(
        self, query: str, messages: list[ChatMessageModel], loopIndex: int
    ) -> PreProcessUserQueryResponseModel:
        if loopIndex > self.RetryLoopIndexLimit:
            raise Exception(
                "Exception while extarcting relation and questions from chunk"
            )

        preProcessResponse: Any = await chatService.Chat(
            modelParams=ChatRequestModel(
                model=CerebrasChatModelEnum.META_LLAMA_17B_MAVERICK,
                maxCompletionTokens=1000,
                messages=messages,
                stream=False,
                temperature=0.3,
                topP=1.0,
                responseFormat={
                    "type": "object",
                    "properties": {
                        "cleanquery": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "PREVIOUS",
                                "ABUSE_LANG_ERROR",
                                "CONTACT_INFO_ERROR",
                                "HMIS",
                            ],
                        },
                    },
                    "required": ["type"],
                    "additionalProperties": False,
                },
                method="cerebras",
            )
        )
        print("preProcessResponse", preProcessResponse)

        chatResponse: Any = {}
        try:
            chatResponse = json.loads(preProcessResponse.content).get("response")
            if (
                chatResponse.get("cleanquery") is None
                or chatResponse.get("cleanquery") == ""
            ):
                messages.append(
                    ChatMessageModel(
                        role=ChatMessageRoleEnum.USER,
                        content="Please generate a valid json object clean query and type",
                    )
                )
                await self.PreProcessUserQuery(
                    messages=messages,
                    loopIndex=loopIndex + 1,
                    query=query,
                )

        except Exception as e:
            messages.append(
                ChatMessageModel(
                    role=ChatMessageRoleEnum.USER,
                    content="Please generate a valid json object clean query and type",
                )
            )
            await self.PreProcessUserQuery(
                messages=messages,
                loopIndex=loopIndex + 1,
                query=query,
            )
        return PreProcessUserQueryResponseModel(
            cleanQuery=chatResponse.get("cleanquery"),
            type=chatResponse.get("type"),
        )

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
                return OpenaiChatModelsEnum.LLAMA_235B_130K

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
        messages: list[ChatMessageModel] = []

        for message in request.messages:
            messages.append(
                ChatMessageModel(
                    role=(
                        ChatMessageRoleEnum.USER
                        if (message.role == "user")
                        else ChatMessageRoleEnum.ASSISTANT
                    ),
                    content=message.content,
                )
            )

        messages.append(
            ChatMessageModel(
                role=ChatMessageRoleEnum.USER,
                content=request.query,
            )
        )

        preProcessMessages = messages.copy()
        preProcessMessages.append(
            ChatMessageModel(
                role=ChatMessageRoleEnum.SYSTEM,
                content=PRE_PROCESS_USER__QUERY_PROMPT,
            )
        )

        preProcessResponse = await self.PreProcessUserQuery(
            query=request.query, messages=preProcessMessages, loopIndex=0
        )

        if preProcessResponse.type == "ABUSE_LANG_ERROR":

            async def abuseStream():
                yield "data: Sorry, your query contains abusive language.\n\n"

            return StreamingResponse(abuseStream(), media_type="text/event-stream")

        elif preProcessResponse.type == "CONTACT_INFO_ERROR":

            async def contactInfo():
                yield "data: Sorry, your query contains personal or confidential information.\n\n"

            return StreamingResponse(contactInfo(), media_type="text/event-stream")

        elif preProcessResponse.type == "PREVIOUS":
            previousMessages: list[ChatMessageModel] = messages.copy()

            previousMessages.append(
                ChatMessageModel(
                    role=ChatMessageRoleEnum.SYSTEM,
                    content="You are **HMIS AI**  your response should be short and concise not more then 100 tokens ",
                )
            )

            response: Any = await chatService.Chat(
                modelParams=ChatRequestModel(
                    model=OpenaiChatModelsEnum.LLAMA_405B_110K,
                    messages=previousMessages,
                    topP=0.9,
                    temperature=1.0,
                    maxCompletionTokens=3000,
                    method=("nvidia"),
                )
            )

            if response is not None:
                return response
            else:

                async def errorStream():
                    yield "data: Sorry, Something went wrong !. Please Try again?\n\n"

                return StreamingResponse(errorStream(), media_type="text/event-stream")

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
                    ch.text   AS chunk_text
                    FROM claims c
                    LEFT JOIN chunks ch ON ch.id = c.chunk_id
                    ORDER BY c.embedding <-> $1
                    LIMIT LEAST(COALESCE($2::int, 10), 10);

                """,
                cast(Any, queryVector).data[0].embedding,
                10,
            )

            for row in rows:
                chunk_text = row.get("chunk_text") or ""
                docs.append(f"chunk: {chunk_text}")

        PROFESSIONAL_SYSTEM_PROMPT = f"""
            You are a professional AI assistant. Your job is to generate a **well-structured Markdown response** using the information provided.

            Retrieved Chunks:
            \n\n" + "\n\n".join({docs}) + "\n\n

            ### Task Instructions:
            - Use only the information from the retrieved chunks to answer.
            - If the retrieved chunks do not contain the answer, respond exactly with:
            > We don't have any information about that. Do you want me to search through other sources for you?

            ### Formatting Guidelines:
            - Output in **plain Markdown only** (no HTML).
            - Present information in a clear, concise, and professional tone.
            - Use bullet points, numbered lists, or headings where appropriate.
            - For links, use Markdown format: ![text](link).
            - **Insert image links naturally at the point of relevance inside the explanation. Never group or place all images at the end of the answer.**
            - Do not include unrelated images.
            - Ensure the response is well-organized, professional, and easy to read.
            - Example inline usage:  
            "The Taj Mahal ![Taj Mahal](https://example.com/taj.jpg) is one of the most famous landmarks in India."
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
