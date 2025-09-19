from apiimplementations import ApiChatImpl
from apimodels import (
    ApiChatRequestModel,
)
from clientservices.services import Chat
from clientservices.models import ChatMessageModel, ChatRequestModel
from clientservices.enums import (
    ChatMessageRoleEnum,
    OpenaiChatModelsEnum,
    CerebrasChatModelEnum,
    GroqChatModelsEnum,
)
from typing import Any
from fastapi.responses import StreamingResponse

chatService = Chat()


class ApiChatService(ApiChatImpl):

    async def ApiChat(self, request: ApiChatRequestModel) -> StreamingResponse:
        PROFESSIONAL_SYSTEM_PROMPT = """
        You are a highly skilled and professional AI assistant specializing in providing clear, concise, and accurate information. Always maintain a polite and respectful tone.
        Maintain confidentiality and respect user privacy at all times.
       
        """

        userMessages: list[ChatMessageModel] = [
            ChatMessageModel(
                role=ChatMessageRoleEnum.SYSTEM,
                content=PROFESSIONAL_SYSTEM_PROMPT,
            )
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

        model: OpenaiChatModelsEnum | CerebrasChatModelEnum | GroqChatModelsEnum = (
            OpenaiChatModelsEnum.LLAMA_405B_110K
        )
        temperature = 0.2
        maxCompletionTokens = 3000

        if request.useCreative:
            model = OpenaiChatModelsEnum.LLAMA_405B_110K
            temperature = 1.2
            maxCompletionTokens = 2000
        elif request.useFlash:
            model = CerebrasChatModelEnum.GPT_OSS_120B
            temperature = 0.3
            maxCompletionTokens = 2000
        elif request.useDeepResearch:
            model = CerebrasChatModelEnum.GPT_OSS_120B
            temperature = 0.2
            maxCompletionTokens = 10000
        elif request.useCode:
            model = OpenaiChatModelsEnum.QWEN_480B_CODER_240K
            temperature = 0.2
            maxCompletionTokens = 20000
        elif request.useWebSearch:
            model = GroqChatModelsEnum.GROQ_COMPOUND
            temperature = 0.2
            maxCompletionTokens = 8100

        response: Any = await chatService.Chat(
            modelParams=ChatRequestModel(
                model=model,
                messages=userMessages,
                temperature=temperature,
                maxCompletionTokens=maxCompletionTokens,
                method=(
                    "groq"
                    if request.useWebSearch
                    else (
                        "cerebras"
                        if request.useFlash
                        else "cerebras" if (request.useDeepResearch) else "openai"
                    )
                ),
            )
        )

        if response is not None:
            return response
        else:

            async def errorStream():
                yield "data: Sorry, Something went wrong !. Please Try again?\n\n"

            return StreamingResponse(errorStream(), media_type="text/event-stream")
