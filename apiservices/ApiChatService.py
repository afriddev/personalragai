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
            ChatMessageModel(role=ChatMessageRoleEnum.USER, content=request.content)
        )

        response: Any = await chatService.Chat(
            modelParams=ChatRequestModel(
                model=OpenaiChatModelsEnum.MISTRAL_NEMOTRON_240K,
                maxCompletionTokens=5000,
                messages=userMessages,
                stream=True,
                temperature=0.7,
                topP=0.9,
                method="nvidia",
            )
        )

        if response is not None:
            return response
        else:

            async def errorStream():
                yield "data: Sorry, Something went wrong !. Please Try again?\n\n"

            return StreamingResponse(errorStream(), media_type="text/event-stream")
