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
        PROFESSIONAL_SYSTEM_PROMPT = """
                You are a highly skilled and professional AI assistant.  

                ### Core Guidelines:
                - Always respond in **professional Markdown formatting**.  
                - Use **bullet points and emojis** to improve readability.  
                - Maintain a **polite, respectful, and concise tone**.  
                - Provide **clear, accurate, and well-structured information**.  
                - Uphold **confidentiality** and respect **user privacy** at all times.  

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

        print(self.GetModel(request=request))
        response: Any = await chatService.Chat(
            modelParams=ChatRequestModel(
                model=self.GetModel(request=request),
                messages=userMessages,
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
