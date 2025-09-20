import json
import cerebras.cloud.sdk
from cerebras.cloud.sdk import AsyncCerebras
from clientservices.enums import (
    ChatResponseStatusEnum,
)
from clientservices.implementations import ChatImpl
from clientservices.models import (
    ChatRequestModel,
    ChatResponseModel,
    ChatDataModel,
    ChatUsageModel,
    ChatChoiceModel,
    ChatChoiceMessageModel,
)
from fastapi.responses import StreamingResponse
from cerebras.cloud.sdk import DefaultAioHttpClient
from typing import Any, cast
from clientservices.workers import (
    GetCerebrasApiKey,
    GetNvidiaApiKey,
    GetNvidiaBaseUrl,
    GetGroqBaseUrl,
    GetGroqApiKey,
)
from openai import AsyncOpenAI


openAiClient = AsyncOpenAI(base_url=GetNvidiaBaseUrl(), api_key=GetNvidiaApiKey())
openAiGroqClient = AsyncOpenAI(base_url=GetGroqBaseUrl(), api_key=GetGroqApiKey())


client = AsyncCerebras(
    api_key=GetCerebrasApiKey(),
    http_client=DefaultAioHttpClient(),
)


class Chat(ChatImpl):

    def HandleApiStatusError(self, statusCode: int) -> ChatResponseModel:
        errorCodes = {
            400: ChatResponseStatusEnum.BAD_REQUEST,
            401: ChatResponseStatusEnum.UNAUTHROZIED,
            403: ChatResponseStatusEnum.PERMISSION_DENIED,
            404: ChatResponseStatusEnum.NOT_FOUND,
        }
        message = errorCodes.get(statusCode, ChatResponseStatusEnum.SERVER_ERROR)
        return ChatResponseModel(status=message)

    async def CerebrasChat(self, modelParams: ChatRequestModel) -> Any:
        createCall = client.chat.completions.create(
            messages=cast(Any, modelParams.messages),
            model=modelParams.model.value[0],
            max_completion_tokens=(
                cast(
                    Any,
                    (
                        modelParams.maxCompletionTokens
                        if modelParams.maxCompletionTokens
                        else modelParams.model.value[1]
                    ),
                )
            ),
            stream=modelParams.stream,
            temperature=modelParams.temperature,
            top_p=modelParams.topP,
            seed=modelParams.seed,
            # reasoning_effort="high",
            response_format=cast(
                Any,
                (
                    None
                    if modelParams.responseFormat is None
                    else {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "schema",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {"response": modelParams.responseFormat},
                                "required": ["response"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ),
            ),
        )
        chatCompletion: Any = await createCall

        return chatCompletion

    async def OpenaiChat(self, modelParams: ChatRequestModel) -> Any:
        createCall = openAiClient.chat.completions.create(
            messages=cast(Any, modelParams.messages),
            model=modelParams.model.value[0],
            max_tokens=(
                cast(
                    Any,
                    (
                        modelParams.maxCompletionTokens
                        if modelParams.maxCompletionTokens
                        else modelParams.model.value[1]
                    ),
                )
            ),
            stream=True,
            temperature=modelParams.temperature,
            top_p=modelParams.topP,
            extra_body={
                "chat_template_kwargs": {"thinking": modelParams.model.value[2]}
            },
        )
        chatCompletion: Any = await createCall

        return chatCompletion

    async def OpenaiGroqChat(self, modelParams: ChatRequestModel) -> Any:
        createCall = openAiGroqClient.chat.completions.create(
            messages=cast(Any, modelParams.messages),
            model=cast(Any, modelParams.model.value),
            max_tokens=8000,
            stream=modelParams.stream,
            temperature=0.2,
            top_p=0.7,
        )
        chatCompletion: Any = await createCall

        return chatCompletion

    async def Chat(
        self, modelParams: ChatRequestModel
    ) -> ChatResponseModel | StreamingResponse:

        try:

            chatCompletion: Any = None

            if modelParams.method == "cerebras":
                chatCompletion = await self.CerebrasChat(modelParams)

            elif modelParams.method == "groq":
                chatCompletion = await self.OpenaiGroqChat(modelParams)

            else:
                chatCompletion = await self.OpenaiChat(modelParams)

            if modelParams.stream:

                async def eventGenerator():
                    startedReasoning = False
                    reasoningStartToken: Any = ""
                    reasoningEndToken: Any = ""
                    reasoningStartIndex = 0
                    try:
                        async for chunk in chatCompletion:
                            if getattr(chunk, "choices", None):
                                delta = getattr(chunk.choices[0], "delta", None)
                                if delta:
                                    content = getattr(delta, "content", None)
                                    reasoningContent = getattr(
                                        delta, "reasoning_content", None
                                    )
                                    if startedReasoning:
                                        reasoningEndToken = reasoningEndToken + content
                                        if "</think>" in reasoningEndToken:
                                            startedReasoning = False
                                            reasoningStartToken = ""

                                        reasoningContent = content
                                        content = None

                                    if reasoningStartIndex < 5 and content:
                                        reasoningStartToken = (
                                            reasoningStartToken + content
                                        )
                                        if "<think>" in reasoningStartToken:
                                            startedReasoning = True
                                            reasoningStartIndex = 5
                                        else:
                                            reasoningStartIndex += 1
    
                                    reasoning = getattr(delta, "reasoning", None)
                                    searchResults = None
                                    searchResultsContent: Any = []
                                    executedTools = getattr(
                                        delta, "executed_tools", None
                                    )
                                    if executedTools and len(executedTools) > 0:
                                        searchResults = executedTools[0].get(
                                            "search_results"
                                        )
                                        results = searchResults.get("results")
                                        if results and len(results) > 0:
                                            for item in results:
                                                searchResultsContent.append(
                                                    {
                                                        "title": item.get("title"),
                                                        "url": item.get("url"),
                                                        "content": item.get("content"),
                                                    }
                                                )
                                    if content:
                                        yield f"data: {json.dumps({'type': 'content', 'data': content})}\n\n"
                                    if reasoning:
                                        yield f"data: {json.dumps({'type': 'reasoning', 'data': reasoning})}\n\n"
                                    if reasoningContent:
                                        yield f"data: {json.dumps({'type': 'reasoning', 'data': reasoningContent})}\n\n"
                                    if len(searchResultsContent) > 0:
                                        yield f"data: {json.dumps({'type': 'searchResults', 'data': searchResultsContent})}\n\n"

                    except Exception as e:
                        yield f"event: error\ndata: {str(e)}\n\n"

                return StreamingResponse(
                    eventGenerator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )

            choices: list[ChatChoiceModel] = []
            for ch in chatCompletion.choices:
                choices.append(
                    ChatChoiceModel(
                        index=ch.index,
                        message=ChatChoiceMessageModel(
                            role=ch.message.role,
                            content=ch.message.content,
                        ),
                    )
                )

            LLMData = ChatDataModel(
                id=chatCompletion.id,
                choices=choices,
                created=chatCompletion.created,
                model=chatCompletion.model,
                usage=ChatUsageModel(
                    promptTokens=chatCompletion.usage.prompt_tokens,
                    completionTokens=chatCompletion.usage.completion_tokens,
                    totalTokens=chatCompletion.usage.total_tokens,
                ),
            )

            return ChatResponseModel(
                status=ChatResponseStatusEnum.SUCCESS,
                content=LLMData.choices[0].message.content,
            )

        except cerebras.cloud.sdk.APIConnectionError as e:
            print(e)
            return ChatResponseModel(status=ChatResponseStatusEnum.SERVER_ERROR)
        except cerebras.cloud.sdk.RateLimitError as e:
            print(e)
            return ChatResponseModel(status=ChatResponseStatusEnum.RATE_LIMIT)
        except cerebras.cloud.sdk.APIStatusError as e:
            print(e)
            return self.HandleApiStatusError(e.status_code)
        except Exception as e:
            print(e)
            return ChatResponseModel(status=ChatResponseStatusEnum.ERROR)
