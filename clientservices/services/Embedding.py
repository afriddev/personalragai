from openai import AsyncOpenAI
from clientservices.workers import GetNvidiaApiKey, GetNvidiaBaseUrl
from clientservices.implementations import EmbeddingImpl
from clientservices.models import (
    EmbeddingRequestModel,
    EmbeddingResponseModel,
    EmbeddingDataModel,
    EmbeddingUsageModel,
)

openAiClient = AsyncOpenAI(base_url=GetNvidiaBaseUrl(), api_key=GetNvidiaApiKey())


class Embedding(EmbeddingImpl):

    async def embed(self, request: EmbeddingRequestModel) -> EmbeddingResponseModel:
        try:
            response = await openAiClient.embeddings.create(
                model=request.model, input=request.texts, dimensions=request.dimensions
            )

            return EmbeddingResponseModel(
                data=[
                    EmbeddingDataModel(embedding=data.embedding, index=i)
                    for i, data in enumerate(response.data)
                ],
                model=request.model,
                usage=EmbeddingUsageModel(
                    prompt_tokens=response.usage.prompt_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            return EmbeddingResponseModel(
                data=None,
                model=request.model,
                usage=None,
            )
