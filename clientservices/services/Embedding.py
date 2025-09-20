from openai import AsyncOpenAI
from clientservices.workers import (
    GetNvidiaApiKey,
    GetNvidiaBaseUrl,
)
from clientservices.implementations import EmbeddingImpl
from clientservices.models import (
    EmbeddingRequestModel,
    EmbeddingResponseModel,
    EmbeddingDataModel,
    EmbeddingUsageModel,
    RerankResponseModel,
    RerankRequestModel,
    RerankResultModel,
)
from typing import cast, Any

from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document


openAiClient = AsyncOpenAI(base_url=GetNvidiaBaseUrl(), api_key=GetNvidiaApiKey())
rerankClient = NVIDIARerank(api_key=GetNvidiaApiKey(), model="")


class Embedding(EmbeddingImpl):

    async def Embed(self, request: EmbeddingRequestModel) -> EmbeddingResponseModel:
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

    async def RerankDocs(self, request: RerankRequestModel) -> RerankResponseModel:

        try:
            rerankClient.model = request.model

            response = rerankClient.compress_documents(
                query=request.query,
                documents=[Document(page_content=doc) for doc in request.docs],
            )
            temp: list[RerankResultModel] = []
            for doc in response:
                temp.append(
                    RerankResultModel(
                        doc=doc.page_content,
                        score=cast(Any, doc).metadata.get("relevance_score"),
                    )
                )
            return RerankResponseModel(results=temp)
        except Exception as e:
            print(e)
            return RerankResponseModel(results=[])
