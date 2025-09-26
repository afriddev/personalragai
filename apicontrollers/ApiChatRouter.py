from fastapi import APIRouter
from apiservices import ApiChatService
from apimodels import ApiChatRequestModel
from fastapi.responses import StreamingResponse
from ragservices.services import BuildRagService


ApiChatRouter = APIRouter()
Chat = ApiChatService()


a = BuildRagService()


@ApiChatRouter.post("/chat")
async def chatAPI(request: ApiChatRequestModel) -> StreamingResponse:
    chatResponse = await Chat.ApiChat(request=request)
    return chatResponse


@ApiChatRouter.get("/a")
async def emb():
    return await a.BuildRagFromPdf(
        file="opd_manual.pdf",
    )
