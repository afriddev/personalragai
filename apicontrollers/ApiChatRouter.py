from fastapi import APIRouter
from apiservices import ApiChatService
from apimodels import ApiChatRequestModel
from fastapi.responses import StreamingResponse

ApiChatRouter = APIRouter()
Chat = ApiChatService()


@ApiChatRouter.post("/chat")
async def chatAPI(request: ApiChatRequestModel) -> StreamingResponse:
    chatResponse = await Chat.ApiChat(request=request)
    return chatResponse
