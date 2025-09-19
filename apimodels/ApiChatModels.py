from pydantic import BaseModel


class ApiChatMessageModel(BaseModel):
    role: str
    id: str
    content: str


class ApiChatRequestModel(BaseModel):
    query: str
    messages: list[ApiChatMessageModel] = []
    useWebSearch: bool = False
    useCode: bool = False
    useDeepResearch: bool = False
    useFlash: bool = False
