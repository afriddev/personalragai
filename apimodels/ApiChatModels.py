from pydantic import BaseModel


class ApiChatMessageModel(BaseModel):
    role: str
    id: str
    useWebSearch: bool = False
    useCodeModel: bool = False
    content: str


class ApiChatRequestModel(BaseModel):
    content: str
    messages: list[ApiChatMessageModel] = []
