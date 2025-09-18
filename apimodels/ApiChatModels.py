from pydantic import BaseModel


class ApiChatMessageModel(BaseModel):
    role: str
    content: str


class ApiChatRequestModel(BaseModel):
    query: str
    messages: list[ApiChatMessageModel] = []
