from pydantic import BaseModel


class ApiChatMessageModel(BaseModel):
    role: str
    query: str
    id: str
    useWebSearch: bool = False
    useCodeModel: bool = False


class ApiChatRequestModel(BaseModel):
    query: str
    messages: list[ApiChatMessageModel] = []
