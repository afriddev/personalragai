
from pydantic import BaseModel

class AllQaResponseModel(BaseModel):
    questions: list[str]
    answers: list[str]


class ExtractTextFromYtResponseModel(BaseModel):
    videoId: str
    chunkText: str
    chunkUrl: str
