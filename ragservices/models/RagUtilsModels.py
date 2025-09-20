
from pydantic import BaseModel

class ExtractQaResponseModel(BaseModel):
    questions: list[str]
    answers: list[str]


class ExtractTextFromYtResponseModel(BaseModel):
    videoId: str
    chunkText: str
    chunkUrl: str
