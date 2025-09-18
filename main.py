from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from clientservices.services import Chat
from apicontrollers import ApiChatRouter
import asyncio
from database import psqlDbClient, mongoClient


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     asyncio.create_task(psqlDbClient.connect())
#     asyncio.create_task(mongoClient["ragai"])
#     yield
#     await asyncio.wait_for(psqlDbClient.close(), timeout=3)
#     await asyncio.wait_for(mongoClient.close(), timeout=3)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ApiChatRouter, prefix="/api/v1")

cerebrasChat = Chat()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
