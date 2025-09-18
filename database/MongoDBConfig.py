from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Any, cast
import os

load_dotenv()


DATABASE_CONNECTION_STRING_MONGODB = os.getenv("DATABASE_CONNECTION_STRING_MONGODB", "")

mongoClient = cast(Any, MongoClient(DATABASE_CONNECTION_STRING_MONGODB))
