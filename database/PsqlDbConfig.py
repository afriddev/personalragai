# config/PsqlDbConfig.py

import os
from database.PsqlDb import PsqlDb
from dotenv import load_dotenv

load_dotenv()

DATABASE_CONNECTION_STRING_PSQL = os.getenv("DATABASE_CONNECTION_STRING_PSQL", "")

psqlDbClient = PsqlDb(DATABASE_CONNECTION_STRING_PSQL)
