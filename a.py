# import asyncio
# from ragservices.services import ChunkInstanceService
# from clientservices.models import ChatMessageModel
# from clientservices.enums import ChatMessageRoleEnum
# from ragservices.utils import EXTARCT_INSTANCE_FROM_CHUNK_PROMPT


# a = ChunkInstanceService()


# chunk = """

# Rahul Verma, a senior data scientist at Infosys Limited, is currently leading a project in collaboration with Microsoft Corporation on building advanced AI models for financial fraud detection. His manager, Meera Iyer, reports directly to the Chief Technology Officer of Infosys, Dr. Ramesh Krishnan. Rahul works in a core analytics team consisting of 15 members, including specialists in natural language processing, computer vision, and statistical modeling. The project, code-named “Sentinel”, is a joint initiative funded by both Infosys and Microsoft, with support from the Reserve Bank of India to ensure compliance with financial regulations. Within the project, Rahul’s responsibilities include designing anomaly detection systems and coordinating with the compliance team, while his colleague Arjun Mehta focuses on building real-time dashboards for transaction monitoring. Microsoft’s technical advisor, John Smith, ensures that the cloud infrastructure on Azure remains scalable and secure. Rahul is also collaborating with the Indian Institute of Technology, Delhi, where Professor Neha Kapoor is providing academic oversight and offering PhD students as research interns. The team is expected to publish at least two research papers in IEEE conferences over the next year. Additionally, Infosys has committed to presenting a demonstration of Sentinel at the upcoming NASSCOM technology summit in Bengaluru. The project timeline extends until December 2026, with periodic reviews by both Infosys and Microsoft executives. Rahul’s career progression is closely tied to the success of Sentinel, and his current designation as “Senior Data Scientist” may be upgraded to “Principal Data Scientist” upon successful delivery.

# """
# asyncio.run(
#     a.ExtractChunkInstance(
#         chunk=chunk,
#         messages=[
#             ChatMessageModel(
#                 role=ChatMessageRoleEnum.SYSTEM,
#                 content=EXTARCT_INSTANCE_FROM_CHUNK_PROMPT,
#             ),
#             ChatMessageModel(
#                 role=ChatMessageRoleEnum.USER,
#                 content=chunk,
#             ),
#         ],
#         retryLimit=3,
#     )
# )


from clientservices.services import Embedding
from clientservices.models import EmbeddingRequestModel
import asyncio

a = Embedding()


asyncio.run(
    a.Embed(
        request=EmbeddingRequestModel(
            model="text-embedding-ada-002",
            texts=[
                "Hello, world!",
                "Bonjour le monde!",
                "Hola, mundo!",
                "Hallo, Welt!",
                "Ciao, mondo!",
                "こんにちは、世界！",
                "안녕하세요, 세계!",
                "你好，世界！",
                "Привет, мир!",
                "مرحبا بالعالم!",
            ],
        )
    )
)
