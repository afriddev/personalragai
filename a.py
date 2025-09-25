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
from clientservices.models import (
    EmbeddingRequestModel,
    FindTopKresultsFromVectorsRequestModel,
    EmbeddingResponseModel,
)
import asyncio

a = Embedding()


async def pro():
    embeddings: EmbeddingResponseModel = await a.Embed(
        request=EmbeddingRequestModel(
            model="baai/bge-m3",
            texts=[
                "OPD is a healthcare organization providing outpatient services through a network of clinics, focusing on affordable care, preventive health, and diagnostics.",
                "OPD operates multi-city clinics offering outpatient treatments, wellness programs, and patient-centered medical care.",
                "OPD delivers accessible healthcare by combining experienced doctors, advanced diagnostic technology, and partnerships with insurance providers.",
                "OPD specializes in outpatient healthcare, emphasizing preventive medicine, chronic disease management, and affordable solutions for patients.",
                "OPD is a trusted healthcare provider offering outpatient consultatizns, diagnostics, and wellness initiatives across several cities.",
                "OPD runs clinics that provide outpatient services, preventive screenings, and community health programs for improved population wellbeing.",
                "OPD combines technology-driven diagnostics with experienced medical staff to deliver affordable outpatient healthcare.",
                "OPD is recognized for its outpatient services, preventive care, and collaboration with insurers to ensure broader patient coverage.",
                "OPD focuses on community health by offering outpatient consultations, affordable diagnostics, and wellness-oriented medical programs.",
            ],
            type="passage",
        )
    )

    embedding1: EmbeddingResponseModel = await a.Embed(
        request=EmbeddingRequestModel(
            model="baai/bge-m3",
            texts=[
                "OPD manages a network of clinics delivering quality healthcare solutions, preventive programs, and modern diagnostic treatments.",
            ],
            type="query",
        )
    )

    rerank = a.FindTopKResultsFromVectors(
        request=FindTopKresultsFromVectorsRequestModel(
            topK=9,
            queryVector=embedding1.data[0].embedding,
            sourceVectors=[data.embedding for data in embeddings.data],
        )
    )
    print(rerank.indeces)
    print(rerank.distances)


asyncio.run(pro())
