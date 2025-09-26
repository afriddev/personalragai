

# from clientservices.services import Embedding
# from clientservices.models import (
#     EmbeddingRequestModel,
#     FindTopKresultsFromVectorsRequestModel,
#     EmbeddingResponseModel,
# )
# import asyncio

# a = Embedding()


# async def pro():
#     embeddings: EmbeddingResponseModel = await a.Embed(
#         request=EmbeddingRequestModel(
#             model="baai/bge-m3",
#             texts=[
#                 "OPD is a healthcare organization providing outpatient services through a network of clinics, focusing on affordable care, preventive health, and diagnostics.",
#                 "OPD operates multi-city clinics offering outpatient treatments, wellness programs, and patient-centered medical care.",
#                 "OPD delivers accessible healthcare by combining experienced doctors, advanced diagnostic technology, and partnerships with insurance providers.",
#                 "OPD specializes in outpatient healthcare, emphasizing preventive medicine, chronic disease management, and affordable solutions for patients.",
#                 "OPD is a trusted healthcare provider offering outpatient consultatizns, diagnostics, and wellness initiatives across several cities.",
#                 "OPD runs clinics that provide outpatient services, preventive screenings, and community health programs for improved population wellbeing.",
#                 "OPD combines technology-driven diagnostics with experienced medical staff to deliver affordable outpatient healthcare.",
#                 "OPD is recognized for its outpatient services, preventive care, and collaboration with insurers to ensure broader patient coverage.",
#                 "OPD focuses on community health by offering outpatient consultations, affordable diagnostics, and wellness-oriented medical programs.",
#             ],
#             type="passage",
#         )
#     )

#     embedding1: EmbeddingResponseModel = await a.Embed(
#         request=EmbeddingRequestModel(
#             model="baai/bge-m3",
#             texts=[
#                 "OPD manages a network of clinics delivering quality healthcare solutions, preventive programs, and modern diagnostic treatments.",
#             ],
#             type="query",
#         )
#     )

#     rerank = a.FindTopKResultsFromVectors(
#         request=FindTopKresultsFromVectorsRequestModel(
#             topK=9,
#             queryVector=embedding1.data[0].embedding,
#             sourceVectors=[data.embedding for data in embeddings.data],
#         )
#     )
#     print(rerank.indeces)
#     print(rerank.distances)


# asyncio.run(pro())
