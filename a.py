from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

query = "What is the GPU memory bandwidth of H100 SXM?"
passages = [
    "The H100 GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth, 7X faster than PCIe Gen5. This innovative design will deliver up to 30X higher aggregate system memory bandwidth to the GPU compared to today's fastest servers and up to10X higher pe rformance for applications running terabytes of data.", 
    "A100 provides up to 20X higher performance over the prior generation and can be partitioned into seven GPU instances to dynamically adjust to shifting demands. The A100 80GB debuts the world's fastest memorry bandwidth at over 2 terabytes per second (TB/s) to run the largest models and datasets.", 
    "Accelerated servers with H100 deliver the compute power—along with 3 terabytes per second (TB/s) of memory bandwidth per GPU and scalability with NVLink and NVSwitch™.", 
]

client = NVIDIARerank(
  model="nvidia/nv-rerankqa-mistral-4b-v3", 
  api_key=api_key,
)

response = client.compress_documents(
  query=query,
  documents=[Document(page_content=passage) for passage in passages]
)

print(response)


# client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

# completion = client.chat.completions.create(
#     model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
#     messages=[
#         {
#             "role": "user",
#             "content": "How to implement graphrag think before answering quesion" , 
#         }
#     ],
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=15000,
#     extra_body={"chat_template_kwargs": {"thinking": True}},
#     stream=True,
# )

# for chunk in completion:
#     print(chunk.choices[0].delta)



# from groq import Groq
# import json
# api_key = os.getenv("GROQ_API_KEY")


# client = Groq()

# response = client.chat.completions.create(
#     model="groq/compound",
#     messages=[
#         {
#             "role": "user",
#             "content": "What happened in AI last week? Provide a list of the most important model releases and updates."
#         }
#     ],
#     stream=True,
    
# )

# for chunk in response:
#     tools = getattr(chunk.choices[0].delta, "executed_tools", None)
#     if tools and len(tools) > 0:
#         tk = getattr(tools[0], "search_results", None)
#         results = getattr(tk, "results", None)
#         if results and len(results) > 0:
#             print(results)

