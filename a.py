from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
# api_key = os.getenv("NVIDIA_API_KEY")

# client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

# completion = client.chat.completions.create(
#     model="meta/llama-3.1-70b-instruct",
#     messages=[
#         {
#             "role": "user",
#             "content": "How to implement graphrag think before answering quesion" *10000, 
#         }
#     ],
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=15000,
#     extra_body={"chat_template_kwargs": {"thinking": True}},
#     stream=True,
# )

# for chunk in completion:
#     reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
#     if reasoning:
#         print(reasoning, end="")



from groq import Groq
import json
api_key = os.getenv("GROQ_API_KEY")


client = Groq()

response = client.chat.completions.create(
    model="groq/compound",
    messages=[
        {
            "role": "user",
            "content": "What happened in AI last week? Provide a list of the most important model releases and updates."
        }
    ],
    stream=True,
    
)

for chunk in response:
    tools = getattr(chunk.choices[0].delta, "executed_tools", None)
    if tools and len(tools) > 0:
        tk = getattr(tools[0], "search_results", None)
        results = getattr(tk, "results", None)
        if results and len(results) > 0:
            print(results)

