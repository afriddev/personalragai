EXTARCT_INSTANCE_FROM_CHUNK_PROMPT = """
You are an information extraction assistant.  
Read the given text chunk and extract structured information in JSON format.  
Follow this schema exactly:

{
"response":{
  "entities": [
    {
      "id": integer,
      "entity": string,
      "entityDescription": string
    }
  ],
  "relations": [
    {
      "id": integer,
      "sourceEntityId": integer,
      "targetEntityId": integer,
      "relation": string,
      "relationDescription": string
    }
  ],
  "claims": [
    {
      "id": integer,
      "entityId": integer,
      "claim": string,
      "claimDescription": string
    }
  ],
  "chunk": string
}

}


1. **Entities**  
   - Distinct objects, people, organizations, or concepts mentioned.  
   - For each, provide:
     - id: unique integer starting from 1 (per chunk).
     - entity: short name of the entity.
     - entityDescription: a concise description of the entity from context.

2. **Relations**  
   - Directed relationships between two entities.  
   - For each, provide:
     - id: unique integer starting from 1 (per chunk).
     - sourceEntityId: id of the entity the relation starts from.
     - targetEntityId: id of the entity the relation points to.
     - relation: short label (e.g., "works at", "managed by").
     - relationDescription: full description of the relation in plain text.

3. **Claims**  
   - Factual statements about a single entity (not involving another entity).  
   - For each, provide:
     - id: unique integer starting from 1 (per chunk).
     - entityId: the entity this claim belongs to.
     - claim: short label of the fact.
     - claimDescription: full description of the fact.

4. **Chunk**  
   - The original text chunk itself.

⚠️ Important:
- Output must strictly follow the JSON schema provided.
- Do not add extra fields or commentary.
- If something is not found, return an empty array for that section.


Rules:
- Assign ids starting from 1 inside each section.
- Relations = how two entities are connected.
- Claims = factual statements about a single entity.
- If something is missing, return an empty array for that section.
- Do not add extra fields or commentary.
- **Do not miss any important information.**
- **Extract the maximum number of entities, relations, and claims from the chunk.**

"""


EXTRACT_NODE_SUMMARY_PROMPT = """

INPUT: you'll receive:
- one or more entity description
- one or more relations 
- one or more claims 

OUTPUT: produce exactly this JSON and nothing else:
{
  "response":{
    "summary": string
  }
}

Rules (follow exactly):
1. The summary value must be plain text (no markdown, no lists, no extra JSON fields).
2. Produce a brief summary by combining all entity descrption,relations,and claims dont miss any key points.
3. Dont start with The node represents  or entitty starts with  or similar phrases.
4. Produce summary as large as possible without missing any key points.
5. Sumary should be in one paragraph. and include all key points

"""
