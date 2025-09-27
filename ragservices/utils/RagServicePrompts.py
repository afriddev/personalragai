EXTARCT_INSTANCE_FROM_CHUNK_PROMPT = r"""
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


EXTRACT_NODE_SUMMARY_PROMPT = r"""

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
6. Dont include any information not present in the input.
7. Dont add any extra fields or commentary.

"""


EXTRACT_QUESTIONS_FROM_CHUNK_PROMPT = r"""
TASK
Return ONLY valid JSON per the schema for ONE input chunk.

INPUT
{ "chunk": "..." }

OUTPUT (conceptual)
{
  "response": {
    "questions": ["..."],
    "chunk": "..."
  }
}

GOAL
- Extract answerable questions from the chunk.
- Echo a minimally cleaned version of the input chunk.

STEPS / REQUIREMENTS
1. Clean the input chunk minimally and echo it in response.chunk:
   - Trim leading/trailing whitespace.
   - Collapse repeated spaces and newlines into a single space.
   - Remove unprintable control characters.
   - Preserve ALL URLs and image links exactly as they appear.
2. Questions:
   - Create as many concise, answerable questions as can be answered using only information present in the chunk.
   - Each question must be directly answerable from the chunk content.
   - Questions should be clear and unambiguous.
   - Do NOT include URLs or image tokens inside questions.
3. General:
   - Do not invent metadata or additional keys.
   - If questions can be extracted, return empty arrays for those fields.

OUTPUT FORMAT RULES
- Output ONLY valid JSON (no markdown, no extra text).
- Use this exact JSON shape: {"response":{"questions":[...],"chunk":"<CLEANED_CHUNK>"}}
- Use DOUBLE QUOTES for JSON keys and string values.
- The JSON must be a single line (no newline characters \n).
- The cleaned chunk string must not include newline escape sequences; it should contain only normal printable characters.
- Do NOT include any extra keys, comments, or metadata.


"""


CLEAN_YT_CHUNK_PROMPT = r"""
TASK:
You are given a YouTube transcript chunk.

INPUT:
A single string called "chunk" containing the transcript text. The text may contain:
- Misspellings
- Broken words or cut sentences
- Mixed languages
- Personal references (e.g., I, we, names)
- Filler and irrelevant words
- YouTube links or other URLs

GOAL:
Produce a single clean paragraph that states the main context of the chunk in neutral, impersonal language.

STEPS:
1. Detect the language of the chunk. If it is not English, translate it to English first.
2. Fix misspellings and broken words automatically.
3. Remove all personal perspective, names, speaker references, filler words, and irrelevant details.
4. Summarize the core content into one concise paragraph (no lists, no headings).
5. Do NOT start with phrases like "The content describes" or "The context discusses". Start directly with the subject matter.
6. Preserve ALL YouTube links or other URLs exactly as they appear in the input.
7. Ensure the paragraph avoids double quotes (") inside the text so JSON stays valid.

OUTPUT FORMAT RULES:
- Output ONLY valid JSON (no markdown, no extra text).
- JSON must follow this exact shape: {"response":{"chunk":"<CLEAN_PARAGRAPH>"}}
- Use DOUBLE QUOTES for JSON keys and values.
- JSON must be a single line (no \n or escape sequences).
- Do NOT add any extra keys, comments, or metadata.

"""