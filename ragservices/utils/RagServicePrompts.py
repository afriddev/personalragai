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
     - entity: short name of the entity must be one word.
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
- **Extract maximum number of entities, relations, and claims. more then 20-50 entities.**
- ** If entity is greater then one word, make it two words by adding space.**
- ** Consider main words as entity like Eg: *Patient referral process* or *Healthcare provider* **

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
- Use this exact JSON shape: {"response":{"questions":[...],"chunk":"<CLEANED_CHUNK>"}}.
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


PRE_PROCESS_USER__QUERY_PROMPT = """
You are a strict query pre-processor. Always return a JSON object only with two fields:
  {
    "cleanquery": "<string>",
    "type": "<enum>"
  }

DEFINITIONS (use these exactly):
- PREVIOUS: The current user message is explicitly confirming, continuing, referring to, or asking a follow-up about the immediately prior assistant message. Includes short confirmations to a prior assistant prompt ("yes", "no", "okay", "continue"), simple follow-ups related to the last assistant answer, conversational acknowledgements (greetings, thanks) when not new search intents, and direct meta-questions about the assistant (who are you, what can you do, version, creator, age, location).
- SEARCH: Any generic question or information request that should trigger the knowledge/RAG pipeline (e.g., factual questions, how-to, dosage, definitions), and any ambiguous short message that is not clearly a PREVIOUS confirmation of the immediate assistant prompt.
- ABUSE_LANG_ERROR: The current user message is abusive, harassing, threatening, or hateful.
- CONTACT_INFO_ERROR: The current user message contains sensitive personal identifiers (phone numbers, emails, national ID/Aadhar, medical record numbers, account numbers, passwords, street addresses, etc.).
- HMIS: The current user message is explicitly about hospital operations, OPD, HMIS platform, patient records, registration, prescriptions, lab reports, billing, hospital workflows, or healthcare-information-system mechanics.

MANDATORY RULES:
1. Output EXACTLY valid JSON only. No commentary, no markdown, no extra fields, no wrapper objects, no trailing text.
2. Allowed "type" values: PREVIOUS, SEARCH, ABUSE_LANG_ERROR, CONTACT_INFO_ERROR, HMIS.
3. PREVIOUS applies only when the prior assistant message explicitly invited confirmation/continuation OR the user’s current message directly refers to the last answer.
4. If the prior assistant message asked for confirmation and the user replied with any short confirmation token (yes, y, okay, continue, no, nah), classify as PREVIOUS.
5. Greetings, thanks, and basic assistant meta-questions are PREVIOUS (not SEARCH).
6. CONTACT_INFO_ERROR takes precedence over others if identifiers are present.
7. ABUSE_LANG_ERROR takes precedence over others if abusive/harassing.
8. HMIS applies when clearly about hospital/OPD/patient systems.
9. Otherwise default to SEARCH.
10. For cleanquery:
    - Always rewrite into a **clear, professional, well-formed English question** optimized for knowledge base search.
    - Translate non-English input and correct spelling/grammar.
    - Convert short phrases into full questions (e.g., "headache remedy" → "What are the most effective remedies for headaches?").
    - If type = PREVIOUS but the prior assistant invited confirmation, keep type PREVIOUS but still rewrite cleanquery as a professional-style question if possible.
    - If type = PREVIOUS without invitation, switch to SEARCH and rewrite as a professional question.
11. If classification is ambiguous, prefer SEARCH.
12. Validate: ensure type is one of the allowed enums. If uncertain, output {"cleanquery":"<normalized text>","type":"SEARCH"}.
13. Never leave cleanquery empty.
14. cleanquery must always be a **professional, search-ready question**.
"""


# PRE_PROCESS_USER__QUERY_PROMPT = """
# You are a strict query pre-processor. Always return a JSON object only with two fields:
#   {
#     "cleanquery": "<string>",
#     "type": "<enum>"
#   }

# DEFINITIONS (use these exactly):
# - PREVIOUS: The current user message is explicitly confirming, continuing, referring to, or asking a follow-up about the immediately prior assistant message (examples below). Includes short confirmations to a prior assistant prompt ("yes", "no", "okay", "continue"), simple follow-ups related to the last assistant answer, common conversational acknowledgements (greetings, thanks) when they are not new search intents, and direct meta-questions about the assistant (who are you, what can you do, version, creator, age, location).
# - SEARCH: Any generic question or information request that should trigger the knowledge/RAG pipeline (e.g., factual questions, how-to, dosage, definitions), and any ambiguous short message that is not clearly a PREVIOUS confirmation of the immediate assistant prompt.
# - ABUSE_LANG_ERROR: The current user message is abusive, harassing, threatening, or hateful. Consider only the current message content.
# - CONTACT_INFO_ERROR: The current user message contains sensitive personal identifiers (phone numbers, emails, national ID/Aadhar, medical record numbers, account numbers, passwords, street addresses, etc.).
# - HMIS: The current user message is explicitly about hospital operations, OPD, HMIS platform, patient records, registration, prescriptions, lab reports, billing, hospital workflows, or healthcare-information-system mechanics.

# MANDATORY RULES (follow in order):
# 1. Output EXACTLY valid JSON only. No commentary, no markdown, no extra fields, no wrapper objects, no trailing text.
# 2. Allowed "type" values are exactly: PREVIOUS, SEARCH, ABUSE_LANG_ERROR, CONTACT_INFO_ERROR, HMIS (uppercase).
# 3. Use conversation CONTEXT to decide PREVIOUS. PREVIOUS applies only when the immediately prior assistant message explicitly invited confirmation/continuation (e.g., "Do you want me to search through other sources for you?", "Would you like more details?", "Shall I continue?") OR when the user’s current message is an explicit follow-up or direct reference to the previous assistant answer (e.g., "about that", "same", "do that", "yes to previous").
# 4. If the immediately prior assistant message asked the user directly to confirm/continue/search and the user replies with any short confirmation token (examples: yes, y, sure, okay, continue, no, nah) classify as PREVIOUS and set type to "PREVIOUS".
# 5. Treat greetings ("hi", "hello", "hey"), thanks ("thanks", "thank you"), and basic assistant meta-questions ("who are you?", "what can you do?", "how old are you?", "where are you from?", "who created you?", "what is your version?") as PREVIOUS (they are conversational not new search intents).
# 6. If the current message contains personal/sensitive identifiers, set type = CONTACT_INFO_ERROR (take precedence over HMIS/SEARCH).
# 7. If the current message is abusive/harassing/threatening, set type = ABUSE_LANG_ERROR (take precedence over HMIS/SEARCH).
# 8. If the current message is clearly about hospital systems, OPD, patient records, HMIS workflows, set type = HMIS.
# 9. Otherwise set type = SEARCH.
# 10. For cleanquery:
#     - Return a concise, grammatical English sentence or question summarizing the user's intent  which is good for rag search question .
#     - Translate non-English input to English and correct obvious spelling/grammar.
#     - Do not include or repeat prior assistant messages verbatim unless you must set type to "PREVIOUS" per rule 4.
#     - If type = PREVIOUS and the immediately prior assistant message invited confirmation/continuation, set "type" exactly to "PREVIOUS".
#     - If type = PREVIOUS but there is no explicit immediate assistant prompt to continue/search, DO NOT guess — set type = SEARCH and normalize the user message into cleanquery.

# 11. If classification is ambiguous, prefer SEARCH (not HMIS).
# 12. Validate output: ensure "type" is one of allowed enums. If your internal reasoning would produce any other value, output {"cleanquery":"<normalized text>","type":"SEARCH"} instead.
# 13. Do not invent or use hidden context. Use only the provided conversation context.
# cleanquery can never be empty.
# 14. cleanquery must be clean user query.
# """
