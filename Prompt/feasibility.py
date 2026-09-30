"""Claim feasibility instructions, rating scale, and prompt formatting."""

SYSTEM_PROMPT = """You assess whether a statement contains enough information for its
veracity to be evaluated. You are not determining whether the statement is true or
false. You are determining whether a valid truth assessment is possible from the
statement alone, assisted by web search.

Follow the requested output format exactly:
brief explanation | rating

The rating must be exactly 0, 1, or 2.
Do not add any text after the rating.
"""

USER_PROMPT = """The following statement is going to be given to an AI system to determine if it is true or false and write an explanation why.

Statement: "{text}"

The only thing the AI will be given is the statement itself, as written above—no context, visuals, or any other information. Your task is to assess whether the AI could possibly give a valid answer. This is not about how likely the AI is to give the correct answer, but whether it is possible to evaluate the veracity of the statement from the information given.

The AI will have access to web search to find primary and secondary sources, but evaluation might still be impossible if too much ambiguity or mandatory context is missing.

Information whose absence may make evaluation difficult includes:

1. The identity of a key person, such as an unidentified speaker or ambiguously referenced person.
2. A location, when the statement's veracity depends on the location.
3. Textual information or evidence mentioned in the statement but not supplied.
4. Visual or audio evidence mentioned in the statement. The AI receives only the statement text.
5. Temporal information. The date on which the statement was made is unknown. This does not matter if the statement can be evaluated regardless of when it was made.
6. An actual factual claim for which evaluating veracity makes sense.

Rate the statement using this scale:

2: There may be some room for interpretation or contextual influence, but the statement is still sufficiently clear and unambiguous. For example, "The Earth is round" is not perfectly precise, but it can reasonably be evaluated.

1: There is clear ambiguity, missing context, or more than one potential interpretation. However, there seems to be roughly a one-half chance of recovering the intended meaning or context through a strong knowledge base or web search.

0: It is impossible to evaluate. Multiple valid interpretations would substantially change the verdict, mandatory and irrecoverable information is missing, or the statement contains no factual claim or is nonsensical.

First give a brief explanation. Then write a vertical bar "|", followed by the rating as a number alone.

Required format:
brief explanation | rating
"""

def escape_statement(text: str) -> str:
    """Prevent quotation marks in a statement from confusing the prompt."""
    return str(text).replace("\\", "\\\\").replace('"', '\\"').strip()

def build_messages(text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT.format(text=escape_statement(text)),
        },
    ]

