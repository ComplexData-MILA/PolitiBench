"""Evidence extraction instructions and few-shot examples."""

import json


EXAMPLE_1_INPUT = """
    Claim:
    The state spent $50 million on a benefits website that never worked.

    Article:
    During a campaign speech, Jordan Lee said the state spent $50
    million on a benefits website that never worked. The original
    contract authorized spending of up to $50 million, but state
    invoices show that $8.7 million had been paid. The website
    experienced several outages during its first month. State records
    show that it nevertheless processed more than 120,000 benefit
    applications during its first year. Officials later approved
    additional repairs to improve its reliability.
    """.strip()

EXAMPLE_1_OUTPUT = {
    "supporting_facts": [],
    "weakening_facts": [
        {
            "fact": (
                "State records show the website processed more than "
                "120,000 benefit applications during its first year."
            ),
            "importance": 5,
        },
        {
            "fact": (
                "State invoices show that $8.7 million had been paid, "
                "not $50 million."
            ),
            "importance": 5,
        },
    ],
    "missing_context": [
        {
            "fact": (
                "The $50 million figure was the contract's maximum "
                "authorized amount rather than actual spending."
            ),
            "importance": 4,
        }
    ],
}

EXAMPLE_2_INPUT = """
    Claim:
    The policy reduced unemployment by 20% in one year.

    Article:
    The governor repeated the claim that the policy reduced
    unemployment by 20%. State labor data show that unemployment
    fell from 6.0% to 5.4% during the year after the policy began.
    That represents a 10% reduction, not 20%. Economists interviewed
    for the article said the available data could not attribute the
    entire decline to the policy.
    """.strip()

EXAMPLE_2_OUTPUT = {
    "supporting_facts": [
        {
            "fact": (
                "State labor data show unemployment fell "
                "after the policy began."
            ),
            "importance": 3,
            "support_type": "partial",
        }
    ],
    "weakening_facts": [
        {
            "fact": (
                "Unemployment fell by 10%, not the claimed 20%."
            ),
            "importance": 5,
        },
        {
            "fact": (
                "Economists could not attribute the entire "
                "unemployment decline to the policy."
            ),
            "importance": 4,
        },
    ],
    "missing_context": [],
}

def build_examples() -> list[dict]:
    return [
        {
            "role": "user",
            "content": EXAMPLE_1_INPUT,
        },
        {
            "role": "assistant",
            "content": json.dumps(
                EXAMPLE_1_OUTPUT,
                ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": EXAMPLE_2_INPUT,
        },
        {
            "role": "assistant",
            "content": json.dumps(
                EXAMPLE_2_OUTPUT,
                ensure_ascii=False,
            ),
        }
    ]

def build_extraction_messages(
    statement: str,
    article: str,
    k: int = 2,
):
    return [
        {
            "role": "system",
            "content": (
                "Extract concise factual evidence from the supplied article "
                "for fact-checking the supplied claim. "
                "Use only the article. Do not invent facts, use outside "
                "knowledge, or provide a verdict.\n\n"

                "Category definitions:\n"
                "- supporting_facts: Independent article evidence showing "
                "that the claim is accurate.\n"
                "- weakening_facts: Article evidence contradicting, "
                "disproving, or seriously weakening the claim.\n"
                "- missing_context: Omitted article information that "
                "materially changes how the claim should be interpreted.\n\n"

                "Importance levels:\n"
                "- 1: minor relevance\n"
                "- 2: useful but weak\n"
                "- 3: important\n"
                "- 4: strong\n"
                "- 5: decisive\n\n"

                "Rules:\n"
                "- Use support_type='full' only when the fact independently "
                "supports the complete central claim.\n"
                "- Use support_type='partial' when the fact supports only a "
                "number, quote, date, person, premise, or subclaim.\n"
                "- Classify each fact by its evidential effect on the claim, "
                "not merely by whether it discusses the same topic.\n"
                "- Supporting evidence must independently verify all or part "
                "of the claim.\n"
                "- A headline, caption, label, or description attached to an image "
                "or video is not evidence that its interpretation is true. Only "
                "independently verifiable features of the media may be supporting evidence.\n"
                "- A speaker, campaign, political party, lawyer, social-media "
                "post, complaint, headline, or report repeating, alleging, "
                "defending, or claiming to have identified something is not "
                "supporting evidence unless the article independently verifies "
                "the underlying information.\n"
                "- Quotation marks alone do not make a claim an attribution claim. "
                "They usually identify the wording whose truth must be checked.\n"
                "- Evidence that a person said, tweeted, posted, or published the "
                "checked words is not support for their truth.\n"
                "- Attribution evidence is relevant only when the proposition's "
                "central factual assertion explicitly concerns whether a person "
                "or organization made or published a particular statement.\n"
                "- Put information in weakening_facts when it directly contradicts "
                "the claim, shows that its evidence is unverified, or provides an "
                "alternative explanation for the alleged event.\n"
                "- Use missing_context only for an omitted qualification or "
                "background fact that materially changes interpretation without "
                "directly contradicting the claim. If it directly undermines the "
                "claim, classify it as weakening_facts.\n"
                "- Do not return semantic duplicates or multiple paraphrases "
                "of the same underlying evidence or conclusion.\n"
                "- Do not extract or mention the article's verdict or rating.\n"
                "- Sort each category from highest to lowest importance."
            ),
        },
        *build_examples(),
        {
            "role": "user",
            "content": (
                f"Claim:\n{statement}\n\n"
                f"Article:\n{article}\n\n"
                f"Extract up to {k} materially distinct facts per category.\n"
                "Include additional facts only when they provide nonredundant evidence.\n"
                "Sort facts from highest to lowest importance.\n"
                "Each fact must be one short sentence of at most 25 words.\n\n"
            ),
        },
    ]

