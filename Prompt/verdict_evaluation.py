"""Shared PolitiFact verdict prompt for both evaluation backends."""

from textwrap import dedent


def build_eval_messages(
        claim: str, 
        evidence: str
        ):
    return [
            {
                "role": "system", 
                "content": dedent(
                    """
                    A conversation between User and Assistant. The user asks a question,
                    and the Assistant solves it. The assistant first thinks about the
                    reasoning process in the mind and then provides the user with the
                    answer.
                    """
                ).strip()
            },
            {
                "role": "user", 
                "content": dedent(
                    f"""
                    You are given a PolitiFact claim and the article's analysis text.
                    Use the article evidence to examine the claim the way a fact-checker
                    would: identify what the claim says, compare it against the evidence,
                    and decide the PolitiFact verdict.

                    Allowed verdict labels:
                    - True
                    - Mostly true
                    - Half true
                    - Mostly false
                    - False
                    - Pants on fire

                    Claim: {claim}
                    Article evidence:
                    {evidence}

                    Return only one allowed verdict label.
                    """
                )
            }
        ]

