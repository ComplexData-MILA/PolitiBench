"""Claim decontextualization instructions and few-shot examples."""

from textwrap import dedent


FEW_SHOT_EXAMPLES = [
    # 1. POLICY/NOUN REFERENCE: "this bill" -> named legislation
    {
        "source": (
            "https://www.snopes.com/fact-check/"
            "house-democrats-insider-trading-bill/"
        ),
        "claim": (
            "Social media users, stated on July 23, 2026 in posts: "
            "“198 Democrats voted against this bill.”"
        ),
        "context": dedent("""
            Posts circulated after a July 22 vote in the U.S. House of
            Representatives. The House passed H.R. 7008, formally named
            the Stop Insider Trading Act, by a vote of 232-198. All 198
            opposing votes came from Democratic representatives.

            The legislation concerned stock trading by members of
            Congress and their spouses. The article also discussed
            disagreements about the bill's restrictions, exemptions and
            other provisions. In the quoted post, “this bill” referred
            specifically to the Stop Insider Trading Act.
        """).strip(),
        "output": (
            "Social media users, stated on July 23, 2026 in posts: "
            "“198 Democrats voted against the Stop Insider Trading Act.”"
        ),
    },

    # 2. PRONOUN REFERENCES: "them" and "they" -> NATO
    # Preserve "Iceland" even though the article discusses whether Trump
    # meant Greenland. That is fact-checking, not augmentation.
    {
        "source": (
            "https://www.snopes.com/fact-check/"
            "trump-greenland-iceland-leavitt/"
        ),
        "claim": (
            "Donald Trump, stated on January 21, 2026 in a World "
            "Economic Forum speech: “Until the last few days when I "
            "told them about Iceland, they loved me.”"
        ),
        "context": dedent("""
            President Donald Trump delivered the statement during an
            address at the World Economic Forum in Davos. The surrounding
            remarks concerned the United States, Greenland and its
            relationship with the North Atlantic Treaty Organization.

            Immediately before and after the quoted sentence, Trump
            discussed NATO members and their reaction to his position.
            In this sentence, both “them” and “they” referred to NATO.
            The article separately examined whether Trump mistakenly
            said Iceland when discussing Greenland, but the spoken word
            in the quotation was Iceland.
        """).strip(),
        "output": (
            "Donald Trump, stated on January 21, 2026 in a World "
            "Economic Forum speech: “Until the last few days when I "
            "told NATO about Iceland, NATO loved me.”"
        ),
    },

    # 3. RELATIVE TIME: "today" -> absolute statement date
    {
        "source": (
            "https://www.snopes.com/fact-check/"
            "trump-uss-lincoln-nda/"
        ),
        "claim": (
            "Social media users, stated on August 17, 2026 in posts: "
            "“Donald Trump announced today that sailors aboard the USS "
            "Abraham Lincoln must sign nondisclosure agreements before "
            "leaving the ship.”"
        ),
        "context": dedent("""
            Social media posts circulating that day claimed President
            Donald Trump had announced a restriction affecting sailors
            aboard the USS Abraham Lincoln. According to the posts, the
            sailors would not be allowed to leave the aircraft carrier
            until they signed legally binding nondisclosure agreements.

            The article investigated the alleged announcement using
            publicly available statements, reporting about the ship and
            information attributed to Navy officials. Regardless of
            whether the alleged announcement was genuine, “today” in the
            circulating claim referred to August 17, 2026.
        """).strip(),
        "output": (
            "Social media users, stated on August 17, 2026 in posts: "
            "“Donald Trump announced on August 17, 2026 that sailors "
            "aboard the USS Abraham Lincoln must sign nondisclosure "
            "agreements before leaving the ship.”"
        ),
    },

    # 4. UNNAMED ENTITY: "the company" -> State Farm
    {
        "source": (
            "https://www.snopes.com/fact-check/"
            "state-farm-mutual-auto-dividend/"
        ),
        "claim": (
            "Social media users, stated on August 20, 2026 in posts: "
            "“The company started distributing $5 billion in cash to "
            "its automobile-insurance customers.”"
        ),
        "context": dedent("""
            Posts and news reports discussed a large distribution being
            made to automobile-insurance customers. The company at the
            center of the reports was State Farm Mutual Automobile
            Insurance Company, commonly known as State Farm.

            The article explained that State Farm had announced a
            distribution totaling approximately $5 billion and described
            which customers could receive money and how amounts would be
            calculated. It also distinguished the distribution from an
            ordinary refund. In the quoted claim, “the company” referred
            to State Farm.
        """).strip(),
        "output": (
            "Social media users, stated on August 20, 2026 in posts: "
            "“State Farm started distributing $5 billion in cash to "
            "its automobile-insurance customers.”"
        ),
    },

    # 6. UNCHANGED: already identifies the jurisdiction, action and time
    {
        "source": (
            "https://www.snopes.com/fact-check/"
            "massachusetts-abortion-until-birth/"
        ),
        "claim": (
            "Social media users, stated on August 11, 2026 in posts: "
            "Massachusetts legalized abortion after 24 weeks of pregnancy."
        ),
        "context": dedent("""
            Posts claimed that Massachusetts had legalized abortion after
            24 weeks of pregnancy. The article examined legislation
            enacted in August 2026 and compared it with the state's
            previous abortion law.

            The reporting described the circumstances under which the law
            permitted an abortion after 24 weeks and explained how the
            legislation changed existing language. It also discussed
            disagreements over descriptions such as “until birth.”
            Although those details may affect a verdict, the original
            claim already identifies the jurisdiction, action, subject
            and relevant time period without relying on an unresolved
            reference.
        """).strip(),
        "output": (
            "Social media users, stated on August 11, 2026 in posts: "
            "Massachusetts legalized abortion after 24 weeks of pregnancy."
        ),
    },

    # 7. UNCHANGED: implausibility or falsity is not ambiguity
    {
        "source": (
            "https://www.snopes.com/fact-check/"
            "nasa-project-anchor-earth-gravity/"
        ),
        "claim": (
            "Social media users, stated on January 12, 2026 in posts: "
            "A leaked NASA document called “Project Anchor” says Earth "
            "will lose gravity for seven seconds on August 12, 2026."
        ),
        "context": dedent("""
            Online posts described an allegedly secret NASA project named
            Project Anchor. They claimed that Earth would temporarily
            lose gravity for seven seconds on August 12, 2026 and that
            NASA knew about the approaching event.

            The article searched NASA materials, examined the supposed
            document and discussed a solar eclipse scheduled for that
            date. It found no scientific mechanism or authentic NASA
            announcement supporting the gravity claim. That evidence is
            relevant to fact-checking, but the claim itself already names
            the organization, alleged document, event, duration and exact
            date. It therefore requires no augmentation.
        """).strip(),
        "output": (
            "Social media users, stated on January 12, 2026 in posts: "
            "A leaked NASA document called “Project Anchor” says Earth "
            "will lose gravity for seven seconds on August 12, 2026."
        ),
    },
    {
        "source": (
            "https://www.snopes.com/fact-check/"
            "ron-johnson-social-security-quote/"
        ),
        "claim": (
            "Social media users, stated on August 21, 2026 in posts: "
            "U.S. Sen. Ron Johnson once said, “Senior citizens have "
            "misused Social Security and that's why I want it to sunset.”"
        ),
        "context": dedent("""
            Social media posts attributed a statement about Social Security
            to Wisconsin Sen. Ron Johnson. The posts claimed that Johnson
            said senior citizens had misused Social Security and that he
            therefore wanted the program to sunset.

            The article examined speeches, interviews, congressional
            records and other publicly available material to determine
            whether Johnson had made the statement. It also reviewed
            remarks Johnson had made about changing how federal programs
            such as Social Security are funded and reconsidered.

            Within the attributed quotation, “it” refers directly to Social
            Security, which is named earlier in the same sentence. The
            first-person pronoun “I” refers to Johnson, who is identified
            in the attribution.
        """).strip(),
        "output": (
            "Social media users, stated on August 21, 2026 in posts: "
            "U.S. Sen. Ron Johnson once said, “Senior citizens have "
            "misused Social Security and that's why I want it to sunset.”"
        ),
    },
    {
        "source": (
            "https://www.snopes.com/fact-check/"
            "trump-girls-kneeling-fake/"
        ),
        "claim": (
            "Social media users, stated in early 2026 in posts: "
            "An image authentically shows Trump smiling in a white tank "
            "top while young girls kneel before him."
        ),
        "context": dedent("""
            Social media users circulated an alleged black-and-white
            photograph in early 2026. Captions accompanying the image
            claimed it had appeared among files connected with the federal
            investigation of Jeffrey Epstein.

            The image purported to show U.S. President Donald Trump smiling
            and wearing a white tank top while several young girls kneeled
            in front of him. The article examined the image's provenance,
            searched for an authentic archival version and considered
            visual indications that it had been artificially generated.

            The article's findings concerned whether the image was
            authentic. Throughout the article and the accompanying posts,
            the surname “Trump” referred to Donald Trump.
        """).strip(),
        "output": (
            "Social media users, stated in early 2026 in posts: "
            "An image authentically shows Trump smiling in a white tank "
            "top while young girls kneel before him."
        ),
    },
]

def format_few_shot_examples(examples: list[dict]) -> str:
    blocks = []

    for number, example in enumerate(examples, start=1):
        block = dedent(f"""
            EXAMPLE {number}

            CLAIM:
            <claim>
            {example["claim"]}
            </claim>

            CONTEXT:
            <context>
            {example["context"]}
            </context>

            OUTPUT:
            {example["output"]}
        """).strip()

        blocks.append(block)

    return "\n\n".join(blocks)

def build_aug_prompt(statement: str, context: str):
    examples_text = format_few_shot_examples(FEW_SHOT_EXAMPLES)

    return [
        {
            "role": "system",
            "content": (
                "You are a claim-augmentation editor. Resolve contextual "
                "ambiguities so the claim can be understood without the "
                "surrounding article, while preserving exactly the proposition "
                "being asserted."
            ),
        },
        {
            "role": "user",
            "content": dedent(f"""
                Augment the CLAIM only when information from the CONTEXT
                is necessary to make the claim standalone and unambiguous.

                WHEN AUGMENTATION IS APPROPRIATE

                Resolve a context-dependent expression when its referent
                is uniquely established by the CONTEXT. This includes:

                - pronouns or demonstratives without a clear referent,
                  such as they, them, it, this or those;
                - vague noun phrases, such as this bill, the company,
                  the policy, the decision or the incident;
                - unnamed people, organizations, laws, programs,
                  documents, events, places or comparison targets;
                - relative time expressions, such as today, yesterday,
                  last year or two years ago;
                - shortened or incomplete names that do not uniquely
                  identify their referent without the article.

                Make an edit only when a specific expression has no clear referent
                within the CLAIM itself. Replace that expression with its uniquely
                supported referent from the CONTEXT and add no other specificity.
                Otherwise, return the CLAIM unchanged.

                PRESERVATION RULES

                - Use only information contained in the CLAIM or explicitly supported
                by the CONTEXT.
                - Preserve everything through the first colon exactly. Edit only the
                claim text after it.
                - Preserve the proposition's meaning exactly.
                - Preserve quantities, polarity, certainty, modality, causality,
                comparisons, attribution and time scope.
                - Preserve the claimant's wording and characterization, even if it is
                informal, awkward, exaggerated, misleading or factually incorrect.
                - When resolving an ambiguity inside quoted text, replace only the
                context-dependent expression and preserve all other quoted wording.
                - Do not correct the claim using the article's evidence or conclusion.
                - Do not add explanations, evidence, qualifications, exceptions,
                verdicts or background information.
                - Do not expand a shortened name or abbreviation merely because the
                CONTEXT provides a fuller or more technical name. Expand it only
                when the original expression is genuinely ambiguous.
                - Do not replace a pronoun when its referent is already clear within
                the same sentence. Do not correct spelling.
                - First-person references such as I, me and my are normally clear when
                the attribution prefix identifies the speaker and should remain
                unchanged.
                - Do not replace what a claim says an image, video, quotation or
                document depicts with the article's conclusion about its
                authenticity.
                - If the claim is already standalone, return it exactly unchanged.
                - If the CONTEXT does not establish exactly one referent with sufficient
                confidence, return the claim exactly unchanged.

                OUTPUT

                Return only one complete, continuous claim. Do not include a label,
                explanation, notes or formatting around it.

                {examples_text}

                NOW PROCESS THE FOLLOWING INPUT

                CLAIM:
                <claim>
                {statement}
                </claim>

                CONTEXT:
                <context>
                {context}
                </context>

                RESULT:
            """).strip(),
        },
    ]

