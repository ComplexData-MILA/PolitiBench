"""Remove explicit verdict language from Snopes article text before augmentation."""

import argparse
import re
import pandas as pd
from bs4 import BeautifulSoup

from pipeline_utils import add_file_arguments, read_dataset, save_outputs

VERDICT_PATTERNS = {
    # --------------------------------------------------------
    # Shared / PolitiFact labels
    # --------------------------------------------------------

    "true": (
        r"(?<!mostly\s)(?<!half\s)\btrue\b|"
        r"\bverdader[oa]\b"
    ),

    "mostly true": (
        r"\bmostly[\s-]+true\b|"
        r"\bmayormente\s+verdader[oa]\b"
    ),

    "half true": (
        r"\bhalf[\s-]+true\b|"
        r"\bmitad\s+verdader[oa]\b"
    ),

    "mostly false": (
        r"\bmostly[\s-]+false\b|"
        r"\bbarely[\s-]+true\b|"
        r"\bmayormente\s+fals[oa]\b"
    ),

    "false": (
        r"(?<!mostly\s)\bfalse\b|"
        r"\bfals[oa]\b"
    ),

    "pants on fire": (
        r"\bpants[\s-]+on[\s-]+fire\b|"
        r"\bridícul[oa](?:\s+y\s+fals[oa])?\b"
    ),

    # --------------------------------------------------------
    # Snopes labels
    # --------------------------------------------------------

    "fake": (
        r"\bfake\b|"
        r"\bfabricated\b"
    ),

    "mixture": (
        r"\bmixture\b|"
        r"\ba\s+mixture\b|"
        r"\bmixed\b"
    ),

    "originated as satire": (
        r"\boriginated\s+as\s+satire\b|"
        r"\boriginates?\s+as\s+satire\b|"
        r"\bstemmed\s+from\s+(?:a\s+)?satire\b"
    ),

    "labeled satire": (
        r"\blabeled\s+satire\b|"
        r"\blabelled\s+satire\b"
    ),

    "miscaptioned": (
        r"\bmiscaptioned\b|"
        r"\bmis[\s-]?captioned\b"
    ),

    "correct attribution": (
        r"\bcorrect\s+attribution\b|"
        r"\bcorrectly\s+attributed\b"
    ),

    "misattributed": (
        r"\bmisattributed\b|"
        r"\bmis[\s-]?attributed\b"
    ),

    "outdated": (
        r"\boutdated\b"
    ),

    "scam": (
        r"\bscam\b"
    ),

    "unproven": (
        r"\bunproven\b|"
        r"\bnot\s+proven\b"
    ),

    "legend": (
        r"\blegend\b"
    ),

    "incorrect attribution": (
        r"\bincorrect\s+attribution\b|"
        r"\bincorrectly\s+attributed\b"
    ),

    "unfounded": (
        r"\bunfounded\b"
    ),

    "unfounded (retired)": (
        r"\bunfounded\b"
    ),

    "unproven (retired)": (
        r"\bunproven\b"
    ),

    "research in progress": (
        r"\bresearch\s+in\s+progress\b"
    ),

    "research in progress (retired)": (
        r"\bresearch\s+in\s+progress\b"
    ),

    "legit": (
        r"\blegit\b|"
        r"\blegitimate\b"
    ),

    "recall": (
        r"\brecall\b|"
        r"\bgenuine\s+recall\b"
    ),
}

RULING_HEADING = re.compile(
    r"""
    (?im)^[ \t]*
    (?:
        our[ \t]+ruling |
        our[ \t]+rating |
        our[ \t]+verdict |
        our[ \t]+conclusion |
        the[ \t]+ruling |
        politifact(?:'s)?[ \t]+ruling |
        ruling |
        rating |
        verdict |
        conclusión |
        nuestra[ \t]+conclusión |
        nuestro[ \t]+veredicto
    )
    [ \t]*:?[ \t]*$
    """,
    flags=re.VERBOSE,
)

RATING_CUE = re.compile(
    r"""
    (?:
        # "we rated", "we have thus rated",
        # "we've labeled", "we are rating",
        # "we have deemed", etc.
        \bwe\b
        (?:
            've |
            ’ve |
            \s+(?:
                have |
                had |
                are |
                were |
                also |
                therefore |
                thus |
                now |
                ultimately |
                accordingly |
                consequently
            )
        ){0,5}
        \s+
        (?:
            rate |
            rated |
            rating |
            rule |
            ruled |
            label |
            labeled |
            labelled |
            deem |
            deemed |
            classify |
            classified |
            categorize |
            categorized
        )
        \b

        |

        # "Snopes rates", "Snopes therefore rates",
        # "Snopes has rated", etc.
        \bsnopes\b
        (?:
            \s+(?:
                has |
                had |
                also |
                therefore |
                thus |
                now |
                ultimately |
                accordingly |
                consequently
            )
        ){0,5}
        \s+
        (?:
            rates? |
            rated |
            rating |
            rules? |
            ruled |
            labels? |
            labeled |
            labelled |
            deems? |
            deemed |
            classifies? |
            classified |
            categorizes? |
            categorized
        )
        \b

        |

        # "we find the statement to be correctly attributed"
        # "we found the claim to be false"
        \bwe\b
        (?:
            \s+(?:
                have |
                had |
                also |
                therefore |
                thus
            )
        ){0,4}
        \s+
        (?:find|found)
        \b
        [^.!?]{0,100}?
        \b(?:to\s+be|as)\b
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

CONCLUSION_CUE = re.compile(
    r"""
    (?:
        \bin\s+short\b |
        \bin\s+summary\b |
        \bin\s+conclusion\b |
        \bbottom\s+line\b |
        \ball\s+in\s+all\b |
        \bas\s+such\b |
        \bfor\s+these\s+reasons\b
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def split_sentences_preserving_punctuation(text):
    """
    Split without consuming closing quotes/parentheses.
    """

    if not text:
        return []

    text = re.sub(
        r'([.!?](?:["\'”’)]*))\s+',
        r'\1<SENTENCE_BOUNDARY>',
        text,
    )

    text = re.sub(
        r"\n+",
        "<SENTENCE_BOUNDARY>",
        text,
    )

    return [
        x.strip()
        for x in text.split("<SENTENCE_BOUNDARY>")
        if x.strip()
    ]


def normalize_verdict(verdict):
    if verdict is None or pd.isna(verdict):
        return None

    return re.sub(
        r"[\s_-]+",
        " ",
        str(verdict).strip().lower(),
    )

DIRECT_CLAIM_JUDGMENT = re.compile(
    r"""
    \b(?:this|the|that)\s+
    (?:claim|statement|assertion)\b
    [^.!?]{0,300}?
    \b(?:is|was)\b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


RATING_TRANSITION = re.compile(
    r"""
    (?:
        [;,]\s*
    )?
    \b(?:
        therefore |
        thus |
        hence |
        consequently |
        accordingly |
        as\s+such |
        for\s+this\s+reason |
        for\s+these\s+reasons
    )\b
    [,:]?\s*
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def remove_rating_clause(sentence, verdict):
    verdict_key = normalize_verdict(verdict)
    verdict_pattern = VERDICT_PATTERNS.get(verdict_key)

    # Unknown label -> conservatively keep everything
    if verdict_pattern is None:
        return sentence

    verdict_matches = list(
        re.finditer(
            verdict_pattern,
            sentence,
            flags=re.IGNORECASE,
        )
    )

    if not verdict_matches:
        return sentence

    # ----------------------------------------------------
    # 1. Explicit Snopes author-rating language
    # ----------------------------------------------------

    cue = RATING_CUE.search(sentence)

    if cue:
        # Require the actual article verdict to occur
        # in/after the rating clause.
        verdict_after_cue = any(
            match.start() >= cue.start()
            for match in verdict_matches
        )

        if verdict_after_cue:
            cut_start = cue.start()

            prefix = sentence[:cue.start()]

            # Preserve evidence before:
            # "... confirmed the video was authentic;
            # therefore, we have rated..."
            transitions = list(
                RATING_TRANSITION.finditer(prefix)
            )

            if transitions:
                transition = transitions[-1]

                # Only treat it as part of the rating clause
                # if it is close to the rating language.
                if (
                    cue.start()
                    - transition.end()
                    <= 50
                ):
                    cut_start = transition.start()

            # Also handle:
            # "evidence here; we rated..."
            semicolon = sentence.rfind(
                ";",
                0,
                cue.start(),
            )

            if (
                cut_start == cue.start()
                and semicolon != -1
                and cue.start() - semicolon <= 120
            ):
                cut_start = semicolon

            # Handle:
            # "... evidence, and we rated..."
            if cut_start == cue.start():
                connector = re.search(
                    r"(?:,\s*)?\b(?:and|so)\s*$",
                    prefix,
                    flags=re.IGNORECASE,
                )

                if connector:
                    cut_start = connector.start()

            kept = sentence[:cut_start].rstrip(
                " ,;:-"
            )

            return kept

    # ----------------------------------------------------
    # 2. Direct meta-verdict:
    # "The claim was false."
    # "The claim that X is mostly true."
    # ----------------------------------------------------

    direct = DIRECT_CLAIM_JUDGMENT.search(sentence)

    if direct:
        # Only treat it as an explicit verdict if the actual
        # verdict comes immediately after "is/was".
        #
        # YES: "The claim was false."
        # YES: "The claim that X is mostly true."
        # NO:  "The claim is based on a false Facebook post."
        tail = sentence[direct.end():]

        direct_verdict = re.match(
            rf"""
            \s*
            (?:
                clearly\s+ |
                indeed\s+ |
                simply\s+
            )?
            (?:{verdict_pattern})
            """,
            tail,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        if direct_verdict:
            return ""

    # ----------------------------------------------------
    # 3. Conclusion wording:
    # "In short, the image is fake."
    # "Bottom line: ... false."
    #
    # Preserve anything before the conclusion cue.
    # ----------------------------------------------------

    conclusion = CONCLUSION_CUE.search(sentence)

    if conclusion:
        kept = sentence[
            :conclusion.start()
        ].rstrip(" ,;:-")

        return kept

    return sentence

def clean_article_for_augmentation(text, verdict):

    if text is None or pd.isna(text):
        return ""

    text = BeautifulSoup(
        str(text),
        "html.parser",
    ).get_text("\n")

    # Remove an explicitly labeled ruling section
    heading_match = RULING_HEADING.search(text)

    if heading_match:
        text = text[:heading_match.start()]

    sentences = split_sentences_preserving_punctuation(
        text
    )

    cleaned_sentences = []

    for sentence in sentences:
        sentence = re.sub(
            r"\s+",
            " ",
            sentence,
        ).strip()

        if not sentence:
            continue

        sentence = remove_rating_clause(
            sentence,
            verdict,
        )

        sentence = sentence.strip()

        if not sentence:
            continue

        cleaned_sentences.append(sentence)

    cleaned_text = " ".join(
        cleaned_sentences
    )

    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text,
    ).strip()

    return cleaned_text


def clean_dataframe(df):
    df = df.copy()
    if "analysis_text_raw" not in df.columns:
        df["analysis_text_raw"] = df["analysis_text"]
    df["analysis_text"] = [
        clean_article_for_augmentation(row["analysis_text"], row["verdict"])
        for row in df.to_dict(orient="records")
    ]
    return df.drop_duplicates("article_url")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_file_arguments(parser, "Data/Snopes_DATA.parquet", "Snopes_Cleaned")
    return parser.parse_args()


def main():
    args = parse_args()
    df = read_dataset(args.input_path, ["analysis_text", "verdict", "article_url"])
    result = clean_dataframe(df)
    save_outputs(result, args.output_dir, args.output_name)
    print(f"Saved {len(result)} articles after leakage cleanup and URL deduplication.")


if __name__ == "__main__":
    main()
