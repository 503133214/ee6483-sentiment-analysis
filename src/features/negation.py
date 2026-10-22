"""Negation-scope marking for bag-of-words features (Das & Chen 2001; Pang, Lee & Vaithyanathan 2002).

Every word after a negator (not, no, never, n't, cannot, without, dont, ...) up to
the next clause boundary gets a `NOT_` prefix, so "not good" becomes "not NOT_good"
and the classifier can learn that NOT_good is a different feature from good.

Clause boundaries: . , ; : ! ? ( ) [ ] a newline, a whitespace-delimited dash,
"--", an em dash or an ellipsis. Double quotes are NOT boundaries (the tokeniser
cannot tell an opening quote from a closing one: `not "old ladyish"`).

The tokeniser is Unicode-aware and keeps decimals (8.99), ordinals (8th) and
apostrophe contractions (don't / don’t) as single tokens, so that after
TfidfVectorizer's own tokenisation the negated and non-negated pipelines see the
same vocabulary except for the NOT_ features.

Only used for TF-IDF models: never apply it before GloVe/BERT tokenisers.
"""
from __future__ import annotations

import re

NEGATORS = {
    "not", "no", "never", "cannot", "without", "nothing", "nobody", "none",
    "neither", "nor", "hardly", "barely", "scarcely", "seldom", "nowhere",
    # apostrophe-less contractions are common in reviews ("didnt fit")
    "dont", "didnt", "doesnt", "cant", "wont", "isnt", "wasnt", "arent", "werent",
    "couldnt", "wouldnt", "shouldnt", "havent", "hasnt", "hadnt", "aint",
}
SCOPE_END = set(".,;:!?()[]")
# whitespace-delimited single dash, doubled dashes, em dash, ellipsis -> clause boundary
CLAUSE_DASH_RE = re.compile(r"(?<=\s)-(?=\s)|-{2,}|[—…]")
# decimals (not followed by a letter) | unicode words with optional 't-style suffix | newline | single punctuation | underscore
TOKEN_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?(?![^\W_])|[^\W_]+(?:['’][^\W\d_]+)?|\n|[^\s\w]|_")


def mark_negation(text: str) -> str:
    text = CLAUSE_DASH_RE.sub(" ; ", text)
    out, negated = [], False
    for tok in TOKEN_RE.findall(text):
        if tok == "\n":
            negated = False
            continue
        if tok[0].isalnum():
            out.append("NOT_" + tok if negated else tok)
            low = tok.lower().replace("’", "'")
            if low in NEGATORS or low.endswith("n't"):
                negated = True
        else:
            if tok in SCOPE_END:
                negated = False
            out.append(tok)
    return " ".join(out)


if __name__ == "__main__":
    for s in [
        "I do not like it, but the price was great.",
        "Not bad at all!",
        "It doesn't fit and I won't buy again",
        'but not "old ladyish". Fine otherwise',
        "didnt like the color\nGreat price though",
        "I haven't noticed any problems with smelling yet--hopefully I won't",
        "Not only charged 8.99 for it, café style, 8th pair",
    ]:
        print(mark_negation(s))
