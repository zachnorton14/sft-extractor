"""Opinion route: turn opinion excerpts into view-and-ground Q/A rows.

Sources excerpts the classifier tagged `opinion` — passages advancing a judgment,
preference, or stance. Extractive: the ANSWER is the verbatim view together with the
ground it rests on; the QUESTION asks what was held or urged. Because an opinion is a
held position, not a fact, the question attributes it — to the thinker/school when that
is on record and nameable, otherwise as a position on the subject — so it stands alone
without ever saying "the author".

Input:  synth/output/excerpts.jsonl (classes contains "opinion")
Output: synth/output/opinion_qa.json

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import random

from synth import corpus, engine

CLASSES = ("opinion",)            # classifier classes this route sources
MAX_SPANS = 3

SYSTEM = f"""\
You are given a short passage from a pre-1930s book that advances an OPINION — a
judgment, preference, or stance the writer argues for. Work in this order.

1. FIND THE VIEW FIRST. Choose the opinion the passage advances together with the
   ground it rests on, and copy it VERBATIM as the answer.
   - Return it as "spans": exact quotations that state the view and (where the passage
     gives it) the reason for it. Prefer ONE span; use up to {MAX_SPANS} only if the
     view and its ground lie apart. Fewest, longest spans.
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
     reorder, or add any word not in the passage.
   - Take WHOLE sentences: begin and end each span at a sentence boundary and keep its
     terminal punctuation, so several spans read as continuous prose when joined.

2. THEN WRITE THE QUESTION the view answers.
   - Ask what was held or urged, NOT a matter of fact: "What did ... hold concerning
     ...?", "What was urged for (or against) ...?", "On what ground was ... defended?".
     The spans must fully give the view (and its ground) that you ask for.
   - ATTRIBUTE the view so the question stands alone: to the thinker, school, party, or
     tradition that held it when that is a matter of record you can name (e.g. "the
     Stoics", "the Free-Trade party", "Dr. Johnson"); otherwise frame it as a position
     on the subject ("What was urged against a standing army in time of peace?"). Do
     NOT say "the author", "the writer", "the passage", "according to".
   - Plain period register, pre-1930s English — no word or idiom that came into use
     after 1930, no modern or conversational phrasing.
   - Self-situating — name the actual subject of the opinion so the question makes sense
     to someone who never saw the passage.
   - Do NOT put the view's distinctive wording in the question; ask for it.

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "..."}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """Opinion excerpts from the materialized corpus (classes contains "opinion"). n
    falsy or >= pool returns the whole pool; otherwise a seeded random sample.
    Requires `classify` write-back to have run."""
    mat = corpus.load_excerpts(cls=CLASSES)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="opinion_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.spans_answer(MAX_SPANS),   # verbatim extraction
    passthrough=("prose_score",),
    extra_body=engine.DISABLE_THINKING,         # no thinking trace: fast, no truncation
)


def load_state():
    return engine.load_state(ROUTE)


def save_state(state):
    engine.save_state(ROUTE, state)


def run_async(excerpts, state):
    return engine.run_async(ROUTE, excerpts, state)


def write_output(excerpts, state):
    return engine.write_output(ROUTE, excerpts, state)


def test_run(excerpts):
    return engine.test_run(ROUTE, excerpts)


def sample_run(excerpts, seed=0):
    return engine.sample_run(ROUTE, excerpts, seed)
