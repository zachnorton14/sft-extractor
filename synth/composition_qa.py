"""Generative route: turn formal period documents into compose-this-document rows.

The fourth content-bound generator, and the first GENERATIVE one. Where narrative asks
the reader to *recount* an episode, this asks the reader to *produce* an artifact — a
statute, a legal pleading, a letter, a speech or resolution, a prayer, a proclamation.
It reads the materialized excerpt corpus (synth/output/excerpts.jsonl from `harvest`)
and keeps the ones the affordance gate tags `composition` (see corpus.is_composition).

It is still EXTRACTIVE, and that is the whole point: the answer is a genuine period
document lifted verbatim, so the model learns to EMIT period-authentic formal prose in
its proper form, with no anachronism risk (it can only select real period text, never
compose modern-sounding text). The question is the one thing that differs from the
recall routes — it is an INSTRUCTION to compose ("Draft an act to ...", "Frame a
resolution ...", "Write a prayer for ..."), naming the genre and subject without
dictating the wording.

  - the ANSWER is the verbatim artifact — prefer ONE long contiguous span that is the
    document (or a self-contained part: a full clause, a full letter, a full stanza of
    the oration). Verified as a literal substring (verbatim_answer); non-verbatim
    dropped.
  - the QUESTION is a compose-instruction: genre + subject, period register, standing
    alone, not reproducing the artifact's wording.

Input:  synth/output/excerpts.jsonl (affordance == composition)
Output: synth/output/composition_qa.json
        [{"doc_index","category","book_category","category_moved","year",
          "prose_score","excerpt","question","answer"}]

Env (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import random

from synth import corpus, engine

AFFORDANCE = "composition"        # this generator handles the generative route
MAX_SPANS = 3                     # prefer one long span; splice only a split document

SYSTEM = f"""\
You are given a short passage from a pre-1930s book that IS, or contains, a formal
COMPOSED DOCUMENT — a statute or act, a legal pleading, a letter, a speech or
resolution, a prayer or devotion, a proclamation, a contract or like instrument. Turn
it into an instruction-and-completion pair in which the reader is asked to COMPOSE such
a document. Work in this order.

1. FIND THE ARTIFACT FIRST. Identify the composed document in the passage and copy it
   VERBATIM as the answer.
   - Return it as "spans": prefer ONE long contiguous quotation that is the document,
     or a self-contained part of it (a whole clause, a whole letter, a whole stanza of
     the oration). Use several exact quotations only when it is genuinely split — never
     more than {MAX_SPANS}, always the fewest, longest spans. Begin at a natural start
     (a heading, salutation, "Whereas", "Resolved, that", "O Lord") and end at a
     natural close.
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
     reorder within a span, or add any word not in the passage — including connecting
     words between spans. You may only select and order the passage's own text.

2. THEN WRITE THE INSTRUCTION that asks for this document.
   - Name the GENRE and the SUBJECT: "Draft an act of Parliament to ...", "Compose a
     letter ...", "Frame a resolution ...", "Write a prayer for ...", "Set out a
     plaintiff's plea that ...". It must call for exactly the kind of document the
     answer is, on its actual subject and occasion.
   - Give the instruction the substance the document needs to be a fitting answer — its
     purpose, parties, or occasion — but do NOT copy the document's wording or dictate
     its phrasing. Ask for the thing; let the answer supply the words.
   - Plain period register: direct, no modern or conversational phrasing, no
     meta-language. Write it in pre-1930s English — period vocabulary, spelling, and
     phrasing; use no word, term, or idiom that came into use after 1930.
   - Stands alone. Never mention the source — no "the passage", "the text", "the
     author", "as given", "above", "described". Phrase it as a task set from scratch.
   - Do NOT reproduce the document's distinctive wording in the instruction.

If the passage is NOT actually a composed document of a recognizable genre — it is
ordinary narration, exposition, or argument — emit NO item for it (omit that index).

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "..."}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """Composition excerpts from the materialized corpus (affordance == composition).

    n falsy or >= pool size returns the whole pool (full run); otherwise a seeded
    random sample, so a --test/--sample with a given --count/--seed varies across
    seeds instead of always taking the first n in file order."""
    mat = corpus.load_excerpts(affordance=AFFORDANCE)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="composition_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.spans_answer(MAX_SPANS),   # verbatim artifact extraction
    passthrough=("prose_score",),
)


# Thin delegates so run.py (and tests) keep the familiar module-level API.
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
