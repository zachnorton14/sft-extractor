"""Calibration route: teach calibrated uncertainty from period text that hedges.

Most routes teach the model to ANSWER. This one teaches it to say "that is not known" —
appropriately, in period voice — instead of confidently making something up. Small models
hallucinate; a model with a pre-1930 worldview especially needs to know the limits of
what its sources settle.

It stays anachronism-safe the same way every other route does: the answer is VERBATIM
period text. It sources passages that already EXPRESS uncertainty (corpus.has_hedge — "it
is not known", "authorities differ", "the cause remains obscure", "a matter of dispute")
and lifts the author's own expression of not-knowing as the answer. The question is
composed to ask the straight question whose honest answer is that hedge — so the model
learns to respond with calibrated uncertainty, not a fabricated fact.

Input:  synth/output/excerpts.jsonl, any factual class, filtered to hedging passages
Output: HF dataset shards calibration_qa/part-*.jsonl (one row per line)
        {"doc_index","category","book_category","year","prose_score","question","answer"}

Env:
    export OPENCODE_API_KEY / DS_API_KEY   # per the active engine provider
"""

import random

from synth import corpus, engine

# Hedges about real, uncertain matters live in the factual/expository classes; source
# broadly across them (an excerpt may carry several) and filter to hedging passages.
CLASSES = ("knowledge", "reasoning", "stem_reasoning", "opinion", "narrative_grounded")
MAX_SPANS = 2                     # the hedge, plus its subject if stated separately

SYSTEM = f"""\
You are given a short passage from a pre-1930s book that, somewhere in it, EXPRESSES
UNCERTAINTY — it says that something is not known, is doubtful or disputed, that
authorities differ, or that a matter cannot be determined. Turn it into a question-and-
answer pair that teaches CALIBRATED UNCERTAINTY: the answer is the author's own
expression of not-knowing, quoted verbatim.

1. FIND THE HEDGE FIRST. Locate where the passage states that something is uncertain,
   unknown, doubtful, or disputed, and copy that statement VERBATIM as the answer.
   - Return it as "spans": one exact quotation from the passage — or up to {MAX_SPANS}
     only if the hedge and the subject it concerns lie in separate places. Prefer one.
   - Take WHOLE sentences: begin and end each span at a sentence boundary and keep its
     terminal punctuation, so the answer stands on its own and plainly voices the doubt.
   - Copy WORD FOR WORD. Do NOT paraphrase, soften, resolve, or explain away the
     uncertainty, and add no word not in the passage.
   - The answer MUST ITSELF express the uncertainty — it has to contain the statement of
     not-knowing / doubt / dispute. If the passage only mentions something uncertain in
     passing without a standing claim of not-knowing, emit NO item for it.

2. THEN WRITE THE QUESTION that this uncertain answer answers.
   - Ask the STRAIGHT question about the uncertain point, as though expecting a definite
     answer — "What is the cause of ...?", "When was ... founded?", "Who first ...?" — so
     that the correct, honest response is precisely the author's expression of
     uncertainty. Do NOT ask "is it known whether ...", and do NOT telegraph the doubt in
     the question; ask the plain question whose honest answer turns out to be "it is not
     known".
   - Self-situating: name the actual subject the uncertainty concerns — the disease,
     place, event, person, work, or quantity — using your own knowledge of it when the
     passage assumes it, so the question makes sense to someone who never saw the passage.
     Never leave a bare "it", "this", or "they".
   - Plain period-schoolbook register, pre-1930s English: period vocabulary and phrasing;
     use no word or idiom that came into use after 1930. Never mention the source — no
     "the passage", "the author", "according to".
   - Do NOT put the answer's distinctive wording in the question.

{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "..."}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """Calibration excerpts: factual-class excerpts that CONTAIN an expression of
    uncertainty (corpus.has_hedge), so the verbatim answer can be the author's own
    not-knowing. n falsy or >= pool returns the whole filtered pool; else a seeded
    sample. Requires `classify` write-back."""
    mat = [r for r in corpus.load_excerpts(cls=CLASSES, drop_broken_math=True)
           if corpus.has_hedge(r["excerpt"])]
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="calibration_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.spans_answer(MAX_SPANS),   # verbatim extraction of the hedge
    passthrough=("prose_score",),
    extra_body=engine.DISABLE_THINKING,
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
