"""Narrative route: turn narrated episodes into reading-comprehension Q/A rows.

The third content-bound generator. It reads the materialized excerpt corpus
(synth/output/excerpts.jsonl from `harvest`) and keeps the ones the affordance gate
tagged `narrative` — passages that recount events: a scene, an episode, an account of
what happened. Like knowledge-QA it is EXTRACTIVE — the answer is a verbatim span of
period text, never composed — so the anachronism guarantee holds: an answer built
only from the passage cannot carry modern phrasing.

What differs from knowledge-QA is the QUESTION framing, not the answer. Two shapes
both fit extraction:
  - RETELL — when the episode is a matter of record ("Washington crossing the
    Delaware", "the burning of Moscow"), the question names it from general knowledge
    and asks the reader to recount it ("Recount how ...", "Tell how ... came about"),
    and the answer is the verbatim narration itself — one long span.
  - COMPREHEND — when the episode is a particular scene the reader has not seen (a
    fiction, an obscure incident), the question sets the scene up from within —
    briefly introducing the actors and circumstance, the way the reasoning route
    introduces persons "described only in the passage" — and asks what happened.

  - the ANSWER is a VERBATIM quotation of what the passage narrates — prefer ONE long
    contiguous span (the account itself); splice more only when the answer is spread
    out. Verified as a literal substring (verbatim_answer); anything not found
    verbatim is dropped.
  - the QUESTION is period-register, stands alone, names the episode when it is on
    record else establishes the scene from within, and never mentions the source.

Input:  synth/output/excerpts.jsonl (affordance == narrative)
Output: synth/output/narrative_qa.json
        [{"doc_index","category","book_category","category_moved","year",
          "prose_score","excerpt","question","answer"}]

Env (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import random

from synth import corpus, engine

AFFORDANCE = "narrative"          # this generator handles the narrative route
MAX_SPANS = 3                     # prefer one long span; splice only a spread account

SYSTEM = f"""\
You are given a short passage from a pre-1930s book that is MEANT to narrate events — a
scene, an episode, an account of what happened.

FIRST, judge whether it actually does. A narrated episode has actors doing or
undergoing things in sequence. If instead the passage is exposition or analysis, a
biographical or catalog entry ("APPLETON, Nathan, manufacturer, was born..."), a list,
table, or section heading, an examination paper, a definition, an argument, or a formal
document, then it has no episode to recount — emit NO item for it (omit that index).
Only proceed when there is a real episode. Better to skip than to force a question the
passage does not narrate.

When there IS an episode, work in this order.

1. FIND THE ANSWER FIRST. Choose the account the passage gives — the narrated episode,
   or the salient action, outcome, or turn within it — and copy it VERBATIM as the
   answer.
   - Return it as "spans": prefer ONE long contiguous quotation that carries the whole
     account. Use several exact quotations only when the answer is genuinely spread out
     — never more than {MAX_SPANS}, always the fewest, longest spans that hold it
     together.
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
     reorder within a span, or add any word not in the passage — including connecting
     words between spans. You may only select and order the passage's own sentences.

2. THEN WRITE THE QUESTION that this account answers. Use whichever shape fits:
   - RETELL — if the episode is a matter of record you can recognize (a known event,
     campaign, voyage, or life — "Washington's crossing of the Delaware", "the retreat
     from Moscow"), NAME it from your own knowledge and ask the reader to recount it:
     "Recount how ...", "Tell how ... came about", "Describe what befell ...".
   - COMPREHEND — otherwise ask what happens in the episode: "What did ... do when
     ...?", "How did ... meet ...?", "What became of ...?". The span must CORRECTLY
     AND COMPLETELY answer it — ask exactly what the account tells; never demand a
     name, motive, or detail it does not supply.
   - Plain period register: direct, no modern or conversational phrasing, no
     meta-language. Write it in pre-1930s English — period vocabulary, spelling, and
     phrasing; use no word, term, or idiom that came into use after 1930.
   - Stands alone. Never mention the source — no "the passage", "the text", "the
     story", "the author", "the narrator", "according to", "described", "mentioned".
     Ask as if of the episode itself.
   - Self-situating — the reader has NOT seen this passage. For a RETELL, the named
     event carries the context. For a COMPREHEND question about a particular scene, SET
     IT UP inside the question: briefly establish the actors and circumstance the
     answer depends on (introduce them by role or name — "a besieged garrison", "the
     young woman Naomi", "a sea-captain pursued by a Spanish squadron") and give just
     enough that the question is unambiguous. Never leave a bare "he", "she", "they",
     "it", or "this" whose antecedent the question itself has not supplied, and do not
     retell the whole scene.
   - Do NOT put the answer, or its distinctive wording, in the question. Ask what
     happened; do not reveal it.

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "..."}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """Narrative excerpts from the materialized corpus (affordance == narrative).

    n falsy or >= pool size returns the whole pool (full run); otherwise a seeded
    random sample, so a --test/--sample with a given --count/--seed varies across
    seeds instead of always taking the first n in file order."""
    mat = corpus.load_excerpts(affordance=AFFORDANCE)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


ROUTE = engine.Route(
    name="narrative_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.spans_answer(MAX_SPANS),   # verbatim extraction
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
