"""Narrative route: turn narrated episodes into reading-comprehension Q/A rows.

The third content-bound generator. It reads the materialized excerpt corpus
(synth/output/excerpts.jsonl from `harvest`) and keeps the ones the affordance gate
tagged `narrative` — passages that recount events: a scene, an episode, an account of
what happened. Like knowledge-QA it is EXTRACTIVE — the answer is a verbatim span of
period text, never composed — so the anachronism guarantee holds: an answer built
only from the passage cannot carry modern phrasing.

What differs from knowledge-QA is the QUESTION framing, not the answer. Two shapes
both fit extraction, and the classifier's real/invented tag (passed to the prompt as
"kind") selects between them:
  - RETELL — grounded episodes that are a matter of record ("Washington crossing the
    Delaware", "the burning of Moscow"): the question names it from general knowledge
    and asks the reader to recount it ("Recount how ...", "Tell how ... came about"),
    and the answer is the verbatim narration itself — one long span. Grounded episodes
    the model does not recognize fall back to COMPREHEND.
  - COMPREHEND — fiction always, and any unrecognized scene the reader has not seen:
    the question sets the scene up from within — briefly introducing the actors and
    circumstance, the way the reasoning route introduces persons "described only in the
    passage" — and asks what happened. A fiction episode is NEVER presented as history.

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

CLASSES = ("narrative_grounded", "narrative_fiction")   # both; the prompt branches on kind
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
   - Take WHOLE sentences: begin and end each span at a sentence boundary and keep its
     terminal punctuation, so several spans read as continuous prose when joined.

2. THEN WRITE THE QUESTION that this account answers. The item's "kind" field tells you
   which shape to use — it is the classifier's judgment of whether the episode is real
   or invented, and it is authoritative:
   - kind "grounded" — the episode is presented as a REAL, historical event. If you
     recognize it as a matter of record (a known event, campaign, voyage, or life —
     "Washington's crossing of the Delaware", "the retreat from Moscow"), NAME it from
     your own knowledge and use RETELL: ask the reader to recount it — "Recount how ...",
     "Tell how ... came about", "Describe what befell ...". If you do NOT recognize the
     specific event, fall back to COMPREHEND.
   - kind "fiction" — the episode is INVENTED (a novel, tale, or story). NEVER name it
     as a matter of record, never claim it happened, and never ask to "recount" it as
     history. ALWAYS use COMPREHEND.
   - COMPREHEND — ask what happens in the episode: "What did ... do when ...?", "How did
     ... meet ...?", "What became of ...?". The span must CORRECTLY AND COMPLETELY
     answer it — ask exactly what the account tells; never demand a name, motive, or
     detail it does not supply.
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

{engine.OCR_REJECT}

Input: JSON array [{{"i": 0, "text": "...", "kind": "grounded" | "fiction"}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "..."}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """Narrative excerpts from the materialized corpus, selected by the classifier
    (classes contains narrative_grounded or narrative_fiction). n falsy or >= pool
    returns the whole pool; otherwise a seeded random sample. Requires `classify`
    write-back to have run."""
    mat = corpus.load_excerpts(cls=CLASSES, drop_broken_math=True)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


def _kind_field(excerpt):
    """Feed the classifier's real/invented judgment into the prompt so RETELL vs
    COMPREHEND follows the tag, not the model's re-derivation. An excerpt carrying the
    grounded tag is treated as grounded (a matter of record may be retold) even if it
    also carries the fiction tag; fiction only when grounded is absent."""
    classes = excerpt.get("classes") or []
    kind = "grounded" if "narrative_grounded" in classes else "fiction"
    return {"kind": kind}


ROUTE = engine.Route(
    name="narrative_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.spans_answer(MAX_SPANS),   # verbatim extraction
    passthrough=("prose_score",),
    extra_body=engine.DISABLE_THINKING,         # no thinking trace: fast, no truncation
    payload_fn=_kind_field,                     # inject narrative kind per item
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
