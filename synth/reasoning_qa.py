"""Reasoning route: extract the author's reasoning chain, verbatim.

The second content-bound generator. It reads the materialized excerpt corpus
(synth/output/excerpts.jsonl from `harvest`) and keeps the ones the affordance gate
tagged `argument` — passages that make a case, derive a result, or reason from
premises. These are passages that CONTAIN reasoning (the author does the reasoning),
not passages that can merely be reasoned about — so the answer is *extracted*, not
generated: the same anachronism guarantee as knowledge-QA, and no reliance on the
model to compose vintage-sounding logic.

  - the ANSWER is the author's reasoning chain, copied VERBATIM. Prefer one long
    contiguous span; but because a chain is naturally spread across a passage, the
    model MAY splice several verbatim spans (up to MAX_SPANS) into one coherent
    chain. Splicing is anachronism-safe: every span is verified as a literal
    substring and spans join with an ellipsis, so the model may only select and
    order — never write a connecting word. Capped, and told to prefer the fewest,
    longest spans, so it can't cherry-pick a disjointed chain.
  - the QUESTION stands alone (no passage attached — a visible passage would make
    extraction a degenerate echo), self-situating, "why does .../ how does it
    follow ...". Answer-first: find the chain, then write the question for it.

Input:  synth/output/excerpts.jsonl (affordance == argument)
Output: synth/output/reasoning_qa.json
        [{"doc_index","category","book_category","category_moved","year",
          "prose_score","excerpt","question","answer"}]

Env (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

from synth import corpus, engine

AFFORDANCE = "argument"          # affordance this generator handles
MAX_SPANS = 4                    # a chain is spread out; more than knowledge-QA's 2

_CLASS_LIST = "\n".join(f"  - {c}" for c in corpus.LOC_CLASSES)

SYSTEM = f"""\
You are given a short passage from a pre-1930s book that argues a point, derives a
result, or reasons from premises. The author's own reasoning is in the text. Work
in this order.

1. EXTRACT THE REASONING CHAIN, VERBATIM. Find where the author reasons from
   premises to a conclusion, and copy that chain WORD FOR WORD as the answer.
   - The reasoning must be a GENERAL argument, derivation, or explanation — a
     principle defended, a result derived, a doctrine, law, or cause argued — that
     stands on its own, apart from any particular scene. It must NOT be the
     motivation of specific characters in a narrated event (why so-and-so did
     such-and-such in a story). If the passage's only reasoning is narrative —
     turning on persons, places, or events known ONLY from the passage itself, that
     you could not name from general knowledge — OMIT this item entirely from the
     output array. A dropped item is far better than a question that assumes the
     reader knows characters or events from a story they cannot see.
   - Return it as "spans": exact quotations from the passage that, read in order,
     form one coherent chain of reasoning (premises through conclusion). Prefer ONE
     long contiguous span. Use several only because the steps are spread out — never
     more than {MAX_SPANS}, and always the fewest, longest spans that hold the chain
     together.
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
     reorder within a span, or add any word not in the passage — including
     connecting words between spans. You may only select and order the author's own
     sentences.

2. THEN WRITE THE QUESTION the chain answers.
   - The chain must CORRECTLY AND COMPLETELY answer your question — ask exactly what
     this reasoning concludes or explains. Do not ask for a step, result, or
     identity the chain does not actually reach. Read the chain as the answer: if it
     does not fully answer, rewrite the QUESTION to fit the chain.
   - Pose a problem that must be reasoned through, not a fact to recall: "Why does
     ...", "How does it follow that ...", "Why must ...?".
   - Write it in pre-1930s English: period vocabulary, spelling, and phrasing. Use
     no word, term, or idiom that came into use after 1930, and no modern
     conversational or academic phrasing — it must read as a question a period
     schoolbook or examiner would actually pose.
   - Stands alone. Never mention the source — no "the passage", "the text",
     "according to", "described", "mentioned".
   - Self-situating — the question must make sense to someone who NEVER saw the
     passage, AND must give the chain's "they / it / this / these" clear
     antecedents. Name the actual subject the reasoning concerns: the thinker,
     doctrine, war, ruler, place, work, statute, or scripture, using your own
     knowledge of the subject when the passage assumes it (e.g. "In Paul's argument
     in the Epistle to the Romans, why...", "Why did Justinian's tax policy...").
     FORBIDDEN — references that mean nothing on their own: a bare "the argument",
     "the author", "the doctrine", "these beings", "such creatures", "this
     process"; or catalog, specimen, or figure numbers. If the reasoning cannot be
     made to stand alone, pick a DIFFERENT chain in the passage that can.
   - Do not put the conclusion or the answer's distinctive wording in the question.

3. Classify the passage's ACTUAL subject (what it is about, not the kind of book it
   came from) into exactly one of these classes, copied verbatim:
{_CLASS_LIST}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["...", "..."], "q": "...", "category": "ONE CLASS"}}, ...]
"""


def source_excerpts(n, **_):
    """Reasoning excerpts from the materialized corpus (affordance == argument)."""
    mat = corpus.load_excerpts(affordance=AFFORDANCE)
    return mat[:n] if n else mat


ROUTE = engine.Route(
    name="reasoning_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.spans_answer(MAX_SPANS),   # verbatim chain extraction
    passthrough=("prose_score",),
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
