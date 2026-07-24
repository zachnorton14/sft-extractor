"""STEM-reasoning route: turn quantitative/physical reasoning excerpts into
step-showing Q/A rows.

Period STEM reasoning lives in the sparse reasoning windows of SCIENCE and
TECHNOLOGY texts (engineering derivations, geometry/surveying exercises, physical
and chemical reasoning). corpus.sample_stem seeks those windows — the ones dense
in reasoning connectives plus quantitative vocabulary — rather than a random one.

Two realities shape the prompt:
  - OCR mangles symbolic equations (exponents, fractions, page numbers bleed in),
    so the answer must reason VERBALLY through the physics/mathematics and must not
    quote or depend on a garbled formula;
  - the target is the qualitative mechanics-and-thermodynamics reasoning a period
    scientist would do, worked step by step.

Input:  corpus.sample_stem (SCIENCE + TECHNOLOGY, stem-seeking windows)
Output: synth/output/stem_reasoning.json
        [{"doc_index","category","book_category","category_moved","year",
          "stem_signal","excerpt","question","answer"}]

Env (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

from synth import corpus, engine

_CLASS_LIST = "\n".join(f"  - {c}" for c in corpus.LOC_CLASSES)

SYSTEM = f"""\
You are given a short passage from a pre-1930s science or engineering text that
reasons quantitatively or physically. Do two things.

1. Write ONE question-answer pair that exercises the STEM REASONING in the passage.

   Question:
   - Pose a problem to reason through — "Why does ...", "How does it follow that
     ...", "Show why ...", "If ..., what happens and why?". Plain period register,
     no modern or conversational phrasing.
   - Stands alone. Never mention the source ("the passage", "the text", "the
     figure", "Fig.", section or page numbers). Ask as if setting an exercise.
   - Self-situating: state the physical setup or quantities involved so the problem
     is well-posed on its own.

   Answer:
   - Work through the reasoning STEP BY STEP — state the principle, apply it to the
     setup, and reach the conclusion. Show the physical or mathematical logic, not
     just the result.
   - Reason VERBALLY. The passage's OCR mangles equations, exponents, and symbols,
     and page numbers bleed into formulas — do NOT quote or rely on any garbled
     expression. Reconstruct the reasoning in words and clean, plainly-written
     relations only (e.g. "the product of two powers of the same base adds their
     exponents"), never a corrupted symbol string.
   - Use only principles and quantities present in the passage; add no outside
     facts. Do not restate the question.

2. Classify the passage's ACTUAL subject into exactly one of these classes, copied
   verbatim:
{_CLASS_LIST}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only: [{{"i": 0, "q": "...", "a": "...", "category": "ONE CLASS"}}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """STEM-reasoning excerpts, sought from SCIENCE + TECHNOLOGY."""
    return corpus.sample_stem(n or 20, seed=seed)


ROUTE = engine.Route(
    name="stem_reasoning",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=engine.composed_answer,           # verbal reconstruction, not extraction
    passthrough=("stem_signal",),
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
