"""STEM-reasoning route: extract the author's reasoning chain, verbatim and in prose.

Period STEM reasoning lives in the sparse reasoning windows of SCIENCE and
TECHNOLOGY texts (engineering derivations, geometry/surveying exercises, physical
and chemical reasoning). corpus.sample_stem seeks those windows — the ones dense
in reasoning connectives plus quantitative vocabulary — rather than a random one.

Like the other content-bound routes, the answer is *extracted*, not generated: the
same anachronism guarantee, and no reliance on the model to compose vintage-sounding
logic. The wrinkle is STEM's OCR — it mangles equations, exponents, symbols, and
bleeds page/figure numbers into formulas — so a verbatim span that carries any of
that would drag corruption into the dataset. The route therefore extracts only the
reasoning the author stated IN WORDS and rejects, whole, any span bearing a math
symbol, a letter-and-number expression, a fraction, or a figure/page reference
(engine.verbal_spans_answer). What survives is the qualitative principle-and-
consequence prose a period scientist wrote out ("the pressure varies inversely as
the volume"). This is path A: if too few passages yield prose-only chains, the
fallback is to let STEM compose (engine.composed_answer) as the documented exception.

Two calls, as in reasoning_qa:
  - call 1 extracts the verbatim prose chain (spans);
  - call 2 writes the standalone question the chain answers.

Input:  corpus.sample_stem (SCIENCE + TECHNOLOGY, stem-seeking windows)
Output: synth/output/stem_reasoning.json
        [{"doc_index","category","book_category","category_moved","year",
          "stem_signal","excerpt","question","answer"}]

Env (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

from synth import corpus, engine

AFFORDANCE = "stem_reasoning"    # affordance this generator handles
MAX_SPANS = 4                    # a chain threads between equations; spread out

# CALL 1 — extract the reasoning chain verbatim, but only where stated in words.
EXTRACT_SYSTEM = f"""\
You are given a short passage from a pre-1930s science or engineering text that
reasons quantitatively or physically. The author's own reasoning is in the text.

EXTRACT THE REASONING CHAIN, STATED IN WORDS, VERBATIM. Find where the author reasons
from a principle to a conclusion, and copy that chain WORD FOR WORD.
  - Return it as "spans": exact quotations from the passage that, read in order,
    form one coherent chain (principle → application → conclusion). Prefer ONE long
    contiguous span. Use several only because the steps are spread out — never more
    than {MAX_SPANS}, and always the fewest, longest spans that hold the chain
    together.
  - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
    reorder within a span, or add any word not in the passage — including connecting
    words between spans. You may only select and order the author's own sentences.
  - CHOOSE SPANS STATED IN WORDS. The OCR has mangled this text's equations,
    exponents, symbols, and numbers, and page and figure numbers bleed into the
    formulas. A span that carries a formula, an equation, a bare math symbol
    (=, ×, ÷, an exponent), a letter-and-number expression (x2, P1V1, H2O), a
    fraction, or a reference to a figure, plate, table, equation, section, or page
    is USELESS on its own and WILL BE DISCARDED. Pick the sentences that state the
    reasoning verbally — the principle and its consequence in prose ("the product of
    two powers of the same base adds their exponents", "the pressure varies inversely
    as the volume", "heat added at constant volume raises the temperature"). If the
    chain cannot be formed from prose alone, omit this item.

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only: [{{"i": 0, "spans": ["..."]}}, ...]
"""

# CALL 2 — write the question, given the passage and the extracted chain (ANSWER).
QUESTION_SYSTEM = """\
You are given a passage from a pre-1930s science or engineering text and a REASONING
CHAIN quoted verbatim from it — this is the ANSWER. Write the single question that
this reasoning chain answers, as if setting an exercise.

- The chain must CORRECTLY AND COMPLETELY answer your question — ask exactly what
  this reasoning concludes or explains. Do not ask for a step, a numeric result, or
  an identity the chain does not reach. Read the chain as the answer: if it does not
  fully answer, adjust the question to fit the chain.
- Pose a problem that must be reasoned through, not a fact to recall: "Why does
  ...", "How does it follow that ...", "Why must ...?", "If ..., what happens and
  why?".
- Write it in pre-1930s English: period vocabulary, spelling, and phrasing. Use no
  word, term, or idiom that came into use after 1930, and no modern conversational
  or academic phrasing — it must read as a question a period examiner would pose.
- The question must STAND ALONE and betray no awareness of a source. Never refer to
  the passage or to whoever wrote it — no "the passage", "the text", "the author",
  "the figure", "according to", "as described", "as shown". Ask about the SUBJECT
  ITSELF, as a matter of physical fact, to someone who never saw the passage.
- SUPPLY THE PHYSICAL SETUP the reader needs so the problem is well-posed and the
  chain's "it / this / these" have clear antecedents: name the bodies, quantities,
  or conditions involved (the gas held at constant temperature, the beam loaded at
  its centre, the current through the coil). Give just enough to make the question
  unambiguous — do not restate the whole passage, and never leave a bare "this
  quantity", "such a body", or "the system".
- Every proper noun must be one the reader can place: either widely recognizable on
  its own (Boyle's law, Carnot's cycle), or introduced with a short descriptor.
- Do NOT reveal the answer: do not state the conclusion, and do not reuse the
  chain's distinctive wording. Ask for the reasoning; make the reader supply it.

Input: JSON array [{"i": 0, "passage": "...", "answer": "..."}, ...]
Output JSON only: [{"i": 0, "q": "..."}, ...]
"""


def source_excerpts(n, seed=0, **_):
    """STEM-reasoning excerpts, sought from SCIENCE + TECHNOLOGY."""
    return corpus.sample_stem(n or 20, seed=seed)


ROUTE = engine.Route(
    name="stem_reasoning",
    system=EXTRACT_SYSTEM,                       # call 1: extract prose chain + classify
    question_system=QUESTION_SYSTEM,             # call 2: write the question
    source=source_excerpts,
    answer_fn=engine.verbal_spans_answer(MAX_SPANS),   # verbatim, prose-only extraction
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


def sample_run(excerpts, seed=0):
    return engine.sample_run(ROUTE, excerpts, seed)
