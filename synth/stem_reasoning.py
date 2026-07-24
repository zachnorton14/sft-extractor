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

import asyncio
import json
from pathlib import Path

import anthropic

from synth import corpus
from synth.knowledge_qa import _pack_batches, _strip_fence

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "synth" / "output"
STATE_DIR = ROOT / "synth" / "state"
STATE_FILE = STATE_DIR / "stem_reasoning.json"

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 16384
CONCURRENCY = 20
TOKEN_BUDGET = 1200

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


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


async def _generate_batch(client, semaphore, batch, state):
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": idx, "text": it["excerpt"]} for idx, it in enumerate(batch)])
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                    messages=[{"role": "user", "content": payload}],
                )
                text = next((b.text for b in msg.content if b.type == "text"), "").strip()
                if not text:
                    raise ValueError(f"empty response (stop_reason={msg.stop_reason}, "
                                     f"blocks={[b.type for b in msg.content]})")
                if msg.stop_reason == "max_tokens":
                    raise ValueError(f"truncated at max_tokens ({len(batch)} in batch)")
                for r in json.loads(_strip_fence(text)):
                    idx = r.get("i")
                    q = (r.get("q") or "").strip()
                    a = (r.get("a") or "").strip()
                    cat = (r.get("category") or "").strip()
                    if isinstance(idx, int) and 0 <= idx < len(keys) and q and a:
                        state[keys[idx]] = {"q": q, "a": a, "category": cat}
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                raise SystemExit(f"\nAPI auth failed: {e}\n"
                                 f"Check ANTHROPIC_API_KEY matches ANTHROPIC_BASE_URL.")
            except json.JSONDecodeError as e:
                if attempt == 4:
                    print(f"  batch of {len(batch)} failed: unparseable: {e}\n"
                          f"    raw[:200]: {text[:200]!r}")
                    return
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 4:
                    print(f"  batch of {len(batch)} failed after 5 attempts: "
                          f"{type(e).__name__}: {str(e)[:200]}")
                    return
                await asyncio.sleep(2 ** attempt)


async def run_async(excerpts, state):
    pending = [e for e in excerpts if str(e["doc_index"]) not in state]
    if not pending:
        print("  nothing pending")
        return
    batches = _pack_batches(pending)
    print(f"  {len(pending)} excerpts, {len(batches)} batches...")
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        await asyncio.gather(*[_generate_batch(client, semaphore, b, state) for b in batches])
    save_state(state)


def write_output(excerpts, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        if not r:
            continue
        content_cat = r.get("category") or e["category"]
        rows.append({
            "doc_index": e["doc_index"],
            "category": content_cat,
            "book_category": e["category"],
            "category_moved": content_cat != e["category"],
            "year": e["year"],
            "stem_signal": e["stem_signal"],
            "excerpt": e["excerpt"],
            "question": r["q"],
            "answer": r["a"],
        })
    out = OUTPUT_DIR / "stem_reasoning.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"  wrote {len(rows)} STEM-reasoning rows -> {out}")
    return out


async def test_run(excerpts):
    state = {}
    batches = _pack_batches(excerpts)
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        await asyncio.gather(*[_generate_batch(client, semaphore, b, state) for b in batches])
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        print("=" * 78)
        print(f"[{e['category']}]  {e['year']}  stem {e['stem_signal']:.2f}")
        print(f"  excerpt : {e['excerpt'][:220]}...")
        if r:
            print(f"  Q       : {r['q']}")
            print(f"  A       : {r['a']}")
        else:
            print("  (failed)")
        print()
