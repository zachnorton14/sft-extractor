"""Reasoning route: turn argument excerpts into step-showing Q/A rows.

The second content-bound generator. It reads the materialized excerpt corpus
(synth/output/excerpts.jsonl from `harvest`) and keeps the ones the affordance gate
tagged `argument` — passages that make a case, derive a result, or reason from
premises. For each, one model call produces a Q/A that exercises the REASONING:

  - the QUESTION poses a problem that must be reasoned through (why, how it
    follows, prove that, if-then), not a fact to recall;
  - the ANSWER reconstructs the reasoning STEP BY STEP along the passage's own
    logic, ending in the conclusion — showing the work, not just the result.

The chain of reasoning is grounded in the passage (no premises from outside it);
only the question's framing may use general knowledge, as in the knowledge route.

Input:  synth/output/excerpts.jsonl (affordance == argument)
Output: synth/output/reasoning_qa.json
        [{"doc_index","category","book_category","category_moved","year",
          "prose_score","excerpt","question","answer"}]

Env (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
from pathlib import Path

import anthropic

from synth import corpus
from synth.knowledge_qa import _estimate_tokens, _pack_batches, _strip_fence

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "synth" / "output"
STATE_DIR = ROOT / "synth" / "state"
STATE_FILE = STATE_DIR / "reasoning_qa.json"

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 16384
CONCURRENCY = 20
TOKEN_BUDGET = 1200
ROUTE = "argument"               # affordance this generator handles

_CLASS_LIST = "\n".join(f"  - {c}" for c in corpus.LOC_CLASSES)

SYSTEM = f"""\
You are given a short passage from a pre-1930s book that argues a point, derives a
result, or reasons from premises. Do two things.

1. Write ONE question-answer pair that exercises the REASONING in the passage.

   Question:
   - Pose a problem that must be reasoned through, not a fact to recall: "Why does
     ...", "How does it follow that ...", "Show that ...", "If ..., what follows and
     why?". Plain period register, no modern or conversational phrasing.
   - Stands alone. Never mention the source — no "the passage", "the text",
     "according to", "described", "mentioned".
   - Self-situating: name the subject, figures, or setting so a reader who cannot
     see the passage knows what is asked. Framing may use your own knowledge; the
     reasoning in the answer may not.

   Answer:
   - Reconstruct the reasoning STEP BY STEP along the passage's own logic, ending in
     the conclusion. Show the intermediate steps, not just the result.
   - Use only premises stated or directly implied in the passage — introduce no
     outside facts. Do not restate the question.

2. Classify the passage's ACTUAL subject (what it is about, not the kind of book it
   came from) into exactly one of these classes, copied verbatim:
{_CLASS_LIST}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only: [{{"i": 0, "q": "...", "a": "...", "category": "ONE CLASS"}}, ...]
"""


def source_excerpts(n, **_):
    """Reasoning excerpts from the materialized corpus (affordance == argument)."""
    mat = corpus.load_excerpts(affordance=ROUTE)
    return mat[:n] if n else mat


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
    done = [0]
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def tracked(batch):
            await _generate_batch(client, semaphore, batch, state)
            done[0] += len(batch)
            if done[0] % 100 < len(batch):
                save_state(state)
                print(f"  {done[0]}/{len(pending)}", flush=True)

        await asyncio.gather(*[asyncio.create_task(tracked(b)) for b in batches])
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
            "prose_score": e["prose_score"],
            "excerpt": e["excerpt"],
            "question": r["q"],
            "answer": r["a"],
        })
    out = OUTPUT_DIR / "reasoning_qa.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"  wrote {len(rows)} reasoning-QA rows -> {out}")
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
        head = (f"[{r['category']}]  (book said {e['category']})"
                if r and r.get("category") and r["category"] != e["category"]
                else f"[{e['category']}]")
        print(f"{head}  {e['year']}  prose {e['prose_score']:.2f}")
        print(f"  excerpt : {e['excerpt'][:200]}...")
        if r:
            print(f"  Q       : {r['q']}")
            print(f"  A       : {r['a']}")
        else:
            print("  (failed)")
        print()
