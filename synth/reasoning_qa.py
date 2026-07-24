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

import asyncio
import json
from pathlib import Path

import anthropic

from synth import corpus
from synth.knowledge_qa import _pack_batches, _strip_fence, verbatim_answer

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "synth" / "output"
STATE_DIR = ROOT / "synth" / "state"
STATE_FILE = STATE_DIR / "reasoning_qa.json"

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 16384
CONCURRENCY = 20
TOKEN_BUDGET = 1200
ROUTE = "argument"               # affordance this generator handles
MAX_SPANS = 4                    # a chain is spread out; more than knowledge-QA's 2

_CLASS_LIST = "\n".join(f"  - {c}" for c in corpus.LOC_CLASSES)

SYSTEM = f"""\
You are given a short passage from a pre-1930s book that argues a point, derives a
result, or reasons from premises. The author's own reasoning is in the text. Work
in this order.

1. EXTRACT THE REASONING CHAIN, VERBATIM. Find where the author reasons from
   premises to a conclusion, and copy that chain WORD FOR WORD as the answer.
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
                    if not (isinstance(idx, int) and 0 <= idx < len(keys)):
                        continue
                    q = (r.get("q") or "").strip()
                    cat = (r.get("category") or "").strip()
                    spans = r.get("spans")
                    if spans is None and r.get("a"):
                        spans = [r["a"]]
                    a = verbatim_answer(spans, batch[idx]["excerpt"], MAX_SPANS)
                    if q and a:                          # a is None if not verbatim -> drop
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
