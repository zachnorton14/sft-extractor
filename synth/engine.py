"""Shared generation engine for the content-bound synth routes.

Every route (knowledge_qa, reasoning_qa, stem_reasoning) is the same pipeline —
sample excerpts, batch them, one model call per batch, parse, verify, write rows,
resume from state — differing only in three things: the SYSTEM prompt, where the
excerpts come from, and how a model result becomes an answer. So a route is just a
`Route` config; all the async / batching / retry / state / output machinery lives
here, once.

    ROUTE = engine.Route(name="knowledge_qa", system=SYSTEM, source=source_fn,
                         answer_fn=extract_spans, passthrough=("prose_score",))
    state = engine.load_state(ROUTE)
    await engine.run_async(ROUTE, excerpts, state)
    engine.write_output(ROUTE, excerpts, state)

Env (same as before):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anthropic

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "synth" / "output"
STATE_DIR = ROOT / "synth" / "state"

MODEL = "claude-haiku-4-5"        # maps to deepseek via ANTHROPIC_BASE_URL
MAX_TOKENS = 16384                # reasoning model: room for thinking + JSON
CONCURRENCY = 20
TOKEN_BUDGET = 1200               # chars/4 per batch


# --- shared parsing / extraction helpers ---------------------------------------

def _estimate_tokens(text):
    return len(text) // 4


def _pack_batches(items, token_budget=TOKEN_BUDGET):
    batches, cur, tok = [], [], 0
    for it in items:
        t = _estimate_tokens(it["excerpt"])
        if cur and tok + t > token_budget:
            batches.append(cur)
            cur, tok = [], 0
        cur.append(it)
        tok += t
    if cur:
        batches.append(cur)
    return batches


def _strip_fence(text):
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    return text


def verbatim_answer(spans, excerpt, max_spans=2):
    """Return the answer built ONLY from exact substrings of `excerpt`, or None.

    Each model-proposed span is located in the excerpt (whitespace-normalized,
    case-insensitive) and the excerpt's own text for that range is used — so the
    result is verbatim by construction, not by trusting what the model typed. Any
    span not found verbatim, or more than max_spans, rejects the whole answer. Spans
    are joined by an ellipsis (a separator, never a word), so no model-written text
    enters the answer.
    """
    if not spans or len(spans) > max_spans:
        return None
    norm = re.sub(r"\s+", " ", excerpt)
    low = norm.lower()
    out = []
    for s in spans:
        if not isinstance(s, str):
            return None
        ns = re.sub(r"\s+", " ", s).strip()
        i = low.find(ns.lower())
        if i == -1:                                  # allow model-appended trailing punct
            ns = ns.rstrip('.,;:"\'')
            i = low.find(ns.lower())
            if i == -1:
                return None                          # not verbatim -> reject
        out.append(norm[i:i + len(ns)])              # store the EXCERPT's exact text
    ans = " … ".join(out).strip()
    return ans[:1].upper() + ans[1:] if ans else None  # capitalize first letter only


def spans_answer(max_spans):
    """answer_fn for extractive routes: verify the model's spans are verbatim."""
    def fn(r, excerpt):
        spans = r.get("spans")
        if spans is None and r.get("a"):             # tolerate single-string form
            spans = [r["a"]]
        return verbatim_answer(spans, excerpt, max_spans)
    return fn


def composed_answer(r, excerpt):
    """answer_fn for composed routes: take the model's answer text as-is."""
    return (r.get("a") or "").strip() or None


# --- route config --------------------------------------------------------------

@dataclass
class Route:
    name: str                     # basename for state/output files
    system: str                   # SYSTEM prompt
    source: Callable              # (n, seed) -> list[excerpt dict]
    answer_fn: Callable           # (result_dict, excerpt_str) -> answer str | None
    passthrough: tuple = ("prose_score",)   # excerpt fields to copy into rows
    model: str = MODEL
    max_tokens: int = MAX_TOKENS
    concurrency: int = CONCURRENCY
    token_budget: int = TOKEN_BUDGET


def state_file(route):
    return STATE_DIR / f"{route.name}.json"


def load_state(route):
    f = state_file(route)
    return json.loads(f.read_text()) if f.exists() else {}


def save_state(route, state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file(route).write_text(json.dumps(state))


# --- generation loop -----------------------------------------------------------

async def _generate_batch(client, semaphore, batch, state, route):
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "text": it["excerpt"]} for i, it in enumerate(batch)])
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=route.model, max_tokens=route.max_tokens,
                    system=route.system, messages=[{"role": "user", "content": payload}],
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
                    a = route.answer_fn(r, batch[idx]["excerpt"])
                    if q and a:                          # a is None -> drop this item
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


async def run_async(route, excerpts, state):
    pending = [e for e in excerpts if str(e["doc_index"]) not in state]
    if not pending:
        print("  nothing pending")
        return
    batches = _pack_batches(pending, route.token_budget)
    print(f"  {len(pending)} excerpts, {len(batches)} batches...")
    done = [0]
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(route.concurrency)

        async def tracked(batch):
            await _generate_batch(client, semaphore, batch, state, route)
            done[0] += len(batch)
            if done[0] % 100 < len(batch):
                save_state(route, state)
                print(f"  {done[0]}/{len(pending)}", flush=True)

        await asyncio.gather(*[asyncio.create_task(tracked(b)) for b in batches])
    save_state(route, state)


def write_output(route, excerpts, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        if not r:
            continue
        content = r.get("category") or e["category"]
        row = {
            "doc_index": e["doc_index"],
            "category": content,
            "book_category": e["category"],
            "category_moved": content != e["category"],
            "year": e.get("year"),
        }
        for f in route.passthrough:
            row[f] = e.get(f)
        row["excerpt"] = e["excerpt"]
        row["question"] = r["q"]
        row["answer"] = r["a"]
        rows.append(row)
    out = OUTPUT_DIR / f"{route.name}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"  wrote {len(rows)} {route.name} rows -> {out}")
    return out


async def test_run(route, excerpts):
    """Generate for a small sample and print each Q/A beside its source excerpt."""
    state = {}
    batches = _pack_batches(excerpts, route.token_budget)
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(route.concurrency)
        await asyncio.gather(*[_generate_batch(client, semaphore, b, state, route)
                               for b in batches])
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        scores = "  ".join(f"{f} {e[f]:.2f}" for f in route.passthrough
                           if isinstance(e.get(f), (int, float)))
        if r and r.get("category") and r["category"] != e["category"]:
            head = f"[{r['category']}]  (book said {e['category']})"
        else:
            head = f"[{e['category']}]"
        print("=" * 78)
        print(f"{head}  {e.get('year')}  {scores}")
        print(f"  excerpt : {e['excerpt'][:200]}...")
        if r:
            print(f"  Q       : {r['q']}")
            print(f"  A       : {r['a']}")
        else:
            print("  (failed)")
        print()
