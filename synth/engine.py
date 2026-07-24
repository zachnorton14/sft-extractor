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
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

WRAP = 88   # column cap for printed / written sample lines

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
    system: str                   # SYSTEM prompt (extraction, or the whole call if one-phase)
    source: Callable              # (n, seed) -> list[excerpt dict]
    answer_fn: Callable           # (result_dict, excerpt_str) -> answer str | None
    passthrough: tuple = ("prose_score",)   # excerpt fields to copy into rows
    question_system: str = None   # if set, generate the question in a SECOND call
                                  # (given passage + answer); else Q and A in one call
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

async def _call(client, semaphore, route, system, payload, n):
    """One model call with retry; returns the parsed JSON list or None."""
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=route.model, max_tokens=route.max_tokens,
                    system=system, messages=[{"role": "user", "content": payload}],
                )
                text = next((b.text for b in msg.content if b.type == "text"), "").strip()
                if not text:
                    raise ValueError(f"empty response (stop_reason={msg.stop_reason}, "
                                     f"blocks={[b.type for b in msg.content]})")
                if msg.stop_reason == "max_tokens":
                    raise ValueError(f"truncated at max_tokens ({n} in batch)")
                return json.loads(_strip_fence(text))
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                raise SystemExit(f"\nAPI auth failed: {e}\n"
                                 f"Check ANTHROPIC_API_KEY matches ANTHROPIC_BASE_URL.")
            except json.JSONDecodeError as e:
                if attempt == 4:
                    print(f"  batch of {n} failed: unparseable: {e}\n"
                          f"    raw[:200]: {text[:200]!r}")
                    return None
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 4:
                    print(f"  batch of {n} failed after 5 attempts: "
                          f"{type(e).__name__}: {str(e)[:200]}")
                    return None
                await asyncio.sleep(2 ** attempt)
    return None


async def _single_batch(client, semaphore, batch, state, route):
    """One-phase: Q, A, and category in a single call (knowledge, stem)."""
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "text": it["excerpt"]} for i, it in enumerate(batch)])
    parsed = await _call(client, semaphore, route, route.system, payload, len(batch))
    for r in parsed or []:
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        q = (r.get("q") or "").strip()
        cat = (r.get("category") or "").strip()
        a = route.answer_fn(r, batch[idx]["excerpt"])
        if q and a:                                  # a is None -> drop this item
            state[keys[idx]] = {"q": q, "a": a, "category": cat}


async def _extract_batch(client, semaphore, batch, state, route):
    """Two-phase call 1: extract the answer (spans) + category from the passage."""
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "text": it["excerpt"]} for i, it in enumerate(batch)])
    parsed = await _call(client, semaphore, route, route.system, payload, len(batch))
    for r in parsed or []:
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        cat = (r.get("category") or "").strip()
        a = route.answer_fn(r, batch[idx]["excerpt"])
        if a:
            state[keys[idx]] = {"a": a, "category": cat}   # question filled in phase 2


async def _question_batch(client, semaphore, batch, state, route):
    """Two-phase call 2: write the question given the passage + its extracted answer."""
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "passage": it["excerpt"], "answer": state[keys[i]]["a"]}
                          for i, it in enumerate(batch)])
    parsed = await _call(client, semaphore, route, route.question_system, payload, len(batch))
    for r in parsed or []:
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        q = (r.get("q") or "").strip()
        if q:
            state[keys[idx]]["q"] = q


async def run_async(route, excerpts, state, save=True):
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(route.concurrency)

        if route.question_system:
            # phase 1 — extract answers for excerpts that don't have one yet
            need_a = [e for e in excerpts if "a" not in state.get(str(e["doc_index"]), {})]
            if need_a:
                batches = _pack_batches(need_a, route.token_budget)
                print(f"  extract: {len(need_a)} excerpts, {len(batches)} batches...")
                await asyncio.gather(*[_extract_batch(client, semaphore, b, state, route)
                                       for b in batches])
                if save:
                    save_state(route, state)
            # phase 2 — write questions for extracted answers that lack one
            need_q = [e for e in excerpts
                      if state.get(str(e["doc_index"]), {}).get("a")
                      and not state[str(e["doc_index"])].get("q")]
            if need_q:
                # payload carries passage + answer, so halve the budget
                batches = _pack_batches(need_q, max(1, route.token_budget // 2))
                print(f"  question: {len(need_q)} answers, {len(batches)} batches...")
                await asyncio.gather(*[_question_batch(client, semaphore, b, state, route)
                                       for b in batches])
                if save:
                    save_state(route, state)
            return

        # one-phase
        pending = [e for e in excerpts if str(e["doc_index"]) not in state]
        if not pending:
            print("  nothing pending")
            return
        batches = _pack_batches(pending, route.token_budget)
        print(f"  {len(pending)} excerpts, {len(batches)} batches...")
        done = [0]

        async def tracked(batch):
            await _single_batch(client, semaphore, batch, state, route)
            done[0] += len(batch)
            if done[0] % 100 < len(batch) and save:
                save_state(route, state)
                print(f"  {done[0]}/{len(pending)}", flush=True)

        await asyncio.gather(*[asyncio.create_task(tracked(b)) for b in batches])
        if save:
            save_state(route, state)


def write_output(route, excerpts, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        if not (r and r.get("q") and r.get("a")):    # both halves required
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


def _wrap(label, text):
    """Wrap `text` to WRAP columns under a `label` prefix, hanging-indented."""
    return textwrap.fill(str(text), width=WRAP, initial_indent=label,
                         subsequent_indent=" " * len(label))


def _pair_lines(route, excerpts, state, excerpt_chars=400):
    """Format generated Q/A pairs beside their excerpts, wrapped to WRAP cols."""
    out = []
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        complete = r and r.get("q") and r.get("a")
        scores = "  ".join(f"{f} {e[f]:.2f}" for f in route.passthrough
                           if isinstance(e.get(f), (int, float)))
        if complete and r.get("category") and r["category"] != e["category"]:
            head = f"[{r['category']}]  (book said {e['category']})"
        else:
            head = f"[{e['category']}]"
        out.append("=" * WRAP)
        out.append(_wrap("", f"{head}  {e.get('year')}  {scores}"))
        out.append(_wrap("  excerpt : ", e["excerpt"][:excerpt_chars] + "..."))
        if complete:
            out.append(_wrap("  Q       : ", r["q"]))
            out.append(_wrap("  A       : ", r["a"]))
        else:
            out.append("  (failed)")
        out.append("")
    return out


async def test_run(route, excerpts):
    """One-off preview: generate a small sample, print it, save nothing."""
    state = {}
    await run_async(route, excerpts, state, save=False)
    print("\n".join(_pair_lines(route, excerpts, state)))


async def sample_run(route, excerpts, seed=0):
    """Prompt-testing record: generate and write a timestamped file to
    samples/<route>/ capturing the prompt(s) used + the pairs, so runs can be
    compared and old prompts recovered. Mirrors the authentic pipeline's `sample`.
    Writes only the file (lines wrapped to WRAP cols); nothing goes to stdout but
    the path confirmation."""
    state = {}
    await run_async(route, excerpts, state, save=False)
    kept = sum(1 for e in excerpts
               if (state.get(str(e["doc_index"])) or {}).get("q")
               and (state.get(str(e["doc_index"])) or {}).get("a"))
    header = [
        f"route: {route.name}", f"model: {route.model}", f"seed:  {seed}",
        f"n:     {len(excerpts)}", f"kept:  {kept}/{len(excerpts)}", "",
    ]
    if route.question_system:
        header += ["--- EXTRACT SYSTEM PROMPT ---", route.system, "",
                   "--- QUESTION SYSTEM PROMPT ---", route.question_system, ""]
    else:
        header += ["--- SYSTEM PROMPT ---", route.system, ""]
    body = _pair_lines(route, excerpts, state, excerpt_chars=400)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = ROOT / "synth" / "samples" / route.name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{ts}_seed{seed}_n{len(excerpts)}.txt"
    path.write_text("\n".join(header + body), encoding="utf-8")
    print(f"wrote sample ({kept}/{len(excerpts)} kept) -> {path}")
    return path
