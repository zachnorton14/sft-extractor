"""LLM-judge verification of Q/A ALIGNMENT for the synthetic SFT dataset.

The answers are verbatim period text (correct by period standards), so this does NOT
fact-check them and does NOT judge era/style (that is the anachronism filter's job). It
asks exactly one thing: does the answer actually, correctly, and completely answer the
composed question? Each pair is scored 1-5. Sampling per route gives you the
misalignment RATE cheaply, so you can decide whether a full judge+filter pass is worth
it before spending on one.

Judge with a model OTHER than the deepseek-chat generator for rigor (--model, and/or
SFT_PROVIDER for a different provider) to avoid self-preference — though "does A answer
Q" is fairly objective, so same-family judging still gives a usable signal.

Multiturn: each assistant turn is judged against its user turn WITH the preceding turns
supplied as context, so follow-ups that corefer ("and why did that happen?") are judged
fairly rather than looking context-bare.

Run:
    python3 -m synth.verify --route knowledge_qa --sample 1000
    python3 -m synth.verify --sample 1500                 # every route
"""

import argparse
import asyncio
import json
import random
from collections import defaultdict

from synth import engine
from synth.ngram_filter import _load_route, ROUTES

BATCH = 40

SYSTEM = """\
You are given a numbered list of QUESTION/ANSWER pairs taken from pre-1930s books. For
each pair judge ONLY whether the ANSWER correctly and completely answers the QUESTION.

Do NOT fact-check against modern knowledge — period beliefs are fine. Do NOT judge style
or era. Judge only: does the answer actually and completely answer the question?

Output ONLY a JSON array of 0/1 — one value per input pair, IN THE SAME ORDER, nothing
else (no keys, no text):
  1 = the answer correctly and completely answers the question
  0 = it does not (wrong subject, incomplete, or it merely REFERS to the answer — "this
      was the ...", "is reported by ..." — without actually stating it)
Return EXACTLY as many values as there are input pairs, e.g. [1,1,0,1,0].

Input: JSON array of pairs [{"q": "...", "a": "..."}, ...]
"""


def _pairs(row):
    """(question, answer) pairs for a row. For multiturn, each assistant turn paired with
    a question that carries the conversation so far as context."""
    convs = row.get("conversations")
    if not convs:
        return [(row.get("question", ""), row.get("answer", ""))]
    out, history = [], []
    for k in range(0, len(convs) - 1, 2):
        u, a = convs[k].get("content", ""), convs[k + 1].get("content", "")
        q = ("\n".join(history) + f"\nUser: {u}").strip() if history else u
        out.append((q, a))
        history += [f"User: {u}", f"Assistant: {a}"]
    return out


def _collect(routes, sample, seed):
    items = []                          # (route, doc_index, question, answer)
    for r in routes:
        rows = _load_route(r)
        if sample and len(rows) > sample:
            rows = random.Random(seed).sample(rows, sample)
        for row in rows:
            di = row.get("doc_index")
            for q, a in _pairs(row):
                if q and a:
                    items.append((r, di, q, a))
    return items


async def _judge(client, sem, route, batch):
    """Judge one batch -> [(route, di, q, a, 0|1), ...]. The model returns a bare 0/1
    array positionally aligned to the input; if the count doesn't match (positional
    drift), split the batch and re-judge so large batches stay safe. Singletons that
    still fail are dropped (left unjudged)."""
    payload = json.dumps([{"q": p[2], "a": p[3]} for p in batch])
    parsed = await engine._call(client, sem, route, SYSTEM, payload, len(batch))
    if (isinstance(parsed, list) and len(parsed) == len(batch)
            and all(v in (0, 1, True, False) for v in parsed)):
        return [(*p, int(v)) for p, v in zip(batch, parsed)]
    if len(batch) <= 1:
        return []                       # unjudged (kept, not dropped, downstream)
    mid = len(batch) // 2
    left = await _judge(client, sem, route, batch[:mid])
    right = await _judge(client, sem, route, batch[mid:])
    return left + right


async def _run(items, model, batch_size, show):
    route = engine.Route(name="verify", system=SYSTEM, source=None, answer_fn=None,
                         model=model, extra_body=engine.DISABLE_THINKING)
    scored = []
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    async with engine.open_client() as client:
        sem = asyncio.Semaphore(engine.CONCURRENCY)

        async def do(batch):
            scored.extend(await _judge(client, sem, route, batch))

        await asyncio.gather(*[asyncio.create_task(do(b)) for b in batches])
    _report(scored, len(items), show)
    return scored


def _report(scored, n_total, show):
    print(f"\njudged {len(scored):,}/{n_total:,} pairs "
          f"({n_total - len(scored):,} unjudged/kept)\n")
    by_route = defaultdict(list)
    for route, di, q, a, s in scored:
        by_route[route].append(s)
    print(f"{'route':22}{'n':>8}{'%misaligned':>14}")
    print("-" * 44)
    for route in sorted(by_route):
        s = by_route[route]
        bad = 100 * sum(1 for x in s if x == 0) / len(s)
        print(f"{route:22}{len(s):>8}{bad:>13.1f}%")
    alls = [s for v in by_route.values() for s in v]
    if alls:
        print("-" * 44)
        print(f"{'ALL':22}{len(alls):>8}"
              f"{100*sum(1 for x in alls if x==0)/len(alls):>13.1f}%")

    lows = [(route, di, q, a) for route, di, q, a, s in scored if s == 0]
    print(f"\n=== {len(lows):,} misaligned (0), showing {min(show, len(lows))} ===")
    for route, di, q, a in lows[:show]:
        print(f"  [{route} {di}]")
        print(f"      Q: {q[:150]}")
        print(f"      A: {a[:150]}")


def main():
    ap = argparse.ArgumentParser(description="LLM-judge Q/A alignment verification")
    ap.add_argument("--route", choices=ROUTES, help="single route (default: all)")
    ap.add_argument("--sample", type=int, default=1000, help="rows/route to judge (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=engine.MODEL, help="judge model (default: engine MODEL)")
    ap.add_argument("--batch", type=int, default=BATCH, help="pairs per judge call (bare 0/1 array; auto-splits on drift)")
    ap.add_argument("--show", type=int, default=20, help="misaligned examples to print")
    args = ap.parse_args()
    routes = [args.route] if args.route else list(ROUTES)
    items = _collect(routes, args.sample, args.seed)
    print(f"collected {len(items):,} pairs from {len(routes)} route(s); judging (batch={args.batch})...")
    asyncio.run(_run(items, args.model, args.batch, args.show))


if __name__ == "__main__":
    main()
