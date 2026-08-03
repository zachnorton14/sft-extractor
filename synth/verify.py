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

BATCH = 8

SYSTEM = """\
You are given a QUESTION and an ANSWER taken verbatim from a pre-1930s book. Judge ONLY
one thing: does the ANSWER actually, correctly, and completely answer the QUESTION as
asked?

Do NOT fact-check against modern knowledge — the answer reflects period sources and may
state period beliefs; that is fine. Do NOT judge writing style, era, or phrasing. Judge
ONLY whether the answer is a correct and complete response to what the question asks.

Score 1-5:
  5: directly and completely answers exactly what is asked
  4: answers it, with a minor gap or a little extra
  3: partially answers, or answers a broader/narrower question than asked
  2: barely related; addresses a different aspect
  1: does not answer the question at all (wrong subject / non sequitur)

Input: JSON array [{"i": 0, "question": "...", "answer": "..."}, ...]
Output JSON only: [{"i": 0, "answered": 4}, ...]   (no prose)
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


async def _run(items, model, show):
    route = engine.Route(name="verify", system=SYSTEM, source=None, answer_fn=None,
                         model=model, extra_body=engine.DISABLE_THINKING)
    scored = []                         # (route, di, q, a, score)
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    async with engine.open_client() as client:
        sem = asyncio.Semaphore(engine.CONCURRENCY)

        async def do(batch):
            payload = json.dumps([{"i": j, "question": p[2], "answer": p[3]}
                                  for j, p in enumerate(batch)])
            parsed = await engine._call(client, sem, route, SYSTEM, payload, len(batch))
            for r in parsed if isinstance(parsed, list) else []:
                j, s = r.get("i"), r.get("answered")
                if isinstance(j, int) and 0 <= j < len(batch) and isinstance(s, (int, float)):
                    scored.append((*batch[j], int(s)))

        await asyncio.gather(*[asyncio.create_task(do(b)) for b in batches])
    _report(scored, len(items), show)
    return scored


def _report(scored, n_total, show):
    print(f"\njudged {len(scored):,}/{n_total:,} pairs (judge={engine.MODEL if scored else '-'})\n")
    by_route = defaultdict(list)
    for route, di, q, a, s in scored:
        by_route[route].append(s)
    print(f"{'route':22}{'n':>7}{'mean':>7}{'%<=2 (misaligned)':>20}")
    print("-" * 56)
    for route in sorted(by_route):
        s = by_route[route]
        mean = sum(s) / len(s)
        low = 100 * sum(1 for x in s if x <= 2) / len(s)
        print(f"{route:22}{len(s):>7}{mean:>7.2f}{low:>19.1f}%")
    alls = [s for v in by_route.values() for s in v]
    if alls:
        print("-" * 56)
        print(f"{'ALL':22}{len(alls):>7}{sum(alls)/len(alls):>7.2f}"
              f"{100*sum(1 for x in alls if x<=2)/len(alls):>19.1f}%")

    lows = [(route, di, q, a, s) for route, di, q, a, s in scored if s <= 2]
    print(f"\n=== {len(lows):,} misaligned (score <= 2), showing {min(show, len(lows))} ===")
    for route, di, q, a, s in lows[:show]:
        print(f"  [{route} {di}] score={s}")
        print(f"      Q: {q[:150]}")
        print(f"      A: {a[:150]}")


def main():
    ap = argparse.ArgumentParser(description="LLM-judge Q/A alignment verification")
    ap.add_argument("--route", choices=ROUTES, help="single route (default: all)")
    ap.add_argument("--sample", type=int, default=1000, help="rows/route to judge (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=engine.MODEL, help="judge model (default: engine MODEL)")
    ap.add_argument("--show", type=int, default=20, help="misaligned examples to print")
    args = ap.parse_args()
    routes = [args.route] if args.route else list(ROUTES)
    items = _collect(routes, args.sample, args.seed)
    print(f"collected {len(items):,} pairs from {len(routes)} route(s); judging...")
    asyncio.run(_run(items, args.model, args.show))


if __name__ == "__main__":
    main()
