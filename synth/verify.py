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
You are given a numbered list of QUESTION/ANSWER pairs from pre-1930s books. For each
pair, judge whether the ANSWER is a RIGHT and COMPLETE answer to the QUESTION — not
merely on the same subject.

Be strict about CORRECTNESS. The answer must actually resolve what the question asks —
the right name, number, date, cause, or outcome — completely and without mismatch. Mark
0 if the answer is on-topic but does not correctly or fully answer: it gives the wrong
specific, omits what was asked, only REFERS to the answer ("this was the ...", "is
reported by ...") without stating it, is cut off, or does not actually match the
question.

Do NOT fact-check against modern knowledge: an answer that correctly reflects what a
pre-1930 source states is CORRECT even if modern science disagrees. You judge whether the
answer correctly answers the question, not whether the period belief is true today. Do
NOT judge style or era.

Study these examples, then apply the SAME standard:
  Q: "Who discovered the circulation of the blood?"
  A: "The circulation of the blood was discovered by Harvey."            -> 1  (states it)
  Q: "What was the original constitution of the government of New-Haven?"
  A: "This was the original, fundamental constitution of New-Haven."     -> 0  (only REFERS
     to it; never says what it was)
  Q: "How many men were lost in the assault on the redoubt?"
  A: "The assault was made at dawn and repulsed with heavy loss."        -> 0  (on topic
     but never gives the number asked)
  Q: "In what year did Harvard place its law school on a graduate basis?"
  A: "Harvard placed its school on a graduate basis in 1896."            -> 1  (right specific)
  Q: "By what river did General Wolfe land before the assault on Quebec?"
  A: "General Wolfe landed by the Hudson before the assault on Quebec."  -> 0  (wrong
     specific — the answer names the wrong thing)
  Q: "What did Fleischmann report as the most extraordinary instance of prepotency?"
  A: "The most extraordinary instance of prepotency is reported by Fleischmann." -> 0
     (attributes it but never states what it was)

Output ONLY a bare JSON array of 0/1 — one value per input pair, IN THE SAME ORDER,
nothing else (no keys, no text):
  1 = correctly and completely answers the question
  0 = does not
Return EXACTLY as many values as there are input pairs, e.g. [1,1,0,1,0].

Input: JSON array of pairs [{"q": "...", "a": "..."}, ...]
"""

# Judge should be deterministic: temperature 0. (extra_body merges into the request; for
# DeepSeek DISABLE_THINKING is empty, for opencode it carries the thinking-off flag.)
JUDGE_EXTRA = {**engine.DISABLE_THINKING, "temperature": 0}


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


async def _judge_all(client, sem, route_cfg, items, batch, label):
    """Judge all items in batches, printing live progress (pairs judged / total) on a
    single updating line."""
    total = len(items)
    bs = [items[i:i + batch] for i in range(0, total, batch)]
    out, done, last = [], [0], [-1]

    async def one(b):
        out.extend(await _judge(client, sem, route_cfg, b))
        done[0] += len(b)
        pct = done[0] * 100 // max(1, total)
        if pct != last[0]:
            last[0] = pct
            print(f"\r  {label}: {done[0]:,}/{total:,} ({pct}%)   ", end="", flush=True)

    await asyncio.gather(*[asyncio.create_task(one(b)) for b in bs])
    print()
    return out


async def _run(items, model, batch_size, show):
    route = engine.Route(name="verify", system=SYSTEM, source=None, answer_fn=None,
                         model=model, extra_body=JUDGE_EXTRA)
    async with engine.open_client() as client:
        sem = asyncio.Semaphore(engine.CONCURRENCY)
        scored = await _judge_all(client, sem, route, items, batch_size, "judging")
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


async def _run_sanity(routes, model, batch, n, seed):
    """Discrimination test: judge N real pairs AND N deliberately-WRONG pairs (each
    question given a different question's answer). A discriminating judge scores real
    pairs ~1 and mismatched pairs ~0. If it passes the mismatches as 1, it is rubber-
    stamping and the pass is worthless."""
    reals = _collect(routes, 0, seed)
    reals = random.Random(seed).sample(reals, min(n, len(reals)))
    perm = [p[3] for p in reals]
    random.Random(seed + 1).shuffle(perm)
    mism = [(r, di, q, a2) for (r, di, q, a), a2 in zip(reals, perm) if a2 != a]
    route = engine.Route(name="verify", system=SYSTEM, source=None, answer_fn=None,
                         model=model, extra_body=JUDGE_EXTRA)
    async with engine.open_client() as client:
        sem = asyncio.Semaphore(engine.CONCURRENCY)
        real_scored = await _judge_all(client, sem, route, reals, batch, "sanity real")
        mism_scored = await _judge_all(client, sem, route, mism, batch, "sanity mismatch")
    keep = 100 * sum(s for *_, s in real_scored) / max(1, len(real_scored))
    caught = 100 * sum(1 for *_, s in mism_scored if s == 0) / max(1, len(mism_scored))
    print(f"\nSANITY (judge={model}, batch={batch}):")
    print(f"  real pairs      : {len(real_scored):>5} judged, {keep:5.1f}% scored 1  "
          f"(want high — real pairs are mostly correct)")
    print(f"  mismatched pairs: {len(mism_scored):>5} judged, {caught:5.1f}% scored 0  "
          f"(DISCRIMINATION — want high; low = rubber-stamping)")


async def _run_filter(routes, model, batch, shard_size):
    """Full filter pass: judge every pair of every route, DROP any row that has a pair
    scored 0 (a whole multiturn conversation goes if any turn fails), and write the kept
    rows to filtered/<route>/ on HF as uniform shards. Reads fresh (bypass cache) and
    processes route-by-route so completed routes are durable if interrupted."""
    from synth import hf_push
    route_cfg = engine.Route(name="verify", system=SYSTEM, source=None, answer_fn=None,
                             model=model, extra_body=JUDGE_EXTRA)
    async with engine.open_client() as client:
        sem = asyncio.Semaphore(engine.CONCURRENCY)
        for ri, route in enumerate(routes, 1):
            tag = f"[{ri}/{len(routes)}] {route}"
            print(f"{tag}: loading from HF...", flush=True)
            rows = _load_route(route, fresh=True)
            items = [(route, row.get("doc_index"), q, a)
                     for row in rows for q, a in _pairs(row) if q and a]
            scored = await _judge_all(client, sem, route_cfg, items, batch, tag)
            drop = {di for (_, di, _, _, sc) in scored if sc == 0}   # any 0 -> drop the row
            kept = [row for row in rows if row.get("doc_index") not in drop]
            print(f"  {tag}: writing filtered/{route}/ ...", flush=True)
            _, nshards, _ = hf_push.write_sharded(f"filtered/{route}", kept, shard_size)
            pct = 100 * len(drop) / max(1, len(rows))
            print(f"  {tag}: {len(rows):,} rows -> kept {len(kept):,} "
                  f"(dropped {len(drop):,}, {pct:.1f}%)  -> filtered/{route}/ ({nshards} shards)\n",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description="LLM-judge Q/A alignment verification")
    ap.add_argument("--route", choices=ROUTES, help="single route (default: all)")
    ap.add_argument("--sample", type=int, default=1000, help="rows/route to judge (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=engine.MODEL, help="judge model (default: engine MODEL)")
    ap.add_argument("--batch", type=int, default=BATCH, help="pairs per judge call (bare 0/1 array; auto-splits on drift)")
    ap.add_argument("--show", type=int, default=20, help="misaligned examples to print")
    ap.add_argument("--sanity", type=int, default=0, metavar="N",
                    help="discrimination test on N real + N shuffled-answer pairs")
    ap.add_argument("--filter", action="store_true",
                    help="FULL pass: judge everything, drop rows scored 0, write filtered/<route>/ to HF")
    ap.add_argument("--shard-size", type=int, default=2000, help="rows per filtered shard")
    args = ap.parse_args()
    routes = [args.route] if args.route else list(ROUTES)
    if args.sanity:
        asyncio.run(_run_sanity(routes, args.model, args.batch, args.sanity, args.seed))
        return
    if args.filter:
        asyncio.run(_run_filter(routes, args.model, args.batch, args.shard_size))
        return
    items = _collect(routes, args.sample, args.seed)
    print(f"collected {len(items):,} pairs from {len(routes)} route(s); judging (batch={args.batch})...")
    asyncio.run(_run(items, args.model, args.batch, args.show))


if __name__ == "__main__":
    main()
