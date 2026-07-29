"""Multiturn route: turn one fact-rich excerpt into an interrelated Q&A conversation.

Where conversational_qa is EXTRACTIVE on both sides (it lifts a real two-party
dialogue verbatim, and is bottlenecked by how little genuine catechism/deposition text
exists), this route is a HYBRID that rides on the huge `knowledge` pool:

  - each USER turn is a model-COMPOSED question (period-schoolbook register), like
    knowledge_qa's question — anachronism on the question side is handled by the
    separate question filter, not here;
  - each ASSISTANT turn is a VERBATIM answer, one or two exact spans lifted straight
    from the excerpt and verified as literal substrings (verbatim_answer). No
    model-written word ever enters an answer, so the answers stay anachronism-safe by
    construction.

The point is MULTI-TURN, not dialogue provenance: the model picks SEVERAL different
facts from one passage and threads them into an interrelated exchange. Only the FIRST
question must stand on its own (knowledge_qa's self-situating rule); follow-ups are
ALLOWED to lean on the context already established in the conversation ("And what
became of it?", "Why was that so?") — that coreference across turns is exactly the
behavior we want to teach, and it is safe because the antecedent now lives in an
earlier turn.

Because the shape is a `conversations` list (not a single question/answer), this route
carries its own run/write loop, reusing the engine's transport, batching, retry,
state, and thinking-off config.

Input:  synth/output/excerpts.jsonl (classes contains "knowledge")
Output: synth/output/multiturn_qa.json
        [{"doc_index","category","year","prose_score","excerpt","conversations":[{role,content}...]}]

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import asyncio
import json
import random
from datetime import datetime

from synth import corpus, engine

CLASSES = ("knowledge",)          # classifier classes this route sources
MIN_TURNS = 4                     # at least two full exchanges — a real conversation
MAX_TURNS = 8                     # cap so one excerpt doesn't overreach
MAX_SPANS = 2                     # verbatim spans per assistant answer

SYSTEM = f"""\
You are given a short fact-rich passage from a pre-1930s book. Build a MULTI-TURN
question-and-answer conversation about it between a curious USER who asks and an
ASSISTANT who answers. Work in this order.

1. Pick SEVERAL DIFFERENT facts the passage states — a definition, a cause, a number,
   a consequence, a name — enough for at least two exchanges. Each fact will become one
   assistant answer, so choose facts that lie in DISTINCT places in the passage.

2. For EACH fact, write the assistant's answer as "spans": a list of ONE exact
   quotation from the passage (up to {MAX_SPANS} only if the fact genuinely lies in
   separate places).
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize, or
     add any word not in the passage. Take WHOLE sentences: begin and end each span at
     a sentence boundary and keep its terminal punctuation.
   - The ASSISTANT ONLY ANSWERS — it never asks a question and never ends on a question.

3. For EACH answer write the USER question ("q") it answers.
   - The span must CORRECTLY AND COMPLETELY answer the question. Do not demand a name,
     number, or identity the span does not supply. Do not reveal the answer, or its
     distinctive wording, in the question.
   - Plain period-schoolbook register, pre-1930s English: period vocabulary, spelling,
     and phrasing. Use no word, term, or idiom that came into use after 1930. Never
     mention the source — no "the passage", "the text", "according to", "mentioned".
   - The FIRST question must SELF-SITUATE — it must make sense to someone who NEVER saw
     the passage. Name the actual subject: the war, battle, place, ruler, work, statute,
     or scripture involved, using your OWN knowledge of the subject when the passage
     assumes it (e.g. "the Revised Statutes of the United States", "at the Battle of
     Williamsburg", "in the Gospel of Luke"). Frame the reader — supply the era, place,
     subject, or situation that makes the span the natural, correct answer.
     FORBIDDEN in the first question — references that mean nothing on their own:
     catalog, specimen, or figure numbers ("specimen 167"); a bare "the statute", "the
     act", "the gun", "the assembly", "the expedition"; a bare "it", "this", "they",
     "these"; or "two specific verses" / "a certain scholar" without naming which. If
     the fact you picked cannot be made to stand alone (it hinges on an item known only
     from the passage), pick a DIFFERENT fact to open with — one that can.
   - LATER questions build on the conversation so far — they MAY and SHOULD use natural
     follow-up phrasing that refers back to what was ALREADY ESTABLISHED in an earlier
     turn ("And why did that happen?", "What became of him afterward?", "How was it
     accomplished?"). But a pronoun or "the ___" is allowed ONLY when its antecedent
     was actually named in a previous turn; never introduce a NEW bare reference the
     conversation has not yet identified. Keep the thread coherent: each question should
     follow from the last answer.

4. Return the exchange as "turns": an ordered list ALTERNATING user, assistant, user,
   assistant …, beginning with the user, {MIN_TURNS} to {MAX_TURNS} turns (two to four
   full exchanges). A user turn carries "q"; an assistant turn carries "spans".
   If the passage does not state enough distinct facts for two exchanges, emit NO item
   for it.

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "turns": [{{"role": "user", "q": "..."}},
                       {{"role": "assistant", "spans": ["exact quotation"]}},
                       {{"role": "user", "q": "..."}},
                       {{"role": "assistant", "spans": ["exact quotation"]}}]}}, ...]
"""


def build_turns(r, excerpt):
    """Validate + rebuild the model's turns: MIN_TURNS..MAX_TURNS, strictly alternating
    from user, each user turn a non-empty composed question, each assistant turn a
    verbatim answer (spans verified as literal substrings). Returns the clean turn list
    or None (drop the row)."""
    turns = r.get("turns")
    if not isinstance(turns, list) or not (MIN_TURNS <= len(turns) <= MAX_TURNS):
        return None
    if len(turns) % 2:                               # must end on an assistant answer
        return None
    out, expect = [], "user"
    for t in turns:
        if not isinstance(t, dict) or t.get("role") != expect:
            return None                              # missing/mis-ordered role -> drop
        if expect == "user":
            q = t.get("q") or t.get("content")
            if not isinstance(q, str) or not q.strip():
                return None
            content = q.strip()
        else:
            spans = t.get("spans")
            if spans is None and t.get("content"):
                spans = [t["content"]]
            content = engine.verbatim_answer(spans, excerpt, MAX_SPANS)
            if not content:
                return None                          # not verbatim -> drop
        out.append({"role": expect, "content": content})
        expect = "assistant" if expect == "user" else "user"
    return out


def source_excerpts(n, seed=0, **_):
    """Multiturn excerpts: the classifier-tagged `knowledge` pool (the same fact-rich
    expository excerpts knowledge_qa uses). n falsy or >= pool returns the whole pool;
    else a seeded sample. Requires `classify` write-back."""
    mat = corpus.load_excerpts(cls=CLASSES)
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


# Config carrier: reuses the engine's Route fields for model / max_tokens / thinking /
# concurrency / batching / state file. answer_fn is unused (custom run below).
ROUTE = engine.Route(
    name="multiturn_qa",
    system=SYSTEM,
    source=source_excerpts,
    answer_fn=lambda r, e: None,
    passthrough=("prose_score",),
    extra_body=engine.DISABLE_THINKING,
)


async def _batch(client, semaphore, batch, state):
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "text": it["excerpt"]} for i, it in enumerate(batch)])
    parsed = await engine._call(client, semaphore, ROUTE, SYSTEM, payload, len(batch))
    for r in parsed if isinstance(parsed, list) else []:
        if not isinstance(r, dict):
            continue
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        turns = build_turns(r, batch[idx]["excerpt"])
        if turns:
            state[keys[idx]] = {"turns": turns}


async def run_async(excerpts, state, save=True):
    async with engine.open_client() as client:
        semaphore = asyncio.Semaphore(ROUTE.concurrency)
        pending = [e for e in excerpts if str(e["doc_index"]) not in state]
        if not pending:
            print("  nothing pending")
            return
        batches = engine._pack_batches(pending, ROUTE.token_budget)
        print(f"  {len(pending)} excerpts, {len(batches)} batches...")
        done = [0]

        async def tracked(batch):
            await _batch(client, semaphore, batch, state)
            done[0] += len(batch)
            if done[0] % 100 < len(batch) and save:
                save_state(state)
                print(f"  {done[0]}/{len(pending)}", flush=True)

        await asyncio.gather(*[asyncio.create_task(tracked(b)) for b in batches])
    if save:
        save_state(state)


def write_output(excerpts, state):
    engine.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        if not (r and r.get("turns")):
            continue
        rows.append({
            "doc_index": e["doc_index"], "category": e.get("category"),
            "year": e.get("year"), "prose_score": e.get("prose_score"),
            "excerpt": e["excerpt"], "conversations": r["turns"],
        })
    out = engine.OUTPUT_DIR / "multiturn_qa.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"  wrote {len(rows)} multiturn_qa rows -> {out}")
    return out


def _conv_lines(excerpts, state):
    out = []
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        turns = r.get("turns") if r else None
        out.append("=" * engine.WRAP)
        out.append(engine._wrap("", f"[{e.get('category')}]  {e.get('year')}  "
                                    f"turns={len(turns) if turns else 0}"))
        out.append(engine._wrap("  excerpt : ", e["excerpt"][:400] + "..."))
        if turns:
            for t in turns:
                label = "  USER  : " if t["role"] == "user" else "  ASST  : "
                out.append(engine._wrap(label, t["content"]))
        else:
            out.append("  (dropped)")
        out.append("")
    return out


async def test_run(excerpts):
    state = {}
    await run_async(excerpts, state, save=False)
    print("\n".join(_conv_lines(excerpts, state)))


async def sample_run(excerpts, seed=0):
    state = {}
    await run_async(excerpts, state, save=False)
    kept = sum(1 for e in excerpts if (state.get(str(e["doc_index"])) or {}).get("turns"))
    header = [f"route: multiturn_qa", f"model: {ROUTE.model}", f"seed:  {seed}",
              f"n:     {len(excerpts)}", f"kept:  {kept}/{len(excerpts)}", "",
              "--- SYSTEM PROMPT ---", SYSTEM, ""]
    d = engine.ROOT / "synth" / "samples" / "routes" / "multiturn_qa"
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = d / f"{ts}_seed{seed}_n{len(excerpts)}.txt"
    path.write_text("\n".join(header + _conv_lines(excerpts, state)), encoding="utf-8")
    print(f"wrote sample ({kept}/{len(excerpts)} kept) -> {path}")
    return path


def load_state():
    return engine.load_state(ROUTE)


def save_state(state):
    engine.save_state(ROUTE, state)
