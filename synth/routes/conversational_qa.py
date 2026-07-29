"""Conversational route: turn dialogue/catechism excerpts into multi-turn rows.

Unlike every other route this one is NOT a single question-answer pair — it is a
multi-turn conversation. The model reads a two-party exchange (a catechism, a
Socratic dialogue, a Q&A book), assigns one voice to the USER role (the one who asks
or leads) and the other to the ASSISTANT role, and returns the turns VERBATIM and
alternating, so the row shows a real back-and-forth spanning many turns.

Still anachronism-safe: every turn's text is verified as a verbatim substring of the
excerpt (whitespace-flexible), so the model may only split, assign, and order the
spoken words — it can write nothing. Rows that aren't a genuine multi-turn two-party
exchange (fewer than MIN_TURNS, non-alternating, or a turn that isn't verbatim) are
dropped.

Because the shape differs (a `conversations` list, not question/answer), this route
carries its own run/write loop, reusing the engine's transport, batching, retry,
state, and thinking-off config.

Input:  synth/output/excerpts.jsonl (classes contains "conversational")
Output: synth/output/conversational_qa.json
        [{"doc_index","category","year","prose_score","excerpt","conversations":[{role,content}...]}]

Env:
    export OPENCODE_API_KEY=<your opencode Go key>   # or put it in ROOT/.env
"""

import asyncio
import json
import random
import re
from collections import Counter
from datetime import datetime

from synth import corpus, engine

CLASSES = ("conversational",)     # classifier classes this route sources
MIN_TURNS = 4                     # at least two full exchanges — a real conversation

# The classifier over-tags novel and play dialogue as `conversational` (quoted speech
# with questions reads as a two-party exchange). Genuine usable material — catechism,
# Socratic dialogue, deposition, debate — has EXPLICIT structure the classifier can't
# see: Q./A. markers, or a repeated short speaker-name label (Euph./Alc., Chairman.).
# Prose fiction lacks these; plays have them but live in LANGUAGE AND LITERATURE, which
# the route excludes. So sourcing filters on structure + category, not the class alone.
_SPEAKER_LABEL = re.compile(r'(?:^|(?<=[.!?”"\'])\s)([A-Z][a-z]{0,9}\.)(?=\s+[A-Z“"\'])')
_QA_MARK = re.compile(r"(?:^|\n|\.\s)\s*(?:Q\.|A\.|Ques\b|Ans\b)", re.I)
_EXCLUDE_CATEGORIES = {"LANGUAGE AND LITERATURE"}


def _is_structured_dialogue(text):
    """True when the excerpt shows explicit two-party structure — Q./A. markers, or a
    short speaker-name label used at least twice (a real turn-taking pattern) — the
    mark of a catechism / Socratic dialogue / deposition / debate rather than prose
    fiction with quoted speech."""
    repeated = sum(v for v in Counter(_SPEAKER_LABEL.findall(text)).values() if v >= 2)
    return len(_QA_MARK.findall(text)) >= 2 or repeated >= 3

SYSTEM = f"""\
You are given a passage of pre-1930s DIALOGUE — a conversation, catechism, or
question-and-answer exchange between TWO voices. Turn it into a multi-turn conversation.

1. Identify the two speakers. Assign one to the USER role — the voice that ASKS the
   questions or leads — and the other to the ASSISTANT role — the voice that ANSWERS. In
   a catechism the questioner is the user.
   - The ASSISTANT NEVER asks a question; it only answers. Every question belongs to a
     USER turn. If one utterance gives an answer and THEN poses the next question, SPLIT
     it — the answer is the assistant's turn, and the following question becomes the next
     USER turn (each half is still a verbatim substring). An assistant turn must not end
     on a question.

2. Return the exchange as "turns": an ordered list, each turn one speaker's utterance
   copied VERBATIM from the passage, ALTERNATING user, assistant, user, assistant …,
   beginning with the user. Keep the natural back-and-forth — several turns, not one
   pair.
   - The FIRST turn (user) must be a question that STANDS ON ITS OWN: it names its own
     subject and assumes no earlier context. Do NOT open mid-exchange — not with "And
     …", "Then …", "But …", nor a bare "it / this / they / these" whose antecedent was
     never given. If the passage's opening question depends on unstated context, BEGIN
     the conversation at a later question that stands alone, or emit no item.
   - Copy WORD FOR WORD. Do NOT paraphrase, rewrite, summarize, translate, or add any
     word. Each turn is the spoken words themselves.
   - Strip ONLY the speaker's label or attribution and the quotation marks that mark
     who is speaking ("said John", "Q.", "A.", "Socrates.", "" ""). Keep a turn's own
     wording exactly, punctuation and spelling included.
   - Drop narration or stage directions that fall BETWEEN turns; keep only what is
     spoken. Each turn's remaining words must appear verbatim and unbroken in the
     passage (do not stitch across an interruption like '"I am," he said, "a
     traveller"').
   - Begin and end each turn at a natural utterance boundary.

3. The exchange must have at least {MIN_TURNS} turns (two full exchanges) and strictly
   alternate the two roles. If the passage is not a genuine two-party spoken exchange —
   it is narration, monologue, exposition, or a single Q&A pair — emit NO item for it.

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "turns": [{{"role": "user", "content": "..."}},
                       {{"role": "assistant", "content": "..."}}, ...]}}, ...]
"""


def _verbatim_turn(content, excerpt):
    """Return the excerpt's own text for `content` (whitespace-flexible match), or
    None if it is not a verbatim, unbroken substring."""
    if not isinstance(content, str) or not content.split():
        return None
    pat = r"\s+".join(re.escape(tok) for tok in content.split())
    m = re.search(pat, excerpt)
    return m.group(0) if m else None


def build_turns(r, excerpt):
    """Validate + rebuild the model's turns: >= MIN_TURNS, strictly alternating from
    user, every turn verbatim. Returns the clean turn list or None (drop the row)."""
    turns = r.get("turns")
    if not isinstance(turns, list) or len(turns) < MIN_TURNS:
        return None
    out, expect = [], "user"
    for t in turns:
        if not isinstance(t, dict) or t.get("role") != expect:
            return None                              # missing/mis-ordered role -> drop
        content = _verbatim_turn(t.get("content"), excerpt)
        if not content:
            return None                              # not verbatim -> drop
        if expect == "assistant" and content.rstrip().endswith("?"):
            return None                              # assistant never asks -> drop
        out.append({"role": expect, "content": content})
        expect = "assistant" if expect == "user" else "user"
    return out


def source_excerpts(n, seed=0, **_):
    """Conversational excerpts: classifier-tagged `conversational`, then narrowed to
    genuine structured two-party exchanges — outside the fiction-heavy LANGUAGE category
    and carrying explicit dialogue structure (see _is_structured_dialogue). This drops
    the novel/play dialogue the classifier over-tags. n falsy or >= pool returns the
    whole filtered pool; else a seeded sample. Requires `classify` write-back."""
    mat = [r for r in corpus.load_excerpts(cls=CLASSES)
           if r.get("category") not in _EXCLUDE_CATEGORIES
           and _is_structured_dialogue(r["excerpt"])]
    if not n or n >= len(mat):
        return mat
    return random.Random(seed).sample(mat, n)


# Config carrier: reuses the engine's Route fields for model / max_tokens / thinking /
# concurrency / batching / state file. answer_fn is unused (custom run below).
ROUTE = engine.Route(
    name="conversational_qa",
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
    out = engine.OUTPUT_DIR / "conversational_qa.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"  wrote {len(rows)} conversational_qa rows -> {out}")
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
    header = [f"route: conversational_qa", f"model: {ROUTE.model}", f"seed:  {seed}",
              f"n:     {len(excerpts)}", f"kept:  {kept}/{len(excerpts)}", "",
              "--- SYSTEM PROMPT ---", SYSTEM, ""]
    d = engine.ROOT / "synth" / "samples" / "routes" / "conversational_qa"
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
