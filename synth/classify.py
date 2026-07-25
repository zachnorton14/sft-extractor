"""Model classifier pass: route every harvested excerpt semantically.

The regex affordance gate (corpus.affordance_label) is a cheap RECALL net — it finds
candidate excerpts but can't make the semantic cuts that matter (grounded vs fiction, a
genuine document vs prose that merely says "whereas", how-to vs description). This pass
reads the materialized corpus and has a model assign each excerpt one or more ROUTE
classes from the fixed taxonomy below, plus a `drop` for unusable spans.

It is the cheapest model pass in the pipeline: the output is a short label list, not
prose, so batches pack hard. It runs once over excerpts.jsonl (resumable), writes the
model labels back as `classes` (multi-label, best-fit first) and `primary` (the highest
claim under CLAIM_ORDER), and leaves the regex `affordance` field untouched so the two
can be compared. Routes migrate from `affordance` to `primary` afterwards.

`primary` follows the priority claim order: rare, specific classes (stem, how-to, verse,
dialogue) claim an excerpt before the elastic catch-alls (knowledge, composition), so a
scarce class is never strip-mined by a class that fits almost anything.

Env (same as the model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your key>          # a NON-reasoning model is cheapest here
"""

import asyncio
import json
import random
from collections import Counter
from types import SimpleNamespace

import anthropic

from synth import corpus, engine

MODEL = engine.MODEL              # override with a cheap non-reasoning model if available
MAX_TOKENS = 8192                 # headroom if the model emits a thinking block
CONCURRENCY = 20
TOKEN_BUDGET = 2000               # ~9 excerpts/batch; output is tiny, input dominates

STATE_FILE = engine.STATE_DIR / "classify.json"

# The routing taxonomy. Order here is documentation; CLAIM_ORDER sets priority.
LABELS = {
    "knowledge":          "states facts, definitions, or explanations about the world; answers what/why/how from general knowledge (expository).",
    "reasoning":          "argues from premises to a conclusion; the author works through an inference chain.",
    "stem_reasoning":     "quantitative or physical reasoning: mathematics, mechanics, engineering, chemistry, a worked physical argument.",
    "narrative_grounded": "recounts REAL events or real persons — history, memoir, biography, travel of record — placeable in the actual world.",
    "narrative_fiction":  "an invented or purely personal scene with no external anchor: a novel, a tale, an unplaceable first-person incident.",
    "opinion":            "advances a judgment, preference, or persuasive stance — what is good, right, or best.",
    "how_to":             "tells how to DO or MAKE something: the steps of a method, recipe, procedure, technique, or drill.",
    "conversational":     "a dialogue, catechism, or question-and-answer exchange — two or more voices, or an explicit Q/A structure.",
    "composition":        "IS a formal composed document: statute/act, legal pleading, letter, speech/oration/resolution, prayer/devotion, proclamation, contract/instrument. Tag ONLY a genuine document — never merely because prose could be imitated.",
    "verse":              "poetry or metrical verse — lines, meter, or rhyme.",
    "drop":               "not usable: OCR gibberish, a bare list/table/index/heading, a fill-in form, a catalog or dictionary index entry, or too fragmentary to answer anything.",
}

# Priority: rarest & most specific first; elastic catch-alls (knowledge, composition)
# last; drop only if nothing else fits. `primary` is the first of these present.
CLAIM_ORDER = [
    "stem_reasoning", "how_to", "verse", "conversational", "reasoning",
    "narrative_grounded", "narrative_fiction", "opinion", "knowledge",
    "composition", "drop",
]

_LABEL_LIST = "\n".join(f'  - "{k}": {v}' for k, v in LABELS.items())

SYSTEM = f"""\
You classify a short passage from a pre-1930s book by what kind of question-answer task
it could source. Choose from these classes ONLY, copied verbatim:
{_LABEL_LIST}

Rules:
- List EVERY class the passage genuinely supports, best fit first. Most passages fit ONE
  or TWO. Do not pad the list.
- "composition" is a GENUINE-document tag, not a fallback — do not add it just because
  any prose could be imitated. A rare, specific class (stem_reasoning, how_to, verse,
  conversational) must be tagged whenever it truly applies; never omit it in favour of a
  broader one.
- narrative splits by whether the events are real: "narrative_grounded" for real events
  or real persons you can place in the world, "narrative_fiction" for invented or
  unplaceable personal scenes.
- If nothing yields a usable task — OCR soup, a bare list or heading, a form, an index
  entry, a fragment — return ["drop"].

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only: [{{"i": 0, "classes": ["class", ...]}}, ...]
"""


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


def primary(classes):
    """The highest-priority class present, under CLAIM_ORDER."""
    for c in CLAIM_ORDER:
        if c in classes:
            return c
    return "drop"


async def _classify_batch(client, semaphore, batch, state, cfg):
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": i, "text": it["excerpt"]} for i, it in enumerate(batch)])
    parsed = await engine._call(client, semaphore, cfg, SYSTEM, payload, len(batch))
    for r in parsed if isinstance(parsed, list) else []:
        if not isinstance(r, dict):
            continue
        idx = r.get("i")
        if not (isinstance(idx, int) and 0 <= idx < len(keys)):
            continue
        classes = [c for c in (r.get("classes") or []) if c in LABELS]
        if classes:
            state[keys[idx]] = classes


async def run_async(records, state, save=True):
    cfg = SimpleNamespace(model=MODEL, max_tokens=MAX_TOKENS)
    pending = [r for r in records if str(r["doc_index"]) not in state]
    if not pending:
        print("  nothing pending")
        return
    batches = engine._pack_batches(pending, TOKEN_BUDGET)
    print(f"  {len(pending)} excerpts, {len(batches)} batches...")
    done = [0]
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def tracked(batch):
            await _classify_batch(client, semaphore, batch, state, cfg)
            done[0] += len(batch)
            if done[0] % 200 < len(batch) and save:
                save_state(state)
                print(f"  {done[0]}/{len(pending)}", flush=True)

        await asyncio.gather(*[asyncio.create_task(tracked(b)) for b in batches])
    if save:
        save_state(state)


def write_back(records, state):
    """Merge model labels (`classes`, `primary`) into excerpts.jsonl in place,
    leaving the regex `affordance` field for comparison."""
    for r in records:
        cl = state.get(str(r["doc_index"]))
        if cl:
            r["classes"] = cl
            r["primary"] = primary(cl)
    with corpus.EXCERPTS_FILE.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = sum(1 for r in records if "primary" in r)
    print(f"wrote labels for {n}/{len(records)} excerpts -> {corpus.EXCERPTS_FILE}")


# regex affordance -> the model primary it should roughly correspond to, for scoring
_AFF_TO_PRIMARY = {
    "expository": "knowledge", "argument": "reasoning", "opinion": "opinion",
    "procedural": "how_to", "composition": "composition",
    "narrative": ("narrative_grounded", "narrative_fiction"),
}


def report(records, state):
    labelled = [(r, state.get(str(r["doc_index"]))) for r in records]
    labelled = [(r, cl) for r, cl in labelled if cl]
    if not labelled:
        print("no excerpts classified yet")
        return
    n = len(labelled)
    prim = Counter(primary(cl) for _, cl in labelled)
    print(f"\nclassified {n}/{len(records)} excerpts")
    print("\nprimary class (claim-order winner):")
    for c in CLAIM_ORDER:
        if prim[c]:
            print(f"  {c:20} {prim[c]:5}  {prim[c]*100//n:3d}%")

    allc = Counter(c for _, cl in labelled for c in cl)
    multi = sum(1 for _, cl in labelled if len(cl) > 1)
    print(f"\nmulti-label: {multi}/{n} ({multi*100//n}%) fit >1 class; "
          f"avg {sum(len(cl) for _, cl in labelled)/n:.2f} labels")
    print("any-label coverage (excerpt counts toward each):")
    for c in CLAIM_ORDER:
        if allc[c]:
            print(f"  {c:20} {allc[c]:5}")

    # how did the regex gate do? primary vs mapped affordance
    agree = moved = 0
    conf = Counter()
    for r, cl in labelled:
        aff = r.get("affordance")
        p = primary(cl)
        exp = _AFF_TO_PRIMARY.get(aff)
        ok = (p in exp) if isinstance(exp, tuple) else (p == exp)
        agree += ok
        if not ok:
            moved += 1
            conf[(aff, p)] += 1
    print(f"\nregex gate vs model primary: {agree}/{n} agree ({agree*100//n}%), "
          f"{moved} reclassified")
    print("top regex -> model disagreements:")
    for (aff, p), k in conf.most_common(15):
        print(f"  {str(aff):14} -> {p:20} {k:4}")
