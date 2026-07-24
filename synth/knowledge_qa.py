"""Knowledge-QA route: turn expository excerpts into grounded Q/A rows.

The first content-bound generator. It sources gated excerpts from the pretrain
corpus (synth/corpus.py), keeps the ones the affordance gate tagged `expository`,
and for each one makes a single model call that distills the passage into a
focused question-answer pair:

  - the ANSWER is a VERBATIM quotation (1-2 exact spans) lifted straight from the
    passage — never composed or paraphrased. This is the anachronism guarantee: an
    answer built only from period text cannot contain modern phrasing. Each span is
    verified as a literal substring of the excerpt (verbatim_answer); anything not
    found verbatim is dropped, not repaired. Spans join with an ellipsis, so no
    model-written word ever enters the answer.
  - the QUESTION is in period-schoolbook register, model-composed and self-situating
    (question-side anachronism is handled by a separate filter, not here).

Only the passage grounds the answer — no outside facts — so a fiction or opinion
passage would produce ungrounded claims; those route elsewhere and are excluded
here by taking only `expository` excerpts.

The model also classifies each passage's ACTUAL subject (content-level), since the
metadata category is book-level and any subject can appear in any book. Rows carry
both: `category` (content, from the model) and `book_category` (from the metadata),
with `category_moved` flagging divergence.

Input:  sampled via corpus.sample_excerpts (coverage-weighted across categories)
Output: synth/output/knowledge_qa.json
        [{"doc_index","category","book_category","category_moved","year",
          "prose_score","excerpt","question","answer"}]

Set environment before running (same as the other model passes):
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
import re
from pathlib import Path

import anthropic

from synth import corpus


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

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "synth" / "output"
STATE_DIR = ROOT / "synth" / "state"
STATE_FILE = STATE_DIR / "knowledge_qa.json"

MODEL = "claude-haiku-4-5"       # maps to deepseek via ANTHROPIC_BASE_URL
MAX_TOKENS = 16384               # reasoning model: needs room for thinking + JSON
CONCURRENCY = 20
TOKEN_BUDGET = 1200              # chars/4 per batch; excerpts are ~150 words each
ROUTE = "expository"             # this generator handles the knowledge-QA route

_CLASS_LIST = "\n".join(f"  - {c}" for c in corpus.LOC_CLASSES)

SYSTEM = f"""\
You are given a short passage from a pre-1930s book. Work in this order.

1. FIND THE ANSWER FIRST. Choose the single most salient fact, definition, or
   explanation the passage states, and copy it VERBATIM as the answer.
   - Return it as "spans": a list of ONE exact quotation from the passage — or up
     to THREE, only if the answer genuinely lies in separate places. Prefer one;
     never more than three. Choose the SHORTEST span(s) that carry the whole fact.
   - Copy WORD FOR WORD. Do NOT paraphrase, summarize, rewrite, correct, modernize,
     reorder, or add any word not in the passage. Pick a span that reads as a clean,
     quotable fact, so a short exact quotation suffices.

2. THEN WRITE THE QUESTION that this answer answers.
   - Plain period-schoolbook register: direct, often "What/Why/How/Of what". No
     modern or conversational phrasing, no meta-language.
   - Stands alone. Never mention the source — no "the passage", "the text",
     "according to", "described", "mentioned". Ask as if from general knowledge.
   - Self-situating — the question must make sense to someone who NEVER saw the
     passage. Name the actual subject: the war, battle, place, ruler, work,
     statute, or scripture involved, using your own knowledge of the subject when
     the passage assumes it (e.g. "the Revised Statutes of the United States", "at
     the Battle of Williamsburg", "in the Gospel of Luke"). FORBIDDEN — references
     that mean nothing on their own: catalog, specimen, or figure numbers
     ("specimen 167"); a bare "the statute", "the act", "the gun", "the assembly",
     "the expedition"; or "two specific verses" / "a certain scholar" without
     naming which. If the chosen fact simply cannot be made to stand alone (it
     hinges on an item known only from the passage), pick a DIFFERENT fact from the
     passage that can.
   - Do NOT put the answer, or the answer's distinctive wording, in the question.
     Ask for the fact; do not reveal it.

3. Classify the passage's ACTUAL subject (what it is about, not the kind of book it
   came from) into exactly one of these classes, copied verbatim:
{_CLASS_LIST}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only:
  [{{"i": 0, "spans": ["exact quotation"], "q": "...", "category": "ONE CLASS"}}, ...]
"""

MAX_SPANS = 3


def source_excerpts(n, alpha=0.5, min_conf=0.7, n_words=150, seed=0):
    """Return expository excerpts for the knowledge-QA route.

    Prefers the materialized corpus (synth/output/excerpts.jsonl from `harvest`) —
    read once, no per-row fetch. Falls back to live coverage-weighted sampling when
    nothing has been harvested yet (handy for a quick --test)."""
    mat = corpus.load_excerpts(affordance=ROUTE)
    if mat:
        return mat[:n] if n else mat
    recs = corpus.sample_excerpts(n, alpha=alpha, min_conf=min_conf,
                                  n_words=n_words, seed=seed)
    return [r for r in recs if r["affordance"] == ROUTE]


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


def _estimate_tokens(text):
    return len(text) // 4


def _pack_batches(items, token_budget=TOKEN_BUDGET):
    batches, cur, cur_tok = [], [], 0
    for it in items:
        tok = _estimate_tokens(it["excerpt"])
        if cur and cur_tok + tok > token_budget:
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append(it)
        cur_tok += tok
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


async def _generate_batch(client, semaphore, batch, state):
    keys = [str(it["doc_index"]) for it in batch]
    payload = json.dumps([{"i": idx, "text": it["excerpt"]} for idx, it in enumerate(batch)])
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": payload}],
                )
                text = next((b.text for b in msg.content if b.type == "text"), "").strip()
                if not text:
                    raise ValueError(
                        f"empty response (stop_reason={msg.stop_reason}, "
                        f"blocks={[b.type for b in msg.content]})"
                    )
                if msg.stop_reason == "max_tokens":
                    raise ValueError(f"truncated at max_tokens ({len(batch)} excerpts in batch)")
                for r in json.loads(_strip_fence(text)):
                    idx = r.get("i")
                    if not (isinstance(idx, int) and 0 <= idx < len(keys)):
                        continue
                    q = (r.get("q") or "").strip()
                    cat = (r.get("category") or "").strip()
                    spans = r.get("spans")
                    if spans is None and r.get("a"):        # tolerate single-string form
                        spans = [r["a"]]
                    a = verbatim_answer(spans, batch[idx]["excerpt"], MAX_SPANS)
                    if q and a:                             # a is None if not verbatim -> drop
                        state[keys[idx]] = {"q": q, "a": a, "category": cat}
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                raise SystemExit(f"\nAPI auth failed: {e}\n"
                                 f"Check ANTHROPIC_API_KEY matches ANTHROPIC_BASE_URL.")
            except json.JSONDecodeError as e:
                if attempt == 4:
                    print(f"  batch of {len(batch)} failed: unparseable response: "
                          f"{e}\n    raw[:200]: {text[:200]!r}")
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
            "category": content_cat,              # content-level, from the model
            "book_category": e["category"],       # book-level, from audit metadata
            "category_moved": content_cat != e["category"],
            "year": e["year"],
            "prose_score": e["prose_score"],
            "excerpt": e["excerpt"],
            "question": r["q"],
            "answer": r["a"],
        })
    out = OUTPUT_DIR / "knowledge_qa.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"  wrote {len(rows)} knowledge-QA rows -> {out}")
    return out


async def test_run(excerpts):
    """Generate for a small sample and print each Q/A beside its source excerpt."""
    state = {}
    batches = _pack_batches(excerpts)
    async with anthropic.AsyncAnthropic() as client:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        await asyncio.gather(*[_generate_batch(client, semaphore, b, state) for b in batches])
    for e in excerpts:
        r = state.get(str(e["doc_index"]))
        print("=" * 78)
        if r and r.get("category") and r["category"] != e["category"]:
            head = f"[{r['category']}]  (book said {e['category']})"
        else:
            head = f"[{e['category']}]"
        print(f"{head}  {e['year']}  prose {e['prose_score']:.2f}")
        print(f"  excerpt : {e['excerpt'][:200]}...")
        if r:
            print(f"  Q       : {r['q']}")
            print(f"  A       : {r['a']}")
        else:
            print("  (failed)")
        print()
