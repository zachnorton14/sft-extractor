"""Knowledge-QA route: turn expository excerpts into grounded Q/A rows.

The first content-bound generator. It sources gated excerpts from the pretrain
corpus (synth/corpus.py), keeps the ones the affordance gate tagged `expository`,
and for each one makes a single model call that distills the passage into a
focused question-answer pair:

  - the ANSWER is concise (1-3 sentences), self-contained, and fully supported by
    the passage — the "trim" of a rambling 150-word window down to something
    usable, done by the model because heuristics can't;
  - the QUESTION is in period-schoolbook register and written from *outside* the
    answer, so it doesn't just echo the answer's vocabulary (the lesson from the
    register experiments).

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
from pathlib import Path

import anthropic

from synth import corpus

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
You are given a short passage from a pre-1930s book. Do two things.

1. Write ONE question-and-answer pair that tests a single fact, definition, or
   explanation found in the passage.
   - The answer must be fully supported by the passage. Do not add any fact that
     is not stated or directly implied there.
   - The answer is a direct, concise response to the question — give only the new
     fact, as briefly as fully answering allows (often a phrase or a single
     sentence). Do NOT restate the question's words or setup; the answer is read
     together with the question, so it need not repeat it to stand on its own.
     (For "...what had filled Italy with horror?" answer "The exactions the Emperor
     had sanctioned and encouraged" — not "The exactions ... had filled all Italy
     with horror and hatred.")
   - Brevity means cutting restatement and padding, NOT substance. For a
     definitional or explanatory question (what is X, why, how), give the
     distinction, reason, or qualification that makes the answer informative — not
     a bare label. (For the meaning of díkaia: "That which is morally right, as
     opposed to merely formal or legal righteousness" — not just "moral".)
   - The question is in the plain register of a period schoolbook: direct, often
     beginning "What", "Why", "How", "Of what". No modern or conversational
     phrasing, no meta-language ("summarize", "explain to me").
   - The question MUST STAND ALONE. Never refer to the source — no "the passage",
     "the text", "according to the passage", "described above", "referred to",
     "mentioned". Ask about the subject directly, as if from general knowledge.
   - Make the question SPECIFIC. If the answer would be ambiguous without knowing a
     particular person, place, time, or situation, name that context in the
     question, drawn from the passage, so a reader knows exactly what is asked
     without seeing the passage. (e.g. not "What caused horror in Italy?" but "When
     Totila's Goths had taken Naples and were marching on Rome, what had filled
     Italy with horror?") Definitional questions need no added context.
   - Put context in the question, but keep the ANSWER out of it — do not state or
     restate the fact you are asking for, and do not reuse the answer's distinctive
     wording.

2. Classify the passage's ACTUAL subject — what it is about, NOT the kind of book
   it may come from — into exactly one of these classes (copy the label verbatim):
{_CLASS_LIST}

Input: JSON array [{{"i": 0, "text": "..."}}, ...]
Output JSON only: [{{"i": 0, "q": "the question", "a": "the answer", "category": "ONE CLASS"}}, ...]
"""


def source_excerpts(n, alpha=0.5, min_conf=0.7, n_words=150, seed=0):
    """Sample coverage-weighted excerpts and keep only the knowledge-QA route."""
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
                    q = (r.get("q") or "").strip()
                    a = (r.get("a") or "").strip()
                    cat = (r.get("category") or "").strip()
                    if isinstance(idx, int) and 0 <= idx < len(keys) and q and a:
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
