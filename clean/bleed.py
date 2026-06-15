"""Pass 0: Detect and recover Q/A bleed pairs using DeepSeek via Anthropic SDK.

Bleed occurs when the extractor incorrectly merges multiple Q/A pairs into one.
This pass detects bleeds and recovers constituent pairs (1→N), passes clean pairs
through unchanged, and discards unrecoverable pairs.

Set environment variables before running:
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "output" / "extracted"
OUTPUT_DIR = ROOT / "output" / "bleed"
STATE_FILE = ROOT / ".bleed_state.json"
MODEL = "claude-opus-4-8"  # maps to deepseek-v4-pro via ANTHROPIC_BASE_URL
MAX_TOKENS = 1024
CONCURRENCY = 20

SYSTEM_PROMPT = """\
You are a dataset quality filter for public-domain Q&A pairs extracted from pre-1930s texts.
Your job is to detect structural bleed — where an OCR extractor incorrectly merged multiple
Q&A pairs into one entry.

A bleed occurs when the Q field contains answer text followed by another question. The key
signal is a pattern of: question → answer prose → question → answer prose → question.

Bleed indicators in the Q field:
- Answer text (declarative sentences that respond to the preceding question)
- Numbered sub-questions mid-field, including OCR-corrupted numbers:
  "go." = 90., "gt." = 91., "gi." = 91., "8o." = 80., "ioo." = 100., etc.

The A field from the input always corresponds to the LAST question in the bleed.
Only include recovered pairs where both Q and A are complete and meaningful.

Do not fix OCR artifacts — preserve all text exactly as given, including corrupted numbers.
Respond with valid JSON only.

If the pair is CLEAN, respond with:
  {"clean": true}

If the pair has bleed and you can recover distinct Q&A pairs, respond with:
  {"pairs": [{"q": "...", "a": "..."}, ...]}

If no clean Q&A can be recovered, respond with:
  {"discard": true}

Examples:

Input — Q: What is a measure? That by which extent is ascertained. go. How many dimensions has extension? Extension has three. gt. Explain how distance is measured by time?
         A: Every circle is divided into 360 degrees.
Output: {"pairs": [{"q": "What is a measure?", "a": "That by which extent is ascertained."}, {"q": "How many dimensions has extension?", "a": "Extension has three."}, {"q": "Explain how distance is measured by time?", "a": "Every circle is divided into 360 degrees."}]}

Input — Q: What is History?
         A: A record of past events.
Output: {"clean": true}

Input — Q: §§§ xh§ merged §§§ unreadable §§§
         A: §§§ garbage
Output: {"discard": true}
"""


def load_all_pairs():
    pairs = []
    for json_file in sorted(INPUT_DIR.glob("*.json")):
        dataset = json_file.stem
        items = json.loads(json_file.read_text())
        for i, item in enumerate(items):
            convs = item["conversations"]
            q = next(c["content"] for c in convs if c["role"] == "user")
            a = next(c["content"] for c in convs if c["role"] == "assistant")
            pairs.append((dataset, i, q, a))
    return pairs


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


def parse_response(text, q, a):
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"clean": True}
    if parsed.get("discard"):
        return {"discard": True}
    if parsed.get("clean"):
        return {"clean": True}
    if "pairs" in parsed and isinstance(parsed["pairs"], list):
        valid = [p for p in parsed["pairs"] if p.get("q") and p.get("a")]
        return {"pairs": valid} if valid else {"discard": True}
    return {"clean": True}


async def process_pair(client, semaphore, dataset, i, q, a, state, progress):
    key = f"{dataset}--{i}"
    async with semaphore:
        for attempt in range(5):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": f"Q: {q}\nA: {a}"}],
                )
                text = next((b.text for b in msg.content if b.type == "text"), "").strip()
                state[key] = parse_response(text, q, a)
                break
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    state[key] = {"clean": True}
                else:
                    await asyncio.sleep(2 ** attempt)

    progress[0] += 1
    if progress[0] % 100 == 0:
        save_state(state)
        print(f"  {progress[0]}/{progress[1]}", flush=True)


async def run_async(pairs, state):
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    pending = [(d, i, q, a) for d, i, q, a in pairs if f"{d}--{i}" not in state]
    progress = [0, len(pairs)]
    tasks = [
        asyncio.create_task(process_pair(client, semaphore, d, i, q, a, state, progress))
        for d, i, q, a in pending
    ]
    await asyncio.gather(*tasks)


def write_output(pairs, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for dataset, i, q, a in pairs:
        datasets.setdefault(dataset, []).append((i, q, a))

    total_written = total_recovered = total_discarded = 0
    for dataset, indexed_pairs in sorted(datasets.items()):
        items = []
        discarded = recovered = 0
        for i, orig_q, orig_a in sorted(indexed_pairs):
            result = state.get(f"{dataset}--{i}")
            if not result or result.get("discard"):
                discarded += 1
                continue
            if result.get("clean"):
                items.append({"conversations": [
                    {"role": "user", "content": orig_q},
                    {"role": "assistant", "content": orig_a},
                ]})
            elif "pairs" in result:
                for pair in result["pairs"]:
                    items.append({"conversations": [
                        {"role": "user", "content": pair["q"]},
                        {"role": "assistant", "content": pair["a"]},
                    ], "recovered": True})
                recovered += 1

        out = OUTPUT_DIR / f"{dataset}.json"
        out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        total_written += len(items)
        total_recovered += recovered
        total_discarded += discarded
        print(f"  {dataset}: {len(items)} pairs ({recovered} bleeds recovered, {discarded} discarded)")

    print(f"\nTotal: {total_written} pairs, {total_recovered} bleeds recovered, {total_discarded} discarded")


async def test_run(pairs, seed, size):
    import random
    random.seed(seed)
    sample = random.sample(pairs, size)
    client = anthropic.AsyncAnthropic()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / "bleed"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_seed{seed}_n{size}.txt"
    header = [f"model: {MODEL}", f"seed:  {seed}", f"n:     {size}", "", "--- SYSTEM PROMPT ---", SYSTEM_PROMPT, "--- END SYSTEM PROMPT ---", ""]
    out_path.write_text("\n".join(header))

    async def process(dataset, i, q, a):
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Q: {q}\nA: {a}"}],
        )
        text = next((b.text for b in msg.content if b.type == "text"), "").strip()
        result = parse_response(text, q, a)

        if result.get("clean"):
            status, detail = "CLEAN", ""
        elif result.get("discard"):
            status, detail = "DISCARD", ""
        else:
            pairs_out = result.get("pairs", [])
            status = f"RECOVERED ({len(pairs_out)} pairs)"
            detail = "\n".join(
                f"  PAIR {j+1} Q: {p['q']}\n  PAIR {j+1} A: {p['a']}"
                for j, p in enumerate(pairs_out)
            )

        lines = [f"[{dataset}--{i}] {status}", f"  IN Q: {q}", f"  IN A: {a}"]
        if detail:
            lines.append(detail)
        lines.append("")

        with out_path.open("a") as f:
            f.write("\n".join(lines) + "\n")

    await asyncio.gather(*[process(d, i, q, a) for d, i, q, a in sample])
    print(f"Wrote {out_path}")
