#!/usr/bin/env python3
"""Pass 1: Fix OCR artifacts in extracted Q&A pairs using DeepSeek via Anthropic SDK.

Set environment variables before running:
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
import sys
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "output" / "bleed"
OUTPUT_DIR = ROOT / "output" / "ocr"
STATE_FILE = ROOT / ".ocr_state.json"
MODEL = "claude-opus-4-8"  # maps to deepseek-v4-pro via ANTHROPIC_BASE_URL
MAX_TOKENS = 2048
CONCURRENCY = 20

SYSTEM_PROMPT = """\
You are an OCR correction assistant for public-domain texts published before 1930.
Your job is to fix mechanical scanning errors. Be aggressive about fixing — the source
texts are legitimate pre-1930s works, so fix confidently.

Fix all of the following:
- Missing or extra spaces (Whatis → What is, "Biography ?" → "Biography?")
- Single-character letter substitutions (ts → is, ln → in, rn → m, 4 → A when clearly a letter)
- Broken hyphenation across lines (na- tions → nations)
- Unicode/encoding artifacts (Â¢ → ¢, â€™ → ', Ã© → é)
- Obvious misrecognized words (tbe → the, liow → how, witb → with)
- Missing terminal period on an answer that is clearly a complete sentence
- Embedded page references or section headers in answer text (e.g. "FRANCE. 107", "6 EGYPT.", "244 COMMON SCHOOL QUESTION BOOK." — remove these entirely)

Never alter grammar or word choice. Fix only what is clearly a scanning artifact.
"Whom", archaic verb forms, and pre-1930s constructions are correct as-is.

Flag with "flagged": true if any word you chose as a replacement could be an anachronism —
a word, term, or concept that did not exist before 1930. Do not flag for space fixes,
punctuation, encoding artifacts, or clear single-character swaps.

Only discard if the text is so severely garbled that no meaningful content can be recovered.
When in doubt, fix and keep. Do not discard pairs that are merely messy.

Respond with valid JSON only:
  Corrected or unchanged:       {"q": "...", "a": "..."}
  Possible anachronism in fix:  {"q": "...", "a": "...", "flagged": true}
  Truly unrecoverable:          {"discard": true}

Examples:

Input — Q: Whatis History? / A: A recital of what has happened respecting na- tions.
Output: {"q": "What is History?", "a": "A recital of what has happened respecting nations."}

Input — Q: What is Biography ? / A: The history of a single individual.
Output: {"q": "What is Biography?", "a": "The history of a single individual."}

Input — Q: What are tbe chief sources ? / A: Records, Monuments, and I^egends.
Output: {"q": "What are the chief sources?", "a": "Records, Monuments, and Legends."}

Input — Q: §§§ xh§ pr¡n§¡p / A: Th§ §l¿ve §§ct¡¿n
Output: {"discard": true}\
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


async def clean_pair(client, semaphore, dataset, i, q, a, state, progress):
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
                if text.startswith("```"):
                    text = text.split("```", 2)[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.rstrip("`").strip()
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    state[key] = {"q": q, "a": a, "flagged": True}
                    break
                if parsed.get("discard"):
                    state[key] = {"discard": True}
                elif "q" not in parsed or "a" not in parsed:
                    state[key] = {"q": q, "a": a, "flagged": True}
                else:
                    state[key] = parsed
                break
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == 4:
                    state[key] = {"q": q, "a": a, "flagged": True}
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
        asyncio.create_task(clean_pair(client, semaphore, d, i, q, a, state, progress))
        for d, i, q, a in pending
    ]
    await asyncio.gather(*tasks)


def write_output(pairs, state):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for dataset, i, q, a in pairs:
        datasets.setdefault(dataset, []).append((i, q, a))

    total_written = total_flagged = total_discarded = 0
    for dataset, indexed_pairs in sorted(datasets.items()):
        items = []
        discarded = 0
        for i, orig_q, orig_a in sorted(indexed_pairs):
            result = state.get(f"{dataset}--{i}")
            if not result or result.get("discard"):
                discarded += 1
                continue
            item = {
                "conversations": [
                    {"role": "user", "content": result["q"]},
                    {"role": "assistant", "content": result["a"]},
                ]
            }
            if result["q"] != orig_q or result["a"] != orig_a:
                item["flagged"] = True
                total_flagged += 1
            items.append(item)

        out = OUTPUT_DIR / f"{dataset}.json"
        out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        total_written += len(items)
        total_discarded += discarded
        flagged = sum(1 for it in items if it.get("flagged"))
        print(f"  {dataset}: {len(items)} pairs ({flagged} flagged, {discarded} discarded)")

    print(f"\nTotal: {total_written} pairs, {total_flagged} flagged, {total_discarded} discarded")


async def test_run(pairs, seed, size):
    import random
    random.seed(seed)
    sample = random.sample(pairs, size)
    client = anthropic.AsyncAnthropic()
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / "ocr"
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_seed{seed}_n{size}.txt"
    header = [f"model: {MODEL}", f"seed:  {seed}", f"n:     {size}", "", "--- SYSTEM PROMPT ---", SYSTEM_PROMPT, "--- END SYSTEM PROMPT ---\n", ""]
    out_path.write_text("\n".join(header))

    async def process(dataset, i, q, a):
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Q: {q}\nA: {a}"}],
        )
        raw = next((b.text for b in msg.content if b.type == "text"), "").strip()
        try:
            parsed = json.loads(raw)
            if parsed.get("discard"):
                out_q, out_a = "[DISCARD]", ""
            else:
                out_q = parsed.get("q", "")
                out_a = parsed.get("a", "")
            flagged = " [FLAGGED]" if parsed.get("flagged") else ""
        except (json.JSONDecodeError, AttributeError):
            out_q, out_a, flagged = "[PARSE ERROR]", raw, ""
        entry = "\n".join([
            f"[{dataset}--{i}]{flagged}",
            f"  IN  Q: {q}",
            f"  IN  A: {a}",
            f"  OUT Q: {out_q}",
            f"  OUT A: {out_a}",
            "",
        ])
        with out_path.open("a") as f:
            f.write(entry + "\n")

    await asyncio.gather(*[process(d, i, q, a) for d, i, q, a in sample])
    print(f"Wrote {out_path}")


def main():
    test = "--test" in sys.argv
    seed_idx = next((i for i, a in enumerate(sys.argv) if a == "--seed"), None)
    seed = int(sys.argv[seed_idx + 1]) if seed_idx is not None else 42
    size_idx = next((i for i, a in enumerate(sys.argv) if a == "--size"), None)
    size = int(sys.argv[size_idx + 1]) if size_idx is not None else 10

    pairs = load_all_pairs()

    if test:
        asyncio.run(test_run(pairs, seed, size))
        return

    state = load_state()

    resolved = sum(1 for d, i, *_ in pairs if f"{d}--{i}" in state)
    pending = len(pairs) - resolved
    print(f"Total: {len(pairs)}  Resolved: {resolved}  Pending: {pending}")

    if pending:
        asyncio.run(run_async(pairs, state))
        save_state(state)

    print("Writing output...")
    write_output(pairs, state)
    print("Done.")


if __name__ == "__main__":
    main()
