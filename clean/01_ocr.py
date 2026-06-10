#!/usr/bin/env python3
"""Pass 1: Fix OCR artifacts in extracted Q&A pairs using DeepSeek via Anthropic SDK.

Set environment variables before running:
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import asyncio
import json
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "output"
OUTPUT_DIR = ROOT / "output" / "ocr"
STATE_FILE = ROOT / ".ocr_state.json"
MODEL = "claude-opus-4-8"  # maps to deepseek-v4-pro via ANTHROPIC_BASE_URL
MAX_TOKENS = 512
CONCURRENCY = 20

SYSTEM_PROMPT = """\
You are an OCR correction assistant for public-domain texts published before 1930.
Fix only mechanical scanning errors within each field. Do not modernize language.

Fix:
- Missing or extra spaces (Whatis → What is, "Biography ?" → "Biography?")
- Single-character letter substitutions (ts → is, ln → in, 4 → A when clearly a letter)
- Broken hyphenation across lines (na- tions → nations)
- Unicode/encoding artifacts (Â¢ → ¢, â€™ → ', Ã© → é)

Never:
- Replace archaic or pre-1930s words with modern equivalents
- Move, reassign, or restructure content between Q and A fields
- Split a pair or merge pairs — one input pair always produces one output pair
- Attempt to fix structural issues: if Q appears to contain embedded answer text, leave both fields exactly as-is

If the pair is so severely garbled that meaningful correction is impossible, respond with:
  {"discard": true}

Flag with "flagged": true ONLY when you inferred a damaged word from context that could plausibly be wrong.
Do NOT flag for: no changes, clear single-char swaps, space/punctuation fixes, hyphen rejoins, Unicode fixes.

Respond with valid JSON only:
  Corrected or unchanged:  {"q": "...", "a": "..."}
  Inferred uncertain word: {"q": "...", "a": "...", "flagged": true}
  Unrecoverable:           {"discard": true}

Examples:

Input — Q: Whatis History? / A: A recital of what has happened respecting na- tions.
Output: {"q": "What is History?", "a": "A recital of what has happened respecting nations."}

Input — Q: What is Biography ? / A: The history of a single individual.
Output: {"q": "What is Biography?", "a": "The history of a single individual."}

Input — Q: What are the chief sources ? / A: Records, Monuments, and I^egends.
Output: {"q": "What are the chief sources?", "a": "Records, Monuments, and Legends.", "flagged": true}

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


def main():
    pairs = load_all_pairs()
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
