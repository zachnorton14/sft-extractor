"""Mine period-authentic deflection phrasings from the excerpt corpus.

The unparseable route needs answers, but no period book answers "xakhavjba" -- so
unlike every graded route, its answers cannot be lifted whole from prose. Authoring
them outright would smuggle in modern register, which is what the whole filter chain
exists to prevent. The compromise: mine the *fragments* verbatim from the corpus and
author only the assembly.

A fragment is reusable only if it is context-free. "I don't know what you mean" is;
"I don't know what you mean by load factor" is not. We keep a clause only when what
follows it is a terminator rather than a complement.

Yield is low by design -- roughly 30-40 distinct clauses across the full corpus --
so the generator composes them combinatorially rather than using them as whole answers.

    python -m synth.robustness.mine --scan-bytes 0     # full corpus
    python -m synth.robustness.mine --scan-bytes 150e6 # quick sample
"""

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCERPTS = ROOT / "synth" / "output" / "excerpts.jsonl"
POOL_FILE = Path(__file__).resolve().parent / "deflection_pool.json"

# Clauses that say "I did not understand you", with no complement attached.
DEFLECTION = re.compile(
    r"\b("
    r"I (?:do not|don't|cannot|can't|could not) (?:quite )?(?:understand|comprehend|follow) you"
    r"|I (?:do not|don't|cannot|can't) (?:quite )?take your meaning"
    r"|I (?:do not|don't) know what you mean"
    r"|I (?:do not|don't|cannot|can't) make (?:it|that|you) out"
    r"|I (?:cannot|can't|do not|don't) (?:quite )?catch your meaning"
    r"|I am at a loss"
    r"|I fail to (?:see|understand|perceive)"
    r"|I have no notion"
    r"|I beg your pardon"
    r"|I did not (?:quite )?(?:hear|catch) you"
    r"|I am no (?:scholar|judge)"
    r"|that is beyond me"
    r"|it is (?:all )?Greek to me"
    r"|I never heard the word"
    r"|there is no such word"
    r"|I (?:do not|don't) understand"
    r"|I (?:cannot|can't) comprehend"
    r")\b",
    re.IGNORECASE,
)

# Clauses that invite the visitor to try again -- the second half of a composed answer.
INVITATION = re.compile(
    r"\b("
    r"(?:Pray|Prithee),? speak plainly"
    r"|speak more plainly"
    r"|say it again"
    r"|say that again"
    r"|tell me plainly"
    r"|what do you mean"
    r"|how do you mean"
    r"|put it another way"
    r"|try me again"
    r"|come again"
    r"|explain yourself"
    r")\b",
    re.IGNORECASE,
)

# Reusable only when a terminator follows, not a complement ("by X", "that Y", ...).
CLEAN_TAIL = re.compile(r"^\s*[.!?,;\"']")

# Clauses the regex finds but that do not mean what we need. Frequency is no defence
# here: "come again" is the single most common invitation match precisely because the
# period sense is "return", not "repeat that" -- every sampled occurrence is
# astronomical recurrence or a parting pleasantry ("till I come again", "Venus would
# come again into conjunction"). Training on it would teach a phrase whose period
# meaning is wrong, which is the exact failure the corpus sourcing exists to prevent.
BLOCKLIST = {
    "come again",        # period sense is "return", not "repeat yourself"
    "explain yourself",  # a challenge in period usage; wrong tone for the persona
    "i am no scholar",   # means "not qualified to judge", not "did not understand you"
    "i am no judge",     # same
}


def _harvest(limit_bytes=0):
    """Stream excerpts, returning Counters of context-free deflection/invitation clauses."""
    if not EXCERPTS.exists():
        raise SystemExit(f"corpus not found: {EXCERPTS}")
    found = {"deflection": collections.Counter(), "invitation": collections.Counter()}
    patterns = (("deflection", DEFLECTION), ("invitation", INVITATION))
    read = rows = 0
    with open(EXCERPTS, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            read += len(line)
            if limit_bytes and read > limit_bytes:
                break
            rows += 1
            try:
                text = json.loads(line).get("excerpt") or ""
            except Exception:
                continue
            for kind, rx in patterns:
                for m in rx.finditer(text):
                    if not CLEAN_TAIL.match(text[m.end():m.end() + 3]):
                        continue # a complement follows -> context-bound
                    found[kind][" ".join(m.group(1).split()).lower()] += 1
    return found, read, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-bytes", type=float, default=0,
                    help="stop after N bytes (0 = whole corpus)")
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop clauses seen fewer than this many times")
    args = ap.parse_args()

    found, read, rows = _harvest(int(args.scan_bytes))
    pool = {
        kind: {c: n for c, n in sorted(counter.items(), key=lambda kv: -kv[1])
               if n >= args.min_count and c not in BLOCKLIST}
        for kind, counter in found.items()
    }
    POOL_FILE.write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")

    print(f"scanned {read/1e6:.0f} MB, {rows:,} excerpts")
    for kind in ("deflection", "invitation"):
        entries = pool[kind]
        print(f"\n{kind}: {len(entries)} distinct, {sum(entries.values()):,} occurrences")
        for clause, n in list(entries.items())[:15]:
            print(f"  {n:5,}  {clause}")
    print(f"\nwrote {POOL_FILE}")


if __name__ == "__main__":
    main()
