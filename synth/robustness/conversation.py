"""Conversation-register rows: what to say when the visitor is being social.

Every graded route is question-and-answer over period prose, so the model has never
been shown what "Hello" looks like, let alone what follows it. Off-distribution it
falls back on base-corpus continuation and rambles at a one-word greeting.

Note the asymmetry: the visitor is a person at a keyboard today, so the INPUT side
carries what they actually type -- "hi", "hey", "what's up", "ok". Only the ANSWER
needs to hold period register, and those clauses are mined verbatim from the corpus
(see mine.py) rather than authored, so the voice stays 1930 even when the prompt is
plainly not.

Surface variants (casing, dropped punctuation) are deliberately NOT baked in here --
the train-time noise transform in nanochat produces those, seeded per epoch, so the
same row can appear clean in one epoch and battered in the next.

    python -m synth.robustness.conversation --count 40 --preview
"""

import argparse
import collections
import json
import random
from pathlib import Path

POOL_FILE = Path(__file__).resolve().parent / "deflection_pool.json"
ROUTE = "conversation_qa"
SCORE = 100

# What a present-day visitor actually types. Modern forms are intentional.
INPUTS = {
    "greeting": ["Hello", "Hi", "Hey", "Hello there", "Hi there", "Good morning",
                 "Good afternoon", "Good evening", "Good day", "Greetings",
                 "Morning", "Evening", "Howdy", "Hey there", "Hello?"],
    "wellbeing": ["How are you", "How are you doing", "How do you do", "How goes it",
                  "How have you been", "Are you well", "How's it going",
                  "How do you fare", "You well"],
    "farewell": ["Goodbye", "Bye", "See you", "Good night", "Farewell", "I must go",
                 "I should be going", "Until next time", "Take care", "Bye for now"],
    "thanks": ["Thanks", "Thank you", "Thanks a lot", "Much appreciated", "Cheers",
               "Thank you kindly", "Ta"],
    "acknowledge": ["I see", "OK", "Okay", "Right", "Sure", "Got it", "Understood",
                    "Mm", "Ah", "Fair enough", "Makes sense"],
    "smalltalk": ["What's up", "What are you up to", "Tell me something",
                  "Let's talk", "Are you there", "Say something", "Talk to me",
                  "What shall we discuss", "Anything interesting"],
    "politeness": ["Please", "Pardon", "Excuse me", "Sorry", "If you please",
                   "Beg pardon", "My apologies"],
    "affirm": ["Yes", "No", "Maybe", "Perhaps", "Indeed", "Certainly", "Not really",
               "I suppose"],
}

# Light authored connective tissue, used only to open a turn back to the visitor.
# Kept deliberately plain so the mined clauses carry the period voice.
_OPENERS = ["What brings you", "What shall we talk of", "Sit, and tell me what you please",
            "What is on your mind", "And what of you", "Tell me what you please"]
_CLOSERS = ["Good day to you", "Until we meet again", "Keep well", "Safe home"]


def load_pool():
    if not POOL_FILE.exists():
        raise SystemExit(
            f"{POOL_FILE.name} missing -- run: python -m synth.robustness.mine --scan-bytes 0"
        )
    pool = json.loads(POOL_FILE.read_text(encoding="utf-8"))
    return {k: list(v) for k, v in pool.items()}


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def compose_answer(rng, cls, pool):
    """Assemble a reply for one input class from mined clauses plus a light join."""
    g = lambda k, fallback: pool.get(k) or fallback

    if cls == "greeting":
        reply = _cap(rng.choice(g("greeting", ["good day"])))
        if rng.random() < 0.6:
            return f"{reply}. {rng.choice(_OPENERS)}?"
        return f"{reply}."

    if cls == "wellbeing":
        state = _cap(rng.choice(g("wellbeing", ["well enough"])))
        if rng.random() < 0.5:
            return f"{state}, thank you. And you?"
        return f"{state}, thank you."

    if cls == "farewell":
        bye = _cap(rng.choice(g("farewell", ["good day"])))
        if rng.random() < 0.45:
            return f"{bye}. {rng.choice(_CLOSERS)}."
        return f"{bye}."

    if cls == "thanks":
        return _cap(rng.choice(g("thanks_reply", ["not at all"]))) + "."

    if cls == "acknowledge":
        ack = _cap(rng.choice(g("acknowledge", ["indeed"])))
        if rng.random() < 0.55:
            return f"{ack}. {rng.choice(_OPENERS)}?"
        return f"{ack}."

    if cls == "smalltalk":
        if rng.random() < 0.5:
            lead = _cap(rng.choice(g("pleased", ["glad to see you"])))
            return f"{lead}. {rng.choice(_OPENERS)}?"
        return f"{rng.choice(_OPENERS)}?"

    if cls == "politeness":
        return _cap(rng.choice(g("thanks_reply", ["not at all"]))) + f". {rng.choice(_OPENERS)}?"

    # affirm / deny -- acknowledge and hand the turn back
    ack = _cap(rng.choice(g("acknowledge", ["indeed"])))
    return f"{ack}. {rng.choice(_OPENERS)}?"


def build_rows(count, seed=1930):
    """Deterministic for a given (count, seed).

    The same greeting deliberately recurs with different replies: there is no single
    right answer to "Hello", and varying it is what stops the model parroting one
    phrase at every visitor. Deduplication is therefore on the (question, answer)
    pair, not the question alone.
    """
    pool = load_pool()
    classes = list(INPUTS)
    # Weight toward greetings and acknowledgements -- the most common real openers.
    weights = {"greeting": 26, "acknowledge": 16, "wellbeing": 14, "smalltalk": 12,
               "farewell": 12, "thanks": 8, "affirm": 7, "politeness": 5}
    w = [weights.get(c, 5) for c in classes]
    rng = random.Random(f"robustness:{ROUTE}:{seed}")
    rows, seen = [], set()
    attempts = 0
    while len(rows) < count and attempts < count * 60:
        attempts += 1
        cls = rng.choices(classes, weights=w, k=1)[0]
        question = rng.choice(INPUTS[cls])
        answer = compose_answer(rng, cls, pool)
        key = (question, answer)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "doc_index": f"robust-{ROUTE}-{len(rows):05d}",
            "category": ROUTE,
            "book_category": "ROBUSTNESS",
            "input_class": cls,
            "question": question,
            "answer": answer,
            "score": SCORE,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=750)
    ap.add_argument("--seed", type=int, default=1930)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    rows = build_rows(args.count, args.seed)
    if args.preview:
        for r in rows[:26]:
            print(f"  [{r['input_class']:12}] {r['question']!r}\n{'':18}-> {r['answer']}")
        dist = collections.Counter(r["input_class"] for r in rows)
        print(f"\n{len(rows)} rows; class distribution:")
        for k, v in dist.most_common():
            print(f"  {k:14} {v}")
        print(f"distinct answers: {len(set(r['answer'] for r in rows))}")
        print(f"distinct questions: {len(set(r['question'] for r in rows))}")
    else:
        print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
