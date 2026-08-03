"""N-gram anachronism tests for the synthetic pre-1930 SFT *questions*.

The ANSWERS are verbatim period text (anachronism-safe by construction), so the risk is
on the QUESTION side (model-composed). These tests flag questions carrying post-1930
vocabulary or phrasing. They are complementary to the vintage-model perplexity filter:
cheap, deterministic, and explainable (they point at the exact offending n-gram).

Three tests (run them together — they catch different things):

  1. modern_unigrams  — question contains a word coined after ~1930 (curated seed list;
                        extend MODERN_UNIGRAMS as test 3 surfaces more).
  2. modern_phrases   — question contains a post-1930 collocation that is period-innocent
                        word-by-word ("cold war", "world war", "machine learning").
  3. out_of_period    — question contains a lowercase word ABSENT from the period
                        vocabulary built from the dataset's own verbatim answers. Data-
                        driven: surfaces candidate anachronisms (also OCR/rare words) to
                        grow the blocklists above. This is the one you iterate on.

Meant to run AFTER character-corruption cleaning (OCR noise inflates test 3's flags).

Run:
    python3 -m synth.ngram_filter --route stem_reasoning --sample 5000
    python3 -m synth.ngram_filter                       # every route, full
"""

import argparse
import json
import random
import re
from collections import Counter

from synth import hf_push

ROUTES = (
    "knowledge_qa", "multiturn_qa", "reasoning_qa", "stem_reasoning",
    "narrative_grounded", "narrative_fiction", "opinion_qa", "how_to_qa",
    "verse_qa", "composition_qa",
)

# --- blocklists (SEED lists — deliberately conservative; extend as test 3 finds more) --

# Words coined after ~1930 (clearly modern). Kept high-precision: excludes words that
# existed pre-1930 even if their modern *sense* is later ("web", "relativity", "quantum",
# "photon", "isotope", "television", "radioactive" are all pre-1930 and were removed).
# Test 3 is how you discover more to add here.
MODERN_UNIGRAMS = {
    "internet", "website", "online", "offline", "email", "e-mail", "download", "upload",
    "wifi", "wi-fi", "bluetooth", "smartphone", "laptop", "software", "app", "apps",
    "blog", "blogger", "podcast", "hashtag", "selfie", "emoji", "google", "googled",
    "googling", "wikipedia", "youtube", "facebook", "twitter", "cyberspace",
    "cybersecurity", "malware", "chatbot", "livestream", "cellphone", "teenager",
    "teenagers", "microchip", "transistor", "cosmonaut", "photocopier", "supermarket",
    "genocide", "napalm", "antibiotic", "antibiotics", "neutron", "radar", "sonar",
    "laser", "spacecraft",
}

# Post-1930 collocations that are period-innocent token-by-token. Match on normalized
# (lowercased) text so spacing/case don't matter. Excludes pre-1930 phrases ("middle
# class", "working class", "human rights", "credit card", "assembly line").
MODERN_PHRASES = (
    "cold war", "world war", "first world war", "second world war", "world war one",
    "world war two", "machine learning", "artificial intelligence", "social media",
    "search engine", "climate change", "global warming", "greenhouse gas",
    "united nations", "european union", "space race", "nuclear weapon", "nuclear weapons",
    "atomic bomb", "hydrogen bomb", "gene therapy", "big bang", "black hole",
    "gross domestic product",
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _words(text):
    return _WORD.findall(text or "")


def _norm(text):
    return " ".join(w.lower() for w in _words(text))


# --- the three tests ----------------------------------------------------------------

def modern_unigrams(question):
    """Return the post-1930 words present in the question (sorted, deduped)."""
    return sorted({w.lower() for w in _words(question)} & MODERN_UNIGRAMS)


def modern_phrases(question):
    """Return the post-1930 collocations present in the question."""
    padded = f" {_norm(question)} "
    return [p for p in MODERN_PHRASES if f" {p} " in padded]


_COUNTS_CACHE = hf_push.ROOT / "synth" / "state" / "period_counts.json"


def period_counts(rebuild=False):
    """Word -> count over EVERY route's verbatim answers (the full period lexicon),
    cached to disk so it is built once. This global scope is essential: a per-route vocab
    is too narrow (stem's technical answers lack common prose words), inflating test 3."""
    if _COUNTS_CACHE.exists() and not rebuild:
        return Counter(json.loads(_COUNTS_CACHE.read_text()))
    c = Counter()
    for route in ROUTES:
        for row in _load_route(route):
            _, ans = _qa(row)
            for t in ans:
                for w in _words(t):
                    c[w.lower()] += 1
    _COUNTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _COUNTS_CACHE.write_text(json.dumps(dict(c)))
    return c


def build_period_vocab(min_count=3, rebuild=False):
    """The set of lowercase words appearing >= min_count times across ALL period answers.
    The frequency floor drops OCR noise and one-off proper nouns."""
    return {w for w, n in period_counts(rebuild).items() if n >= min_count}


def _clean_tok(w):
    wl = w.lower()
    if wl.endswith("'s"):
        wl = wl[:-2]
    return wl.strip("'-")


def out_of_period(question, vocab, min_len=3):
    """Return lowercase question words absent from the period vocab. Skips proper nouns
    (capitalized, non-sentence-initial), short words, and hyphen-compounds whose parts
    are all period words ("organ-systems") — those are legitimate, not anachronisms."""
    out = []
    for i, w in enumerate(_words(question)):
        if i > 0 and w[0].isupper():                 # likely proper noun -> skip
            continue
        wl = _clean_tok(w)
        if len(wl) < min_len or wl in vocab:
            continue
        parts = [p for p in wl.split("-") if p]
        if len(parts) > 1 and all(p in vocab for p in parts):
            continue                                 # compound of period words
        out.append(wl)
    return sorted(set(out))


# --- data loading -------------------------------------------------------------------

def _load_route(route):
    from huggingface_hub import hf_hub_download
    api = hf_push._api()
    shards = sorted(f for f in api.list_repo_files(hf_push.HF_REPO, repo_type="dataset")
                    if f.startswith(f"{route}/") and f.endswith(".jsonl"))
    rows = []
    for f in shards:
        path = hf_hub_download(hf_push.HF_REPO, f, repo_type="dataset")
        with open(path, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def _qa(row):
    """(question strings, period strings) for a row — handles both single-turn and
    multiturn shapes. User turns are the (composed) questions; assistant turns are the
    verbatim period text."""
    convs = row.get("conversations")
    if convs:
        q = [m["content"] for m in convs if m.get("role") == "user"]
        a = [m["content"] for m in convs if m.get("role") == "assistant"]
    else:
        q, a = [row.get("question", "")], [row.get("answer", "")]
    return q, a


# --- runner -------------------------------------------------------------------------

def run(routes, sample=0, seed=0, min_count=3, show=12, rebuild=False):
    """Run the three tests over the questions of `routes` (optionally a seeded sample),
    using the GLOBAL period vocab (all routes' answers, cached). Prints a report."""
    vocab = build_period_vocab(min_count, rebuild)
    questions = []                      # (route, doc_index, text)
    for r in routes:
        rows = _load_route(r)
        if sample and len(rows) > sample:
            rows = random.Random(seed).sample(rows, sample)
        for row in rows:
            qs, _ = _qa(row)
            for q in qs:
                questions.append((r, row.get("doc_index"), q))

    print(f"period vocab: {len(vocab):,} words (>= {min_count} uses, all routes)")
    print(f"questions tested: {len(questions):,} from {len(routes)} route(s)\n")

    hits = {"modern_unigrams": [], "modern_phrases": [], "out_of_period": []}
    oov_counter = Counter()
    for route, di, q in questions:
        u, p = modern_unigrams(q), modern_phrases(q)
        o = out_of_period(q, vocab)
        if u:
            hits["modern_unigrams"].append((route, di, q, u))
        if p:
            hits["modern_phrases"].append((route, di, q, p))
        if o:
            hits["out_of_period"].append((route, di, q, o))
            oov_counter.update(o)

    for name in ("modern_unigrams", "modern_phrases", "out_of_period"):
        rows = hits[name]
        pct = 100 * len(rows) / max(1, len(questions))
        print(f"=== {name}: {len(rows):,} flagged ({pct:.2f}%) ===")
        for route, di, q, terms in rows[:show]:
            print(f"  [{route} {di}] {terms}")
            print(f"      {q[:140]}")
        print()

    if oov_counter:
        print("top out-of-period tokens (candidates for MODERN_UNIGRAMS / review):")
        for w, n in oov_counter.most_common(40):
            print(f"  {n:>5}  {w}")
    return hits


def main():
    ap = argparse.ArgumentParser(description="n-gram anachronism tests over the SFT questions")
    ap.add_argument("--route", choices=ROUTES, help="single route (default: all)")
    ap.add_argument("--sample", type=int, default=0, help="test a seeded sample of N rows/route (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=3, help="period-vocab frequency floor")
    ap.add_argument("--rebuild-vocab", action="store_true", help="rebuild the cached global period vocab")
    args = ap.parse_args()
    run([args.route] if args.route else list(ROUTES),
        sample=args.sample, seed=args.seed, min_count=args.min_count, rebuild=args.rebuild_vocab)


if __name__ == "__main__":
    main()
