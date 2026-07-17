"""Sample the pretrain corpus (jbduran/think-dataset) to feel out excerpt sourcing.

We source from the *uncleaned* think-dataset rather than the -clean-1930s variant
because only the uncleaned repo ships per-document metadata (`audit_metadata.jsonl`)
aligned row-for-row with the parquet. The clean repo is text-only and re-sharded
through a shuffle, so its rows can't be joined back to a category. Anachronism
filtering is therefore ours to do downstream, at excerpt granularity.

Each parquet row is a whole document (a book; median ~80k words), single `text`
column. Metadata for document at (shard S, row r) is audit record
`offset(S) + r`, where offset(S) is the cumulative doc count of shards < S taken
from manifest.json. Key metadata fields:

  topic_or_subject_gen        LoC class, classifier-placed (the subject axis)
  topic_or_subject_score_gen  classifier confidence -- low = suspect label
  resolved_year, title_src, author_src, word_count

Reads remote parquet with DuckDB over HTTP range requests (no bulk download);
caches manifest.json + audit_metadata.jsonl under .cache/ (127 MB, one-time).

    python -m synth.corpus --docs 12 --excerpt-words 60,120,240

Requires: duckdb (pip install duckdb).
"""

import argparse
import json
import random
import re
import textwrap
import urllib.request
from pathlib import Path

import duckdb

# Small function-word set: running prose is ~35-45% stopwords; indexes, tables,
# and title pages fall far below that, which is one of the region-quality signals.
_STOP = set(
    "the of and to a in is that it for as was with his he be not by this had she "
    "at they or an which we you from are his her their its but have has one all "
    "were their when there been who will more no if out so up would about into".split()
)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["\'(]?[A-Z0-9])')

ROOT = Path(__file__).parent.parent
CACHE = ROOT / ".cache"
DATASET = "jbduran/think-dataset"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
SHARD_URL = BASE + "/shard_{:05d}.parquet"
N_SHARDS = 473

META_FIELDS = ("topic_or_subject_gen", "topic_or_subject_score_gen",
               "resolved_year", "title_src", "author_src", "word_count")


def _connect():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con


def _download(url, dest):
    CACHE.mkdir(exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  downloading {url.rsplit('/', 1)[-1]} -> {dest} ...", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def _offsets():
    """Return {shard_index: cumulative_doc_offset} and total doc count."""
    man = _download(BASE + "/manifest.json", CACHE / "manifest.json")
    shards = sorted(json.loads(man.read_text())["shards"], key=lambda s: s["index"])
    offs, cum = {}, 0
    for s in shards:
        offs[s["index"]] = cum
        cum += s["num_docs"]
    return offs, cum


def _audit():
    """Return a list of slim metadata dicts, one per document, in global order.

    Parses the 127 MB audit jsonl once, then caches a slim projection so later
    runs are fast. `shard` is carried so alignment can be self-checked.
    """
    slim = CACHE / "audit_slim.json"
    if slim.exists():
        return json.loads(slim.read_text())
    raw = _download(BASE + "/audit_metadata.jsonl", CACHE / "audit_metadata.jsonl")
    recs = []
    with open(raw, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec = {k: d.get(k) for k in META_FIELDS}
            rec["shard"] = d.get("output_shard_index")
            recs.append(rec)
    slim.write_text(json.dumps(recs))
    return recs


def _load_index():
    """Return (offsets, audit, con) with an alignment self-check."""
    offs, total = _offsets()
    audit = _audit()
    if len(audit) != total:
        print(f"  WARN: audit records {len(audit)} != manifest total {total}")
    # spot-check that offset(S) really lands on the first record of shard S
    for s in (1, N_SHARDS // 2):
        base = offs.get(s)
        if base is not None and base < len(audit) and audit[base]["shard"] != s:
            print(f"  WARN: alignment off at shard {s}: "
                  f"audit[{base}].shard = {audit[base]['shard']}")
    return offs, audit, _connect()


def _shard_starts():
    """Return ([(global_offset, shard_index, num_docs), ...], total) sorted by offset."""
    man = _download(BASE + "/manifest.json", CACHE / "manifest.json")
    shards = sorted(json.loads(man.read_text())["shards"], key=lambda s: s["index"])
    starts, cum = [], 0
    for s in shards:
        starts.append((cum, s["index"], s["num_docs"]))
        cum += s["num_docs"]
    return starts, cum


def _locate(global_index, starts):
    """Map a global document index to (shard_index, local_row)."""
    import bisect
    offsets = [s[0] for s in starts]
    i = bisect.bisect_right(offsets, global_index) - 1
    off, shard, _ = starts[i]
    return shard, global_index - off


def _category_index(audit):
    """Return {category: [global_index, ...]}, cached."""
    cache = CACHE / "category_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    idx = {}
    for gi, m in enumerate(audit):
        idx.setdefault(m.get("topic_or_subject_gen") or "UNKNOWN", []).append(gi)
    cache.write_text(json.dumps(idx))
    return idx


def _fetch_text(con, shard, local_row):
    row = con.execute(
        f"SELECT text FROM read_parquet('{SHARD_URL.format(shard)}') LIMIT 1 OFFSET {local_row}"
    ).fetchone()
    return row[0] if row else None


def sample_by_category(category, n_text=3, seed=0):
    """Return up to n_text (text, meta) pairs whose documents are in `category`."""
    _, audit, con = _load_index()
    idx = _category_index(audit)
    starts, _ = _shard_starts()
    gis = idx.get(category)
    if not gis:
        raise ValueError(f"unknown category {category!r}; try --list-categories")
    rng = random.Random(seed)
    docs = []
    for gi in rng.sample(gis, min(n_text, len(gis))):
        shard, local = _locate(gi, starts)
        docs.append((_fetch_text(con, shard, local), audit[gi]))
    return docs


def sample_documents(n, seed=0):
    """Return up to n (text, meta) pairs drawn from random shards.

    Takes the head rows of each chosen shard (cheap) and attaches the aligned
    metadata record; spreads across shards so the sample isn't one book cluster.
    """
    offs, audit, con = _load_index()
    rng = random.Random(seed)
    shards = rng.sample(range(N_SHARDS), min(N_SHARDS, max(1, n)))
    out, per = [], max(1, n // len(shards) + 1)
    for s in shards:
        rows = con.execute(
            f"SELECT text FROM read_parquet('{SHARD_URL.format(s)}') LIMIT {per}"
        ).fetchall()
        base = offs.get(s)
        for r, (t,) in enumerate(rows):
            meta = audit[base + r] if base is not None and base + r < len(audit) else {}
            out.append((t, meta))
            if len(out) >= n:
                return out
    return out


def _split_sentences(text):
    """Split into sentences on .!? boundaries (whitespace normalized). OCR makes
    this imperfect, but the region-quality score rejects the spans where it fails
    worst (indexes, tables, front matter have few real sentence boundaries)."""
    return [s for s in _SENT_SPLIT.split(re.sub(r"\s+", " ", text).strip()) if s]


def region_quality(text):
    """Model-free 0-1 prose score. High = running prose; low = index / table /
    title page / OCR sludge. Returns (score, detail)."""
    chars = [c for c in text if not c.isspace()]
    words = re.findall(r"[A-Za-z]+", text)
    if not chars or not words:
        return 0.0, {}
    alpha = sum(c.isalpha() for c in chars) / len(chars)      # prose ~.85+
    digit = sum(c.isdigit() for c in chars) / len(chars)      # prose ~0; tables high
    stop = sum(w.lower() in _STOP for w in words) / len(words) # prose ~.35+
    sents = _split_sentences(text)
    msl = sum(len(s.split()) for s in sents) / len(sents) if sents else 0.0  # prose 15-30
    clamp = lambda x: max(0.0, min(1.0, x))
    score = (clamp((alpha - 0.60) / 0.30)     # penalize symbol/digit-heavy spans
             + clamp((0.05 - digit) / 0.05)   # penalize numeric density
             + clamp((stop - 0.20) / 0.15)    # reward function-word density
             + clamp((msl - 8) / 12)) / 4     # reward complete sentences
    return score, {"alpha": alpha, "digit": digit, "stop": stop, "msl": msl}


def prose_excerpt(text, n_words=150, rng=None, tries=12, floor=0.6):
    """Cut a sentence-bounded excerpt of ~n_words that clears the prose bar.

    Builds candidates from runs of whole sentences (so no cut lands mid-sentence),
    scores each with region_quality, and returns the first above `floor` — or the
    best seen. Skips the first 10% of sentences (front matter). Returns (text, score)."""
    rng = rng or random
    sents = _split_sentences(text)
    if not sents:
        return "", 0.0
    wc = [len(s.split()) for s in sents]
    n = len(sents)
    lo = min(n // 10, n - 1)
    best = ("", -1.0)
    for _ in range(tries):
        i = rng.randint(lo, n - 1)
        j, words = i, 0
        while j < n and words < n_words:
            words += wc[j]
            j += 1
        ex = " ".join(sents[i:j])
        score = region_quality(ex)[0]
        if score > best[1]:
            best = (ex, score)
        if score >= floor:
            break
    return best


def _stats(values):
    v = sorted(values)
    def pct(q):
        return v[int(q * (len(v) - 1))]
    return min(v), pct(0.5), pct(0.9), max(v)


def report(n=10, seed=0, excerpt_words=(60, 120, 240), show=3):
    """Sample n documents, print length + category distribution, and preview
    excerpt windows for the first `show` documents."""
    from collections import Counter
    docs = sample_documents(n, seed=seed)
    rng = random.Random(seed)

    cc = [len(t) for t, _ in docs]
    ww = [len(t.split()) for t, _ in docs]
    print(f"\nsampled {len(docs)} documents from {DATASET}")
    print(f"  chars  min {_stats(cc)[0]:>8,}  p50 {_stats(cc)[1]:>8,}  "
          f"p90 {_stats(cc)[2]:>9,}  max {_stats(cc)[3]:>9,}")
    print(f"  words  min {_stats(ww)[0]:>8,}  p50 {_stats(ww)[1]:>8,}  "
          f"p90 {_stats(ww)[2]:>9,}  max {_stats(ww)[3]:>9,}")

    tally = Counter(m.get("topic_or_subject_gen", "?") for _, m in docs)
    print("  categories in sample:")
    for cat, k in tally.most_common():
        print(f"    {k:>3}  {cat}")

    for t, m in docs[:show]:
        print("\n" + "=" * 78)
        cat = m.get("topic_or_subject_gen", "?")
        score = m.get("topic_or_subject_score_gen")
        score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
        print(f"[{cat}]  conf {score_s}  {m.get('resolved_year','?')}  "
              f"{len(t.split()):,} words")
        print(f"  title: {str(m.get('title_src',''))[:90]}")
        for nw in excerpt_words:
            ex, score = prose_excerpt(t, nw, rng)
            print(f"\n--- excerpt ~{nw} words ({len(ex.split())} actual, prose {score:.2f}) ---")
            print(textwrap.fill(ex, width=78))


def list_categories():
    """Print the taxonomy with per-category document counts (metadata only)."""
    _, audit, _ = _load_index()
    idx = _category_index(audit)
    top = max(len(v) for v in idx.values())
    print(f"{len(idx)} categories, {sum(len(v) for v in idx.values()):,} documents:")
    for cat, gis in sorted(idx.items(), key=lambda kv: -len(kv[1])):
        bar = "#" * int(40 * len(gis) / top)
        print(f"  {len(gis):>7,}  {cat:44} {bar}")


def report_by_category(category, n_text=3, seed=0, excerpt_words=(120, 240)):
    """Profile a category from metadata (free), then show excerpts for a few docs."""
    _, audit, _ = _load_index()
    idx = _category_index(audit)
    gis = idx.get(category)
    if not gis:
        print(f"unknown category {category!r}; try --list-categories")
        return
    conf = [audit[g].get("topic_or_subject_score_gen") for g in gis]
    conf = [c for c in conf if isinstance(c, (int, float))]
    years = [audit[g].get("resolved_year") for g in gis]
    years = [y for y in years if isinstance(y, int)]
    wc = [audit[g].get("word_count") for g in gis]
    wc = [w for w in wc if isinstance(w, int)]

    print(f"\n=== {category} ===")
    print(f"  documents: {len(gis):,}")
    if conf:
        lo, p50, _, _ = _stats(conf)
        suspect = sum(c < 0.5 for c in conf)
        print(f"  label confidence: min {lo:.2f}  p50 {p50:.2f}  "
              f"below 0.5: {suspect} ({suspect/len(conf):.0%})")
    if years:
        print(f"  years: {_stats(years)[0]}–{_stats(years)[3]} (p50 {_stats(years)[1]})")
    if wc:
        s = _stats(wc)
        print(f"  word_count: p50 {s[1]:,}  p90 {s[2]:,}")

    rng = random.Random(seed)
    print("  sample titles:")
    for g in rng.sample(gis, min(8, len(gis))):
        m = audit[g]
        sc = m.get("topic_or_subject_score_gen")
        sc = f"{sc:.2f}" if isinstance(sc, (int, float)) else "?"
        print(f"    [{sc}] {m.get('resolved_year','?')}  {str(m.get('title_src',''))[:80]}")

    docs = sample_by_category(category, n_text=n_text, seed=seed)
    ewin = random.Random(seed)
    for t, m in docs:
        if not t:
            continue
        print("\n" + "-" * 78)
        print(f"[{m.get('title_src','')[:70]}]  {len(t.split()):,} words")
        for nw in excerpt_words:
            ex, score = prose_excerpt(t, nw, ewin)
            print(f"\n  --- excerpt ~{nw} words ({len(ex.split())} actual, prose {score:.2f}) ---")
            print(textwrap.fill(ex, width=76, initial_indent="  ", subsequent_indent="  "))


def main():
    ap = argparse.ArgumentParser(description="Sample the pretrain corpus for excerpt sizing")
    ap.add_argument("--docs", type=int, default=10, help="number of documents to sample")
    ap.add_argument("--category", type=str, default=None, help="sample within one LoC category")
    ap.add_argument("--list-categories", action="store_true", help="print taxonomy with counts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--excerpt-words", type=str, default="60,120,240",
                    help="comma-separated excerpt lengths to preview")
    ap.add_argument("--show", type=int, default=3, help="documents to show excerpt windows for")
    args = ap.parse_args()
    ewords = [int(x) for x in args.excerpt_words.split(",")]
    if args.list_categories:
        list_categories()
    elif args.category:
        report_by_category(args.category, n_text=args.show, seed=args.seed, excerpt_words=ewords)
    else:
        report(n=args.docs, seed=args.seed, excerpt_words=ewords, show=args.show)


if __name__ == "__main__":
    main()
