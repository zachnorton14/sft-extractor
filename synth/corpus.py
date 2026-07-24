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
from collections import Counter
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
# The 20 Library of Congress main classes used as the subject taxonomy (the
# `topic_or_subject_gen` values, minus "UNKNOWN"). Book-level in the metadata;
# used here as the fixed label set for content-level classification too.
LOC_CLASSES = [
    "LANGUAGE AND LITERATURE",
    "PHILOSOPHY. PSYCHOLOGY. RELIGION",
    "LAW",
    "SCIENCE",
    "HISTORY OF THE AMERICAS",
    "SOCIAL SCIENCES",
    "AUXILIARY SCIENCES OF HISTORY",
    "AGRICULTURE",
    "POLITICAL SCIENCE",
    "EDUCATION",
    "TECHNOLOGY",
    "GEOGRAPHY. ANTHROPOLOGY. RECREATION",
    "FINE ARTS",
    "MEDICINE",
    "MUSIC AND BOOKS ON MUSIC",
    "WORLD HISTORY AND HISTORY OF EUROPE, ASIA, AFRICA, AUSTRALIA, NEW ZEALAND, ETC.",
    "NAVAL SCIENCE",
    "GENERAL WORKS",
    "MILITARY SCIENCE",
    "BIBLIOGRAPHY. LIBRARY SCIENCE. INFORMATION RESOURCES (GENERAL)",
]

DATASET = "jbduran/think-dataset"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
SHARD_URL = BASE + "/shard_{:05d}.parquet"
N_SHARDS = 473

META_FIELDS = ("topic_or_subject_gen", "topic_or_subject_score_gen",
               "resolved_year", "title_src", "author_src", "word_count")


def _connect():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar=false;")
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


EXCERPTS_FILE = ROOT / "synth" / "output" / "excerpts.jsonl"
HARVEST_STATE = ROOT / "synth" / "state" / "harvest.json"


def load_excerpts(affordance=None, min_prose=0.0):
    """Read the materialized excerpt corpus (one JSON object per line), optionally
    filtered to a single affordance / a prose floor. This is what every generation
    route consumes — sourced once by harvest(), never re-fetched per row."""
    if not EXCERPTS_FILE.exists():
        return []
    out = []
    for line in EXCERPTS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if affordance and r.get("affordance") != affordance:
            continue
        if r.get("prose_score", 0) < min_prose:
            continue
        out.append(r)
    return out


def relabel_excerpts():
    """Re-tag the affordance of every excerpt in EXCERPTS_FILE in place, using the
    current affordance_label — for when the routing rules change (a new affordance,
    a tuned gate) without re-harvesting. Rewrites the file and reports the delta."""
    if not EXCERPTS_FILE.exists():
        print("no excerpts file")
        return
    recs = [json.loads(l) for l in EXCERPTS_FILE.read_text().splitlines() if l.strip()]
    changed = Counter()
    for r in recs:
        new = affordance_label(r["excerpt"])
        if new != r.get("affordance"):
            changed[f"{r.get('affordance')} -> {new}"] += 1
            r["affordance"] = new
    with EXCERPTS_FILE.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"relabeled {sum(changed.values())}/{len(recs)} excerpts")
    for k, v in changed.most_common():
        print(f"  {k}: {v}")
    aff = Counter(r["affordance"] for r in recs)
    print("  by affordance: " + "  ".join(f"{a}={k}" for a, k in aff.most_common()))


def _load_harvest_state():
    if HARVEST_STATE.exists():
        s = json.loads(HARVEST_STATE.read_text())
        return set(s.get("processed", [])), Counter(s.get("counts", {}))
    return set(), Counter()


def _save_harvest_state(processed, counts):
    HARVEST_STATE.parent.mkdir(parents=True, exist_ok=True)
    HARVEST_STATE.write_text(json.dumps(
        {"processed": sorted(processed), "counts": dict(counts)}))


def harvest(total, alpha=0.5, min_conf=0.7, n_words=150, seed=0, max_shards=N_SHARDS):
    """Shard-major sweep: read each shard's text column ONCE and cut gated
    excerpts to fill the tempered coverage quotas, tagging every affordance.

    Far cheaper than per-row OFFSET fetches (one sequential scan per shard vs. one
    range read per excerpt) and resumable: processed shards and per-category counts
    persist in the harvest state, excerpts append to EXCERPTS_FILE as JSONL.
    """
    offs, audit, con = _load_index()
    starts, _ = _shard_starts()
    idx = _category_index(audit)
    elig = {c: len(g) for c, g in _eligible_by_category(audit, idx, min_conf).items()}
    quotas = _tempered_targets(elig, total, alpha)

    processed, got = _load_harvest_state()
    seen = {r["doc_index"] for r in load_excerpts()}  # dedup backstop
    EXCERPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    order = list(range(N_SHARDS))
    rng.shuffle(order)

    def _done():
        return all(got[c] >= quotas.get(c, 0) for c in quotas)

    swept = 0
    with open(EXCERPTS_FILE, "a", encoding="utf-8") as fh:
        for s in order:
            if _done() or swept >= max_shards:
                break
            if s in processed:
                continue
            base = offs.get(s)
            rows = con.execute(
                f"SELECT text FROM read_parquet('{SHARD_URL.format(s)}')"
            ).fetchall()
            new = 0
            for r, (t,) in enumerate(rows):
                gi = base + r
                if gi in seen:
                    continue
                m = audit[gi]
                cat = m.get("topic_or_subject_gen") or "UNKNOWN"
                if cat not in quotas or got[cat] >= quotas[cat]:
                    continue
                if (m.get("topic_or_subject_score_gen") or 0) < min_conf:
                    continue
                ex, score, label = prose_excerpt(t, n_words, rng)
                if not ex:
                    continue
                fh.write(json.dumps({
                    "doc_index": gi, "shard": s,
                    "category": cat, "affordance": label,
                    "confidence": m.get("topic_or_subject_score_gen"),
                    "year": m.get("resolved_year"), "title": m.get("title_src"),
                    "prose_score": round(score, 3), "n_words": len(ex.split()),
                    "excerpt": ex,
                }, ensure_ascii=False) + "\n")
                got[cat] += 1
                seen.add(gi)
                new += 1
            fh.flush()
            processed.add(s)
            swept += 1
            _save_harvest_state(processed, got)
            filled = sum(min(got[c], quotas[c]) for c in quotas)
            print(f"  shard {s:>3}: +{new:>3}  ({filled}/{sum(quotas.values())} quota, "
                  f"{swept} shards swept)", flush=True)

    total_have = sum(got.values())
    print(f"harvest: {total_have} excerpts across {len([c for c in got if got[c]])} "
          f"categories -> {EXCERPTS_FILE}")
    aff = Counter(r["affordance"] for r in load_excerpts())
    print("  by affordance: " + "  ".join(f"{a}={k}" for a, k in aff.most_common()))
    return total_have


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


_BARE_NUM = re.compile(r"^[\d.,;:\-—\s]+$")
_ROMAN = re.compile(r"^[ivxlcdm]{1,8}$", re.I)


def strip_lines(text):
    """Remove structural/boilerplate lines OCR leaves embedded in body text:
    bare page numbers, all-caps running headers (optionally with a page number),
    and footnote/citation refs. Conservative — only strong structural signals,
    so running prose is left intact. Run before sentence-splitting."""
    out = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            out.append(raw)
            continue
        letters = [c for c in s if c.isalpha()]
        words = s.split()
        drop = False
        if _BARE_NUM.match(s) or _ROMAN.match(s) or not letters:
            drop = True                                   # page number / all digits
        elif s[0] in "*†‡" and len(words) <= 12:
            drop = True                                   # footnote / citation ref
        elif len(words) <= 10 and len(s) <= 80 and \
                sum(c.isupper() for c in letters) / len(letters) > 0.7:
            drop = True                                   # all-caps running header
        if not drop:
            out.append(raw)
    return "\n".join(out)


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


# Opening words that signal the excerpt continues from text it doesn't contain:
# demonstratives, third-person pronouns, and continuation connectives. Existential
# "it/there/here" are excluded — they open self-contained sentences too often.
_REF_OPENERS = {
    "this", "that", "these", "those", "he", "she", "they", "him", "her", "his",
    "their", "them", "such", "thus", "hence", "therefore", "however", "moreover",
    "nevertheless", "besides", "furthermore", "accordingly", "consequently",
    "meanwhile", "likewise", "which", "who", "whom", "whose",
}
_DOT_LEADER = re.compile(r"\.{2,}|_{2,}")             # form blanks / TOC dot leaders
_ORDINAL = re.compile(r"\b(1st|2nd|3rd|\d+th|firstly|secondly|thirdly)\b", re.I)
_OPINION = re.compile(r"\b(pleasant\w*|best|worst|admirable|agreeable|beautiful|"
                      r"ought|should|prefer\w*|delightful|charming|excellent|finest|noble)\b", re.I)
_REASON = re.compile(r"\b(because|therefore|thus|hence|since|consequently|inasmuch)\b", re.I)
_SPEECH = re.compile(r'\b(said|asked|replied|cried|answered|exclaimed|'
                     r'demanded|told|spoke|quoth)\b', re.I)
_QUOTE = re.compile(r'["“”‘’]')

# Formal composed-document markers. An excerpt that IS such a document (statute,
# legal pleading, letter, oration/resolution, devotion, instrument) is feedstock for
# the generative route: the answer is the verbatim artifact, the question asks to
# compose one. Two tiers so ordinary prose that merely says "whereas" or "the said
# man" isn't swept in — a STRONG phrase qualifies alone; WEAK markers need three hits.
_DOC_STRONG = re.compile(
    r"\b(be it enacted|be it ordained|by the authority of the same|"
    r"in this present parliament|know all men by these presents|in witness whereof|"
    r"this indenture|given under my hand|resolved,? that|be it resolved|"
    r"your obedient servant|yours (faithfully|truly|sincerely|obediently)|"
    r"i have the honour|we beseech thee|vouchsafe)\b", re.I)
_DOC_WEAK = re.compile(
    r"\b(whereas|provided,? that|an act (to|for)|enacting|plaintiff|defendant|"
    r"aforesaid|hereinbefore|hereinafter|the said |to wit|whereof|indictment|"
    r"complainant|dear sir|my dear \w+|i beg to|i am, sir|mr\.? president|"
    r"fellow[- ]citizens|gentlemen of the|o lord|almighty god|thy servant)\b", re.I)


def is_composition(text):
    """True when the excerpt reads as a formal composed document — statute, pleading,
    letter, oration, devotion, or legal instrument — rather than narration or
    exposition. A strong, unambiguous phrase qualifies alone; otherwise three weak
    genre markers must fire, so prose that merely says 'whereas' isn't caught."""
    if _DOC_STRONG.search(text):
        return True
    return len(_DOC_WEAK.findall(text)) >= 3


def narrative_signal(text):
    """0-1 score for story/dialogue prose. Reported speech is the strong cue (past
    tense alone appears in historical argument too, so it's weighted lightly). High
    score = a scene with speakers and events, which has no general reasoning to
    extract and reads as context-bare in a standalone question.

    Quotation marks count as dialogue ONLY alongside a speech verb: scare-quotes on
    single terms fill expository prose too ("the child is a 'visualizer'"), and were
    pulling exposition into the narrative pool."""
    w = text.split()
    if len(w) < 20:
        return 0.0
    speech = len(_SPEECH.findall(text))
    quotes = len(_QUOTE.findall(text)) if speech else 0
    past = len(re.findall(r"\b\w+ed\b", text)) / len(w)
    dlg = (speech + quotes) / len(w)
    return min(1.0, dlg * 12 + past * 2)


_VOWEL = re.compile(r"[aeiouy]", re.I)


def _foreign_count(text):
    """Count characters from scripts that shouldn't appear in pre-1930 English
    OCR — CJK/kana/Hangul/fullwidth, plus replacement and zero-width chars. Their
    presence signals OCR mode-confusion. Latin-1 accents and typographic symbols
    (é, æ, °, £, §) are intentionally NOT counted; Greek is left alone too (it
    appears legitimately in period scientific/classical texts)."""
    n = 0
    for ch in text:
        o = ord(ch)
        if o == 0xFFFD or o in (0x200B, 0x200C, 0x200D, 0xFEFF):
            n += 1
        elif (0x3000 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF
              or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF
              or 0xFF00 <= o <= 0xFFEF):
            n += 1
    return n


def _gibberish_ratio(text):
    """Fraction of tokens that are OCR nonsense: digit-letter mashes ("97en") or
    length>=4 with no vowel (consonant runs)."""
    toks = re.findall(r"[A-Za-z0-9]+", text)
    if not toks:
        return 0.0
    bad = 0
    for t in toks:
        has_d = any(c.isdigit() for c in t)
        has_a = any(c.isalpha() for c in t)
        if (has_d and has_a) or (len(t) >= 4 and not _VOWEL.search(t)):
            bad += 1
    return bad / len(toks)


def is_garbage(text):
    """True for OCR-garbled spans that read as prose to the region score but are
    unusable: foreign-script contamination or heavy nonsense-token density."""
    return _foreign_count(text) >= 2 or _gibberish_ratio(text) > 0.10


def _first_word(text):
    m = re.match(r"\s*([A-Za-z]+)", text)
    return m.group(1).lower() if m else ""


def is_self_contained(text):
    """False if the excerpt opens on an unresolved reference (dangling pronoun,
    demonstrative, or continuation connective) — no question can be answered
    without the missing antecedent."""
    return _first_word(text) not in _REF_OPENERS


def has_affordance(text):
    """False for structurally dead spans — fill-in forms and dot-leader tables —
    that no question can be answered from."""
    return len(_DOT_LEADER.findall(text)) < 2


def affordance_label(text):
    """Coarse routing tag for a surviving excerpt (low-stakes: mislabels only
    misroute, they don't drop data)."""
    # Checked first: a formal document can carry past tense, quotes, and opinion
    # words that would otherwise misroute it to narrative/argument/opinion.
    if is_composition(text):
        return "composition"
    fp = len(re.findall(r"\b(i|we|my|our|me|us)\b", text, re.I))
    past = len(re.findall(r"\b\w+ed\b", text))
    if _ORDINAL.search(text) and past < 6:
        return "procedural"
    if fp >= 2 and _OPINION.search(text):
        return "opinion"
    # dialogue-heavy scenes are narrative, not argument — checked BEFORE `argument`
    # because narration is full of causal "because/since/for" that _REASON matches.
    if narrative_signal(text) >= 0.3:
        return "narrative"
    if _REASON.search(text):
        return "argument"
    if past >= 8 and re.search(r"\b1[5-9]\d\d\b", text):
        return "narrative"
    return "expository"


# STEM reasoning lives in windows that combine reasoning connectives with
# quantitative/physical vocabulary. Both are required (their product scores it):
# vocabulary alone is a description, connectives alone are rhetoric.
_REASON_CONN = re.compile(
    r"\b(therefore|hence|thus|since|because|consequently|whence|accordingly|"
    r"it follows|for this reason|so that|if|then|let|suppose|assume|given)\b", re.I)
_STEM_VOCAB = re.compile(
    r"\b(equal|equals|angle|triangle|square|circle|ratio|proportion|multiply|"
    r"divide|product|quotient|sum|difference|root|velocity|force|pressure|weight|"
    r"mass|volume|density|temperature|heat|energy|current|voltage|resistance|"
    r"equation|formula|theorem|proposition|proof|quantity|degrees?|per\s*cent)\b"
    r"|[=×÷]", re.I)


def stem_signal(text):
    """0-1 score for STEM-reasoning density: reasoning structure AND quantitative
    vocabulary, as a product so both must be present."""
    if len(text.split()) < 20:
        return 0.0
    conn = min(len(_REASON_CONN.findall(text)), 5) / 5
    vocab = min(len(_STEM_VOCAB.findall(text)), 8) / 8
    return conn * vocab


def prose_excerpt(text, n_words=150, rng=None, tries=16, floor=0.7, prefer="prose"):
    """Cut a sentence-bounded excerpt of ~n_words that clears the prose bar.

    Builds candidates from runs of whole sentences (so no cut lands mid-sentence),
    scores each with region_quality, and returns the first above `floor` — or the
    best seen. Skips the first 10% of sentences (front matter). Returns (text, score)."""
    rng = rng or random
    sents = _split_sentences(strip_lines(text))
    if not sents:
        return "", 0.0, ""
    wc = [len(s.split()) for s in sents]
    n = len(sents)
    lo = min(n // 10, n - 1)

    def window(i):
        j, words = i, 0
        while j < n and words < n_words:
            words += wc[j]
            j += 1
        return " ".join(sents[i:j])

    if prefer == "stem":
        # STEM reasoning is sparse (~few % of sentences), so random windows miss it.
        # Anchor the search on sentences carrying a reasoning connective, window
        # around each, and keep the gated window with the highest stem signal.
        anchors = [i for i in range(lo, n) if _REASON_CONN.search(sents[i])]
        best = ("", -1.0)
        for i in anchors[:200]:
            ex = window(max(lo, i - 1))
            if not is_self_contained(ex) or not has_affordance(ex) or is_garbage(ex):
                continue
            if region_quality(ex)[0] < 0.5:       # looser floor: math lowers region
                continue
            sig = stem_signal(ex)
            if sig > best[1]:
                best = (ex, sig)
        ex, sig = best
        if not ex or sig <= 0:
            return "", 0.0, ""
        return ex, round(sig, 3), "stem_reasoning"

    best = ("", -1.0)
    for _ in range(tries):
        i = rng.randint(lo, n - 1)
        ex = window(i)
        if not is_self_contained(ex) or not has_affordance(ex) or is_garbage(ex):
            continue                              # hard-drop: resample another window
        score = region_quality(ex)[0]
        if score > best[1]:
            best = (ex, score)
        if score >= floor:
            break
    ex, score = best
    if not ex:
        return "", 0.0, ""                        # no window survived the gate
    return ex, score, affordance_label(ex)


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
            ex, score, label = prose_excerpt(t, nw, rng)
            if not ex:
                print(f"\n--- excerpt ~{nw} words: (dropped — no self-contained window) ---")
                continue
            print(f"\n--- excerpt ~{nw} words ({len(ex.split())} actual, {label}, prose {score:.2f}) ---")
            print(textwrap.fill(ex, width=78))


def _tempered_targets(eligible, total, alpha):
    """target_c ∝ eligible_c ** alpha, normalized to `total`, capped at eligible_c."""
    W = sum(n ** alpha for n in eligible.values() if n > 0)
    return {c: (min(round(total * n ** alpha / W), n) if n > 0 else 0)
            for c, n in eligible.items()}


def _eligible_by_category(audit, idx, min_conf):
    """{category: [global_index, ...]} of docs whose label confidence >= min_conf."""
    return {
        c: [g for g in gis if (audit[g].get("topic_or_subject_score_gen") or 0) >= min_conf]
        for c, gis in idx.items() if c != "UNKNOWN"
    }


def plan_targets(total, alpha=0.5, min_conf=0.5):
    """Compute per-category excerpt quotas via a tempered distribution.

    alpha=1 mirrors the corpus skew, alpha=0 is uniform, alpha≈0.5 flattens skew
    while keeping big fields larger. Returns (targets, eligible_counts,
    corpus_counts, audit, index).
    """
    _, audit, _ = _load_index()
    idx = _category_index(audit)
    corpus = {c: len(g) for c, g in idx.items() if c != "UNKNOWN"}
    elig = {c: len(g) for c, g in _eligible_by_category(audit, idx, min_conf).items()}
    return _tempered_targets(elig, total, alpha), elig, corpus, audit, idx


def report_coverage(total, alpha=0.5, min_conf=0.5):
    """Print corpus share vs. target share so the rebalance is visible before
    any excerpt is fetched."""
    targets, eligible, corpus, _, _ = plan_targets(total, alpha, min_conf)
    corpus_total = sum(corpus.values())
    plan_total = sum(targets.values())
    print(f"coverage plan: total≈{plan_total} (asked {total}), alpha={alpha}, "
          f"min_conf={min_conf}")
    print(f"  {'category':44} {'corpus%':>8} {'target%':>8} {'target':>7} {'eligible':>9}")
    for c in sorted(targets, key=lambda c: -targets[c]):
        cap = "  (capped)" if targets[c] == eligible[c] and eligible[c] else ""
        print(f"  {c:44} {100*corpus[c]/corpus_total:7.1f}% "
              f"{100*targets[c]/max(plan_total,1):7.1f}% {targets[c]:7} {eligible[c]:9,}{cap}")


STEM_CATEGORIES = ["SCIENCE", "TECHNOLOGY"]


def sample_stem(n=20, seed=0, min_signal=0.4, min_conf=0.7, n_words=170,
                categories=STEM_CATEGORIES):
    """Draw ~n STEM-reasoning excerpts by seeking the reasoning-dense window in
    each doc from the STEM categories, keeping those above `min_signal`.

    Fetch-based for now (the reasoning window differs from the harvest's prose
    window, so it needs its own selection); materialize via the harvest later.
    """
    _, audit, con = _load_index()
    idx = _category_index(audit)
    starts, _ = _shard_starts()
    rng = random.Random(seed)
    per = max(1, n // len(categories))
    out = []
    for cat in categories:
        gis = [g for g in idx.get(cat, [])
               if (audit[g].get("topic_or_subject_score_gen") or 0) >= min_conf]
        if not gis:
            continue
        # oversample docs (many yield no strong stem window) and keep the hits
        for gi in rng.sample(gis, min(per * 4, len(gis))):
            if len([r for r in out if r["category"] == cat]) >= per:
                break
            shard, local = _locate(gi, starts)
            text = _fetch_text(con, shard, local)
            if not text:
                continue
            ex, sig, label = prose_excerpt(text, n_words, rng, prefer="stem")
            if not ex or sig < min_signal:
                continue
            m = audit[gi]
            out.append({
                "doc_index": gi, "category": cat, "affordance": "stem_reasoning",
                "stem_signal": sig, "confidence": m.get("topic_or_subject_score_gen"),
                "year": m.get("resolved_year"), "title": m.get("title_src"),
                "prose_score": sig, "n_words": len(ex.split()), "excerpt": ex,
            })
    return out


def sample_excerpts(n=20, alpha=0.5, min_conf=0.5, n_words=150, seed=0):
    """Draw ~n excerpts across categories weighted by the tempered coverage plan.

    Allocates n across categories via the same tempering as plan_targets, then
    pulls that many confidence-passing docs per category and cuts a prose excerpt
    from each. Returns a list of excerpt records. Loads the index once.
    """
    _, audit, con = _load_index()
    idx = _category_index(audit)
    starts, _ = _shard_starts()
    elig = _eligible_by_category(audit, idx, min_conf)
    targets = _tempered_targets({c: len(g) for c, g in elig.items()}, n, alpha)
    rng = random.Random(seed)
    out = []
    for cat in sorted(targets, key=lambda c: -targets[c]):
        k, gis = targets[cat], elig[cat]
        if k <= 0 or not gis:
            continue
        for gi in rng.sample(gis, min(k, len(gis))):
            shard, local = _locate(gi, starts)
            text = _fetch_text(con, shard, local)
            if not text:
                continue
            ex, score, label = prose_excerpt(text, n_words, rng)
            if not ex:
                continue                          # every window hit a hard-drop
            m = audit[gi]
            out.append({
                "category": cat,
                "affordance": label,
                "confidence": m.get("topic_or_subject_score_gen"),
                "year": m.get("resolved_year"),
                "title": m.get("title_src"),
                "doc_index": gi,
                "n_words": len(ex.split()),
                "prose_score": round(score, 3),
                "excerpt": ex,
            })
    return out


def report_excerpts(n=20, alpha=0.5, min_conf=0.5, n_words=150, seed=0):
    """Sample plan-weighted excerpts and print them for inspection."""
    from collections import Counter
    recs = sample_excerpts(n, alpha=alpha, min_conf=min_conf, n_words=n_words, seed=seed)
    tally = Counter(r["category"] for r in recs)
    aff = Counter(r["affordance"] for r in recs)
    print(f"sampled {len(recs)} excerpts  (alpha={alpha}, min_conf={min_conf}, "
          f"~{n_words} words each)")
    print("  by affordance: " + "  ".join(f"{a}={k}" for a, k in aff.most_common()))
    for cat, k in tally.most_common():
        print(f"    {k:>3}  {cat}")
    for r in recs:
        print("\n" + "=" * 78)
        print(f"[{r['category']} / {r['affordance']}]  conf {r['confidence']:.2f}  "
              f"{r['year']}  prose {r['prose_score']:.2f}  {r['n_words']}w")
        print(f"  {str(r['title'])[:88]}")
        print(textwrap.fill(r["excerpt"], width=78))


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
            ex, score, label = prose_excerpt(t, nw, ewin)
            if not ex:
                print(f"\n  --- excerpt ~{nw} words: (dropped — no self-contained window) ---")
                continue
            print(f"\n  --- excerpt ~{nw} words ({len(ex.split())} actual, {label}, prose {score:.2f}) ---")
            print(textwrap.fill(ex, width=76, initial_indent="  ", subsequent_indent="  "))


def main():
    ap = argparse.ArgumentParser(description="Sample the pretrain corpus for excerpt sizing")
    ap.add_argument("--docs", type=int, default=10, help="number of documents to sample")
    ap.add_argument("--category", type=str, default=None, help="sample within one LoC category")
    ap.add_argument("--list-categories", action="store_true", help="print taxonomy with counts")
    ap.add_argument("--coverage", action="store_true", help="print the coverage plan (targets)")
    ap.add_argument("--excerpts", action="store_true",
                    help="sample plan-weighted excerpts across categories and print them")
    ap.add_argument("--total", type=int, default=2000, help="excerpt budget for the coverage plan")
    ap.add_argument("--alpha", type=float, default=0.5, help="tempering: 1=corpus, 0=uniform")
    ap.add_argument("--min-conf", type=float, default=0.5, help="label-confidence floor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--excerpt-words", type=str, default="60,120,240",
                    help="comma-separated excerpt lengths to preview")
    ap.add_argument("--show", type=int, default=3, help="documents to show excerpt windows for")
    args = ap.parse_args()
    ewords = [int(x) for x in args.excerpt_words.split(",")]
    if args.list_categories:
        list_categories()
    elif args.coverage:
        report_coverage(args.total, alpha=args.alpha, min_conf=args.min_conf)
    elif args.excerpts:
        report_excerpts(n=args.docs, alpha=args.alpha, min_conf=args.min_conf,
                        n_words=ewords[0], seed=args.seed)
    elif args.category:
        report_by_category(args.category, n_text=args.show, seed=args.seed, excerpt_words=ewords)
    else:
        report(n=args.docs, seed=args.seed, excerpt_words=ewords, show=args.show)


if __name__ == "__main__":
    main()
