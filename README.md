# sft-extractor

Builds a supervised fine-tuning (SFT) dataset from public-domain textbooks and catechisms published before 1930. All source texts are from Internet Archive.

## Pipeline

```
data/*.txt
  → run.py extract   → output/extracted/*.json   (19,284 pairs)
  → run.py bleed     → output/bleed/*.json        (19,473 pairs)
  → run.py ocr       → output/ocr/*.json          (18,382 pairs)
  → run.py pair      → output/paired/*.json       (13,157 conversations)
  → run.py enrich    → output/enriched/*.json     (12,264 conversations)
  → run.py filter    → output/filtered/*.json     (12,257 conversations)
```

1. **Extract** — parse raw OCR text into Q&A pairs per dataset (regex-based, no model)
2. **Bleed** (pass 0) — detect and recover structural bleed via LLM; pairs where the Q field contains embedded answer text followed by more questions are split into constituent pairs (1→N), unrecoverable pairs are discarded
3. **OCR** (pass 1) — fix OCR scanning artifacts (broken hyphenation, character substitutions, encoding garbage, stray page headers) via LLM; pairs where any substitution could be an anachronism are flagged for review
4. **Pair** (pass 2) — group dependent Q&A pairs into multi-turn conversations; a sliding window model detects whether each question depends on the prior pair (pronoun reference, demonstratives, quoted terms); chains are reconstructed algorithmically
5. **Enrich** (pass 3) — detect context-bare and domain-ambiguous opening questions; bare follow-ups are attached to their predecessor; ambiguous openers are rewritten with minimal domain context
6. **Filter** (pass 4) — rule-based removal of residual garbage: music notation OCR placeholders, fingering chart artifacts, and bleed remnants (7 pairs removed, 0.06%)

## Final dataset statistics

| Metric | Count |
|--------|-------|
| Total conversations | 12,257 |
| Single-turn | 9,103 |
| Multi-turn | 3,154 |
| Enriched (question rewritten) | 147 |
| Datasets | 27 |

**Multi-turn conversation length distribution:**

| Turns | Conversations |
|-------|--------------|
| 2 | 1,887 |
| 3 | 640 |
| 4 | 282 |
| 5 | 115 |
| 6–10 | 199 |
| 11+ | 31 |

## Dataset breakdown

| Dataset | Conversations | Multi-turn | Enriched |
|---------|--------------|------------|---------|
| 1001-questions | 592 | 240 | 6 |
| advanced-questions | 547 | 71 | 1 |
| agriculture | 213 | 57 | 7 |
| astronomy | 331 | 121 | 5 |
| botany | 136 | 51 | 1 |
| brewers-guide | 1,216 | 289 | 9 |
| chemistry | 712 | 177 | 9 |
| civil-war | 1,198 | 314 | 0 |
| common-core | 2,194 | 425 | 0 |
| constitution | 116 | 52 | 3 |
| electricity | 629 | 139 | 0 |
| engineering | 659 | 229 | 0 |
| ethics | 178 | 66 | 1 |
| familiar-things | 385 | 174 | 3 |
| grammar | 251 | 163 | 0 |
| investors | 400 | 2 | 0 |
| laborers | 110 | 73 | 2 |
| logic | 145 | 82 | 15 |
| music | 128 | 38 | 16 |
| mythology | 301 | 148 | 4 |
| new-york-bar | 620 | 1 | 2 |
| patriotism | 7 | 6 | 0 |
| school-bulletin | 352 | 43 | 10 |
| seeleys | 726 | 142 | 48 |
| stokers | 30 | 10 | 0 |
| symbological | 10 | 7 | 0 |
| world-history | 71 | 34 | 5 |

## Output format

Each file is a JSON array of conversation objects:

```json
[
  {
    "conversations": [
      { "role": "user",      "content": "What is the capital of France?" },
      { "role": "assistant", "content": "Paris." }
    ]
  }
]
```

Additional fields that may appear:
- `"chained": true` — multi-turn conversation (from pair or enrich pass)
- `"enriched": true` — opening question was rewritten for context
- `"recovered": true` — pair was split from a bleed
- `"flagged": true` — OCR correction may contain an anachronism

## Usage

All pipeline operations go through `run.py`:

```bash
python3 run.py <command> [options]
```

### Extract

```bash
python3 run.py extract <name> data/<file>.txt output/extracted/<file>.json
```

Available extractors: `advanced_questions`, `common_core`, `brewers_guide`, `familiar_things`, `1001_questions`, `logic`, `seeleys`, `stokers`, `symbological`, `agriculture`, `astronomy`, `botany`, `chemistry`, `civil_war`, `constitution`, `electricity`, `engineering`, `ethics`, `grammar`, `mythology`, `new_york_bar`, `patriotism`, `school_bulletin`, `investors`, `music`, `laborers`, `world_history`

### Bleed pass

```bash
python3 run.py bleed                        # full run (resumable)
python3 run.py bleed --test --count 10      # test sample
python3 run.py bleed --retry-bleed          # second pass on output/ocr/
```

### OCR pass

```bash
python3 run.py ocr                          # full run (resumable)
python3 run.py ocr --test --count 10        # test sample
python3 run.py ocr --retry-flagged          # reprocess flagged pairs
```

### Pair pass

```bash
python3 run.py pair                         # full run (resumable)
python3 run.py pair --test --count 10       # test sample
```

### Enrich pass

```bash
python3 run.py enrich                       # full run (resumable)
python3 run.py enrich --test --count 20     # test sample
```

### Filter pass

```bash
python3 run.py filter                       # run and write output/filtered/
python3 run.py filter --sample --count 20   # preview what would be dropped
```

### Sample any stage

```bash
python3 run.py sample filtered --count 20 --seed 42
```

Stages: `extracted`, `bleed`, `ocr`, `paired`, `enriched`, `filtered`

### Environment

Model passes (bleed, ocr, pair, enrich) use DeepSeek via the Anthropic SDK compatibility layer:

```bash
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_API_KEY=<your deepseek key>
```

All model passes are resumable — progress is saved to state files (`.bleed_state.json`, `.ocr_state.json`, `.pair_state.json`, `.enrich_state.json`) and re-running skips already-processed items.

## Domains

27 source texts covering: world history, American history, natural science, chemistry, astronomy, botany, agriculture, engineering, electricity, steam engineering, grammar, logic, ethics, mythology, music theory, investment law, labor economics, constitutional law, New York bar examination, American civics, biblical interpretation, and literature.
