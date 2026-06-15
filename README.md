# sft-extractor

Builds a supervised fine-tuning (SFT) dataset from public-domain textbooks and catechisms published before 1930. All source texts are from Internet Archive.

## Pipeline

```
data/*.txt
  → run.py extract   → output/extracted/*.json
  → run.py bleed     → output/bleed/*.json
  → run.py ocr       → output/ocr/*.json
```

1. **Extract** — parse raw OCR text into Q&A pairs per dataset (regex-based, no model)
2. **Bleed** (pass 0) — detect and recover structural bleed via LLM; pairs where the Q field contains embedded answer text followed by more questions are split into constituent pairs (1→N), unrecoverable pairs are discarded
3. **OCR** (pass 1) — fix OCR scanning artifacts (broken hyphenation, character substitutions, encoding garbage, stray page headers) via LLM; pairs where any substitution could be an anachronism are flagged for review

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

Bleed output may include `"recovered": true` on pairs that were split from a bleed. OCR output may include `"flagged": true` on pairs where content was changed and could contain an anachronism.

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
# Full run (resumable)
python3 run.py bleed

# Test on a sample (writes to samples/bleed/)
python3 run.py bleed --test --seed 42 --size 10
```

### OCR pass

```bash
# Full run (resumable — reads from output/bleed/)
python3 run.py ocr

# Test on a sample (writes to samples/ocr/)
python3 run.py ocr --test --seed 42 --size 10
```

### Sample any stage (no model)

Randomly samples pairs from any pipeline stage directory and writes a readable text file to `samples/<stage>/`:

```bash
python3 run.py sample extracted --seed 42 --size 20
python3 run.py sample bleed    --seed 42 --size 20
python3 run.py sample ocr      --seed 42 --size 20
```

### Environment

Model passes (bleed, ocr) use DeepSeek via the Anthropic SDK compatibility layer:

```bash
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_API_KEY=<your deepseek key>
```

Both passes are resumable: progress is saved to `.bleed_state.json` / `.ocr_state.json` and re-running skips already-processed pairs.

## Datasets

27 source texts covering history, science, law, grammar, music, mythology, agriculture, engineering, and more.
