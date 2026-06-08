# sft-extractor

Builds a supervised fine-tuning (SFT) dataset from public-domain textbooks and catechisms published before 1930. All source texts are from Internet Archive.

## Pipeline

```
data/*.txt  →  extract.py  →  output/*.json  →  clean.py  →  output/cleaned/*.json  →  (combine)  →  output/conversations/*.json
```

1. **Extract** — parse raw OCR text into Q&A pairs per dataset
2. **Clean** — fix OCR artifacts and encoding issues via Anthropic Batches API; omit unrecoverable pairs
3. **Combine** *(planned)* — merge sequential questions that rely on each other into multi-turn conversations

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

## Usage

### Extract

```bash
python extract.py <extractor> data/<file>.txt output/<file>.json
```

Available extractors: `common_core`, `brewers_guide`, `familiar_things`, `1001_questions`, `logic`, `seeleys`, `stokers`, `symbological`, `agriculture`, `astronomy`, `botany`, `chemistry`, `civil_war`, `constitution`, `electricity`, `engineering`, `ethics`, `grammar`, `mythology`, `new_york_bar`, `patriotism`, `school_bulletin`, `investors`, `music`, `laborers`, `world_history`

### Clean

Requires `ANTHROPIC_API_KEY`. Submits all `output/*.json` to the Anthropic Batches API (Haiku, 50% discount) and writes cleaned files to `output/cleaned/`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python clean.py
```

If interrupted, re-running resumes the in-progress batch automatically.

## Datasets

26 source texts covering history, science, law, grammar, music, mythology, agriculture, engineering, and more.
