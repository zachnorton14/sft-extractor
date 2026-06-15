#!/usr/bin/env python3
"""Unified CLI for the sft-extractor pipeline.

Usage:
  python run.py extract <name> <input> <output>   Run a single extractor
  python run.py bleed [--test] [--seed N] [--size N]  Run/test bleed pass
  python run.py ocr   [--test] [--seed N] [--size N]  Run/test OCR pass
  python run.py sample <stage> [--seed N] [--size N]  Sample raw pairs from any stage

Stages: extracted | bleed | ocr

Set environment variables for model passes:
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent

from extractors.base import (
    AdvancedQuestionsExtractor,
    AgricultureExtractor,
    AstronomyExtractor,
    BotanyExtractor,
    BrewersGuideExtractor,
    ChemistryExtractor,
    CivilWarExtractor,
    CommonCoreExtractor,
    ConstitutionExtractor,
    ElectricityExtractor,
    EngineeringExtractor,
    EthicsExtractor,
    GrammarExtractor,
    FamiliarThingsExtractor,
    InvestorsExtractor,
    LaborersExtractor,
    LogicExtractor,
    MusicExtractor,
    MythologyExtractor,
    NewYorkBarExtractor,
    PatriotismExtractor,
    Questions1001Extractor,
    SchoolBulletinExtractor,
    SeeleysExtractor,
    StokersExtractor,
    SymbologicalExtractor,
    WorldHistoryExtractor,
)

EXTRACTORS = {
    "advanced_questions": AdvancedQuestionsExtractor,
    "common_core": CommonCoreExtractor,
    "brewers_guide": BrewersGuideExtractor,
    "familiar_things": FamiliarThingsExtractor,
    "1001_questions": Questions1001Extractor,
    "logic": LogicExtractor,
    "seeleys": SeeleysExtractor,
    "stokers": StokersExtractor,
    "symbological": SymbologicalExtractor,
    "agriculture": AgricultureExtractor,
    "astronomy": AstronomyExtractor,
    "botany": BotanyExtractor,
    "chemistry": ChemistryExtractor,
    "civil_war": CivilWarExtractor,
    "constitution": ConstitutionExtractor,
    "electricity": ElectricityExtractor,
    "engineering": EngineeringExtractor,
    "ethics": EthicsExtractor,
    "grammar": GrammarExtractor,
    "mythology": MythologyExtractor,
    "new_york_bar": NewYorkBarExtractor,
    "patriotism": PatriotismExtractor,
    "school_bulletin": SchoolBulletinExtractor,
    "investors": InvestorsExtractor,
    "music": MusicExtractor,
    "laborers": LaborersExtractor,
    "world_history": WorldHistoryExtractor,
}

OUTPUT_DIRS = {
    "extracted": ROOT / "output" / "extracted",
    "bleed":     ROOT / "output" / "bleed",
    "ocr":       ROOT / "output" / "ocr",
}


def cmd_extract(args):
    if args.name not in EXTRACTORS:
        print(f"Unknown extractor: {args.name}. Available: {', '.join(EXTRACTORS)}")
        sys.exit(1)
    EXTRACTORS[args.name](args.input, args.output).run()


def cmd_bleed(args):
    from clean import bleed
    pairs = bleed.load_all_pairs()
    if args.test:
        asyncio.run(bleed.test_run(pairs, args.seed, args.size))
        return
    state = bleed.load_state()
    resolved = sum(1 for d, i, *_ in pairs if f"{d}--{i}" in state)
    pending = len(pairs) - resolved
    print(f"Total: {len(pairs)}  Resolved: {resolved}  Pending: {pending}")
    if pending:
        asyncio.run(bleed.run_async(pairs, state))
        bleed.save_state(state)
    print("Writing output...")
    bleed.write_output(pairs, state)
    print("Done.")


def cmd_ocr(args):
    from clean import ocr
    pairs = ocr.load_all_pairs()
    if args.test:
        asyncio.run(ocr.test_run(pairs, args.seed, args.size))
        return
    state = ocr.load_state()
    resolved = sum(1 for d, i, *_ in pairs if f"{d}--{i}" in state)
    pending = len(pairs) - resolved
    print(f"Total: {len(pairs)}  Resolved: {resolved}  Pending: {pending}")
    if pending:
        asyncio.run(ocr.run_async(pairs, state))
        ocr.save_state(state)
    print("Writing output...")
    ocr.write_output(pairs, state)
    print("Done.")


def cmd_sample(args):
    directory = OUTPUT_DIRS[args.stage]
    if not directory.exists():
        print(f"Directory not found: {directory}. Has this stage been run yet?")
        sys.exit(1)

    pairs = []
    for json_file in sorted(directory.glob("*.json")):
        dataset = json_file.stem
        items = json.loads(json_file.read_text())
        for i, item in enumerate(items):
            convs = item["conversations"]
            q = next(c["content"] for c in convs if c["role"] == "user")
            a = next(c["content"] for c in convs if c["role"] == "assistant")
            pairs.append((dataset, i, q, a))

    random.seed(args.seed)
    sample = random.sample(pairs, min(args.size, len(pairs)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / args.stage
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_seed{args.seed}_n{args.size}.txt"

    lines = [f"stage: {args.stage}", f"seed:  {args.seed}", f"n:     {args.size}", ""]
    for dataset, i, q, a in sample:
        lines += [f"[{dataset}--{i}]", f"  Q: {q}", f"  A: {a}", ""]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="sft-extractor pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract
    p_ext = sub.add_parser("extract", help="Run an extractor")
    p_ext.add_argument("name", choices=sorted(EXTRACTORS), metavar="name",
                       help=f"Extractor name. Choices: {', '.join(sorted(EXTRACTORS))}")
    p_ext.add_argument("input", help="Path to input .txt file")
    p_ext.add_argument("output", help="Path to output .json file")

    # bleed
    p_bleed = sub.add_parser("bleed", help="Run bleed detection/recovery pass (pass 0)")
    p_bleed.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_bleed.add_argument("--seed", type=int, default=42)
    p_bleed.add_argument("--size", type=int, default=10)

    # ocr
    p_ocr = sub.add_parser("ocr", help="Run OCR correction pass (pass 1)")
    p_ocr.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_ocr.add_argument("--seed", type=int, default=42)
    p_ocr.add_argument("--size", type=int, default=10)

    # sample
    p_sample = sub.add_parser("sample", help="Sample raw pairs from any pipeline stage")
    p_sample.add_argument("stage", choices=list(OUTPUT_DIRS))
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--size", type=int, default=10)

    args = parser.parse_args()

    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "bleed":
        cmd_bleed(args)
    elif args.cmd == "ocr":
        cmd_ocr(args)
    elif args.cmd == "sample":
        cmd_sample(args)


if __name__ == "__main__":
    main()
