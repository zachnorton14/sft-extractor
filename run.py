#!/usr/bin/env python3
"""Unified CLI for the sft-extractor pipeline.

Usage:
  python run.py extract <name> <input> <output>   Run a single extractor
  python run.py bleed [--test] [--seed N] [--count N]  Run/test bleed pass
  python run.py ocr   [--test] [--seed N] [--count N]  Run/test OCR pass
  python run.py pair  [--test] [--seed N] [--count N]  Run/test pair pass
  python run.py sample <stage> [--seed N] [--count N]  Sample from any stage

Stages: extracted | bleed | ocr | paired

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
    "paired":    ROOT / "output" / "paired",
    "enriched":  ROOT / "output" / "enriched",
    "filtered":  ROOT / "output" / "filtered",
    "scored":    ROOT / "output" / "scored",
}


def cmd_extract(args):
    if args.name not in EXTRACTORS:
        print(f"Unknown extractor: {args.name}. Available: {', '.join(EXTRACTORS)}")
        sys.exit(1)
    EXTRACTORS[args.name](args.input, args.output).run()


def cmd_bleed(args):
    from clean import bleed
    if args.retry_bleed:
        pairs = bleed.load_ocr_pairs()
        state = bleed.load_retry_state()
        asyncio.run(bleed.retry_bleed(pairs, state))
        print("Writing output...")
        bleed.write_retry_output(state)
        print("Done. Re-run `pair` to rebuild multi-turn conversations.")
        return
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
    if args.test and args.retry_flagged:
        state = ocr.load_state()
        asyncio.run(ocr.test_retry_flagged(pairs, state, args.seed, args.size))
        return
    if args.test:
        asyncio.run(ocr.test_run(pairs, args.seed, args.size))
        return
    state = ocr.load_state()

    if args.retry_flagged:
        asyncio.run(ocr.retry_flagged(pairs, state))
        print("Writing output...")
        ocr.write_output(pairs, state)
        print("Done.")
        return

    resolved = sum(1 for d, i, *_ in pairs if f"{d}--{i}" in state)
    pending = len(pairs) - resolved
    print(f"Total: {len(pairs)}  Resolved: {resolved}  Pending: {pending}")
    if pending:
        asyncio.run(ocr.run_async(pairs, state))
        ocr.save_state(state)
    print("Writing output...")
    ocr.write_output(pairs, state)
    print("Done.")


def cmd_filter(args):
    from clean import filter as f
    if args.sample:
        f.sample_dropped(n=args.size, seed=args.seed)
        return
    f.run()


def cmd_score(args):
    from clean import score
    datasets = score.load_all_conversations()
    if args.test:
        asyncio.run(score.test_run(datasets, args.seed, args.size))
        return
    state = score.load_state()
    resolved = sum(1 for d, items in datasets.items() for i in range(len(items))
                   if f"{d}--{i}" in state)
    total = sum(len(items) for items in datasets.values())
    pending = total - resolved
    print(f"Total: {total}  Resolved: {resolved}  Pending: {pending}")
    if pending:
        asyncio.run(score.run_async(datasets, state))
        score.save_state(state)
    print("Writing output...")
    score.write_output(datasets, state, filter_pct=args.filter_pct)
    print("Done.")


def cmd_enrich(args):
    from clean import enrich
    datasets = enrich.load_all_conversations()
    if args.test:
        asyncio.run(enrich.test_run(datasets, args.seed, args.size))
        return
    state = enrich.load_state()
    asyncio.run(enrich.run_async(datasets, state))
    print("Writing output...")
    enrich.write_output(datasets, state)
    print("Done.")


def cmd_pair(args):
    from clean import pair
    datasets = pair.load_all_pairs()
    if args.test:
        asyncio.run(pair.test_run(datasets, args.seed, args.size))
        return
    state = pair.load_state()
    all_keys = {f"{d}--w{w}" for d, pairs in datasets.items()
                for w in range(len(pair.make_windows(pairs)))}
    resolved = sum(1 for k in all_keys if k in state)
    pending = len(all_keys) - resolved
    print(f"Total windows: {len(all_keys)}  Resolved: {resolved}  Pending: {pending}")
    if pending:
        asyncio.run(pair.run_async(datasets, state))
        pair.save_state(state)
    print("Writing output...")
    pair.write_output(datasets, state)
    print("Done.")





def cmd_sample(args):
    directory = OUTPUT_DIRS[args.stage]
    if not directory.exists():
        print(f"Directory not found: {directory}. Has this stage been run yet?")
        sys.exit(1)

    convs_list = []
    for json_file in sorted(directory.glob("*.json")):
        dataset = json_file.stem
        items = json.loads(json_file.read_text())
        for i, item in enumerate(items):
            convs_list.append((dataset, i, item["conversations"], item.get("chained", False)))

    random.seed(args.seed)
    sample = random.sample(convs_list, min(args.size, len(convs_list)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "samples" / args.stage
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_seed{args.seed}_n{args.size}.txt"

    lines = [f"stage: {args.stage}", f"seed:  {args.seed}", f"n:     {args.size}", ""]
    for dataset, i, convs, chained in sample:
        label = f"[{dataset}--{i}]" + (" [multi-turn]" if chained else "")
        lines.append(label)
        turns = [(c["role"], c["content"]) for c in convs]
        for role, content in turns:
            prefix = "  Q:" if role == "user" else "  A:"
            lines.append(f"{prefix} {content}")
        lines.append("")
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
    p_bleed.add_argument("--count", type=int, default=10, dest="size", metavar="N")
    p_bleed.add_argument("--retry-bleed", action="store_true",
                         help="Re-run bleed detection on output/ocr/ and update in place")

    # ocr
    p_ocr = sub.add_parser("ocr", help="Run OCR correction pass (pass 1)")
    p_ocr.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_ocr.add_argument("--seed", type=int, default=42)
    p_ocr.add_argument("--count", type=int, default=10, dest="size", metavar="N")
    p_ocr.add_argument("--retry-flagged", action="store_true",
                        help="Re-process only currently flagged pairs using greedy bin-packing batches")

    # pair
    p_pair = sub.add_parser("pair", help="Run dependency-chain grouping pass (pass 2)")
    p_pair.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_pair.add_argument("--seed", type=int, default=42)
    p_pair.add_argument("--count", type=int, default=10, dest="size", metavar="N")

    # enrich
    p_enrich = sub.add_parser("enrich", help="Detect and enrich context-bare opening questions (pass 3)")
    p_enrich.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_enrich.add_argument("--seed", type=int, default=42)
    p_enrich.add_argument("--count", type=int, default=20, dest="size", metavar="N")

    # filter
    p_filter = sub.add_parser("filter", help="Rule-based garbage filtering (pass 4)")
    p_filter.add_argument("--sample", action="store_true", help="Preview what would be dropped without writing output")
    p_filter.add_argument("--seed", type=int, default=42)
    p_filter.add_argument("--count", type=int, default=20, dest="size", metavar="N")

    # score
    p_score = sub.add_parser("score", help="Score conversations and filter bottom N% (pass 4)")
    p_score.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_score.add_argument("--seed", type=int, default=42)
    p_score.add_argument("--count", type=int, default=20, dest="size", metavar="N")
    p_score.add_argument("--filter-pct", type=float, default=0.05, metavar="F",
                         help="Fraction to drop (default 0.05 = bottom 5%%)")

    # sample
    p_sample = sub.add_parser("sample", help="Sample conversations from any pipeline stage")
    p_sample.add_argument("stage", choices=list(OUTPUT_DIRS))
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--count", type=int, default=10, dest="size", metavar="N")

    args = parser.parse_args()

    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "bleed":
        cmd_bleed(args)
    elif args.cmd == "ocr":
        cmd_ocr(args)
    elif args.cmd == "pair":
        cmd_pair(args)
    elif args.cmd == "enrich":
        cmd_enrich(args)
    elif args.cmd == "filter":
        cmd_filter(args)
    elif args.cmd == "score":
        cmd_score(args)
    elif args.cmd == "sample":
        cmd_sample(args)


if __name__ == "__main__":
    main()
