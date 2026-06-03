import sys
from extractors.common_core import CommonCoreExtractor
from extractors.brewers_guide import BrewersGuideExtractor

EXTRACTORS = {
    "common_core": CommonCoreExtractor,
    "brewers_guide": BrewersGuideExtractor,
}

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract.py <extractor> <input.txt> <output.json>")
        print(f"Extractors: {', '.join(EXTRACTORS)}")
        sys.exit(1)

    name, input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    if name not in EXTRACTORS:
        print(f"Unknown extractor: {name}. Available: {', '.join(EXTRACTORS)}")
        sys.exit(1)

    EXTRACTORS[name](input_path, output_path).run()
