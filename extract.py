import sys
from extractors.base import (
    AgricultureExtractor,
    AstronomyExtractor,
    BrewersGuideExtractor,
    ChemistryExtractor,
    CommonCoreExtractor,
    EngineeringExtractor,
    FamiliarThingsExtractor,
    InvestorsExtractor,
    LaborersExtractor,
    MusicExtractor,
    Questions1001Extractor,
    StokersExtractor,
    SymbologicalExtractor,
    WorldHistoryExtractor,
)

EXTRACTORS = {
    "common_core": CommonCoreExtractor,
    "brewers_guide": BrewersGuideExtractor,
    "familiar_things": FamiliarThingsExtractor,
    "1001_questions": Questions1001Extractor,
    "stokers": StokersExtractor,
    "symbological": SymbologicalExtractor,
    "agriculture": AgricultureExtractor,
    "astronomy": AstronomyExtractor,
    "chemistry": ChemistryExtractor,
    "engineering": EngineeringExtractor,
    "investors": InvestorsExtractor,
    "music": MusicExtractor,
    "laborers": LaborersExtractor,
    "world_history": WorldHistoryExtractor,
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
