import sys
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
