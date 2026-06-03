import json

class BaseExtractor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def extract_pairs(self, text):
        raise NotImplementedError

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            text = f.read()

        pairs = self.extract_pairs(text)

        records = [
            {
                "conversations": [
                    {"role": "user", "content": pair["q"]},
                    {"role": "assistant", "content": pair["a"]}
                ]
            }
            for pair in pairs
        ]

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"Extracted {len(pairs)} Q&A pairs → {self.output_path}")

