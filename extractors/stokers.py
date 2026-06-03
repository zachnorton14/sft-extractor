import re
from extractors.base import BaseExtractor

class StokersExtractor(BaseExtractor):
    def extract_pairs(self, text):
        # Strip inline page markers like [8] before parsing
        text = re.sub(r'\[\d+\]', '', text)

        pattern = re.compile(
            r'^\d+\.\s+Question\.—(.+?)\n\nAnswer\.—(.+?)(?=\n\n\d+\.\s+Question\.|\Z)',
            re.MULTILINE | re.DOTALL
        )

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r" {2,}", " ", q_text.strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs
