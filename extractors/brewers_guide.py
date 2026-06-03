import re
from extractors.base import BaseExtractor

class BrewersGuideExtractor(BaseExtractor):
    def extract_pairs(self, text):
        pattern = re.compile(r'^Q\.\s+(.+?)\n\nA\.\s+(.+?)(?=\n\nQ\.|\Z)', re.MULTILINE | re.DOTALL)

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r" {2,}", " ", q_text.strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs
