import re
from extractors.base import BaseExtractor

class Questions1001Extractor(BaseExtractor):
    def extract_pairs(self, text):
        pattern = re.compile(r"^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)", re.MULTILINE | re.DOTALL)

        pairs = []
        for _num, block in pattern.findall(text):
            idx = block.rfind("?")
            if idx == -1:
                continue
            question = re.sub(r" {2,}", " ", block[:idx + 1].strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", block[idx + 1:].strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs
