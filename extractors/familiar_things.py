import re
from extractors.base import BaseExtractor

class FamiliarThingsExtractor(BaseExtractor):
    def extract_pairs(self, text):
        # Format: single line ending with '?' followed by blank line then answer
        # Answers may include indented glossary definitions
        pattern = re.compile(r'^([^\n]+\?)\n\n(.+?)(?=\n\n[^\n]+\?|\Z)', re.MULTILINE | re.DOTALL)

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = q_text.strip()
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs
