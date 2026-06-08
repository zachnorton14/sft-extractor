import re
from extractors.base import BaseExtractor

def _is_header(paragraph):
    lines = [l.strip() for l in paragraph.strip().split('\n') if l.strip()]
    if not lines:
        return True
    # Protect Q/A blocks
    if re.match(r'^(Question|Answer|Q|A)\.', lines[0]):
        return False
    # Short paragraphs that are mostly uppercase are chapter headers / page markers
    if len(lines) > 3:
        return False
    text = ' '.join(lines)
    alpha = sum(1 for c in text if c.isalpha())
    upper = sum(1 for c in text if c.isupper())
    return alpha > 0 and upper / alpha > 0.65 and len(text) < 80

def _preprocess(text):
    # Rejoin hyphenated line breaks: "re- \nentering" → "reentering"
    text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
    # Rejoin within-line OCR hyphen-space artifacts: "gar- den" → "garden"
    text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
    # Strip chapter header paragraphs
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p for p in paragraphs if not _is_header(p)]
    text = '\n\n'.join(paragraphs)
    # Collapse any remaining excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

class SymbologicalExtractor(BaseExtractor):
    def extract_pairs(self, text):
        text = _preprocess(text)

        pattern = re.compile(
            r'(?:Question|Q)\.\s+(.+?)\n\n(?:Answer|A)\.\s+(.+?)(?=\n\n(?:Question|Q)\.|\Z)',
            re.MULTILINE | re.DOTALL
        )

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r" {2,}", " ", q_text.strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs
