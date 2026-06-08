import json
import re

_GUTENBERG_END = re.compile(r'\*{3}\s*END\b.*', re.DOTALL)

def strip_gutenberg_boilerplate(text, stop_markers=None):
    text = _GUTENBERG_END.split(text)[0]
    if stop_markers:
        for marker in stop_markers:
            text = text.split(marker)[0]
    return text


class BaseExtractor:
    """
    Base class for extractors. Subclasses should implement the extract_pairs method to return a list of
    {"q": ..., "a": ...} dicts.
    """
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


class CatechismExtractor(BaseExtractor):
    """
    Many catechisms are formatted with 'Question. Question text' followed by 'Answer. Answer text',
    repeated for each Q&A pair. This extractor handles that format.
    """
    PATTERN = re.compile(
        r'(?:Question|Q)\.\s+(.+?)\n\n(?:Answer|A)\.\s+(.+?)(?=\n\n(?:Question|Q)\.|\Z)',
        re.MULTILINE | re.DOTALL
    )

    def preprocess(self, text):
        return text

    def extract_pairs(self, text):
        text = self.preprocess(text)

        pairs = []
        for q_text, a_text in self.PATTERN.findall(text):
            question = re.sub(r" {2,}", " ", q_text.strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class BrewersGuideExtractor(CatechismExtractor):
    pass


def _is_symbological_header(paragraph):
    """
    Detect if a paragraph is a header like 'Question 1.' or 'Q.1' that should be removed before parsing.
    """
    lines = [l.strip() for l in paragraph.strip().split('\n') if l.strip()]
    if not lines:
        return True
    if re.match(r'^(Question|Answer|Q|A)\.', lines[0]):
        return False
    if len(lines) > 3:
        return False
    text = ' '.join(lines)
    alpha = sum(1 for c in text if c.isalpha())
    upper = sum(1 for c in text if c.isupper())
    return alpha > 0 and upper / alpha > 0.65 and len(text) < 80


class SymbologicalExtractor(CatechismExtractor):
    """
    The Symbological is formatted with section headers like 'Question 1.' that break up the text. 
    We need to remove those before parsing.
    """
    def preprocess(self, text):
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)
        return re.sub(r'\n{3,}', '\n\n', text)
    
class CommonCoreExtractor(BaseExtractor):
    """
    Common Core is formatted with numbered questions like '1. Question text' followed by answer text,
    repeated for each Q&A pair.
    """
    def extract_pairs(self, text):
        pattern = re.compile(r"^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)", re.MULTILINE | re.DOTALL)
        matches = pattern.findall(text)

        pairs = []
        for _num, block in matches:
            idx = block.rfind("?")
            if idx == -1:
                continue
            question = re.sub(r" {2,}", " ", block[:idx + 1].strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", block[idx + 1:].strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs

class FamiliarThingsExtractor(BaseExtractor):
    """
    Familiar Things is formatted with questions as single lines ending with '?' followed by a blank line and
    then the answer text, repeated for each Q&A pair. Answers may include indented glossary definitions that
    should be included in the answer text.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text, stop_markers=['\nINDEX.'])
        pattern = re.compile(r'^([^\n]+\?)\n\n(.+?)(?=\n\n[^\n]+\?|\Z)', re.MULTILINE | re.DOTALL)

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = q_text.strip()
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs

class Questions1001Extractor(BaseExtractor):
    """
    1001 Questions is formatted with numbered questions like '1. Question text' followed by answer text starting
    with 'Answer.' on the next line, repeated for each Q&A pair.
    """
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
    
class StokersExtractor(BaseExtractor):
    """
    Stokers is formatted with numbered questions like '1. Question text' followed by answer text starting
    with 'Answer.' on the next line, repeated for each Q&A pair. However, the question text may include inline
    page markers like [8] that should be removed before parsing.
    """
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


class InvestorsExtractor(BaseExtractor):
    """
    The Investor's Catechism. Format: 'What is TERM ?' / answer, similar to FamiliarThings.
    Key terms appear in ALL CAPS in questions and are lowercased on extraction.
    Front matter (index) precedes the content; page headers are interspersed as noise.
    """
    _PAGE_HEADER = re.compile(r'\n\n[\w]+ THE INVESTOR\'S CATECHISM[\w ]*\n\n|\n\nTHE INVESTOR\'S CATECHISM[\w ]*\n\n')

    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        # Drop everything before the first question
        first_q = re.search(r'\nWhat ', text)
        if first_q:
            text = text[first_q.start():]
        # Strip page headers
        text = self._PAGE_HEADER.sub('\n\n', text)
        # Normalize single-line Q&A: "What is X? Answer." → two paragraphs
        text = re.sub(r'^(What [^\n]+?\?)\s+([A-Z])', r'\1\n\n\2', text, flags=re.MULTILINE)

        pattern = re.compile(r'^(What [^\n]+\?)\n\n(.+?)(?=\nWhat |\Z)', re.MULTILINE | re.DOTALL)

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r'\b([A-Z]{2,})\b', lambda m: m.group().lower(), q_text.strip())
            question = re.sub(r" {2,}", " ", question)
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class WorldHistoryExtractor(BaseExtractor):
    """
    Catechism of Universal History. Format: 'N. Q. question' / 'A. answer'.
    Q. sometimes OCRs as Q,. or Q,.. Single-line items ('N. Q. question? A. answer.')
    also occur. Standalone page numbers and CHAP. headers are interspersed as noise.
    """
    def extract_pairs(self, text):
        # Fix OCR hyphens
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        # Strip standalone page numbers and short OCR artifacts between blank lines
        text = re.sub(r'\n\n\d{1,3}\n\n', '\n\n', text)
        text = re.sub(r'\n\n[a-zA-Z]{1,2}\n\n', '\n\n', text)
        # Strip CHAP. header lines
        text = re.sub(r'\n\nCHAP\.[^\n]*\n\n', '\n\n', text)
        # Normalize single-line Q&A: "N. Q. question? A. answer" → two paragraphs
        text = re.sub(r'^(\d+\.\s+Q[.,]+\s+.+?\?)\s+(A\.\s+)', r'\1\n\n\2', text, flags=re.MULTILINE)

        pattern = re.compile(
            r'^\d+\.\s+Q[.,]+\s+(.+?)\n\nA\.\s+(.+?)(?=\n\n\d+\.\s+Q[.,]|\Z)',
            re.MULTILINE | re.DOTALL
        )

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r" {2,}", " ", q_text.strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs
