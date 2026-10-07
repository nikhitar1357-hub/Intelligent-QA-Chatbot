"""
preprocessing.py - text cleaning, tokenisation and document loading.

We deliberately avoid NLTK/spaCy downloads so the project runs fully offline.
The tokeniser, stop-word list and light stemmer below are small but sufficient
for TF-IDF retrieval on an educational corpus.
"""
import re
from pathlib import Path

# A compact English stop-word list (common words that carry little meaning).
# NOTE: question words (what, who, when...) are stop words for RETRIEVAL only.
STOPWORDS = set("""
a about above after again all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few
for from further had has have having he her here hers him his how i if in into is
it its just me more most my no nor not of off on once only or other our out over
own same she should so some such than that the their them then there these they
this those through today current currently now to too under until up us very was we were what when where which
while who whom why will with would you your tell me please explain define
""".split())


def clean_text(text: str) -> str:
    """Step 2 of the pipeline: normalise raw text.

    - lower-case everything
    - replace anything that is not a letter/number/space with a space
    - collapse multiple spaces
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_stem(word: str) -> str:
    """Very light suffix stripping so 'networks' matches 'network'
    and 'reduced' matches 'reduce'. Intentionally simple and easy to explain.
    """
    for suffix in ("ing", "ers", "er", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[:-3] + "y" if suffix == "ies" else word[: -len(suffix)]
            break
    # drop a trailing 'e' so 'reduce' and 'reduced' both become 'reduc'
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True):
    """Step 3 of the pipeline: split cleaned text into a list of tokens."""
    tokens = clean_text(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [simple_stem(t) for t in tokens]
    return tokens


def preprocess_for_tfidf(text: str) -> str:
    """Return the tokens joined back into a string (what TfidfVectorizer expects)."""
    return " ".join(tokenize(text))


def split_sentences(text: str):
    """Split a paragraph into sentences using punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_documents(doc_dir):
    """Step 1 of the pipeline: load every .txt file and split into paragraphs.

    Returns a list of dicts:
        {"doc": file name, "para_id": index, "text": paragraph text}
    Each paragraph becomes one retrievable unit ("passage").
    """
    passages = []
    for path in sorted(Path(doc_dir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        for i, para in enumerate(paragraphs):
            passages.append({"doc": path.stem, "para_id": i, "text": para})
    return passages
