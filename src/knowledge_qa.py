"""
knowledge_qa.py - Knowledge-based QA over the structured JSON knowledge base.

Matching strategy (simple and explainable):
    1. Every KB entry has a canonical question plus optional aliases.
    2. The user question is compared with all of them using
       (a) TF-IDF cosine similarity  and  (b) difflib string similarity.
    3. The final score is a weighted blend: 0.7*TF-IDF + 0.3*fuzzy. If it passes a threshold the
       stored answer is returned exactly (structured knowledge = exact facts).
"""
import re
from difflib import SequenceMatcher

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import clean_text, preprocess_for_tfidf
from .utils import KB_PATH, load_json


class KnowledgeQA:
    def __init__(self, kb_path=KB_PATH, threshold=0.45):
        self.threshold = threshold
        self.entries = load_json(kb_path)["entries"]

        # Flatten: one searchable row per question phrasing (question + aliases)
        self.rows = []
        for e in self.entries:
            for phrasing in [e["question"]] + e.get("aliases", []):
                self.rows.append({"text": phrasing, "entry": e})

        corpus = [preprocess_for_tfidf(r["text"]) for r in self.rows]
        self.known_tokens = {t for c in corpus for t in c.split()}
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def categories(self):
        return sorted({e["category"] for e in self.entries})

    def answer(self, question: str):
        """Return the best KB match as a dict, or None if below threshold."""
        q_clean = preprocess_for_tfidf(question)
        if not q_clean:
            return None

        # (a) TF-IDF cosine similarity to every phrasing
        tfidf_scores = cosine_similarity(self.vectorizer.transform([q_clean]), self.matrix).ravel()

        # (b) character-level similarity (catches small wording differences)
        # Compared on content words only (stop-words removed) so that shared
        # words like "what is the" do not inflate the similarity.
        fuzzy_scores = np.array([SequenceMatcher(None, q_clean, preprocess_for_tfidf(r["text"])).ratio()
                                 for r in self.rows])

        combined = 0.7 * tfidf_scores + 0.3 * fuzzy_scores

        # Out-of-vocabulary guard: if the question contains words that never
        # appear anywhere in the KB (e.g. "Java"), we should not pretend to know
        # the answer, so the score is scaled by the share of known words.
        tokens = q_clean.split()
        known = [t for t in tokens if t in self.known_tokens]
        combined = combined * (len(known) / len(tokens))
        best = int(np.argmax(combined))
        score = float(combined[best])
        if score < self.threshold:
            return None
        entry = self.rows[best]["entry"]
        return {
            "answer": entry["answer"],
            "category": entry["category"],
            "matched_question": entry["question"],
            "confidence": score,
            "kb_id": entry["id"],
        }
