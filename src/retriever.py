"""
retriever.py - IR-based factoid QA engine (TF-IDF + cosine similarity).

Pipeline (each step is marked in the code):
    1. Load documents        -> preprocessing.load_documents
    2. Clean text            -> preprocessing.clean_text
    3. Tokenisation          -> preprocessing.tokenize
    4. TF-IDF vectorisation  -> TfidfVectorizer
    5. Cosine similarity     -> cosine_similarity(query_vec, passage_matrix)
    6. Retrieve top passage  -> argsort of similarity scores
    7. Generate answer       -> best-matching sentence inside the passage
    8. Confidence score      -> cosine similarity of the winning passage
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import load_documents, preprocess_for_tfidf, split_sentences
from .utils import DOC_DIR


class TfidfRetriever:
    def __init__(self, doc_dir=DOC_DIR, min_score=0.15):
        self.min_score = min_score          # below this we say "not found"

        # Step 1: load documents (each paragraph = one passage)
        self.passages = load_documents(doc_dir)

        # Steps 2-3: clean + tokenise every passage
        corpus = [preprocess_for_tfidf(p["text"]) for p in self.passages]

        # Step 4: TF-IDF. ngram_range=(1,2) also captures phrases such as
        # "neural network" or "supervised learning".
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus)

    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int = 3):
        """Return the top_k passages for a question with similarity scores."""
        # Apply the SAME cleaning + tokenising to the question
        q_vec = self.vectorizer.transform([preprocess_for_tfidf(question)])

        # Step 5: cosine similarity between question and every passage
        scores = cosine_similarity(q_vec, self.matrix).ravel()

        # Definition questions usually target the opening paragraph of a
        # document, so nudge paragraph 0 slightly (a simple IR "prior").
        if re.match(r"\s*(what|what's|define|explain|describe)\b", question.lower()):
            for i, p in enumerate(self.passages):
                if p["para_id"] == 0:
                    scores[i] *= 1.35

        # Step 6: highest scores first
        best = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in best:
            item = dict(self.passages[idx])
            item["score"] = float(scores[idx])
            results.append(item)
        return results

    # ------------------------------------------------------------------
    def extract_answer(self, question: str, passage_text: str) -> str:
        """Step 7: pick the sentence in the passage that best matches the question.

        Same TF-IDF idea, but applied at sentence level (the vectorizer that
        was trained on passages is reused so the vocabulary stays consistent).
        """
        sentences = split_sentences(passage_text)
        if len(sentences) == 1:
            return sentences[0]
        q_vec = self.vectorizer.transform([preprocess_for_tfidf(question)])
        s_vecs = self.vectorizer.transform([preprocess_for_tfidf(s) for s in sentences])
        sims = cosine_similarity(q_vec, s_vecs).ravel()
        # Definition questions ("what is / define / explain X") are usually
        # answered by the first sentence of a passage, so give it a bonus.
        if re.match(r"\s*(what|what's|define|explain|describe)\b", question.lower()):
            sims[0] += 0.30
        else:
            sims[0] += 0.02
        return sentences[int(np.argmax(sims))]

    # ------------------------------------------------------------------
    def answer(self, question: str, top_k: int = 3):
        """Full IR-QA: retrieve -> extract -> score (Step 8).

        Returns a dict, or None when nothing relevant is found.
        """
        results = self.retrieve(question, top_k=top_k)
        top = results[0]
        if top["score"] < self.min_score:
            return None
        return {
            "answer": self.extract_answer(question, top["text"]),
            "context": top["text"],
            "source": top["doc"],
            "confidence": top["score"],      # Step 8
            "candidates": results,
        }
