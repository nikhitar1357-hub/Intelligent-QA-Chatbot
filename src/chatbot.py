"""
chatbot.py - the orchestrator that ties every module together.

Data flow for one user message
    User Interface
      -> Query Processing   (clean text, resolve follow-ups like "who invented it?")
      -> Intent Detection   (small-talk vs. real question)
      -> Dialogue Manager   (handles small talk, memory, fallback)
      -> Structured QA Engine  (Knowledge Base, exact facts)   [mode: Knowledge QA]
      -> IR Retrieval Engine   (TF-IDF over documents)          [mode: IR QA]
      -> Context Retrieval  (best passage + source)
      -> Answer Generator   (final answer text)
      -> Response Formatter (uniform Response object for the UI)
"""
import re
from dataclasses import dataclass, field

from .dialogue import DialogueManager
from .knowledge_qa import KnowledgeQA
from .retriever import TfidfRetriever
from .utils import Timer

MODE_DIALOGUE = "Dialogue"
MODE_KB = "Knowledge QA"
MODE_IR = "IR QA"
MODE_FALLBACK = "Fallback"

PRONOUNS = {"it", "this", "that", "they", "them", "its", "these", "those"}


@dataclass
class Response:
    """Uniform result object returned to the UI (Response Formatter output)."""
    text: str
    mode: str
    confidence: float = 0.0
    context: str = ""
    source: str = ""
    elapsed: float = 0.0
    resolved_query: str = ""
    extra: dict = field(default_factory=dict)


class Chatbot:
    def __init__(self):
        self.dialogue = DialogueManager()
        self.kb = KnowledgeQA()
        self.ir = TfidfRetriever()

    # ---- Query Processing ------------------------------------------------
    def process_query(self, text: str) -> str:
        """Tidy the query and resolve simple follow-ups using memory.

        Example: after "What is Python?", the question "Who invented it?"
        becomes "Who invented it? What is Python?" so that retrieval sees the topic.
        """
        query = text.strip()
        words = re.findall(r"[a-zA-Z']+", query.lower())
        last = self.dialogue.state.last_topic
        if last and len(words) <= 7 and PRONOUNS.intersection(words):
            # replace the pronoun with the subject of the previous question
            subject = re.sub(r"^(what|who|when|where|which|why|how)\s+(is|are|was|were|does|do|did)?\s*(a|an|the)?\s*",
                             "", last.strip().rstrip("?"), flags=re.I)
            resolved = " ".join(subject if w.lower().strip("?.,!") in PRONOUNS else w
                                for w in query.split())
            return resolved
        return query

    # ---- Main entry point ------------------------------------------------
    def ask(self, text: str) -> Response:
        with Timer() as t:
            resp = self._route(text)
        resp.elapsed = t.elapsed
        self.dialogue.add_turn("bot", resp.text)
        return resp

    def _route(self, text: str) -> Response:
        self.dialogue.add_turn("user", text)
        if not text.strip():
            return Response("Please type a question.", MODE_DIALOGUE, 1.0)

        # 1) Intent detection -> Dialogue Manager for small talk
        intent, match = self.dialogue.detect_intent(text)
        if intent:
            return Response(self.dialogue.respond(intent, match), MODE_DIALOGUE, 1.0,
                            extra={"intent": intent})

        # 2) Query processing (follow-up resolution using memory)
        query = self.process_query(text)

        # 3) Structured QA first (exact facts), then IR over documents
        kb_hit = self.kb.answer(query)
        if kb_hit and kb_hit["confidence"] >= 0.75:
            return self._kb_response(kb_hit, query, text)

        ir_hit = self.ir.answer(query)
        if ir_hit:
            self.dialogue.set_topic(text)
            return Response(ir_hit["answer"], MODE_IR, ir_hit["confidence"],
                            context=ir_hit["context"], source=ir_hit["source"],
                            resolved_query=query,
                            extra={"candidates": ir_hit["candidates"]})

        # weaker KB match is still better than nothing
        if kb_hit:
            return self._kb_response(kb_hit, query, text)

        # 4) Fallback
        return Response(self.dialogue.fallback(), MODE_FALLBACK, 0.0, resolved_query=query)

    def _kb_response(self, hit, query, original) -> Response:
        self.dialogue.set_topic(original)
        return Response(hit["answer"], MODE_KB, hit["confidence"],
                        context=f"Matched KB question: \"{hit['matched_question']}\"",
                        source=f"knowledge_base.json · {hit['category']}",
                        resolved_query=query)

    def reset(self):
        self.dialogue.reset()
