"""
dialogue.py - Simple rule-based dialogue manager with conversation memory.

Responsibilities
    * Intent detection for small talk (greeting, thanks, bye, how are you ...)
    * Keeping the dialogue state (history, user name, last topic)
    * Producing friendly responses and the fallback message
"""
import random
import re
from dataclasses import dataclass, field

# intent -> regex patterns (checked in this order)
INTENT_PATTERNS = [
    ("bye",        r"^(bye|goodbye|good bye|see you|see ya|take care|exit|quit)\b"),
    ("thanks",     r"^(thanks|thank you|thank u|thx|ty|much appreciated)\b"),
    ("how_are_you", r"\b(how are you|how r u|how do you do|how's it going|how is it going|what's up|whats up)\b"),
    ("greeting",   r"^(hi|hii+|hello|hey|hola|namaste|good morning|good afternoon|good evening|greetings)\b"),
    ("set_name",   r"\b(my name is|i am called|call me)\s+([a-z]+)"),
    ("get_name",   r"\b(what is my name|what's my name|do you know my name|who am i)\b"),
    ("who_are_you", r"\b(who are you|what are you|what is your name|what's your name|your name)\b"),
    ("help",       r"^(help|what can you do|how can you help|what do you do)\b"),
    ("affirm",     r"^(ok|okay|cool|nice|great|awesome|good|fine|alright)\W*$"),
]

RESPONSES = {
    "greeting": ["Hello! 👋 I'm your AI/ML study assistant. Ask me anything about AI, Python, ML or Karnataka.",
                 "Hi there! How can I help you today?",
                 "Hey! Ready to answer your questions about AI and ML."],
    "how_are_you": ["I'm doing great, thank you for asking! How can I help you?",
                    "All systems running smoothly 😊 What would you like to know?"],
    "thanks": ["You're welcome! Feel free to ask more questions.",
               "Happy to help! Anything else?"],
    "bye": ["Goodbye! Have a great day 👋", "Bye! Come back any time you want to learn something."],
    "who_are_you": ["I'm an intelligent QA chatbot built with information retrieval, a structured knowledge base and a dialogue manager."],
    "help": ["I can: (1) answer questions from my text documents (IR QA), (2) answer factual questions from my knowledge base (Knowledge QA), and (3) chat in a friendly way. Try 'What is machine learning?' or 'Who invented Python?'"],
    "affirm": ["Great! What would you like to ask next?"],
}

FALLBACK = ("I couldn't find the answer in my knowledge base. "
            "Please try asking differently.")


@dataclass
class DialogueState:
    """Everything the bot remembers about the current conversation."""
    history: list = field(default_factory=list)    # [(speaker, text), ...]
    user_name: str = ""
    last_topic: str = ""       # last question that was answered successfully
    turns: int = 0


class DialogueManager:
    def __init__(self):
        self.state = DialogueState()

    # -- state handling --------------------------------------------------
    def reset(self):
        self.state = DialogueState()

    def add_turn(self, speaker: str, text: str):
        self.state.history.append((speaker, text))
        if speaker == "user":
            self.state.turns += 1

    def set_topic(self, topic: str):
        self.state.last_topic = topic

    # -- intent detection ------------------------------------------------
    def detect_intent(self, text: str):
        """Return (intent, match) for small talk, or (None, None)."""
        t = text.lower().strip()
        # Long messages are almost certainly real questions, not small talk
        if len(t.split()) > 8 and "my name is" not in t:
            return None, None
        for intent, pattern in INTENT_PATTERNS:
            m = re.search(pattern, t)
            if m:
                return intent, m
        return None, None

    # -- response generation ---------------------------------------------
    def respond(self, intent: str, match=None) -> str:
        name = self.state.user_name
        if intent == "set_name":
            self.state.user_name = match.group(2).capitalize()
            return f"Nice to meet you, {self.state.user_name}! What would you like to learn today?"
        if intent == "get_name":
            return f"Your name is {name}." if name else "I don't know your name yet. Tell me by saying 'My name is ...'."
        reply = random.choice(RESPONSES[intent])
        # Use memory: personalise greetings/thanks/bye if we know the name
        if name and intent in ("greeting", "bye", "thanks"):
            reply = reply.replace("Hello!", f"Hello {name}!").replace("Hi there!", f"Hi {name}!") \
                         .replace("Hey!", f"Hey {name}!").replace("Goodbye!", f"Goodbye {name}!") \
                         .replace("Bye!", f"Bye {name}!")
        return reply

    @staticmethod
    def fallback() -> str:
        return FALLBACK
