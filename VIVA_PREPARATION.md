# Viva Preparation — Intelligent QA / Chatbot System

## Module-by-module explanation (plain language)

- **Query Processing** — cleans the user's text and, if it contains a pronoun like
  "it" right after a previous question was answered, swaps the pronoun for the
  earlier topic so follow-up questions still make sense.
- **Intent Detection** — a set of regex patterns that recognise small talk
  (hi, thanks, bye, how are you) before the message is ever treated as a question.
- **Dialogue Manager** — owns the conversation's short-term memory (the user's name,
  the last topic) and produces friendly, sometimes personalised, responses.
- **IR Retrieval Engine** — turns every document paragraph and the question into
  TF-IDF vectors and ranks paragraphs by cosine similarity.
- **Knowledge Base** — a JSON file of question/answer/category/alias records that
  acts as the system's "structured memory" of exact facts.
- **Structured QA Engine** — matches the user's question against the knowledge base
  using a blend of TF-IDF and fuzzy string similarity.
- **Context Retrieval** — the step inside the IR engine that hands the single
  best-matching passage forward to answer extraction.
- **Answer Generator** — picks the one sentence inside the retrieved passage that
  best matches the question.
- **Response Formatter** — wraps every answer (regardless of source) into one
  consistent object (text, mode, confidence, context, source) for the UI.

## 30 Viva Questions & Answers

1. **What is the difference between IR-based QA and knowledge-based QA?**
   IR-based QA retrieves and extracts an answer from free text; knowledge-based QA
   looks up an exact, pre-written answer from a structured source.

2. **What is TF-IDF?**
   A weighting scheme that scores a word highly in a document if it appears often
   there (term frequency) but rarely across the whole collection (inverse document
   frequency), so common words are down-weighted and distinctive words stand out.

3. **Why use cosine similarity instead of, say, Euclidean distance?**
   Cosine similarity measures the angle between two vectors, ignoring their length,
   so it is not biased by document length the way Euclidean distance would be.

4. **What does a cosine similarity of 0 mean? Of 1?**
   0 means the vectors share no weighted terms at all (completely dissimilar
   direction); 1 means they point in exactly the same direction (maximally similar).

5. **Why is the mean confidence around 0.6 rather than close to 1.0?**
   TF-IDF vectors for a short question only share a fraction of the vocabulary of a
   whole passage, so even a correct match rarely approaches a cosine similarity of 1.
   This is expected behaviour, not an error.

6. **What is a stop word, and why remove it?**
   A very common word (the, is, of, what) that carries little topic-specific meaning;
   removing it stops the similarity score from being dominated by words every
   document shares.

7. **What is stemming, and why is a simple rule-based stemmer used here instead of a
   library like NLTK's Porter stemmer?**
   Stemming reduces a word to an approximate root form (e.g. "networks" → "network")
   so related word forms match. A small custom stemmer was used to keep the project
   fully offline with no extra downloads, while still covering the vocabulary of the
   project's own documents.

8. **What is an n-gram, and why use unigrams and bigrams (ngram_range=(1,2))?**
   An n-gram is a sequence of n consecutive tokens. Bigrams (2-word phrases like
   "neural network") let the retriever match multi-word concepts, not just single
   words, giving it a little more context sensitivity.

9. **How does the system decide which passage's sentence to return as the answer?**
   It re-applies the same TF-IDF comparison at the sentence level: every sentence of
   the winning passage is scored against the question, and the top-scoring sentence
   is returned (with a small bonus for the first sentence on definition questions).

10. **What is the structured knowledge base, and what format is it stored in?**
    A JSON file of 62 question-answer entries, each with a category and optional
    alias phrasings, acting as the system's source of exact facts.

11. **Why blend TF-IDF similarity with fuzzy string matching for knowledge-base
    matching instead of using just one?**
    TF-IDF handles different wording of the same idea; fuzzy (character-level)
    matching catches small typos or near-identical phrasings that TF-IDF alone might
    miss. Blending both makes matching more robust.

12. **What is the out-of-vocabulary guard in knowledge_qa.py, and why is it needed?**
    It scales down the match score if the question contains content words that never
    appear anywhere in the knowledge base, which stops the bot from confidently
    answering a question about something (e.g. a different person or technology) it
    was never actually told about.

13. **Why does the chatbot try the Knowledge Base before the IR engine?**
    Structured facts are exact, so a confident knowledge-base match is preferred over
    a retrieved passage, which is only an approximate, extracted answer.

14. **What happens if neither the Knowledge Base nor the IR engine is confident?**
    The system returns a polite fallback message rather than guessing, so the user is
    never given a confident-sounding but unsupported answer.

15. **What is dialogue management, and why does a chatbot need it?**
    It is the component that controls conversational flow — recognising small talk,
    remembering context across turns and deciding how to respond — so the system
    feels like a conversation partner rather than a single-shot search box.

16. **How does the dialogue manager detect intents?**
    With an ordered list of regular expressions, each mapped to an intent such as
    greeting, thanks, bye or how-are-you; the first matching pattern wins.

17. **How does the chatbot resolve a follow-up question like "Who invented it?"**
    It keeps the previous question as "last topic" in the dialogue state; if the new
    message is short and contains a pronoun, the pronoun is replaced with the subject
    of the last topic before the question is routed.

18. **What is the difference between structured and unstructured knowledge?**
    Structured knowledge is organised in a fixed format (e.g. question-answer pairs
    or database rows) and can be queried precisely; unstructured knowledge is free
    text that must be searched and interpreted, which is less precise but far more
    abundant.

19. **Why was Streamlit chosen for the interface?**
    It lets a complete, interactive, chat-style web interface be built in pure
    Python with no separate frontend code, which suits a student project's scope and
    time budget while still producing a professional-looking demo.

20. **What does the confidence score actually represent for an IR answer?**
    The cosine similarity between the question's TF-IDF vector and the winning
    passage's TF-IDF vector — a proxy for how lexically well-matched the two are.

21. **What are precision, recall and F1-score, in the context of this project's
    routing evaluation?**
    For a given mode (e.g. IR QA), precision is the fraction of questions the system
    routed to that mode that truly belonged there; recall is the fraction of
    questions that truly belonged there that the system actually routed there; F1 is
    their harmonic mean.

22. **Why is a 28-question test set "experimental" rather than a general benchmark?**
    It was hand-built from the same dataset the system was given, so the perfect
    scores show the pipeline works correctly as designed — they do not claim the
    system would perform this well on arbitrary, unseen real-world questions.

23. **What is the risk of overfitting the evaluation to the system's own dataset?**
    The numbers may look better than real-world performance, since the test
    questions were written with knowledge of what the system can answer; this is
    disclosed explicitly in the report rather than presented as a general claim.

24. **What would need to change to deploy this with semantic (not just lexical)
    retrieval?**
    Replace TF-IDF with dense sentence embeddings (e.g. from a transformer model)
    and use a similarity search library such as FAISS, so paraphrased questions with
    no shared vocabulary could still be matched.

25. **Why does the project avoid paid LLM APIs?**
    To keep it fully offline, free to run, reproducible for any evaluator, and to
    keep the techniques (TF-IDF, cosine similarity, rule-based matching) transparent
    and explainable, which suits the stated IR/knowledge-based QA scope of the
    assignment.

26. **What is a limitation of the TF-IDF based IR engine?**
    It matches on shared vocabulary only, so a question that is phrased very
    differently from the source text (a paraphrase with no overlapping words) may not
    be retrieved correctly.

27. **What is a limitation of the rule-based dialogue manager?**
    It can only recognise intents that match its predefined patterns; truly novel
    phrasings of small talk may fall through to the QA modules instead.

28. **How is the knowledge base kept consistent when the same fact can be asked in
    different ways?**
    Each entry can store alias phrasings in addition to its canonical question, and
    all of them are indexed, so different wordings of the same question still match
    the same stored fact.

29. **What does the Response Formatter do, and why keep it separate?**
    It converts whichever module answered (Dialogue, Knowledge QA or IR QA) into one
    consistent `Response` object, so the user interface code does not need to know
    the internal details of each module — only Claude's content changes, not the
    response shape.

30. **What would you improve first if you continued this project?**
    Likely the retrieval engine first — moving from lexical TF-IDF to semantic
    sentence embeddings would directly address the main limitation (missed
    paraphrases) without changing the rest of the architecture.
