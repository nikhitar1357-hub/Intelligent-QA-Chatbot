"""
evaluation.py - measures how well the chatbot works on data/sample_questions.csv.

Metrics
    * Routing precision / recall / F1 (did the bot choose the right mode?)
    * Retrieval accuracy for IR questions (top-1 and top-3 document hit)
    * Answer accuracy (does the answer contain an expected keyword?)
    * Mean confidence and average response time
"""
import pandas as pd

from .chatbot import Chatbot
from .utils import QUESTIONS_PATH

MODES = ["Dialogue", "Knowledge QA", "IR QA", "Fallback"]


def run_evaluation(path=QUESTIONS_PATH):
    bot = Chatbot()
    df = pd.read_csv(path).fillna("")
    records = []
    for _, row in df.iterrows():
        bot.reset()                                  # each test question is independent
        r = bot.ask(row["question"])
        keywords = [k.strip().lower() for k in str(row["answer_keywords"]).split("|") if k.strip()]
        answer_ok = any(k in r.text.lower() for k in keywords) if keywords else True

        top1 = top3 = None
        if row["expected_mode"] == "IR QA":          # retrieval test, independent of routing
            cands = bot.ir.retrieve(row["question"], top_k=3)
            top1 = cands[0]["doc"] == row["expected_source"]
            top3 = any(c["doc"] == row["expected_source"] for c in cands)

        records.append({
            "id": row["id"], "question": row["question"],
            "expected_mode": row["expected_mode"], "predicted_mode": r.mode,
            "mode_correct": r.mode == row["expected_mode"],
            "answer_correct": answer_ok, "confidence": r.confidence,
            "time_ms": r.elapsed * 1000, "ir_top1": top1, "ir_top3": top3,
            "answer": r.text,
        })
    return pd.DataFrame(records)


def per_mode_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Precision, recall and F1 for each mode (one-vs-rest)."""
    rows = []
    for m in MODES:
        tp = ((results.predicted_mode == m) & (results.expected_mode == m)).sum()
        fp = ((results.predicted_mode == m) & (results.expected_mode != m)).sum()
        fn = ((results.predicted_mode != m) & (results.expected_mode == m)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"Mode": m, "Precision": p, "Recall": r, "F1": f1,
                     "Support": int((results.expected_mode == m).sum())})
    out = pd.DataFrame(rows)
    macro = {"Mode": "Macro average", "Precision": out.Precision.mean(),
             "Recall": out.Recall.mean(), "F1": out.F1.mean(), "Support": out.Support.sum()}
    return pd.concat([out, pd.DataFrame([macro])], ignore_index=True)


def summary(results: pd.DataFrame) -> dict:
    ir = results[results.expected_mode == "IR QA"]
    return {
        "Questions evaluated": len(results),
        "Routing accuracy": results.mode_correct.mean(),
        "Answer accuracy": results.answer_correct.mean(),
        "IR retrieval top-1 accuracy": ir.ir_top1.astype(float).mean(),
        "IR retrieval top-3 accuracy": ir.ir_top3.astype(float).mean(),
        "Mean confidence": results.confidence.mean(),
        "Average response time (ms)": results.time_ms.mean(),
    }
