"""Run the eval set against several RAG configurations and score answers with an LLM judge.

Usage:  python run_eval.py            (needs ANTHROPIC_API_KEY)
Output: a markdown table printed to the console and saved to eval_results.md
"""
import json
import os
import re

from rag import MODEL, PROVIDER, RagIndex, generate

JUDGE_MODEL = os.getenv("JUDGE_MODEL", MODEL)

# Add or change configurations here to compare them.
CONFIGS = [
    {"name": "chunk300_k3_v1", "chunk_size": 300, "top_k": 3, "prompt_version": "v1"},
    {"name": "chunk300_k3_v2", "chunk_size": 300, "top_k": 3, "prompt_version": "v2"},
    {"name": "chunk800_k3_v2", "chunk_size": 800, "top_k": 3, "prompt_version": "v2"},
]

JUDGE_PROMPT = """You are grading a support assistant. Score each item from 1 to 5.

Question: {question}
Reference answer: {reference}
Retrieved context:
{context}
Assistant answer: {answer}

Scores:
- faithfulness: is every claim in the answer supported by the retrieved context? (5 = fully supported, 1 = invented or contradicted). If the question is marked UNANSWERABLE in the reference, a correct "not found in the documentation" answer gets 5.
- relevance: does the answer address the question that was asked?
- correctness: does the answer agree with the reference answer?

Reply with JSON only, for example: {{"faithfulness": 4, "relevance": 5, "correctness": 3}}"""


def judge(question, reference, context, answer) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, context=context, answer=answer)
    text = generate(prompt, max_tokens=100, model=JUDGE_MODEL)["text"]
    match = re.search(r"\{.*?\}", text, re.S)
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"faithfulness": 1, "relevance": 1, "correctness": 1}  # unparseable judge output counts as a fail


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "eval_set.json"), encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    for cfg in CONFIGS:
        index = RagIndex(cfg["chunk_size"], cfg["top_k"], cfg["prompt_version"])
        totals = {"faithfulness": 0, "relevance": 0, "correctness": 0}
        latency = 0
        for item in items:
            out = index.answer(item["question"])
            scores = judge(item["question"], item["reference"], out["context"], out["answer"])
            for k in totals:
                totals[k] += scores.get(k, 1)
            latency += out["latency_ms"]
            print(f"[{cfg['name']}] {item['question'][:50]!r} -> {scores}")
        n = len(items)
        pct = {k: round(v / n / 5 * 100) for k, v in totals.items()}
        rows.append((cfg["name"], pct, round(latency / n)))

    lines = [
        f"Eval set: {len(items)} questions. Answers and judge: {PROVIDER} / {MODEL}. "
        "Scores are the average judge score (1 to 5) as a percentage.",
        "",
        "| Config | Faithfulness | Relevance | Correctness | Avg latency (ms) |",
        "|---|---|---|---|---|",
    ]
    for name, pct, lat in rows:
        lines.append(f"| {name} | {pct['faithfulness']}% | {pct['relevance']}% | {pct['correctness']}% | {lat} |")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(here, "eval_results.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
