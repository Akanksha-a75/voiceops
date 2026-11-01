"""Score the agent on practice/questions.csv.

python -m src.eval retrieval [--rewrite] [--k 5]   # no answering, retrieval only
python -m src.eval step1 [--limit N] [--asr]       # prompt-only baseline
python -m src.eval step2 [--limit N] [--asr] [--k 5]  # RAG
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd

ROOT = os.getenv("DATASET_DIR", "data_local/Capstone_Project_Dataset_FTTT_2026")
Q_PATH = os.path.join(ROOT, "practice", "questions.csv")
OUT_DIR = "sample_outputs"

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm(s):
    s = str(s).translate(DEV_DIGITS).lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    return re.sub(r"\s+", " ", s)


def split_semi(v):
    return [x.strip() for x in str(v).split(" ; ") if x.strip()]


def missing_facts(reply, must_include):
    t = norm(reply)
    missing = []
    for fact in split_semi(must_include):
        alts = [a.strip() for a in fact.split(" / ") if a.strip()]
        if not any(norm(a) in t for a in alts):
            missing.append(fact)
    return missing


def forbidden_hits(reply, must_not):
    t = norm(reply)
    return [p for p in split_semi(must_not) if norm(p) in t]


def acceptable(row):
    raw = row.get("acceptable_decisions") or row.get("expected_decision") or "answer"
    return [x.strip() for x in str(raw).split(";") if x.strip()]


def retrieval_scores(row, sources):
    gold = [g.strip() for g in str(row.get("gold_evidence", "")).split(";") if g.strip()]
    if not gold:
        return None
    got = set(sources)
    return sum(1 for g in gold if g in got) / len(gold)


def load_questions(limit=None):
    df = pd.read_csv(Q_PATH, dtype=str, keep_default_na=False)
    return df.head(limit) if limit else df


def run_retrieval(args):
    from . import qa
    kb = qa.get_kb()
    df = load_questions(args.limit)
    rows, scores = [], []
    for _, r in df.iterrows():
        text = r["asr_text"] if args.asr else r["question"]
        q = qa.search_query(text, args.rewrite)
        chunks = kb.search(q, k=args.k)
        prods = kb.products_for(text)
        src = [c["id"] for c in chunks] + [f'products.csv:{p["sku"]}' for p in prods]
        rec = retrieval_scores(r, src)
        if rec is not None:
            scores.append(rec)
        rows.append({"id": r["id"], "gold": r["gold_evidence"], "retrieved": " ; ".join(src),
                     "recall": rec})
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, "retrieval_results.csv"), index=False)
    bad = out[(out["recall"].notna()) & (out["recall"] < 1)]
    print(f"retrieval recall@{args.k} (incl. linked sections): "
          f"{sum(scores) / max(len(scores), 1):.3f} over {len(scores)} questions "
          f"(rewrite={'on' if args.rewrite else 'off'})")
    for _, b in bad.iterrows():
        print(f'  MISS {b["id"]}: gold=[{b["gold"]}] got=[{b["retrieved"]}]')


def run_answers(args):
    from . import qa
    df = load_questions(args.limit)
    fn = qa.baseline_answer if args.mode == "step1" else (
        lambda q: qa.rag_answer(q, k=args.k, use_rewrite=not args.no_rewrite))
    rows = []
    for n, (_, r) in enumerate(df.iterrows(), 1):
        text = r["asr_text"] if args.asr else r["question"]
        res = fn(text)
        miss = missing_facts(res["reply"], r["must_include"])
        bad = forbidden_hits(res["reply"], r["must_not_include"])
        dec_ok = res["decision"] in acceptable(r)
        passed = (not miss) and (not bad) and dec_ok
        rec = retrieval_scores(r, res["sources"]) if args.mode == "step2" else None
        rows.append({
            "id": r["id"], "intent": r["intent"], "language": r["language"],
            "question": text, "decision": res["decision"],
            "expected": r["expected_decision"], "reply": res["reply"],
            "sources": " ; ".join(res["sources"]), "missing_facts": " | ".join(miss),
            "forbidden": " | ".join(bad), "decision_ok": dec_ok, "passed": passed,
            "retrieval_recall": rec,
        })
        print(f"[{n}/{len(df)}] {r['id']} {'PASS' if passed else 'FAIL'}"
              f"{'' if passed else '  missing=' + str(miss) + ' forbidden=' + str(bad) + ' dec=' + res['decision']}")
        time.sleep(args.sleep)
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_asr" if args.asr else ""
    out.to_csv(os.path.join(OUT_DIR, f"{args.mode}_results{suffix}.csv"), index=False)

    n = len(out)
    summary = {
        "n": n,
        "passed": int(out["passed"].sum()),
        "pass_rate": round(float(out["passed"].mean()), 3),
        "facts_ok": int(((out["missing_facts"] == "") & (out["forbidden"] == "")).sum()),
        "decision_ok": int(out["decision_ok"].sum()),
    }
    if args.mode == "step2":
        rec = pd.to_numeric(out["retrieval_recall"], errors="coerce").dropna()
        summary["retrieval_recall"] = round(float(rec.mean()), 3) if len(rec) else None
    print("\nSUMMARY", json.dumps(summary))
    print(out.groupby("intent")["passed"].agg(["sum", "count"]).to_string())

    path = os.path.join(OUT_DIR, "scores.json")
    scores = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    scores[args.mode + suffix] = summary
    json.dump(scores, open(path, "w", encoding="utf-8"), indent=2)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retrieval", "step1", "step2"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--asr", action="store_true", help="use asr_text instead of question")
    ap.add_argument("--rewrite", action="store_true", help="retrieval mode: use LLM rewrite")
    ap.add_argument("--no-rewrite", action="store_true", help="step2: skip LLM query rewrite")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()
    if args.mode == "retrieval":
        run_retrieval(args)
    else:
        run_answers(args)


if __name__ == "__main__":
    main()