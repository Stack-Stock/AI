# step1e_build_match_candidate_sets.py

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\SSAFY\Desktop\seohyung\Stack&Stock\projects\article_embedding_V2.0")
OUT_DIR = PROJECT_ROOT / "outputs_step1"

INPUT_JSONL = OUT_DIR / "structured_articles_step1d_plus.jsonl"

OUT_HIGH = OUT_DIR / "match_candidates_high_confidence.jsonl"
OUT_REVIEW = OUT_DIR / "match_candidates_review_needed.jsonl"
OUT_DISCARD = OUT_DIR / "discard_articles.jsonl"
OUT_SUMMARY = OUT_DIR / "step1e_match_candidate_summary.json"

def is_high_confidence(row: dict) -> bool:
    return bool(
        row.get("is_match_candidate", False)
        and row.get("final_industry_for_matching", "UNKNOWN") != "UNKNOWN"
        and row.get("article_type") in {"direct_company_news", "macro_policy_news", "industry_news"}
        and row.get("is_noise_article", False) is False
        and (
            row.get("is_company_direct", False)
            or row.get("company_mention_zone") in {"title", "lead"}
            or int(row.get("candidate_quality_score", 0)) >= 7
        )
    )

def is_review_needed(row: dict) -> bool:
    return bool(
        row.get("is_match_candidate", False)
        and not is_high_confidence(row)
        and row.get("unknown_bucket") == "recoverable_unknown"
    )

def is_discard(row: dict) -> bool:
    return bool(
        row.get("unknown_bucket") == "discard_unknown"
        or row.get("is_noise_article", False)
    )

def main():
    summary = {
        "total_rows": 0,
        "high_confidence_count": 0,
        "review_needed_count": 0,
        "discard_count": 0,
        "overlap_guard_fail": 0,
        "high_article_type_counter": Counter(),
        "review_reason_counter": Counter(),
        "high_industry_counter": Counter()
    }

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, \
         OUT_HIGH.open("w", encoding="utf-8") as f_high, \
         OUT_REVIEW.open("w", encoding="utf-8") as f_review, \
         OUT_DISCARD.open("w", encoding="utf-8") as f_discard:

        for line in fin:
            row = json.loads(line)
            summary["total_rows"] += 1

            high = is_high_confidence(row)
            review = is_review_needed(row)
            discard = is_discard(row)

            flags = sum([high, review, discard])
            if flags > 1:
                summary["overlap_guard_fail"] += 1

            if high:
                f_high.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["high_confidence_count"] += 1
                summary["high_article_type_counter"][row.get("article_type", "UNKNOWN")] += 1
                summary["high_industry_counter"][row.get("final_industry_for_matching", "UNKNOWN")] += 1
            elif review:
                f_review.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["review_needed_count"] += 1
                summary["review_reason_counter"][row.get("recoverable_reason", "UNKNOWN")] += 1
            elif discard:
                f_discard.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["discard_count"] += 1
            else:
                # 어디에도 안 들어간 애들은 일단 review로 보냄
                f_review.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["review_needed_count"] += 1
                summary["review_reason_counter"]["fallback_review_bucket"] += 1

    summary_dump = {
        **summary,
        "high_article_type_counter": dict(summary["high_article_type_counter"]),
        "review_reason_counter": dict(summary["review_reason_counter"]),
        "high_industry_counter": dict(summary["high_industry_counter"]),
    }

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary_dump, f, ensure_ascii=False, indent=2)

    print("=== STEP1-E DONE ===")
    print(json.dumps(summary_dump, ensure_ascii=False, indent=2))
    print("saved:", OUT_HIGH)
    print("saved:", OUT_REVIEW)
    print("saved:", OUT_DISCARD)
    print("saved:", OUT_SUMMARY)

if __name__ == "__main__":
    main()