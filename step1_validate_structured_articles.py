# step1_validate_structured_articles.py

import json
from collections import Counter
from pathlib import Path

INPUT_JSONL = Path("./outputs_step1/structured_articles_step1.jsonl")

REQUIRED_FIELDS = [
    "article_id",
    "title",
    "body",
    "published_date",
    "industries",
    "top_industry",
    "article_type",
    "is_company_direct",
    "company_match_score",
    "company_centrality_score",
    "company_mention_zone",
    "cause_keywords",
    "cause_event_type_candidates",
    "market_scope",
    "article_signal_strength",
    "is_noise_article",
    "is_price_recap_article",
    "article_match_text",
]

def main():
    total = 0
    missing_counter = Counter()
    article_type_counter = Counter()
    bad_rows = []

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            row = json.loads(line)

            for field in REQUIRED_FIELDS:
                if field not in row:
                    missing_counter[field] += 1

            article_type_counter[row.get("article_type", "UNKNOWN")] += 1

            if row.get("top_industry") == "UNKNOWN":
                bad_rows.append(("UNKNOWN_INDUSTRY", row.get("article_id"), row.get("title", "")[:60]))

            if not row.get("article_match_text"):
                bad_rows.append(("EMPTY_MATCH_TEXT", row.get("article_id"), row.get("title", "")[:60]))

            if row.get("is_company_direct") and row.get("company_centrality_score", 0) < 4:
                bad_rows.append(("DIRECT_BUT_LOW_CENTRALITY", row.get("article_id"), row.get("title", "")[:60]))

    print("=== VALIDATION RESULT ===")
    print("total_rows:", total)
    print("missing_counter:", dict(missing_counter))
    print("article_type_counter:", dict(article_type_counter))
    print("bad_row_samples:", bad_rows[:20])

if __name__ == "__main__":
    main()