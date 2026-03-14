import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

# =========================
# 경로
# =========================

PROJECT_ROOT = Path(r"C:\Users\SSAFY\Desktop\seohyung\Stack&Stock\projects\article_embedding_V2.0")
OUT_DIR = PROJECT_ROOT / "outputs_step1"

HIGH_JSONL = OUT_DIR / "match_candidates_high_confidence.jsonl"
REVIEW_JSONL = OUT_DIR / "match_candidates_review_needed.jsonl"

COMPANY_DICT = PROJECT_ROOT / "company_dictionary.csv"

OUT_ALL = PROJECT_ROOT / "company_candidates_dict_all_v3.csv"
OUT_TOP = PROJECT_ROOT / "company_candidates_dict_top_by_industry_v3.csv"

TOP_N = 10


# =========================
# company → industry
# =========================

def load_company_industry_map():

    df = pd.read_csv(COMPANY_DICT)

    mapping = {}

    for _, r in df.iterrows():

        name = str(r["company_name"]).strip()
        industry = str(r["industry"]).strip()

        mapping[name] = industry

    return mapping


# =========================
# jsonl loader
# =========================

def load_jsonl(path):

    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()
            if not line:
                continue

            rows.append(json.loads(line))

    return rows


# =========================
# stats update
# =========================

def update_stats(stats, row, source, company_industry):

    company = row.get("target_company")

    if company not in company_industry:
        return

    industry = company_industry[company]

    key = (company, industry)

    if key not in stats:

        stats[key] = {
            "company_candidate": company,
            "top_industry": industry,

            "article_count_total": 0,
            "article_count_high": 0,
            "article_count_review": 0,

            "direct_company_news_count": 0,
            "macro_policy_news_count": 0,
            "industry_news_count": 0,

            "company_direct_count": 0,
            "high_signal_count": 0,

            "candidate_quality_score_sum": 0,
            "article_signal_strength_sum": 0,
            "company_match_score_sum": 0,
            "cause_keyword_score_sum": 0,

            "months": set(),
        }

    s = stats[key]

    s["article_count_total"] += 1

    if source == "high":
        s["article_count_high"] += 1
    else:
        s["article_count_review"] += 1

    atype = row.get("article_type")

    if atype == "direct_company_news":
        s["direct_company_news_count"] += 1
    elif atype == "macro_policy_news":
        s["macro_policy_news_count"] += 1
    elif atype == "industry_news":
        s["industry_news_count"] += 1

    if row.get("is_company_direct"):
        s["company_direct_count"] += 1

    if float(row.get("article_signal_strength", 0)) >= 6:
        s["high_signal_count"] += 1

    s["candidate_quality_score_sum"] += float(row.get("candidate_quality_score", 0))
    s["article_signal_strength_sum"] += float(row.get("article_signal_strength", 0))
    s["company_match_score_sum"] += float(row.get("company_match_score", 0))
    s["cause_keyword_score_sum"] += float(row.get("cause_keyword_score", 0))

    date = str(row.get("published_date", ""))

    if len(date) >= 7:
        s["months"].add(date[:7])


# =========================
# finalize
# =========================

def finalize(stats):

    rows = []

    for _, s in stats.items():

        n = s["article_count_total"]

        month_count = len(s["months"])

        avg_q = s["candidate_quality_score_sum"] / n
        avg_sig = s["article_signal_strength_sum"] / n
        avg_match = s["company_match_score_sum"] / n
        avg_cause = s["cause_keyword_score_sum"] / n

        score = (
            s["article_count_high"] * 3.0
            + s["article_count_review"] * 1.2
            + s["direct_company_news_count"] * 2.5
            + s["macro_policy_news_count"] * 1.5
            + s["industry_news_count"] * 1.0
            + s["company_direct_count"] * 2.0
            + s["high_signal_count"] * 1.5
            + month_count * 1.2
            + avg_q
            + avg_sig * 0.8
            + avg_match * 0.7
            + avg_cause * 0.5
        )

        rows.append({
            "company_candidate": s["company_candidate"],
            "top_industry": s["top_industry"],
            "article_count_total": n,
            "article_count_high": s["article_count_high"],
            "article_count_review": s["article_count_review"],
            "month_count": month_count,
            "source_score": score
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["top_industry", "source_score"],
        ascending=[True, False]
    )

    df["industry_rank"] = (
        df.groupby("top_industry")["source_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    return df


# =========================
# main
# =========================

def main():

    company_industry = load_company_industry_map()

    high = load_jsonl(HIGH_JSONL)
    review = load_jsonl(REVIEW_JSONL)

    stats = {}

    for r in high:
        update_stats(stats, r, "high", company_industry)

    for r in review:
        update_stats(stats, r, "review", company_industry)

    df = finalize(stats)

    df.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

    df_top = (
        df.groupby("top_industry")
        .head(TOP_N)
        .reset_index(drop=True)
    )

    df_top.to_csv(OUT_TOP, index=False, encoding="utf-8-sig")

    print("=== STEP1-F v2 DONE ===")
    print("companies:", len(df))
    print("industries:", df["top_industry"].nunique())


if __name__ == "__main__":
    main()