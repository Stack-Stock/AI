# step1f_rebuild_company_candidates.py

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

# =========================
# 0. 경로
# =========================
PROJECT_ROOT = Path(r"C:\Users\SSAFY\Desktop\seohyung\Stack&Stock\projects\article_embedding_V2.0")
OUT_DIR = PROJECT_ROOT / "outputs_step1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIGH_JSONL = OUT_DIR / "match_candidates_high_confidence.jsonl"
REVIEW_JSONL = OUT_DIR / "match_candidates_review_needed.jsonl"

OUT_ALL = PROJECT_ROOT / "company_candidates_dict_all_v2.csv"
OUT_TOP = PROJECT_ROOT / "company_candidates_dict_top_by_industry_v2.csv"

TOP_N_PER_INDUSTRY = 10

# =========================
# 1. 유틸
# =========================
def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except:
        return default

def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except:
        return default

# =========================
# 2. 기사 1건 -> 회사 점수 반영
# =========================
def update_company_stats(stats: dict, row: dict, source_type: str):
    company = row.get("target_company")
    industry = row.get("final_industry_for_matching", "UNKNOWN")

    if not company or industry == "UNKNOWN":
        return

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

            "candidate_quality_score_sum": 0.0,
            "article_signal_strength_sum": 0.0,
            "company_match_score_sum": 0.0,
            "cause_keyword_score_sum": 0.0,

            "month_set": set(),
            "article_id_set": set(),
        }

    s = stats[key]

    article_id = row.get("article_id")
    if article_id in s["article_id_set"]:
        return

    s["article_id_set"].add(article_id)
    s["article_count_total"] += 1

    if source_type == "high":
        s["article_count_high"] += 1
    elif source_type == "review":
        s["article_count_review"] += 1

    article_type = row.get("article_type")
    if article_type == "direct_company_news":
        s["direct_company_news_count"] += 1
    elif article_type == "macro_policy_news":
        s["macro_policy_news_count"] += 1
    elif article_type == "industry_news":
        s["industry_news_count"] += 1

    if bool(row.get("is_company_direct", False)):
        s["company_direct_count"] += 1

    if safe_float(row.get("article_signal_strength"), 0.0) >= 6.0:
        s["high_signal_count"] += 1

    s["candidate_quality_score_sum"] += safe_float(row.get("candidate_quality_score"), 0.0)
    s["article_signal_strength_sum"] += safe_float(row.get("article_signal_strength"), 0.0)
    s["company_match_score_sum"] += safe_float(row.get("company_match_score"), 0.0)
    s["cause_keyword_score_sum"] += safe_float(row.get("cause_keyword_score"), 0.0)

    published_date = str(row.get("published_date", "")).strip()
    if len(published_date) >= 7:
        s["month_set"].add(published_date[:7])

# =========================
# 3. 최종 점수 계산
# =========================
def finalize_stats(stats: dict):
    rows = []

    for _, s in stats.items():
        n = s["article_count_total"]
        if n == 0:
            continue

        month_count = len(s["month_set"])

        avg_candidate_quality_score = s["candidate_quality_score_sum"] / n
        avg_article_signal_strength = s["article_signal_strength_sum"] / n
        avg_company_match_score = s["company_match_score_sum"] / n
        avg_cause_keyword_score = s["cause_keyword_score_sum"] / n

        source_score = (
            s["article_count_high"] * 3.0
            + s["article_count_review"] * 1.2
            + s["direct_company_news_count"] * 2.5
            + s["macro_policy_news_count"] * 1.5
            + s["industry_news_count"] * 1.0
            + s["company_direct_count"] * 2.0
            + s["high_signal_count"] * 1.5
            + month_count * 1.2
            + avg_candidate_quality_score * 1.0
            + avg_article_signal_strength * 0.8
            + avg_company_match_score * 0.7
            + avg_cause_keyword_score * 0.5
        )

        row = {
            "company_candidate": s["company_candidate"],
            "top_industry": s["top_industry"],

            "article_count_total": s["article_count_total"],
            "article_count_high": s["article_count_high"],
            "article_count_review": s["article_count_review"],

            "direct_company_news_count": s["direct_company_news_count"],
            "macro_policy_news_count": s["macro_policy_news_count"],
            "industry_news_count": s["industry_news_count"],

            "company_direct_count": s["company_direct_count"],
            "high_signal_count": s["high_signal_count"],

            "avg_candidate_quality_score": round(avg_candidate_quality_score, 4),
            "avg_article_signal_strength": round(avg_article_signal_strength, 4),
            "avg_company_match_score": round(avg_company_match_score, 4),
            "avg_cause_keyword_score": round(avg_cause_keyword_score, 4),

            "month_count": month_count,
            "source_score": round(source_score, 4),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # 산업 내 rank / percentile
    df = df.sort_values(["top_industry", "source_score"], ascending=[True, False]).reset_index(drop=True)

    df["industry_rank"] = (
        df.groupby("top_industry")["source_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    def pct_rank_desc(s):
        return s.rank(method="average", pct=True)

    # 점수가 높을수록 1에 가깝게
    df["industry_percentile"] = (
        df.groupby("top_industry")["source_score"]
        .transform(pct_rank_desc)
        .round(4)
    )

    # 최종 후보 점수
    df["final_candidate_score"] = (
        df["source_score"] * 0.8
        + df["industry_percentile"] * 20
    ).round(4)

    df = df.sort_values(
        ["top_industry", "final_candidate_score", "article_count_high", "month_count"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)

    return df

# =========================
# 4. 메인
# =========================
def main():
    high_rows = load_jsonl(HIGH_JSONL)
    review_rows = load_jsonl(REVIEW_JSONL)

    print(f"[info] high_rows: {len(high_rows)}")
    print(f"[info] review_rows: {len(review_rows)}")

    stats = {}

    for row in high_rows:
        update_company_stats(stats, row, source_type="high")

    for row in review_rows:
        update_company_stats(stats, row, source_type="review")

    df_all = finalize_stats(stats)

    if df_all.empty:
        print("[warn] no company candidates generated")
        return

    df_all.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

    df_top = (
        df_all.sort_values(
            ["top_industry", "final_candidate_score", "article_count_high", "month_count"],
            ascending=[True, False, False, False]
        )
        .groupby("top_industry", group_keys=False)
        .head(TOP_N_PER_INDUSTRY)
        .reset_index(drop=True)
    )

    df_top.to_csv(OUT_TOP, index=False, encoding="utf-8-sig")

    print("\n=== STEP1-F DONE ===")
    print("all_candidates:", len(df_all))
    print("top_candidates:", len(df_top))
    print("industries:", df_all["top_industry"].nunique())
    print("saved:", OUT_ALL)
    print("saved:", OUT_TOP)

    print("\n[top 3 by industry preview]")
    for industry, g in df_top.groupby("top_industry"):
        print(f"\n[{industry}]")
        preview = g.head(3)[
            ["company_candidate", "final_candidate_score", "article_count_high", "month_count"]
        ]
        print(preview.to_string(index=False))

if __name__ == "__main__":
    main()