import json
from pathlib import Path

import pandas as pd

# =========================
# 0. 경로 / 설정
# =========================
INPUT_CAND = Path("./outputs_step3/event_article_candidates_topk.csv")

OUT_DIR = Path("./outputs_step3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUT_DIR / "event_article_reranked_all.csv"
OUT_TOP3 = OUT_DIR / "event_article_top3.csv"
OUT_TOP1 = OUT_DIR / "event_article_top1.csv"
OUT_TOP1_STRICT = OUT_DIR / "event_article_top1_strict.csv"
OUT_TOP1_GOLD = OUT_DIR / "event_article_top1_gold.csv"
OUT_TOP1_SILVER = OUT_DIR / "event_article_top1_silver.csv"

TOP_N_FINAL = 3

# =========================
# 1. 점수 설정
# =========================
W_EMBED = 100.0
W_TARGET_MATCH = 35.0
W_TITLE_MATCH = 25.0
W_CANDIDATE_MATCH = 12.0
W_INDUSTRY_MATCH = 10.0
W_COMPANY_SPECIFIC = 12.0
W_DIRECT_COMPANY_NEWS = 10.0
W_HIGH_CONFIDENCE = 5.0

P_GENERIC_TITLE = 25.0
P_SECTOR_WIDE = 8.0
P_MARKET_WIDE = 15.0
P_OTHER_TARGET_COMPANY = 45.0
P_MACRO_POLICY = 6.0
P_NON_STOCK_CONTEXT = 30.0

TITLE_PATTERNS_STRONG = [
    "MK시그널",
    "골든크로스",
    "매도신호",
    "매수신호",
    "수익률",
    "HOT종목",
    "특징주",
    "관련주",
    "마감 시황",
    "주식 초고수",
    "상승 종목",
    "하락 종목",
    "인기검색TOP5",
    "인기검색",
    "매니저의 HOT종목",
]

TITLE_PATTERNS_WEAK = [
    "매일경제TV",
    "총수익률",
    "1위",
]

NON_STOCK_CONTEXT_PATTERNS = [
    "프로기전",
    "바둑",
    "기전",
    "대국",
    "결승",
    "우승",
    "준우승",
]

# =========================
# 2. 유틸
# =========================
def safe_str(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def contains_any(text: str, patterns) -> bool:
    t = safe_str(text)
    return any(p in t for p in patterns)


def calc_date_score(abs_day_diff: int) -> float:
    d = safe_int(abs_day_diff, 9999)
    if d <= 1:
        return 15.0
    if d <= 3:
        return 10.0
    if d <= 5:
        return 5.0
    if d <= 7:
        return 0.0
    if d <= 10:
        return -8.0
    if d <= 14:
        return -18.0
    return -35.0


def calc_scope_score(market_scope: str) -> float:
    scope = safe_str(market_scope)
    if scope == "company_specific":
        return W_COMPANY_SPECIFIC
    if scope == "sector_wide":
        return -P_SECTOR_WIDE
    if scope == "market_wide":
        return -P_MARKET_WIDE
    return 0.0


def calc_article_type_score(article_type: str) -> float:
    at = safe_str(article_type)
    if at == "direct_company_news":
        return W_DIRECT_COMPANY_NEWS
    if at == "macro_policy_news":
        return -P_MACRO_POLICY
    return 0.0


def calc_source_group_score(source_group: str) -> float:
    sg = safe_str(source_group)
    if sg == "high_confidence":
        return W_HIGH_CONFIDENCE
    return 0.0


def calc_title_penalty(title: str, is_generic_title: int) -> float:
    penalty = 0.0
    if safe_int(is_generic_title) == 1:
        penalty += P_GENERIC_TITLE
    if contains_any(title, TITLE_PATTERNS_STRONG):
        penalty += 20.0
    if contains_any(title, TITLE_PATTERNS_WEAK):
        penalty += 10.0
    return penalty


def calc_other_target_company_penalty(event_company: str, target_company: str) -> float:
    event_company = safe_str(event_company)
    target_company = safe_str(target_company)
    if target_company and target_company != event_company:
        return P_OTHER_TARGET_COMPANY
    return 0.0


def calc_non_stock_context_penalty(title: str, article_match_text: str) -> float:
    title = safe_str(title)
    body_text = safe_str(article_match_text)
    text = f"{title} {body_text}"
    if contains_any(text, NON_STOCK_CONTEXT_PATTERNS):
        return P_NON_STOCK_CONTEXT
    return 0.0


def build_reason(row) -> str:
    reasons = []

    if safe_int(row.get("target_match")) == 1:
        reasons.append("target_match")
    if safe_int(row.get("title_match")) == 1:
        reasons.append("title_match")
    if safe_int(row.get("candidate_match")) == 1:
        reasons.append("candidate_match")
    if safe_int(row.get("industry_match")) == 1:
        reasons.append("industry_match")

    ms = safe_str(row.get("market_scope"))
    if ms:
        reasons.append(f"scope={ms}")

    at = safe_str(row.get("article_type"))
    if at:
        reasons.append(f"type={at}")

    reasons.append(f"abs_day_diff={safe_int(row.get('abs_day_diff'))}")
    reasons.append(f"embed={safe_float(row.get('embedding_similarity')):.4f}")

    return ", ".join(reasons)


# =========================
# 3. 로드
# =========================
print("[LOAD] reading candidate file...")
df = pd.read_csv(INPUT_CAND)

print(f"[INFO] candidate rows: {len(df)}")

if df.empty:
    print("[WARN] input is empty")
    for out in [OUT_ALL, OUT_TOP3, OUT_TOP1, OUT_TOP1_STRICT, OUT_TOP1_GOLD, OUT_TOP1_SILVER]:
        pd.DataFrame().to_csv(out, index=False, encoding="utf-8-sig")
        print(f"saved: {out}")
    raise SystemExit(0)

# 수치형 보정
for col in [
    "embedding_similarity",
    "return_1d",
    "volume_ratio",
    "event_strength",
    "article_signal_strength",
    "candidate_quality_score",
    "day_diff",
    "abs_day_diff",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

for col in [
    "candidate_rank_by_embedding",
    "title_match",
    "target_match",
    "candidate_match",
    "industry_match",
    "is_generic_title",
    "is_within_preferred_window",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# =========================
# 4. 점수 계산
# =========================
print("[RERANK] calculating scores...")

df["score_embed"] = df["embedding_similarity"] * W_EMBED
df["score_target_match"] = df["target_match"] * W_TARGET_MATCH
df["score_title_match"] = df["title_match"] * W_TITLE_MATCH
df["score_candidate_match"] = df["candidate_match"] * W_CANDIDATE_MATCH
df["score_industry_match"] = df["industry_match"] * W_INDUSTRY_MATCH

df["score_scope"] = df["market_scope"].apply(calc_scope_score)
df["score_article_type"] = df["article_type"].apply(calc_article_type_score)
df["score_source_group"] = df["source_group"].apply(calc_source_group_score)
df["score_date"] = df["abs_day_diff"].apply(calc_date_score)

df["penalty_title"] = df.apply(
    lambda r: calc_title_penalty(r.get("title"), r.get("is_generic_title")),
    axis=1,
)

df["penalty_other_target"] = df.apply(
    lambda r: calc_other_target_company_penalty(r.get("company"), r.get("target_company")),
    axis=1,
)

df["penalty_non_stock_context"] = df.apply(
    lambda r: calc_non_stock_context_penalty(r.get("title"), r.get("article_match_text")),
    axis=1,
)

df["score_article_signal"] = df["article_signal_strength"].clip(lower=0, upper=15) * 0.8
df["score_candidate_quality"] = df["candidate_quality_score"].clip(lower=0, upper=15) * 0.5

df["final_score"] = (
    df["score_embed"]
    + df["score_target_match"]
    + df["score_title_match"]
    + df["score_candidate_match"]
    + df["score_industry_match"]
    + df["score_scope"]
    + df["score_article_type"]
    + df["score_source_group"]
    + df["score_date"]
    + df["score_article_signal"]
    + df["score_candidate_quality"]
    - df["penalty_title"]
    - df["penalty_other_target"]
    - df["penalty_non_stock_context"]
)

df["rerank_reason"] = df.apply(build_reason, axis=1)

# =========================
# 5. 정렬 / 전체 저장
# =========================
sort_cols = [
    "event_id",
    "final_score",
    "target_match",
    "title_match",
    "candidate_match",
    "industry_match",
    "embedding_similarity",
]
ascending = [True, False, False, False, False, False, False]

df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
df["candidate_rank_final"] = df.groupby("event_id").cumcount() + 1

df.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
print(f"saved: {OUT_ALL}")

# =========================
# 6. top3 / top1
# =========================
top3_df = df[df["candidate_rank_final"] <= TOP_N_FINAL].copy()
top3_df.to_csv(OUT_TOP3, index=False, encoding="utf-8-sig")
print(f"saved: {OUT_TOP3}")

top1_df = df.groupby("event_id", as_index=False).head(1).copy()
top1_df.to_csv(OUT_TOP1, index=False, encoding="utf-8-sig")
print(f"saved: {OUT_TOP1}")

# =========================
# 7. strict / gold / silver
# =========================
strict_df = top1_df[
    (top1_df["final_score"] >= 120)
    & (top1_df["target_match"] == 1)
    & (top1_df["candidate_match"] == 1)
    & (top1_df["market_scope"] == "company_specific")
    & (top1_df["article_type"] == "direct_company_news")
    & (top1_df["abs_day_diff"] <= 7)
].copy()

gold_df = top1_df[
    (top1_df["final_score"] >= 140)
    & (top1_df["target_match"] == 1)
    & (top1_df["title_match"] == 1)
    & (top1_df["candidate_match"] == 1)
    & (top1_df["market_scope"] == "company_specific")
    & (top1_df["article_type"] == "direct_company_news")
    & (top1_df["abs_day_diff"] <= 5)
].copy()

silver_df = top1_df[
    (top1_df["final_score"] >= 110)
    & (top1_df["target_match"] == 1)
    & (top1_df["candidate_match"] == 1)
    & (top1_df["abs_day_diff"] <= 10)
].copy()

strict_df.to_csv(OUT_TOP1_STRICT, index=False, encoding="utf-8-sig")
gold_df.to_csv(OUT_TOP1_GOLD, index=False, encoding="utf-8-sig")
silver_df.to_csv(OUT_TOP1_SILVER, index=False, encoding="utf-8-sig")

print(f"saved: {OUT_TOP1_STRICT}")
print(f"saved: {OUT_TOP1_GOLD}")
print(f"saved: {OUT_TOP1_SILVER}")

# =========================
# 8. 로그
# =========================
event_count = df["event_id"].nunique()

print("\n=== DONE ===")
print(f"candidate_rows: {len(df)}")
print(f"unique_events: {event_count}")
print(f"top3_rows: {len(top3_df)}")
print(f"top1_rows: {len(top1_df)}")
print(f"strict_top1_rows: {len(strict_df)}")
print(f"gold_rows: {len(gold_df)}")
print(f"silver_rows: {len(silver_df)}")

print("\n[Sample gold]")
if len(gold_df):
    sample_cols = [
        "event_id",
        "company",
        "event_date",
        "article_id",
        "title",
        "final_score",
        "abs_day_diff",
        "rerank_reason",
    ]
    print(gold_df[sample_cols].head(15).to_string(index=False))
else:
    print("no gold rows")

print("\n[Sample silver]")
if len(silver_df):
    sample_cols = [
        "event_id",
        "company",
        "event_date",
        "article_id",
        "title",
        "final_score",
        "abs_day_diff",
        "rerank_reason",
    ]
    print(silver_df[sample_cols].head(15).to_string(index=False))
else:
    print("no silver rows")