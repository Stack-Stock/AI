import pandas as pd
from pathlib import Path

# =========================
# 경로
# =========================
STEP2_DIR = Path("./outputs_step2")

EVENTS_FILE = STEP2_DIR / "stock_events_refined.csv"
STRICT_MATCH_FILE = STEP2_DIR / "matched_event_articles_top1_strict.csv"

OUT_MAIN = STEP2_DIR / "case_dataset_main.csv"
OUT_REVIEW = STEP2_DIR / "case_dataset_review.csv"

# =========================
# 설정
# =========================
MAIN_MIN_MATCH_SCORE = 90

WEAK_TITLE_PATTERNS = [
    "지속가능경영보고서",
    "보고서 발간",
    "코스피",
    "금융 투자법",
    "들썩",
    "강세",
    "수혜주",
]

# =========================
# util
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

def contains_weak_title(title: str) -> bool:
    title = safe_str(title)
    return any(p in title for p in WEAK_TITLE_PATTERNS)

def return_to_pct(x: float) -> float:
    return round(x * 100, 2)

def get_direction_text(event_type: str, real_return: float) -> str:
    event_type = safe_str(event_type)
    if event_type in ("SPIKE_UP_1D", "STREAK_UP3"):
        return "상승"
    if event_type in ("SPIKE_DOWN_1D", "STREAK_DOWN3"):
        return "하락"
    if event_type == "VOLUME_SURGE":
        if real_return > 0:
            return "상승"
        elif real_return < 0:
            return "하락"
        return "변동"
    return "변동"

def clean_title_for_case(title: str, company: str) -> str:
    title = safe_str(title)
    company = safe_str(company)

    # 너무 긴 제목 축약
    if len(title) > 48:
        title = title[:48].rstrip() + "..."
    return f"{company} | {title}"

def build_event_summary(event_date, company: str, event_type: str, real_return: float, volume_ratio: float) -> str:
    date_text = str(event_date)
    direction = get_direction_text(event_type, real_return)
    pct = abs(return_to_pct(real_return))

    if safe_str(event_type) == "VOLUME_SURGE":
        return f"{date_text} {company}는 거래량 급증과 함께 주가가 {pct}% {direction}했다."
    return f"{date_text} {company} 주가는 {pct}% {direction}했다."

def build_reason(company: str, article_title: str, event_type: str, real_return: float) -> str:
    direction = get_direction_text(event_type, real_return)
    article_title = safe_str(article_title)

    if "실적" in article_title or "영업이익" in article_title or "매출" in article_title:
        return f"{article_title} 재료가 부각되며 {company} 주가가 {direction}한 것으로 해석된다."
    if "목표가" in article_title:
        return f"증권가 평가 변화가 반영되며 {company} 주가가 {direction}한 것으로 해석된다."
    if "관세" in article_title or "정책" in article_title:
        return f"정책·규제 관련 기대 또는 우려가 반영되며 {company} 주가가 {direction}한 것으로 해석된다."
    if "수주" in article_title or "계약" in article_title:
        return f"수주·계약 관련 재료가 부각되며 {company} 주가가 {direction}한 것으로 해석된다."
    if "임상" in article_title or "신약" in article_title:
        return f"신약·임상 관련 재료가 반영되며 {company} 주가가 {direction}한 것으로 해석된다."
    if "AI" in article_title or "반도체" in article_title:
        return f"AI·반도체 관련 기대감 또는 우려가 반영되며 {company} 주가가 {direction}한 것으로 해석된다."

    return f"{article_title} 관련 재료가 부각되며 {company} 주가가 {direction}한 것으로 해석된다."

# =========================
# 1. 데이터 로드
# =========================
events = pd.read_csv(EVENTS_FILE)
matches = pd.read_csv(STRICT_MATCH_FILE)

events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.date
matches["event_date"] = pd.to_datetime(matches["event_date"], errors="coerce").dt.date
matches["article_date"] = pd.to_datetime(matches["article_date"], errors="coerce").dt.date

print(f"[0] refined events: {len(events)}")
print(f"[1] strict matches: {len(matches)}")

# =========================
# 2. merge
# =========================
df = events.merge(
    matches,
    on=["event_id", "company", "ticker", "industry", "event_date", "event_type"],
    how="inner",
    suffixes=("_event", "_match")
).copy()

print(f"[2] merged rows: {len(df)}")

# =========================
# 3. 컬럼 정리
# =========================
if "return_1d" not in df.columns:
    df["return_1d"] = 0.0
if "close_price" not in df.columns:
    df["close_price"] = 0.0
if "volume_ratio" not in df.columns:
    df["volume_ratio"] = 0.0
if "match_score" not in df.columns:
    df["match_score"] = 0.0
if "target_match" not in df.columns:
    df["target_match"] = 0
if "title_match" not in df.columns:
    df["title_match"] = 0

# 숫자 정리
num_cols = ["return_1d", "close_price", "volume_ratio", "event_strength", "match_score"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

for col in ["target_match", "title_match", "candidate_match", "industry_match"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# =========================
# 4. 케이스 텍스트 생성
# =========================
df["real_return"] = df["return_1d"]
df["real_return_pct"] = df["real_return"].apply(return_to_pct)

df["case_title"] = df.apply(
    lambda x: clean_title_for_case(x["article_title"], x["company"]),
    axis=1
)

df["event_summary"] = df.apply(
    lambda x: build_event_summary(
        x["event_date"],
        x["company"],
        x["event_type"],
        safe_float(x["real_return"]),
        safe_float(x["volume_ratio"])
    ),
    axis=1
)

df["reason"] = df.apply(
    lambda x: build_reason(
        x["company"],
        x["article_title"],
        x["event_type"],
        safe_float(x["real_return"])
    ),
    axis=1
)

# =========================
# 5. main / review 분리
# =========================
main_mask = (
    (df["match_score"] >= MAIN_MIN_MATCH_SCORE) &
    ((df["target_match"] == 1) | (df["title_match"] == 1)) &
    (~df["article_title"].apply(contains_weak_title))
)

main_df = df[main_mask].copy().reset_index(drop=True)
review_df = df[~main_mask].copy().reset_index(drop=True)

print(f"[3] main cases: {len(main_df)}")
print(f"[4] review cases: {len(review_df)}")

# =========================
# 6. case_id 부여
# =========================
main_df = main_df.sort_values(["event_date", "company", "match_score"], ascending=[True, True, False]).reset_index(drop=True)
review_df = review_df.sort_values(["event_date", "company", "match_score"], ascending=[True, True, False]).reset_index(drop=True)

main_df["case_id"] = [f"CASE_{i:05d}" for i in range(1, len(main_df) + 1)]
review_df["case_id"] = [f"REVIEW_CASE_{i:05d}" for i in range(1, len(review_df) + 1)]

# =========================
# 7. 최종 컬럼
# =========================
main_cols = [
    "case_id",
    "event_id",
    "company",
    "ticker",
    "industry",
    "event_date",
    "event_type",
    "real_return",
    "real_return_pct",
    "close_price",
    "volume_ratio",
    "event_strength",
    "article_id",
    "article_date",
    "article_title",
    "article_url",
    "case_title",
    "event_summary",
    "reason",
]

review_cols = main_cols + [
    "article_pool",
    "article_industry",
    "target_company",
    "company_candidates",
    "target_match",
    "title_match",
    "candidate_match",
    "industry_match",
    "match_score",
]

main_df = main_df[main_cols].copy()
review_df = review_df[review_cols].copy()

# =========================
# 8. 저장
# =========================
main_df.to_csv(OUT_MAIN, index=False, encoding="utf-8-sig")
review_df.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

print("saved:", OUT_MAIN)
print("saved:", OUT_REVIEW)
print()
print("[main industry distribution]")
print(main_df["industry"].value_counts())