import pandas as pd
from pathlib import Path

STEP3_DIR = Path("./outputs_step3")
STEP4_DIR = Path("./outputs_step4")
STEP5_DIR = Path("./outputs_step5")

STEP4_DIR.mkdir(exist_ok=True)
STEP5_DIR.mkdir(exist_ok=True)

INPUT_FILE = STEP3_DIR / "event_article_top1_gold.csv"
STOCK_UNIVERSE_FILE = STEP5_DIR / "stock_universe_top30.csv"
OUTPUT_FILE = STEP4_DIR / "case_selection_final.csv"

TARGET_TOTAL = 90
MAX_PER_COMPANY = 3
MIN_FINAL_SCORE = 120
MIN_EVENT_GAP = 7

UP_RATIO = 0.6
DOWN_RATIO = 0.4


def normalize_title(title):
    if pd.isna(title):
        return ""
    return " ".join(str(title).strip().split())


def normalize_text(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def normalize_ticker(v):
    if pd.isna(v):
        return ""
    return str(v).strip().upper()


def event_too_close(date, selected_dates):
    for d in selected_dates:
        if abs((date - d).days) < MIN_EVENT_GAP:
            return True
    return False


def ensure_column(df, col, default=None):
    if col not in df.columns:
        df[col] = default
    return df


print("[LOAD]")
df = pd.read_csv(INPUT_FILE)
stock_universe = pd.read_csv(STOCK_UNIVERSE_FILE)

print("event rows:", len(df))
print("stock universe rows:", len(stock_universe))

# =========================
# 0. stock universe 정규화
# =========================
stock_universe["company"] = stock_universe["company"].apply(normalize_text)
stock_universe["ticker"] = stock_universe["ticker"].apply(normalize_ticker)

stock_universe = stock_universe.drop_duplicates(subset=["company", "ticker"]).copy()

valid_pairs = set(
    zip(stock_universe["company"], stock_universe["ticker"])
)
valid_companies = set(stock_universe["company"])

print("valid company+ticker pairs:", len(valid_pairs))
print("valid companies:", len(valid_companies))

# =========================
# 1. 기본 전처리
# =========================
df["final_score"] = pd.to_numeric(df.get("final_score"), errors="coerce").fillna(0)
df["event_date"] = pd.to_datetime(df.get("event_date"), errors="coerce")

if "real_return" not in df.columns:
    if "return_1d" in df.columns:
        df["real_return"] = pd.to_numeric(df["return_1d"], errors="coerce").fillna(0)
    else:
        df["real_return"] = 0.0

if "article_title" not in df.columns:
    if "title" in df.columns:
        df["article_title"] = df["title"]
    else:
        df["article_title"] = ""

if "article_url" not in df.columns:
    if "url" in df.columns:
        df["article_url"] = df["url"]
    else:
        df["article_url"] = ""

if "article_published_at" not in df.columns:
    if "published_at" in df.columns:
        df["article_published_at"] = df["published_at"]
    else:
        df["article_published_at"] = ""

ensure_column(df, "industry", "")
ensure_column(df, "event_type", "")
ensure_column(df, "ticker", "")
ensure_column(df, "article_type", "")
ensure_column(df, "market_scope", "")
ensure_column(df, "body", "")
ensure_column(df, "volume_ratio", 0.0)
ensure_column(df, "event_strength", 0.0)
ensure_column(df, "article_match_text", "")
ensure_column(df, "rerank_reason", "")

if "article_body_summary" not in df.columns:
    if "body_summary" in df.columns:
        df["article_body_summary"] = df["body_summary"]
    else:
        df["article_body_summary"] = ""

if "cause_keywords" not in df.columns:
    df["cause_keywords"] = ""

if "target_match" not in df.columns:
    df["target_match"] = 1
if "title_match" not in df.columns:
    df["title_match"] = 1

ensure_column(df, "abs_day_diff", 999)
ensure_column(df, "embedding_similarity", 0.0)

# 문자열 정규화
df["company"] = df["company"].apply(normalize_text)
df["ticker"] = df["ticker"].apply(normalize_ticker)
df["article_title"] = df["article_title"].fillna("").astype(str)
df["article_url"] = df["article_url"].fillna("").astype(str)
df["article_published_at"] = df["article_published_at"].fillna("").astype(str)
df["industry"] = df["industry"].fillna("").astype(str)

# reason_seed 우선순위
if "reason_seed" not in df.columns:
    if "article_body_summary" in df.columns:
        df["reason_seed"] = df["article_body_summary"].fillna("")
    elif "body_summary" in df.columns:
        df["reason_seed"] = df["body_summary"].fillna("")
    elif "article_match_text" in df.columns:
        df["reason_seed"] = df["article_match_text"].fillna("")
    elif "body" in df.columns:
        df["reason_seed"] = df["body"].fillna("").astype(str).str[:500]
    elif "rerank_reason" in df.columns:
        df["reason_seed"] = df["rerank_reason"].fillna("")
    else:
        df["reason_seed"] = df["article_title"].fillna("")

# =========================
# 2. 품질 필터
# =========================
df = df[df["final_score"] >= MIN_FINAL_SCORE]
df = df[df["target_match"] == 1]
df = df[df["title_match"] == 1]
df = df.dropna(subset=["company", "event_date", "article_id"])

print("after quality filter:", len(df))

# =========================
# 3. stock_universe_top30 기준 필터
#    company+ticker exact match 우선
# =========================
pair_mask = df.apply(lambda r: (r["company"], r["ticker"]) in valid_pairs, axis=1)
company_only_mask = df["company"].isin(valid_companies)

matched_by_pair = df[pair_mask].copy()
fallback_company_only = df[(~pair_mask) & company_only_mask].copy()

print("matched by company+ticker:", len(matched_by_pair))
print("fallback matched by company only:", len(fallback_company_only))

# company+ticker exact match가 우선이고,
# ticker 누락/불일치 데이터가 있을 경우 company-only fallback도 허용
df = pd.concat([matched_by_pair, fallback_company_only], ignore_index=True)

# stock_universe 정보로 stock_id/industry 보정
df = df.merge(
    stock_universe[["stock_id", "company", "ticker", "industry"]].rename(
        columns={"industry": "industry_stock"}
    ),
    on=["company", "ticker"],
    how="left"
)

# pair 매칭 실패한 fallback row 처리:
# 같은 company가 stock_universe에 하나만 있는 경우 그 ticker / stock_id / industry를 자동 보정
company_unique_map = (
    stock_universe.groupby("company")
    .agg(
        universe_ticker=("ticker", lambda x: list(pd.unique(x))),
        universe_stock_id=("stock_id", lambda x: list(pd.unique(x))),
        universe_industry=("industry", lambda x: list(pd.unique(x))),
    )
    .reset_index()
)

df = df.merge(company_unique_map, on="company", how="left")

def fill_from_company_unique(row):
    if pd.notna(row.get("stock_id")):
        return row

    tickers = row.get("universe_ticker")
    stock_ids = row.get("universe_stock_id")
    industries = row.get("universe_industry")

    if isinstance(tickers, list) and len(tickers) == 1:
        row["ticker"] = tickers[0]
        row["stock_id"] = stock_ids[0] if isinstance(stock_ids, list) and len(stock_ids) == 1 else None
        if not row.get("industry"):
            row["industry"] = industries[0] if isinstance(industries, list) and len(industries) == 1 else row.get("industry", "")
    return row

df = df.apply(fill_from_company_unique, axis=1)

# industry 보정
df["industry"] = df["industry"].fillna("")
df["industry_stock"] = df["industry_stock"].fillna("")
df["industry"] = df.apply(
    lambda r: r["industry_stock"] if r["industry_stock"] else r["industry"],
    axis=1
)

# 최종적으로 stock_universe_top30에 매칭된 row만 유지
df = df[df["stock_id"].notna()].copy()

print("after stock universe filter:", len(df))
print("unique companies after stock universe filter:", df["company"].nunique())

# =========================
# 4. 중복 제거
# =========================
df = df.sort_values("final_score", ascending=False)
df = df.drop_duplicates(subset=["article_id"])

df["title_norm"] = df["article_title"].apply(normalize_title)
df = df.drop_duplicates(subset=["company", "title_norm"])

print("after dedup:", len(df))

# =========================
# 5. 방향 생성
# =========================
df["direction"] = df["real_return"].apply(lambda x: "up" if x > 0 else "down")

# =========================
# 6. 정렬
# =========================
df = df.sort_values(
    ["final_score", "abs_day_diff", "embedding_similarity"],
    ascending=[False, True, False]
).reset_index(drop=True)

# =========================
# 7. 방향별 후보 선택
# =========================
target_up = round(TARGET_TOTAL * UP_RATIO)
target_down = TARGET_TOTAL - target_up

selected_indices = set()
selected_dates_by_company = {}
company_counts = {}

up_selected = []
down_selected = []


def can_select(row):
    company = row["company"]
    event_date = row["event_date"]

    if company_counts.get(company, 0) >= MAX_PER_COMPANY:
        return False

    if company not in selected_dates_by_company:
        selected_dates_by_company[company] = []

    if event_too_close(event_date, selected_dates_by_company[company]):
        return False

    return True


def register_selection(idx, row):
    company = row["company"]
    event_date = row["event_date"]

    selected_indices.add(idx)
    selected_dates_by_company.setdefault(company, []).append(event_date)
    company_counts[company] = company_counts.get(company, 0) + 1


for idx, row in df.iterrows():
    if row["direction"] != "up":
        continue
    if len(up_selected) >= target_up:
        break
    if not can_select(row):
        continue

    up_selected.append(row.to_dict())
    register_selection(idx, row)

for idx, row in df.iterrows():
    if row["direction"] != "down":
        continue
    if len(down_selected) >= target_down:
        break
    if idx in selected_indices:
        continue
    if not can_select(row):
        continue

    down_selected.append(row.to_dict())
    register_selection(idx, row)

selected_total = len(up_selected) + len(down_selected)

if selected_total < TARGET_TOTAL:
    for idx, row in df.iterrows():
        if selected_total >= TARGET_TOTAL:
            break
        if idx in selected_indices:
            continue
        if not can_select(row):
            continue

        if row["direction"] == "up":
            up_selected.append(row.to_dict())
        else:
            down_selected.append(row.to_dict())

        register_selection(idx, row)
        selected_total += 1

selected_rows = up_selected + down_selected
selected_df = pd.DataFrame(selected_rows)

# =========================
# 8. stock_id 포함 + Step5 입력용 컬럼 정리
# =========================
output_cols = [
    "stock_id",
    "event_id",
    "company",
    "ticker",
    "industry",
    "event_date",
    "event_type",
    "real_return",
    "volume_ratio",
    "event_strength",
    "article_id",
    "article_title",
    "article_url",
    "article_published_at",
    "article_type",
    "market_scope",
    "article_match_text",
    "article_body_summary",
    "body",
    "cause_keywords",
    "rerank_reason",
    "reason_seed",
    "final_score",
    "target_match",
    "title_match",
    "abs_day_diff",
    "embedding_similarity",
    "direction",
]

numeric_cols = [
    "stock_id",
    "real_return",
    "volume_ratio",
    "event_strength",
    "final_score",
    "abs_day_diff",
    "embedding_similarity",
]

for col in output_cols:
    ensure_column(selected_df, col, 0 if col in numeric_cols else "")

selected_df = selected_df[output_cols].copy()

# stock_id 정수화
selected_df["stock_id"] = pd.to_numeric(selected_df["stock_id"], errors="coerce")
selected_df = selected_df[selected_df["stock_id"].notna()].copy()
selected_df["stock_id"] = selected_df["stock_id"].astype(int)

selected_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT_FILE)
print("total selected:", len(selected_df))
print("[DEBUG] stock_id null count:", selected_df["stock_id"].isna().sum())

print("\ncompany distribution:")
print(selected_df["company"].value_counts())

print("\ndirection distribution:")
print(selected_df["direction"].value_counts())

print("\n[reason_seed sample]")
sample_cols = ["company", "ticker", "stock_id", "article_title", "article_body_summary", "reason_seed"]
print(selected_df[sample_cols].head(5).to_string(index=False))