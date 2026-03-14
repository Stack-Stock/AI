import pandas as pd
from pathlib import Path

STEP3_DIR = Path("./outputs_step3")
STEP4_DIR = Path("./outputs_step4")
STEP4_DIR.mkdir(exist_ok=True)

INPUT_FILE = STEP3_DIR / "event_article_top1_gold.csv"
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
print("rows:", len(df))

# =========================
# 1. 기본 전처리
# =========================
df["final_score"] = pd.to_numeric(df.get("final_score"), errors="coerce").fillna(0)
df["event_date"] = pd.to_datetime(df.get("event_date"), errors="coerce")

# return 컬럼 정리
if "real_return" not in df.columns:
    if "return_1d" in df.columns:
        df["real_return"] = pd.to_numeric(df["return_1d"], errors="coerce").fillna(0)
    else:
        df["real_return"] = 0.0

# article title 정리
if "article_title" not in df.columns:
    if "title" in df.columns:
        df["article_title"] = df["title"]
    else:
        df["article_title"] = ""

# article url 정리
if "article_url" not in df.columns:
    if "url" in df.columns:
        df["article_url"] = df["url"]
    else:
        df["article_url"] = ""

# published_at 정리
if "article_published_at" not in df.columns:
    if "published_at" in df.columns:
        df["article_published_at"] = df["published_at"]
    else:
        df["article_published_at"] = ""

# industry 보정
ensure_column(df, "industry", "")

# event_type 보정
ensure_column(df, "event_type", "")

# ticker 보정
ensure_column(df, "ticker", "")

# article_type / market_scope 보정
ensure_column(df, "article_type", "")
ensure_column(df, "market_scope", "")

# reason_seed 보정
if "reason_seed" not in df.columns:
    if "body_summary" in df.columns:
        df["reason_seed"] = df["body_summary"].fillna("")
    elif "article_body_summary" in df.columns:
        df["reason_seed"] = df["article_body_summary"].fillna("")
    elif "article_match_text" in df.columns:
        df["reason_seed"] = df["article_match_text"].fillna("")
    elif "body" in df.columns:
        df["reason_seed"] = df["body"].fillna("").astype(str).str[:500]
    elif "rerank_reason" in df.columns:
        df["reason_seed"] = df["rerank_reason"].fillna("")
    else:
        df["reason_seed"] = df["article_title"].fillna("")

# target_match / title_match 없으면 완화
if "target_match" not in df.columns:
    df["target_match"] = 1
if "title_match" not in df.columns:
    df["title_match"] = 1

# 정렬용 컬럼 없으면 생성
ensure_column(df, "abs_day_diff", 999)
ensure_column(df, "embedding_similarity", 0.0)

# =========================
# 2. 품질 필터
# =========================
df = df[df["final_score"] >= MIN_FINAL_SCORE]
df = df[df["target_match"] == 1]
df = df[df["title_match"] == 1]
df = df.dropna(subset=["company", "event_date", "article_id"])

print("after quality filter:", len(df))

# =========================
# 3. 중복 제거
# =========================
df = df.sort_values("final_score", ascending=False)
df = df.drop_duplicates(subset=["article_id"])

df["title_norm"] = df["article_title"].apply(normalize_title)
df = df.drop_duplicates(subset=["company", "title_norm"])

print("after dedup:", len(df))

# =========================
# 4. 방향 생성
# =========================
df["direction"] = df["real_return"].apply(lambda x: "up" if x > 0 else "down")

# =========================
# 5. 정렬
# =========================
df = df.sort_values(
    ["final_score", "abs_day_diff", "embedding_similarity"],
    ascending=[False, True, False]
).reset_index(drop=True)

# =========================
# 6. 방향별 후보 선택
#    - 우선 목표 비율만 맞추고
#    - 남는 자리는 전체 후보에서 추가 충원
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


# 1차: up 목표치 채우기
for idx, row in df.iterrows():
    if row["direction"] != "up":
        continue
    if len(up_selected) >= target_up:
        break
    if not can_select(row):
        continue

    up_selected.append(row.to_dict())
    register_selection(idx, row)

# 2차: down 목표치 채우기
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

# 3차: 남는 자리 있으면 방향 상관없이 추가 충원
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
# 7. Step5 입력용 컬럼 정리
# =========================
output_cols = [
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
    "reason_seed",
    "final_score",
    "target_match",
    "title_match",
    "abs_day_diff",
    "embedding_similarity",
    "direction",
]

for col in output_cols:
    ensure_column(selected_df, col, "" if col not in ["real_return", "volume_ratio", "event_strength", "final_score", "abs_day_diff", "embedding_similarity"] else 0)

selected_df = selected_df[output_cols].copy()
selected_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT_FILE)
print("total selected:", len(selected_df))
print("\ncompany distribution:")
print(selected_df["company"].value_counts())

print("\ndirection distribution:")
print(selected_df["direction"].value_counts())