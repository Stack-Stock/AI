import pandas as pd
from pathlib import Path

# =========================
# 경로 설정
# =========================
IN_RAW = Path("./outputs_step2/stock_events_raw.csv")

OUT_DIR = Path("./outputs_step2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REFINED = OUT_DIR / "stock_events_refined.csv"
OUT_SUMMARY_STOCK = OUT_DIR / "stock_events_refined_summary_by_stock.csv"
OUT_SUMMARY_TYPE = OUT_DIR / "stock_events_refined_summary_by_type.csv"

# =========================
# 정제 파라미터
# =========================
CLUSTER_GAP_DAYS = 2 # 같은 종목에서 ±2일 이내면 같은 이벤트 클러스터
MAX_EVENTS_PER_STOCK = 8 # 종목당 최종 유지 개수
CORPORATE_ACTION_ABS_RETURN = 0.35

# volume 단독 이벤트 최소 조건
MIN_ABS_RETURN_FOR_VOLUME = 0.04
MIN_VOLUME_RATIO_FOR_VOLUME = 5.0

EVENT_PRIORITY = {
    "SPIKE_UP_1D": 4,
    "SPIKE_DOWN_1D": 4,
    "VOLUME_SURGE": 3,
    "STREAK_UP3": 2,
    "STREAK_DOWN3": 2,
}

# =========================
# 유틸
# =========================
def safe_priority(event_type: str) -> int:
    return EVENT_PRIORITY.get(str(event_type), 0)

def safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

# =========================
# 1. raw 로드
# =========================
df = pd.read_csv(IN_RAW)

if df.empty:
    raise ValueError(f"Input file is empty: {IN_RAW}")

# 날짜/숫자형 정리
df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

numeric_cols = [
    "close_price",
    "return_1d",
    "return_pre_3d",
    "return_pre_5d",
    "volume_ratio",
    "gap_ratio",
    "event_strength",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["company", "ticker", "industry", "event_date", "event_type"]).copy()

# 보조 컬럼
df["event_priority"] = df["event_type"].map(EVENT_PRIORITY).fillna(0).astype(int)
df["abs_return_1d"] = df["return_1d"].abs()
df["corporate_action_suspect"] = (df["abs_return_1d"] >= CORPORATE_ACTION_ABS_RETURN).astype(int)

print(f"[0] raw events: {len(df)}")

# =========================
# 2. 같은 종목 + 같은 날짜 중복 정리
#    우선순위/강도 기준으로 1건만 유지
# =========================
df_same_day = df.sort_values(
    by=["company", "event_date", "event_priority", "event_strength", "abs_return_1d"],
    ascending=[True, True, False, False, False]
).copy()

df_same_day = df_same_day.drop_duplicates(
    subset=["company", "event_date"],
    keep="first"
).reset_index(drop=True)

print(f"[1] after same-day dedup: {len(df_same_day)}")

# =========================
# 3. 약한 volume_surge 제거
# =========================
is_volume = df_same_day["event_type"] == "VOLUME_SURGE"

keep_volume_mask = (
    (df_same_day["abs_return_1d"] >= MIN_ABS_RETURN_FOR_VOLUME) |
    (df_same_day["volume_ratio"] >= MIN_VOLUME_RATIO_FOR_VOLUME)
)

df_filtered = df_same_day[
    (~is_volume) | (keep_volume_mask)
].copy().reset_index(drop=True)

print(f"[2] after weak volume_surge filter: {len(df_filtered)}")

# =========================
# 4. 종목별 클러스터링
#    같은 종목에서 날짜 차이가 2일 이하면 같은 클러스터
# =========================
cluster_rows = []

for company, g in df_filtered.groupby("company", sort=False):
    g = g.sort_values("event_date").copy().reset_index(drop=True)

    cluster_id = 1
    cluster_ids = []

    prev_date = None
    for _, row in g.iterrows():
        curr_date = row["event_date"]

        if prev_date is None:
            cluster_ids.append(cluster_id)
        else:
            diff_days = (curr_date - prev_date).days
            if diff_days <= CLUSTER_GAP_DAYS:
                cluster_ids.append(cluster_id)
            else:
                cluster_id += 1
                cluster_ids.append(cluster_id)

        prev_date = curr_date

    g["cluster_local_id"] = cluster_ids
    g["cluster_id"] = g["company"].astype(str) + "_C" + g["cluster_local_id"].astype(str).str.zfill(3)
    cluster_rows.append(g)

df_clustered = pd.concat(cluster_rows, ignore_index=True)

# 클러스터 크기
cluster_size_map = (
    df_clustered.groupby("cluster_id")
    .size()
    .rename("cluster_size")
    .reset_index()
)

df_clustered = df_clustered.merge(cluster_size_map, on="cluster_id", how="left")

print(f"[3] clustered rows: {len(df_clustered)}")
print(f"[3] unique clusters: {df_clustered['cluster_id'].nunique()}")

# =========================
# 5. 클러스터 대표 이벤트 선택
# =========================
df_clustered = df_clustered.sort_values(
    by=["cluster_id", "event_strength", "event_priority", "abs_return_1d", "event_date"],
    ascending=[True, False, False, False, True]
).copy()

df_clustered["is_cluster_representative"] = 0
rep_index = (
    df_clustered.groupby("cluster_id", as_index=False)
    .head(1)
    .index
)
df_clustered.loc[rep_index, "is_cluster_representative"] = 1

df_repr = df_clustered[df_clustered["is_cluster_representative"] == 1].copy().reset_index(drop=True)

print(f"[4] cluster representatives: {len(df_repr)}")

# =========================
# 6. 종목당 상위 이벤트 제한
# =========================
df_repr = df_repr.sort_values(
    by=["company", "event_strength", "event_priority", "abs_return_1d", "event_date"],
    ascending=[True, False, False, False, True]
).copy()

df_repr["stock_event_rank"] = (
    df_repr.groupby("company").cumcount() + 1
)

df_final = df_repr[df_repr["stock_event_rank"] <= MAX_EVENTS_PER_STOCK].copy().reset_index(drop=True)

print(f"[5] after top-{MAX_EVENTS_PER_STOCK} per stock: {len(df_final)}")

# =========================
# 7. 최종 event_id 재부여
# =========================
df_final = df_final.sort_values(
    by=["event_strength", "event_date"],
    ascending=[False, True]
).reset_index(drop=True)

df_final["event_id"] = [f"REF_EV_{i:05d}" for i in range(1, len(df_final) + 1)]

# =========================
# 8. 최종 컬럼 정리
# =========================
final_cols = [
    "event_id",
    "company",
    "ticker",
    "industry",
    "event_date",
    "event_type",
    "close_price",
    "return_1d",
    "return_pre_3d",
    "return_pre_5d",
    "volume_ratio",
    "gap_ratio",
    "event_strength",
    "event_priority",
    "cluster_id",
    "cluster_size",
    "is_cluster_representative",
    "corporate_action_suspect",
    "stock_event_rank",
]

df_final = df_final[final_cols].copy()

# 날짜 문자열화
df_final["event_date"] = pd.to_datetime(df_final["event_date"]).dt.date

# =========================
# 9. 요약 파일 생성
# =========================
summary_by_stock = (
    df_final.groupby(["industry", "company"], as_index=False)
    .agg(
        event_count=("event_id", "count"),
        avg_strength=("event_strength", "mean"),
        max_strength=("event_strength", "max"),
        corporate_action_suspect_count=("corporate_action_suspect", "sum"),
    )
    .sort_values(["industry", "event_count", "max_strength"], ascending=[True, False, False])
)

summary_by_type = (
    df_final.groupby("event_type", as_index=False)
    .agg(
        count=("event_id", "count"),
        avg_strength=("event_strength", "mean"),
        max_strength=("event_strength", "max"),
    )
    .sort_values("count", ascending=False)
)

# =========================
# 10. 저장
# =========================
df_final.to_csv(OUT_REFINED, index=False, encoding="utf-8-sig")
summary_by_stock.to_csv(OUT_SUMMARY_STOCK, index=False, encoding="utf-8-sig")
summary_by_type.to_csv(OUT_SUMMARY_TYPE, index=False, encoding="utf-8-sig")

print("saved:", OUT_REFINED)
print("saved:", OUT_SUMMARY_STOCK)
print("saved:", OUT_SUMMARY_TYPE)
print("final refined events:", len(df_final))
print()
print("[event_type distribution]")
print(df_final["event_type"].value_counts())