import json
from pathlib import Path

import pandas as pd


# =========================
# 0. 경로 설정
# =========================
STEP5_DIR = Path("./outputs_step5")
STEP5_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV = STEP5_DIR / "game_case_generated.csv"

OUTPUT_DIR = Path("./outputs_stats")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLAT_OUTPUT_CSV = OUTPUT_DIR / "case_stock_returns_flat.csv"
STATS_OUTPUT_CSV = OUTPUT_DIR / "stock_return_stats.csv"
CASE_SUMMARY_OUTPUT_CSV = OUTPUT_DIR / "case_event_stock_summary.csv"
CASE_STOCK_MATRIX_CSV = OUTPUT_DIR / "case_stock_return_matrix_1_80.csv"
STOCK_CASE_MATRIX_CSV = OUTPUT_DIR / "stock_case_return_matrix_1_30x80.csv"


# =========================
# 1. JSON 안전 파싱
# =========================
def safe_parse_json(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []

    return []


# =========================
# 2. up_down_json 펼치기
# =========================
def flatten_up_down_json(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for idx, row in df.iterrows():
        case_uid = row.get("case_uid")
        event_id = row.get("event_id")
        base_company = row.get("company")
        base_ticker = row.get("ticker")
        base_industry = row.get("industry")
        event_date = row.get("event_date")
        event_type = row.get("event_type")
        case_real_return = row.get("real_return")

        # case 순번: 1부터 시작
        case_no = idx + 1

        items = safe_parse_json(row.get("up_down_json"))

        for item in items:
            rows.append({
                "case_no": case_no,
                "case_uid": case_uid,
                "event_id": event_id,
                "base_company": base_company,
                "base_ticker": base_ticker,
                "base_industry": base_industry,
                "event_date": event_date,
                "event_type": event_type,
                "case_real_return": case_real_return,

                "stock_id": item.get("stock_id"),
                "company": item.get("company"),
                "ticker": item.get("ticker"),
                "industry": item.get("industry"),
                "json_real_return": item.get("real_return"),
                "game_return": item.get("game_return"),
                "is_event_stock": item.get("is_event_stock", 0),
            })

    flat_df = pd.DataFrame(rows)

    numeric_cols = [
        "case_no",
        "stock_id",
        "json_real_return",
        "game_return",
        "is_event_stock",
        "case_real_return",
    ]
    for col in numeric_cols:
        if col in flat_df.columns:
            flat_df[col] = pd.to_numeric(flat_df[col], errors="coerce")

    return flat_df


# =========================
# 3. 종목별 통계 계산
# =========================
def build_stock_stats(flat_df: pd.DataFrame) -> pd.DataFrame:
    def pos_ratio(series):
        s = series.dropna()
        if len(s) == 0:
            return None
        return (s > 0).mean()

    def neg_ratio(series):
        s = series.dropna()
        if len(s) == 0:
            return None
        return (s < 0).mean()

    def zero_ratio(series):
        s = series.dropna()
        if len(s) == 0:
            return None
        return (s == 0).mean()

    def event_ratio(series):
        s = series.dropna()
        if len(s) == 0:
            return None
        return (s == 1).mean()

    group_cols = ["stock_id", "company", "ticker", "industry"]
    grouped = flat_df.groupby(group_cols, dropna=False)

    stats_df = grouped.agg(
        total_rows=("case_uid", "count"),
        unique_case_count=("case_uid", "nunique"),
        event_stock_count=("is_event_stock", "sum"),

        avg_game_return=("game_return", "mean"),
        median_game_return=("game_return", "median"),
        std_game_return=("game_return", "std"),
        min_game_return=("game_return", "min"),
        max_game_return=("game_return", "max"),

        avg_case_real_return=("case_real_return", "mean"),
        median_case_real_return=("case_real_return", "median"),
    ).reset_index()

    ratio_df = grouped["game_return"].agg(
        positive_ratio=pos_ratio,
        negative_ratio=neg_ratio,
        zero_ratio=zero_ratio,
    ).reset_index()

    event_ratio_df = grouped["is_event_stock"].agg(
        event_stock_ratio=event_ratio
    ).reset_index()

    abs_impact_df = grouped["game_return"].apply(
        lambda x: x.dropna().abs().mean() if len(x.dropna()) > 0 else None
    ).reset_index(name="avg_abs_game_return")

    stats_df = stats_df.merge(ratio_df, on=group_cols, how="left")
    stats_df = stats_df.merge(event_ratio_df, on=group_cols, how="left")
    stats_df = stats_df.merge(abs_impact_df, on=group_cols, how="left")

    stats_df = stats_df.sort_values(
        by=["avg_abs_game_return", "avg_game_return"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return stats_df


# =========================
# 4. 케이스별 이벤트 종목 요약
# =========================
def build_case_event_stock_summary(flat_df: pd.DataFrame) -> pd.DataFrame:
    event_df = flat_df[flat_df["is_event_stock"] == 1].copy()

    if event_df.empty:
        return pd.DataFrame(columns=[
            "case_no", "case_uid", "event_id", "base_company",
            "event_date", "event_type", "event_stock_count",
            "avg_event_stock_game_return", "min_event_stock_game_return",
            "max_event_stock_game_return"
        ])

    summary = event_df.groupby(
        ["case_no", "case_uid", "event_id", "base_company", "event_date", "event_type"],
        dropna=False
    ).agg(
        event_stock_count=("stock_id", "count"),
        avg_event_stock_game_return=("game_return", "mean"),
        min_event_stock_game_return=("game_return", "min"),
        max_event_stock_game_return=("game_return", "max"),
    ).reset_index()

    return summary


# =========================
# 5. case x stock 매트릭스
#    행 = case 1~80
#    열 = stock_id 1~30
# =========================
def build_case_stock_matrix(flat_df: pd.DataFrame) -> pd.DataFrame:
    matrix_df = flat_df.pivot_table(
        index=["case_no", "case_uid"],
        columns="stock_id",
        values="game_return",
        aggfunc="first"
    ).reset_index()

    desired_stock_ids = list(range(1, 31))
    existing_stock_ids = [sid for sid in desired_stock_ids if sid in matrix_df.columns]

    matrix_df = matrix_df[["case_no", "case_uid"] + existing_stock_ids]

    rename_map = {sid: f"stock_id_{int(sid)}" for sid in existing_stock_ids}
    matrix_df = matrix_df.rename(columns=rename_map)

    matrix_df = matrix_df.sort_values("case_no").reset_index(drop=True)

    return matrix_df


# =========================
# 6. stock x case 매트릭스
#    행 = stock_id 1~30
#    열 = case 1~80
#    + 종목명 포함
# =========================
def build_stock_case_matrix(flat_df: pd.DataFrame) -> pd.DataFrame:
    pivot_df = flat_df.pivot_table(
        index=["stock_id", "company"],
        columns="case_no",
        values="game_return",
        aggfunc="first"
    ).reset_index()

    rename_map = {}
    for col in pivot_df.columns:
        if isinstance(col, (int, float)) and not pd.isna(col):
            rename_map[col] = f"case_{int(col)}"

    pivot_df = pivot_df.rename(columns=rename_map)

    desired_case_cols = [f"case_{i}" for i in range(1, 81)]
    existing_case_cols = [c for c in desired_case_cols if c in pivot_df.columns]

    pivot_df = pivot_df[["stock_id", "company"] + existing_case_cols]
    pivot_df = pivot_df.sort_values("stock_id").reset_index(drop=True)

    return pivot_df


# =========================
# 7. 실행
# =========================
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # 원본 케이스가 80개보다 많아도 앞 80개만 쓰고 싶으면 아래 주석 해제
    # df = df.head(80).copy()

    flat_df = flatten_up_down_json(df)

    if flat_df.empty:
        raise ValueError("up_down_json을 펼친 결과가 비어 있습니다. JSON 구조를 확인해주세요.")

    stats_df = build_stock_stats(flat_df)
    case_summary_df = build_case_event_stock_summary(flat_df)
    case_stock_matrix_df = build_case_stock_matrix(flat_df)
    stock_case_matrix_df = build_stock_case_matrix(flat_df)

    flat_df.to_csv(FLAT_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    stats_df.to_csv(STATS_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    case_summary_df.to_csv(CASE_SUMMARY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    case_stock_matrix_df.to_csv(CASE_STOCK_MATRIX_CSV, index=False, encoding="utf-8-sig")
    stock_case_matrix_df.to_csv(STOCK_CASE_MATRIX_CSV, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("완료")
    print(f"- 펼친 데이터 저장: {FLAT_OUTPUT_CSV}")
    print(f"- 종목별 통계 저장: {STATS_OUTPUT_CSV}")
    print(f"- 케이스별 이벤트 종목 요약 저장: {CASE_SUMMARY_OUTPUT_CSV}")
    print(f"- 케이스 x 종목 매트릭스 저장: {CASE_STOCK_MATRIX_CSV}")
    print(f"- 종목 x 케이스 매트릭스 저장: {STOCK_CASE_MATRIX_CSV}")
    print("=" * 70)

    print("\n[종목별 통계 상위 10개]")
    print(stats_df.head(10).to_string(index=False))

    print("\n[case x stock 매트릭스 상위 5행]")
    print(case_stock_matrix_df.head().to_string(index=False))

    print("\n[stock x case 매트릭스 상위 5행]")
    print(stock_case_matrix_df.head().to_string(index=False))


if __name__ == "__main__":
    main()