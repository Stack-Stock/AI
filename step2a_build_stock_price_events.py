import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

# =========================
# 경로 설정
# =========================
CANDIDATES_CSV = Path("./company_candidates_dict_top_by_industry_v3.csv")
TICKERS_CSV = Path("./company_tickers.csv")

OUT_DIR = Path("./outputs_step2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVENTS = OUT_DIR / "stock_events_raw.csv"

START_DATE = "2025-01-01"
END_DATE = "2026-01-01"

# =========================
# 이벤트 기준
# =========================
SPIKE_UP_THRESHOLD = 0.07
SPIKE_DOWN_THRESHOLD = -0.07
VOLUME_SURGE_RATIO = 3.0

# =========================
# 유틸
# =========================
def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance가 MultiIndex 컬럼으로 반환하는 경우를 처리.
    예:
      ('Close', '005930.KS') -> 'Close'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


# =========================
# 데이터 로드
# =========================
candidates = pd.read_csv(CANDIDATES_CSV)
tickers = pd.read_csv(TICKERS_CSV)

df = candidates.merge(
    tickers,
    on="company_candidate",
    how="left"
)

df = df.dropna(subset=["ticker"]).copy()

print("stocks to process:", len(df))

# =========================
# 이벤트 저장 리스트
# =========================
events = []
event_id_counter = 1

# =========================
# 종목별 처리
# =========================
for _, row in tqdm(df.iterrows(), total=len(df)):
    company = row["company_candidate"]
    ticker = row["ticker"]
    industry = row["top_industry"]

    try:
        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=False
        )
        print(company, ticker, data.columns)
    except Exception as e:
        print(f"[download error] {company} {ticker}: {e}")
        continue

    if data.empty:
        print(f"[empty] {company} {ticker}")
        continue

    data = flatten_yf_columns(data)
    data = data.reset_index()

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(data.columns)):
        print(f"[missing columns] {company} {ticker}: {list(data.columns)}")
        continue

    data = data.sort_values("Date").reset_index(drop=True)

    # 숫자형 강제 변환
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Close", "Volume"]).reset_index(drop=True)

    if len(data) < 25:
        print(f"[too short] {company} {ticker}: {len(data)} rows")
        continue

    # =========================
    # 지표 계산
    # =========================
    data["return_1d"] = data["Close"].pct_change()
    data["volume_ma20"] = data["Volume"].rolling(20, min_periods=5).mean()
    data["volume_ratio"] = data["Volume"] / data["volume_ma20"]
    data["gap_ratio"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)

    data["return_pre_3d"] = data["Close"] / data["Close"].shift(3) - 1
    data["return_pre_5d"] = data["Close"] / data["Close"].shift(5) - 1

    # =========================
    # 이벤트 탐지
    # =========================
    for i in range(2, len(data)):
        date = data.loc[i, "Date"]
        close_price = safe_float(data.loc[i, "Close"])
        ret_1d = safe_float(data.loc[i, "return_1d"])
        volume_ratio = safe_float(data.loc[i, "volume_ratio"])
        gap_ratio = safe_float(data.loc[i, "gap_ratio"])
        return_pre_3d = safe_float(data.loc[i, "return_pre_3d"])
        return_pre_5d = safe_float(data.loc[i, "return_pre_5d"])

        if close_price is None or ret_1d is None:
            continue

        event_type = None

        # 1) 급등
        if ret_1d >= SPIKE_UP_THRESHOLD:
            event_type = "SPIKE_UP_1D"

        # 2) 급락
        elif ret_1d <= SPIKE_DOWN_THRESHOLD:
            event_type = "SPIKE_DOWN_1D"

        # 3) 거래량 급증
        elif volume_ratio is not None and volume_ratio >= VOLUME_SURGE_RATIO:
            event_type = "VOLUME_SURGE"

        # 4) 3일 연속 상승
        elif (
            pd.notna(data.loc[i, "return_1d"])
            and pd.notna(data.loc[i - 1, "return_1d"])
            and pd.notna(data.loc[i - 2, "return_1d"])
            and data.loc[i, "return_1d"] > 0
            and data.loc[i - 1, "return_1d"] > 0
            and data.loc[i - 2, "return_1d"] > 0
        ):
            event_type = "STREAK_UP3"

        # 5) 3일 연속 하락
        elif (
            pd.notna(data.loc[i, "return_1d"])
            and pd.notna(data.loc[i - 1, "return_1d"])
            and pd.notna(data.loc[i - 2, "return_1d"])
            and data.loc[i, "return_1d"] < 0
            and data.loc[i - 1, "return_1d"] < 0
            and data.loc[i - 2, "return_1d"] < 0
        ):
            event_type = "STREAK_DOWN3"

        if event_type is None:
            continue

        # 이벤트 강도
        # 가격 변동 + 거래량 + 갭 반영
        ret_score = abs(ret_1d) * 100
        vol_score = min(volume_ratio if volume_ratio is not None else 0, 10)
        gap_score = abs(gap_ratio) * 100 if gap_ratio is not None else 0

        event_strength = ret_score + vol_score + gap_score

        events.append({
            "event_id": f"EV_{event_id_counter:05d}",
            "company": company,
            "ticker": ticker,
            "industry": industry,
            "event_date": pd.to_datetime(date).date(),
            "event_type": event_type,
            "close_price": close_price,
            "return_1d": ret_1d,
            "return_pre_3d": return_pre_3d,
            "return_pre_5d": return_pre_5d,
            "volume_ratio": volume_ratio,
            "gap_ratio": gap_ratio,
            "event_strength": event_strength
        })

        event_id_counter += 1

# =========================
# 저장
# =========================
events_df = pd.DataFrame(events)

if events_df.empty:
    print("No events found.")
else:
    events_df = events_df.sort_values(
        ["event_strength", "event_date"],
        ascending=[False, True]
    ).reset_index(drop=True)

    events_df.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig")

    print("saved:", OUT_EVENTS)
    print("total events:", len(events_df))