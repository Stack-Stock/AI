import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import voyageai

# =========================
# 0. 경로 / 설정
# =========================
ARTICLE_EMBED_FILE = Path("./outputs_step3/article_embeddings_voyage_large4.parquet")
EVENT_FILE = Path("./outputs_step2/stock_events_refined.csv")

OUT_DIR = Path("./outputs_step3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "event_article_candidates_topk.csv"
OUT_JSONL = OUT_DIR / "event_article_candidates_topk.jsonl"

MODEL_NAME = "voyage-4-large"
EVENT_BATCH_SIZE = 64

TOP_CANDIDATES = 50

ONLY_REPRESENTATIVE_EVENT = True
EXCLUDE_CORPORATE_ACTION_SUSPECT = True

MAX_ABS_DAY_DIFF = 21
PREFER_BEFORE_DAYS = 14
PREFER_AFTER_DAYS = 5

GENERIC_TITLE_PATTERNS = [
    "특징주",
    "관련주",
    "희비",
    "인기 검색",
    "주식 초고수",
    "상승 종목",
    "하락 종목",
    "마감 시황",
    "MK시그널",
    "HOT종목",
    "인기검색",
    "TOP5",
    "시황저격",
    "골든크로스",
    "매도신호",
]


# =========================
# 1. 유틸
# =========================
def safe_str(v):
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip() if v is not None else ""


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


def safe_json_list(v):
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return []


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return mat / norms


def infer_direction(event_type: str, return_1d: float) -> str:
    et = safe_str(event_type).upper()
    if "UP" in et:
        return "up"
    if "DOWN" in et:
        return "down"
    if return_1d > 0:
        return "up"
    if return_1d < 0:
        return "down"
    return "flat"


def contains_generic_title(title: str) -> bool:
    t = safe_str(title)
    return any(p in t for p in GENERIC_TITLE_PATTERNS)


def has_company_in_title(title: str, company: str) -> bool:
    title = safe_str(title)
    company = safe_str(company)
    if not title or not company:
        return False
    return company in title


def target_company_match(target_company: str, company: str) -> bool:
    return safe_str(target_company) == safe_str(company)


def candidate_company_match(company_candidates, company: str) -> bool:
    cands = safe_json_list(company_candidates)
    company = safe_str(company)
    return company in cands


def calc_day_diff(article_date, event_date):
    if pd.isna(article_date) or pd.isna(event_date):
        return 9999
    return int((article_date - event_date).days)


def build_event_text(row) -> str:
    company = safe_str(row.get("company"))
    ticker = safe_str(row.get("ticker"))
    industry = safe_str(row.get("industry"))
    event_date = safe_str(row.get("event_date"))
    event_type = safe_str(row.get("event_type"))

    return_1d = safe_float(row.get("return_1d"))
    return_pre_3d = safe_float(row.get("return_pre_3d"))
    return_pre_5d = safe_float(row.get("return_pre_5d"))
    volume_ratio = safe_float(row.get("volume_ratio"))
    gap_ratio = safe_float(row.get("gap_ratio"))
    event_strength = safe_float(row.get("event_strength"))
    event_priority = safe_int(row.get("event_priority"))
    stock_event_rank = safe_int(row.get("stock_event_rank"))

    direction = infer_direction(event_type, return_1d)

    parts = [
        company,
        f"ticker={ticker}" if ticker else "",
        f"industry={industry}" if industry else "",
        f"event_date={event_date}" if event_date else "",
        f"event_type={event_type}" if event_type else "",
        f"direction={direction}",
        f"return_1d={return_1d:.4f}",
        f"return_pre_3d={return_pre_3d:.4f}",
        f"return_pre_5d={return_pre_5d:.4f}",
        f"volume_ratio={volume_ratio:.4f}",
        f"gap_ratio={gap_ratio:.4f}",
        f"event_strength={event_strength:.4f}",
        f"event_priority={event_priority}",
        f"stock_event_rank={stock_event_rank}",
    ]
    return " | ".join([p for p in parts if p])


# =========================
# 2. 기사 임베딩 로드
# =========================
print("[LOAD] article embeddings...")
article_df = pd.read_parquet(ARTICLE_EMBED_FILE).copy()

if "embedding" not in article_df.columns:
    raise RuntimeError("article embedding file에 embedding 컬럼이 없습니다.")

article_df["published_date"] = pd.to_datetime(article_df["published_date"], errors="coerce")

article_embeddings = np.vstack(article_df["embedding"].values).astype(np.float32)
article_embeddings = l2_normalize(article_embeddings)

print(f"[INFO] article rows: {len(article_df)}")
print(f"[INFO] article embedding dim: {article_embeddings.shape[1]}")

# body/body_summary가 없더라도 죽지 않게 보정
for col in ["body", "body_summary", "cause_keywords", "article_match_text"]:
    if col not in article_df.columns:
        article_df[col] = ""


# =========================
# 3. 이벤트 로드 / 필터
# =========================
print("[LOAD] events...")
event_df = pd.read_csv(EVENT_FILE).copy()

event_df["event_date"] = pd.to_datetime(event_df["event_date"], errors="coerce")
event_df = event_df.dropna(subset=["event_date"]).copy()

print(f"[INFO] raw event rows: {len(event_df)}")

if ONLY_REPRESENTATIVE_EVENT and "is_cluster_representative" in event_df.columns:
    event_df = event_df[event_df["is_cluster_representative"] == 1].copy()
    print(f"[INFO] representative only rows: {len(event_df)}")

if EXCLUDE_CORPORATE_ACTION_SUSPECT and "corporate_action_suspect" in event_df.columns:
    event_df = event_df[event_df["corporate_action_suspect"] != 1].copy()
    print(f"[INFO] corporate action filtered rows: {len(event_df)}")

event_df = event_df.reset_index(drop=True)

if event_df.empty:
    raise RuntimeError("필터링 후 이벤트가 0건입니다.")


# =========================
# 4. 이벤트 텍스트 / 임베딩
# =========================
event_df["event_match_text"] = event_df.apply(build_event_text, axis=1)

print("\n[Sample event_match_text]")
for x in event_df["event_match_text"].head(3).tolist():
    print("-", x)

api_key = os.getenv("VOYAGE_API_KEY")
if not api_key:
    raise RuntimeError("VOYAGE_API_KEY 환경변수 설정 필요")

client = voyageai.Client(api_key=api_key)

event_texts = event_df["event_match_text"].fillna("").tolist()
event_embeddings_list = []

print("\n[EMBED] events...")
for i in range(0, len(event_texts), EVENT_BATCH_SIZE):
    batch = event_texts[i:i + EVENT_BATCH_SIZE]
    result = client.embed(
        batch,
        model=MODEL_NAME,
        input_type="document"
    )
    event_embeddings_list.extend(result.embeddings)
    print(f"embedded {i + len(batch)} / {len(event_texts)}")

event_embeddings = np.array(event_embeddings_list, dtype=np.float32)
event_embeddings = l2_normalize(event_embeddings)


# =========================
# 5. 후보 생성
# =========================
print("\n[CANDIDATES] generating top-k candidates...")

sim_matrix = np.matmul(event_embeddings, article_embeddings.T)

rows = []

for i in range(sim_matrix.shape[0]):
    event_row = event_df.iloc[i]
    event_date = event_row["event_date"]
    company = safe_str(event_row.get("company"))
    industry = safe_str(event_row.get("industry"))

    sims = sim_matrix[i]
    cand_idx = np.argsort(-sims)

    picked = 0
    for art_idx in cand_idx:
        article_row = article_df.iloc[art_idx]

        article_date = article_row.get("published_date")
        day_diff = calc_day_diff(article_date, event_date)
        abs_day_diff = abs(day_diff)

        if abs_day_diff > MAX_ABS_DAY_DIFF:
            continue

        title = safe_str(article_row.get("title"))
        target_company = safe_str(article_row.get("target_company"))
        company_candidates = article_row.get("company_candidates")
        article_industry = safe_str(article_row.get("final_industry_for_matching"))
        market_scope = safe_str(article_row.get("market_scope"))
        article_type = safe_str(article_row.get("article_type"))

        rows.append({
            "event_id": safe_str(event_row.get("event_id")),
            "company": company,
            "ticker": safe_str(event_row.get("ticker")),
            "industry": industry,
            "event_date": safe_str(event_row.get("event_date").date()),
            "event_type": safe_str(event_row.get("event_type")),
            "return_1d": safe_float(event_row.get("return_1d")),
            "volume_ratio": safe_float(event_row.get("volume_ratio")),
            "event_strength": safe_float(event_row.get("event_strength")),
            "cluster_id": safe_str(event_row.get("cluster_id")),
            "stock_event_rank": safe_int(event_row.get("stock_event_rank")),

            "article_id": safe_str(article_row.get("article_id")),
            "source_group": safe_str(article_row.get("source_group")),
            "title": title,
            "body": safe_str(article_row.get("body")),
            "body_summary": safe_str(article_row.get("body_summary")),
            "published_at": safe_str(article_row.get("published_at")),
            "published_date": safe_str(article_row.get("published_date").date()) if not pd.isna(article_date) else "",
            "url": safe_str(article_row.get("url")),
            "article_industry": article_industry,
            "article_type": article_type,
            "market_scope": market_scope,
            "target_company": target_company,
            "company_candidates": json.dumps(safe_json_list(company_candidates), ensure_ascii=False),
            "cause_keywords": safe_str(article_row.get("cause_keywords")),
            "article_signal_strength": safe_float(article_row.get("article_signal_strength")),
            "candidate_quality_score": safe_float(article_row.get("candidate_quality_score")),

            "embedding_similarity": float(sims[art_idx]),
            "candidate_rank_by_embedding": picked + 1,

            "day_diff": day_diff,
            "abs_day_diff": abs_day_diff,
            "is_within_preferred_window": int((-PREFER_BEFORE_DAYS <= day_diff <= PREFER_AFTER_DAYS)),
            "title_match": int(has_company_in_title(title, company)),
            "target_match": int(target_company_match(target_company, company)),
            "candidate_match": int(candidate_company_match(company_candidates, company)),
            "industry_match": int(article_industry == industry),
            "is_generic_title": int(contains_generic_title(title)),

            "event_match_text": safe_str(event_row.get("event_match_text")),
            "article_match_text": safe_str(article_row.get("article_match_text")),
        })

        picked += 1
        if picked >= TOP_CANDIDATES:
            break

    if (i + 1) <= 5 or (i + 1) % 20 == 0 or (i + 1) == len(event_df):
        print(f"[progress] {i+1}/{len(event_df)} events done | rows_so_far={len(rows)}")


# =========================
# 6. 저장
# =========================
cand_df = pd.DataFrame(rows)

if cand_df.empty:
    print("[WARN] no candidates found")
    cand_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        pass
else:
    cand_df = cand_df.sort_values(
        ["event_id", "candidate_rank_by_embedding"],
        ascending=[True, True]
    ).reset_index(drop=True)

    cand_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for _, row in cand_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

print("\n=== DONE ===")
print(f"saved: {OUT_CSV}")
print(f"saved: {OUT_JSONL}")
print(f"candidate_rows: {len(cand_df)}")
print(f"unique_events: {cand_df['event_id'].nunique() if len(cand_df) else 0}")

if len(cand_df):
    print("\n[Sample result]")
    sample_cols = [
        "event_id", "company", "event_date", "article_id", "title",
        "body_summary", "embedding_similarity", "candidate_rank_by_embedding",
        "day_diff", "title_match", "target_match", "industry_match"
    ]
    print(cand_df[sample_cols].head(20).to_string(index=False))