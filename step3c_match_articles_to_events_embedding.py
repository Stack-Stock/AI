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

OUT_CSV = OUT_DIR / "event_article_matches_topk.csv"
OUT_JSONL = OUT_DIR / "event_article_matches_topk.jsonl"

MODEL_NAME = "voyage-4-large"
TOP_K = 5
EVENT_BATCH_SIZE = 64

# 선택 옵션
ONLY_REPRESENTATIVE_EVENT = True
EXCLUDE_CORPORATE_ACTION_SUSPECT = True


# =========================
# 1. 유틸
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


def build_event_text(row) -> str:
    company = safe_str(row.get("company"))
    industry = safe_str(row.get("industry"))
    event_type = safe_str(row.get("event_type"))
    event_date = safe_str(row.get("event_date"))
    ticker = safe_str(row.get("ticker"))

    return_1d = safe_float(row.get("return_1d"))
    return_pre_3d = safe_float(row.get("return_pre_3d"))
    return_pre_5d = safe_float(row.get("return_pre_5d"))
    volume_ratio = safe_float(row.get("volume_ratio"))
    gap_ratio = safe_float(row.get("gap_ratio"))
    event_strength = safe_float(row.get("event_strength"))
    event_priority = safe_int(row.get("event_priority"))
    stock_event_rank = safe_int(row.get("stock_event_rank"))

    direction = infer_direction(event_type, return_1d)

    # 수치들을 너무 길지 않게 반올림
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
article_df = pd.read_parquet(ARTICLE_EMBED_FILE)

if "embedding" not in article_df.columns:
    raise RuntimeError("article embedding file에 'embedding' 컬럼이 없습니다.")

article_embeddings = np.vstack(article_df["embedding"].values).astype(np.float32)
article_embeddings = l2_normalize(article_embeddings)

print(f"[INFO] article rows: {len(article_df)}")
print(f"[INFO] article embedding dim: {article_embeddings.shape[1]}")


# =========================
# 3. 이벤트 로드 / 필터
# =========================
print("[LOAD] events...")
event_df = pd.read_csv(EVENT_FILE)

print(f"[INFO] raw event rows: {len(event_df)}")

if ONLY_REPRESENTATIVE_EVENT and "is_cluster_representative" in event_df.columns:
    event_df = event_df[event_df["is_cluster_representative"] == 1].copy()
    print(f"[INFO] representative only rows: {len(event_df)}")

if EXCLUDE_CORPORATE_ACTION_SUSPECT and "corporate_action_suspect" in event_df.columns:
    event_df = event_df[event_df["corporate_action_suspect"] != 1].copy()
    print(f"[INFO] excluding corporate action suspect rows: {len(event_df)}")

event_df = event_df.reset_index(drop=True)

if len(event_df) == 0:
    raise RuntimeError("필터링 후 이벤트가 0건입니다.")


# =========================
# 4. 이벤트 텍스트 생성
# =========================
event_df["event_match_text"] = event_df.apply(build_event_text, axis=1)

print("\n[Sample event_match_text]")
for x in event_df["event_match_text"].head(3).tolist():
    print("-", x)


# =========================
# 5. 이벤트 임베딩 생성
# =========================
API_KEY = os.getenv("VOYAGE_API_KEY")
if not API_KEY:
    raise RuntimeError("VOYAGE_API_KEY 환경변수 설정 필요")

client = voyageai.Client(api_key=API_KEY)

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

print(f"[INFO] event rows: {len(event_df)}")
print(f"[INFO] event embedding dim: {event_embeddings.shape[1]}")


# =========================
# 6. 이벤트 ↔ 기사 매칭
# =========================
print("\n[MATCH] event to article top-k...")

# cosine similarity = normalized dot product
sim_matrix = np.matmul(event_embeddings, article_embeddings.T)

matches = []

for i in range(sim_matrix.shape[0]):
    sims = sim_matrix[i]
    top_idx = np.argsort(-sims)[:TOP_K]

    event_row = event_df.iloc[i]

    for rank, art_idx in enumerate(top_idx, start=1):
        article_row = article_df.iloc[art_idx]

        matches.append({
            "event_id": safe_str(event_row.get("event_id")),
            "company": safe_str(event_row.get("company")),
            "ticker": safe_str(event_row.get("ticker")),
            "industry": safe_str(event_row.get("industry")),
            "event_date": safe_str(event_row.get("event_date")),
            "event_type": safe_str(event_row.get("event_type")),
            "return_1d": safe_float(event_row.get("return_1d")),
            "volume_ratio": safe_float(event_row.get("volume_ratio")),
            "event_strength": safe_float(event_row.get("event_strength")),
            "cluster_id": safe_str(event_row.get("cluster_id")),
            "stock_event_rank": safe_int(event_row.get("stock_event_rank")),

            "article_id": safe_str(article_row.get("article_id")),
            "source_group": safe_str(article_row.get("source_group")),
            "title": safe_str(article_row.get("title")),
            "published_at": safe_str(article_row.get("published_at")),
            "published_date": safe_str(article_row.get("published_date")),
            "url": safe_str(article_row.get("url")),
            "article_industry": safe_str(article_row.get("final_industry_for_matching")),
            "article_type": safe_str(article_row.get("article_type")),
            "target_company": safe_str(article_row.get("target_company")),
            "article_signal_strength": safe_float(article_row.get("article_signal_strength")),
            "candidate_quality_score": safe_float(article_row.get("candidate_quality_score")),

            "similarity": float(sims[art_idx]),
            "rank": rank,

            "event_match_text": safe_str(event_row.get("event_match_text")),
            "article_match_text": safe_str(article_row.get("article_match_text")),
        })

match_df = pd.DataFrame(matches)


# =========================
# 7. 저장
# =========================
match_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

with OUT_JSONL.open("w", encoding="utf-8") as f:
    for row in matches:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\n=== DONE ===")
print(f"saved: {OUT_CSV}")
print(f"saved: {OUT_JSONL}")
print(f"events_matched: {len(event_df)}")
print(f"total_rows: {len(match_df)}")

print("\n[Sample result]")
sample_cols = [
    "event_id", "company", "event_date", "event_type",
    "article_id", "title", "similarity", "rank"
]
print(match_df[sample_cols].head(10).to_string(index=False))