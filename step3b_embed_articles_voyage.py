import os
import time
from pathlib import Path

import pandas as pd
import voyageai

# =========================
# 0. 설정
# =========================

INPUT_CSV = Path("./outputs_step3/article_embedding_source.csv")

OUT_DIR = Path("./outputs_step3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "article_embeddings_voyage_large4.parquet"

MODEL_NAME = "voyage-4-large"
BATCH_SIZE = 64


# =========================
# 1. API 키
# =========================

API_KEY = os.getenv("VOYAGE_API_KEY")

if not API_KEY:
    raise RuntimeError("VOYAGE_API_KEY 환경변수 설정 필요")

client = voyageai.Client(api_key=API_KEY)


# =========================
# 2. 데이터 로드
# =========================

print("loading article source...")

df = pd.read_csv(INPUT_CSV)

texts = df["article_match_text"].fillna("").tolist()

print("rows:", len(texts))


# =========================
# 3. 임베딩 생성
# =========================

all_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):

    batch = texts[i:i + BATCH_SIZE]

    try:

        result = client.embed(
            batch,
            model=MODEL_NAME,
            input_type="document"
        )

        all_embeddings.extend(result.embeddings)

    except Exception as e:

        print("API error:", e)
        print("retrying...")

        time.sleep(5)

        result = client.embed(
            batch,
            model=MODEL_NAME,
            input_type="document"
        )

        all_embeddings.extend(result.embeddings)

    print(f"embedded {i + len(batch)} / {len(texts)}")


# =========================
# 4. 저장
# =========================

df["embedding"] = all_embeddings

df.to_parquet(OUT_FILE, index=False)

print("\n=== DONE ===")
print("saved:", OUT_FILE)
print("rows:", len(df))
print("embedding_dim:", len(all_embeddings[0]))