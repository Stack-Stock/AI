# step2c_match_articles_to_events_v2.py

import json
from pathlib import Path
from datetime import timedelta

import pandas as pd

# =========================
# 경로
# =========================
STEP1_DIR = Path("./outputs_step1")
STEP2_DIR = Path("./outputs_step2")

EVENT_FILE = STEP2_DIR / "stock_events_refined.csv"

HIGH_CONF_FILE = STEP1_DIR / "match_candidates_high_confidence.jsonl"
REVIEW_FILE = STEP1_DIR / "match_candidates_review_needed.jsonl"

OUT_MATCHED = STEP2_DIR / "matched_event_articles_v2.csv"
OUT_TOP1_STRICT = STEP2_DIR / "matched_event_articles_top1_strict.csv"

# =========================
# 설정
# =========================
MATCH_WINDOW_BEFORE = 2
MATCH_WINDOW_AFTER = 1

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
]

MIN_STRICT_SCORE = 70

# 진행 로그
LOG_EVERY_N_EVENTS = 10
PRINT_EMPTY_EVENT = False
PRINT_CANDIDATE_COUNT = True

# =========================
# util
# =========================
def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                if i <= 5:
                    print(f"[jsonl parse skip] {path.name} line={i} err={e}")
    return rows


def contains_generic_title(title: str) -> bool:
    title = "" if title is None else str(title)
    for p in GENERIC_TITLE_PATTERNS:
        if p in title:
            return True
    return False


def safe_list(v):
    return v if isinstance(v, list) else []


def safe_str(v):
    if v is None:
        return ""
    return str(v).strip()


# =========================
# 0. load events
# =========================
print("[load] reading events...")
events = pd.read_csv(EVENT_FILE)
events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
events = events.dropna(subset=["event_date"]).copy()
events = events.sort_values(["event_date", "company"]).reset_index(drop=True)

print(f"[0] refined events: {len(events)}")

# =========================
# 1. load articles
# =========================
print("[load] reading high confidence articles...")
high_conf_articles = load_jsonl(HIGH_CONF_FILE)
for a in high_conf_articles:
    a["_article_pool"] = "high_confidence"

print(f"[1-1] high_confidence loaded: {len(high_conf_articles)}")

print("[load] reading review-needed articles...")
review_articles = load_jsonl(REVIEW_FILE)
for a in review_articles:
    a["_article_pool"] = "review_needed"

print(f"[1-2] review_needed loaded: {len(review_articles)}")

articles_raw = high_conf_articles + review_articles
print(f"[1] loaded articles total: {len(articles_raw)}")

# usable filtering
usable_articles = []
for a in articles_raw:
    if a.get("is_noise_article"):
        continue
    if not a.get("published_date"):
        continue
    usable_articles.append(a)

print(f"[2] usable articles: {len(usable_articles)}")

# =========================
# 2. articles -> DataFrame
# =========================
print("[prep] building article dataframe...")

articles_df = pd.DataFrame(usable_articles).copy()

articles_df["published_date"] = pd.to_datetime(articles_df["published_date"], errors="coerce")
articles_df = articles_df.dropna(subset=["published_date"]).copy()

articles_df["title"] = articles_df["title"].fillna("")
articles_df["url"] = articles_df["url"].fillna("")
articles_df["target_company"] = articles_df["target_company"].fillna("")
articles_df["top_industry_final"] = articles_df["top_industry_final"].fillna("")
articles_df["cause_keyword_score"] = pd.to_numeric(
    articles_df.get("cause_keyword_score", 0), errors="coerce"
).fillna(0)

if "company_candidates" not in articles_df.columns:
    articles_df["company_candidates"] = [[] for _ in range(len(articles_df))]
else:
    articles_df["company_candidates"] = articles_df["company_candidates"].apply(safe_list)

# generic 기사 미리 제거
before_generic = len(articles_df)
articles_df = articles_df[~articles_df["title"].apply(contains_generic_title)].copy()
after_generic = len(articles_df)

print(f"[prep] generic-title removed: {before_generic - after_generic}")
print(f"[prep] article dataframe ready: {len(articles_df)}")

# 날짜 범위 로그
if not articles_df.empty:
    print(
        f"[prep] article date range: "
        f"{articles_df['published_date'].min().date()} ~ {articles_df['published_date'].max().date()}"
    )

# =========================
# 3. matching
# =========================
print("[match] start...")

rows = []
total_events = len(events)
matched_event_count = 0

for idx, (_, event) in enumerate(events.iterrows(), start=1):
    company = safe_str(event["company"])
    industry = safe_str(event["industry"])
    event_date = pd.to_datetime(event["event_date"])

    window_start = event_date - pd.Timedelta(days=MATCH_WINDOW_BEFORE)
    window_end = event_date + pd.Timedelta(days=MATCH_WINDOW_AFTER)

    # 1차: 날짜 필터
    cand = articles_df[
        (articles_df["published_date"] >= window_start) &
        (articles_df["published_date"] <= window_end)
    ].copy()

    if PRINT_CANDIDATE_COUNT and (idx <= 5 or idx % LOG_EVERY_N_EVENTS == 0):
        print(
            f"[match] {idx}/{total_events} "
            f"{company} {event_date.date()} "
            f"date-window candidates={len(cand)}"
        )

    if cand.empty:
        if PRINT_EMPTY_EVENT:
            print(f"[empty] no date candidates: {company} {event_date.date()}")
        continue

    # 2차: strong company gate
    cand["title_match"] = cand["title"].str.contains(company, regex=False)
    cand["target_match"] = cand["target_company"].eq(company)
    cand["candidate_match"] = cand["company_candidates"].apply(lambda x: company in x)

    strong = cand[(cand["title_match"]) | (cand["target_match"])].copy()

    if strong.empty:
        if PRINT_EMPTY_EVENT:
            print(f"[empty] no strong company match: {company} {event_date.date()}")
        continue

    # 3차: industry / score
    strong["industry_match"] = strong["top_industry_final"].eq(industry)

    strong["match_score"] = 0
    strong.loc[strong["target_match"], "match_score"] += 50
    strong.loc[strong["title_match"], "match_score"] += 40
    strong.loc[strong["candidate_match"], "match_score"] += 10
    strong.loc[strong["industry_match"], "match_score"] += 20
    strong["match_score"] += strong["cause_keyword_score"]

    strong = strong.sort_values(
        ["match_score", "target_match", "title_match", "candidate_match", "industry_match", "published_date"],
        ascending=[False, False, False, False, False, False]
    ).copy()

    if not strong.empty:
        matched_event_count += 1

    # 모든 후보 저장
    for _, a in strong.iterrows():
        rows.append({
            "event_id": event["event_id"],
            "company": company,
            "ticker": event["ticker"],
            "industry": industry,
            "event_date": event_date.date(),
            "event_type": event["event_type"],
            "event_strength": event["event_strength"],

            "article_id": a.get("article_id", ""),
            "article_title": a.get("title", ""),
            "article_url": a.get("url", ""),
            "article_pool": a.get("_article_pool", ""),
            "article_date": pd.to_datetime(a["published_date"]).date(),
            "article_industry": a.get("top_industry_final", ""),
            "target_company": a.get("target_company", ""),
            "company_candidates": ",".join(a.get("company_candidates", [])[:10]),

            "title_match": int(bool(a["title_match"])),
            "target_match": int(bool(a["target_match"])),
            "candidate_match": int(bool(a["candidate_match"])),
            "industry_match": int(bool(a["industry_match"])),
            "cause_keyword_score": a.get("cause_keyword_score", 0),
            "match_score": a["match_score"],
        })

    if idx <= 5 or idx % LOG_EVERY_N_EVENTS == 0 or idx == total_events:
        print(
            f"[progress] {idx}/{total_events} done | "
            f"matched_events_so_far={matched_event_count} | "
            f"rows_so_far={len(rows)}"
        )

print("[match] finished")

# =========================
# 4. save all matches
# =========================
df = pd.DataFrame(rows)

print(f"[result] matched rows: {len(df)}")

if df.empty:
    print("[result] no matches found")
    df.to_csv(OUT_MATCHED, index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(OUT_TOP1_STRICT, index=False, encoding="utf-8-sig")
    print("saved:", OUT_MATCHED)
    print("saved:", OUT_TOP1_STRICT)
else:
    df = df.sort_values(["event_id", "match_score"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(OUT_MATCHED, index=False, encoding="utf-8-sig")
    print("saved:", OUT_MATCHED)

    # =========================
    # 5. strict top1
    # =========================
    strict_rows = []

    print("[strict] building strict top1...")
    grouped = df.groupby("event_id", sort=False)

    total_groups = df["event_id"].nunique()
    for g_idx, (event_id, g) in enumerate(grouped, start=1):
        g = g[g["match_score"] >= MIN_STRICT_SCORE].copy()
        if len(g) == 0:
            continue
        strict_rows.append(g.iloc[0])

        if g_idx <= 5 or g_idx % 100 == 0 or g_idx == total_groups:
            print(f"[strict] {g_idx}/{total_groups} processed | strict_so_far={len(strict_rows)}")

    strict_df = pd.DataFrame(strict_rows)
    strict_df.to_csv(OUT_TOP1_STRICT, index=False, encoding="utf-8-sig")

    coverage = 0.0 if len(events) == 0 else len(strict_df) / len(events)

    print(f"[result] strict matches: {len(strict_df)}")
    print(f"[result] coverage: {coverage:.4f}")
    print("saved:", OUT_TOP1_STRICT)