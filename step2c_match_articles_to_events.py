import json
from pathlib import Path
from datetime import timedelta

import pandas as pd

# =========================
# 경로 설정
# =========================
EVENTS_CSV = Path("./outputs_step2/stock_events_refined.csv")

HIGH_CONF_JSONL = Path("./outputs_step1/match_candidates_high_confidence.jsonl")
REVIEW_JSONL = Path("./outputs_step1/match_candidates_review_needed.jsonl")

OUT_DIR = Path("./outputs_step2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MATCHED = OUT_DIR / "matched_event_articles.csv"
OUT_MATCHED_TOP1 = OUT_DIR / "matched_event_articles_top1.csv"
OUT_REVIEW = OUT_DIR / "event_article_review_sheet.csv"

# =========================
# 매칭 파라미터
# =========================
WINDOW_BEFORE_DAYS = 2
WINDOW_AFTER_DAYS = 1
TOP_K_PER_EVENT = 3

# 점수 가중치
W_DATE_DIFF = 8.0
W_COMPANY_TARGET = 20.0
W_COMPANY_CANDIDATE = 14.0
W_COMPANY_TEXT = 8.0
W_INDUSTRY_MATCH = 6.0
W_HIGH_CONF = 5.0
W_SIGNAL = 2.5
W_QUALITY = 1.5
W_EVENT_TYPE_HINT = 4.0
W_TITLE_KEYWORD = 3.0
W_BODY_KEYWORD = 1.5

TITLE_KEYWORDS_UP = [
    "급등", "상승", "강세", "호실적", "실적", "수주", "계약", "확대", "성장", "최대", "돌파",
    "인수", "합병", "신제품", "출시", "기대", "흑자", "상향", "AI", "반도체"
]
TITLE_KEYWORDS_DOWN = [
    "급락", "하락", "약세", "적자", "부진", "우려", "리스크", "충격", "관세", "소송", "규제",
    "중단", "감소", "악화", "정정", "실패"
]

BODY_KEYWORDS_UP = [
    "실적", "호조", "성장", "수주", "계약", "증가", "확대", "개선", "기대", "흑자"
]
BODY_KEYWORDS_DOWN = [
    "부진", "감소", "적자", "악화", "우려", "리스크", "규제", "하회", "충격"
]

EVENT_TYPE_HINT_MAP = {
    "SPIKE_UP_1D": ["earnings", "contract", "mna", "product_launch", "policy_sentiment", "investment_capex"],
    "SPIKE_DOWN_1D": ["risk", "regulation", "guidance_cut", "rumor_or_expectation", "policy_sentiment"],
    "VOLUME_SURGE": ["policy_sentiment", "rumor_or_expectation", "investment_capex", "contract"],
    "STREAK_UP3": ["earnings", "contract", "product_launch", "policy_sentiment"],
    "STREAK_DOWN3": ["risk", "regulation", "guidance_cut", "policy_sentiment"],
}

# =========================
# 유틸
# =========================
def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def to_list(v):
    if isinstance(v, list):
        return v
    if pd.isna(v) if not isinstance(v, list) else False:
        return []
    return []


def safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def normalize_text(v):
    if v is None:
        return ""
    return str(v).strip()


def contains_company_text(company: str, title: str, body: str) -> bool:
    company = normalize_text(company)
    title = normalize_text(title)
    body = normalize_text(body)
    if not company:
        return False
    return (company in title) or (company in body)


def get_article_industry(article: dict) -> str:
    v = article.get("final_industry_for_matching")
    if v:
        return str(v)
    v = article.get("top_industry_final")
    if v:
        return str(v)
    return "UNKNOWN"


def get_date_score(diff_days: int) -> float:
    # 0일차 최고, 이후 감점
    diff = abs(diff_days)
    if diff == 0:
        return W_DATE_DIFF
    if diff == 1:
        return W_DATE_DIFF * 0.7
    if diff == 2:
        return W_DATE_DIFF * 0.4
    return 0.0


def get_title_keyword_score(event_type: str, title: str) -> float:
    title = normalize_text(title)
    if event_type in ("SPIKE_UP_1D", "STREAK_UP3"):
        return sum(1 for kw in TITLE_KEYWORDS_UP if kw in title) * W_TITLE_KEYWORD
    if event_type in ("SPIKE_DOWN_1D", "STREAK_DOWN3"):
        return sum(1 for kw in TITLE_KEYWORDS_DOWN if kw in title) * W_TITLE_KEYWORD
    return 0.0


def get_body_keyword_score(event_type: str, body: str) -> float:
    body = normalize_text(body)
    if event_type in ("SPIKE_UP_1D", "STREAK_UP3"):
        return sum(1 for kw in BODY_KEYWORDS_UP if kw in body) * W_BODY_KEYWORD
    if event_type in ("SPIKE_DOWN_1D", "STREAK_DOWN3"):
        return sum(1 for kw in BODY_KEYWORDS_DOWN if kw in body) * W_BODY_KEYWORD
    return 0.0


def get_event_type_hint_score(event_type: str, cause_types: list) -> float:
    hints = EVENT_TYPE_HINT_MAP.get(event_type, [])
    cause_types = [str(x) for x in cause_types] if isinstance(cause_types, list) else []
    overlap = len(set(hints) & set(cause_types))
    return overlap * W_EVENT_TYPE_HINT


# =========================
# 1. 이벤트 로드
# =========================
events = pd.read_csv(EVENTS_CSV)
events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
events = events.dropna(subset=["event_date"]).copy()

print(f"[0] refined events: {len(events)}")

# =========================
# 2. 기사 로드
# =========================
high_rows = load_jsonl(HIGH_CONF_JSONL)
review_rows = load_jsonl(REVIEW_JSONL)

for row in high_rows:
    row["_article_pool"] = "high_confidence"

for row in review_rows:
    row["_article_pool"] = "review_needed"

article_rows = high_rows + review_rows
print(f"[1] loaded articles: {len(article_rows)}")

# DataFrame 변환
articles = pd.DataFrame(article_rows)

if articles.empty:
    raise ValueError("No articles loaded from step1 jsonl files.")

# 기본 컬럼 정리
articles["published_date"] = pd.to_datetime(articles["published_date"], errors="coerce")
articles["title"] = articles["title"].fillna("")
articles["body"] = articles["body"].fillna("")
articles["url"] = articles["url"].fillna("")
articles["article_id"] = articles["article_id"].astype(str)

# 기사 품질 관련
if "article_signal_strength" not in articles.columns:
    articles["article_signal_strength"] = 0.0
if "candidate_quality_score" not in articles.columns:
    articles["candidate_quality_score"] = 0.0
if "is_match_candidate" not in articles.columns:
    articles["is_match_candidate"] = True

articles["article_signal_strength"] = pd.to_numeric(articles["article_signal_strength"], errors="coerce").fillna(0.0)
articles["candidate_quality_score"] = pd.to_numeric(articles["candidate_quality_score"], errors="coerce").fillna(0.0)

# 불필요한 기사 1차 컷
articles = articles[
    articles["published_date"].notna() &
    (articles["is_match_candidate"] == True)
].copy()

print(f"[2] usable articles: {len(articles)}")

# =========================
# 3. 이벤트별 기사 매칭
# =========================
matches = []

for _, ev in events.iterrows():
    event_id = ev["event_id"]
    company = str(ev["company"])
    ticker = str(ev["ticker"])
    industry = str(ev["industry"])
    event_date = pd.to_datetime(ev["event_date"])
    event_type = str(ev["event_type"])
    event_strength = safe_float(ev.get("event_strength", 0.0))
    return_1d = safe_float(ev.get("return_1d", 0.0))
    volume_ratio = safe_float(ev.get("volume_ratio", 0.0))

    start_date = event_date - timedelta(days=WINDOW_BEFORE_DAYS)
    end_date = event_date + timedelta(days=WINDOW_AFTER_DAYS)

    cand = articles[
        (articles["published_date"] >= start_date) &
        (articles["published_date"] <= end_date)
    ].copy()

    if cand.empty:
        continue

    local_rows = []

    for _, ar in cand.iterrows():
        article_id = str(ar.get("article_id", ""))
        title = normalize_text(ar.get("title", ""))
        body = normalize_text(ar.get("body", ""))
        url = normalize_text(ar.get("url", ""))
        article_date = pd.to_datetime(ar.get("published_date"))
        article_pool = normalize_text(ar.get("_article_pool", ""))
        target_company = normalize_text(ar.get("target_company", ""))
        company_candidates = ar.get("company_candidates", [])
        article_industry = get_article_industry(ar)
        cause_types = ar.get("cause_event_type_candidates", [])
        signal_strength = safe_float(ar.get("article_signal_strength", 0.0))
        quality_score = safe_float(ar.get("candidate_quality_score", 0.0))

        if not isinstance(company_candidates, list):
            company_candidates = []

        # 회사 매칭 여부
        company_target_match = int(target_company == company and target_company != "")
        company_candidate_match = int(company in company_candidates)
        company_text_match = int(contains_company_text(company, title, body))

        # 회사 관련성 없는 기사 제거
        if (company_target_match + company_candidate_match + company_text_match) == 0:
            continue

        # 산업 매칭
        industry_match = int(article_industry == industry)

        # 날짜 차이
        diff_days = (article_date.date() - event_date.date()).days
        date_score = get_date_score(diff_days)

        # 가중 점수
        match_score = 0.0
        match_score += date_score
        match_score += company_target_match * W_COMPANY_TARGET
        match_score += company_candidate_match * W_COMPANY_CANDIDATE
        match_score += company_text_match * W_COMPANY_TEXT
        match_score += industry_match * W_INDUSTRY_MATCH
        match_score += (article_pool == "high_confidence") * W_HIGH_CONF
        match_score += signal_strength * W_SIGNAL
        match_score += quality_score * W_QUALITY
        match_score += get_event_type_hint_score(event_type, cause_types)
        match_score += get_title_keyword_score(event_type, title)
        match_score += get_body_keyword_score(event_type, body)

        local_rows.append({
            "event_id": event_id,
            "company": company,
            "ticker": ticker,
            "industry": industry,
            "event_date": event_date.date(),
            "event_type": event_type,
            "event_strength": event_strength,
            "return_1d": return_1d,
            "volume_ratio": volume_ratio,

            "article_id": article_id,
            "article_date": article_date.date(),
            "article_title": title,
            "article_url": url,
            "article_pool": article_pool,
            "article_industry": article_industry,
            "target_company": target_company,
            "company_candidates": ", ".join(company_candidates[:10]) if company_candidates else "",
            "article_signal_strength": signal_strength,
            "candidate_quality_score": quality_score,
            "cause_event_type_candidates": ", ".join(cause_types[:10]) if isinstance(cause_types, list) else "",

            "date_diff_days": diff_days,
            "company_target_match": company_target_match,
            "company_candidate_match": company_candidate_match,
            "company_text_match": company_text_match,
            "industry_match": industry_match,
            "match_score": match_score,
        })

    if not local_rows:
        continue

    local_df = pd.DataFrame(local_rows)
    local_df = local_df.sort_values(
        by=[
            "match_score",
            "company_target_match",
            "company_candidate_match",
            "industry_match",
            "article_signal_strength",
            "candidate_quality_score",
            "article_date",
        ],
        ascending=[False, False, False, False, False, False, False]
    ).reset_index(drop=True)

    local_df["article_rank_for_event"] = range(1, len(local_df) + 1)
    local_df = local_df[local_df["article_rank_for_event"] <= TOP_K_PER_EVENT].copy()

    matches.append(local_df)

# =========================
# 4. 결과 취합
# =========================
if not matches:
    print("No matched articles found.")
    empty_cols = [
        "event_id", "company", "ticker", "industry", "event_date", "event_type",
        "article_id", "article_date", "article_title", "article_url", "match_score"
    ]
    pd.DataFrame(columns=empty_cols).to_csv(OUT_MATCHED, index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=empty_cols).to_csv(OUT_MATCHED_TOP1, index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=empty_cols).to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")
else:
    matched_df = pd.concat(matches, ignore_index=True)

    # 전체 정렬
    matched_df = matched_df.sort_values(
        by=["event_date", "company", "article_rank_for_event", "match_score"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    # top1
    top1_df = matched_df[matched_df["article_rank_for_event"] == 1].copy().reset_index(drop=True)

    # review sheet
    review_cols = [
        "event_id", "company", "industry", "event_date", "event_type", "event_strength",
        "article_rank_for_event", "article_date", "article_title", "article_url",
        "article_pool", "target_company", "company_candidates", "article_industry",
        "date_diff_days", "company_target_match", "company_candidate_match",
        "company_text_match", "industry_match", "article_signal_strength",
        "candidate_quality_score", "match_score"
    ]
    review_df = matched_df[review_cols].copy()

    matched_df.to_csv(OUT_MATCHED, index=False, encoding="utf-8-sig")
    top1_df.to_csv(OUT_MATCHED_TOP1, index=False, encoding="utf-8-sig")
    review_df.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

    print("saved:", OUT_MATCHED)
    print("saved:", OUT_MATCHED_TOP1)
    print("saved:", OUT_REVIEW)
    print("matched rows:", len(matched_df))
    print("matched unique events:", top1_df['event_id'].nunique())
    print("coverage ratio:", round(top1_df["event_id"].nunique() / len(events), 4))
    print()
    print("[article_pool distribution]")
    print(matched_df["article_pool"].value_counts())