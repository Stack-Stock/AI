import csv
import json
import re
from collections import Counter
from pathlib import Path

# =========================
# 0. 경로 설정
# =========================
PROJECT_ROOT = Path(r"C:\Users\SSAFY\Desktop\seohyung\Stack&Stock\projects\article_embedding_V2.0")
OUT_DIR = PROJECT_ROOT / "outputs_step1"

INPUT_JSONL = OUT_DIR / "structured_articles_step1c.jsonl"

OUT_JSONL = OUT_DIR / "structured_articles_step1d.jsonl"
OUT_REVIEW_CSV = OUT_DIR / "structured_articles_step1d_review.csv"
OUT_SUMMARY_JSON = OUT_DIR / "structured_articles_step1d_summary.json"

# =========================
# 1. 규칙 사전
# =========================
CAUSE_EVENT_RULES = {
    "earnings_positive": {"실적", "호실적", "어닝서프라이즈", "영업이익", "매출 증가", "흑자", "가이던스 상향"},
    "earnings_negative": {"적자", "실적 부진", "어닝쇼크", "영업손실", "매출 감소", "적자전환", "가이던스 하향"},
    "contract_order": {"수주", "공급계약", "계약 체결", "납품", "발주"},
    "investment_capex": {"투자", "증설", "CAPEX", "시설투자", "공장 증설"},
    "approval_regulation": {"승인", "허가", "FDA", "인허가", "규제", "제재", "행정처분"},
    "shareholder_action": {"배당", "자사주", "소각", "분할", "합병", "인수"},
    "artist_activity": {"컴백", "투어", "앨범", "음원", "아이돌", "배우", "예능", "드라마", "영화 개봉"},
    "overseas_expansion": {"해외", "수출", "글로벌", "북미", "유럽", "동남아", "현지 진출"},
    "policy_sentiment": {"정책", "지원", "수혜", "관세", "정부", "예산", "보조금"},
    "rumor_or_expectation": {"기대감", "루머", "관측", "전망", "예상", "가능성", "수혜 기대"}
}

PRICE_RECAP_KEYWORDS = {
    "주가", "급등", "급락", "상승", "하락", "강세", "약세", "상한가", "하한가",
    "장중", "종가", "마감", "급반등", "랠리"
}

NOISE_KEYWORDS = {
    "포토", "화보", "인터뷰", "피플", "칼럼", "사설", "기자수첩", "기자24시",
    "CEO 라운지", "슬기로운", "방영덕의", "추동훈의", "노영우의", "김성회의"
}

MACRO_POLICY_KEYWORDS = {
    "정부", "정책", "금리", "관세", "규제", "대선", "예산", "법안", "금융당국", "한국은행", "연준"
}

INDUSTRY_NEWS_HINTS = {
    "업계", "산업", "시장", "전반", "관련 업계", "업황", "섹터", "동향"
}

# =========================
# 2. 유틸
# =========================
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def split_sentences(text: str, max_sentences: int = 2):
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!다요])\s+", text)
    parts = [normalize_space(x) for x in parts if normalize_space(x)]
    return parts[:max_sentences]

def contains_any(text: str, keywords: set) -> int:
    return sum(1 for kw in keywords if kw in text)

# =========================
# 3. 회사 중심성
# =========================
def detect_company_mention_zone(title: str, body: str, company_candidate_rows: list):
    if not company_candidate_rows:
        return "none"

    top = company_candidate_rows[0]
    if top.get("title_hits", 0) > 0:
        return "title"
    if top.get("lead_hits", 0) > 0:
        return "lead"
    if top.get("body_hits", 0) > 0:
        return "body"
    return "none"

def compute_company_directness(row: dict):
    target_company = row.get("target_company")
    company_match_score = float(row.get("company_match_score", 0.0))
    company_rows = row.get("company_candidate_rows", []) or []

    mention_zone = detect_company_mention_zone(
        title=row.get("title", ""),
        body=row.get("body", ""),
        company_candidate_rows=company_rows
    )

    num_companies = len(row.get("company_candidates", []) or [])

    centrality = 0.0
    if mention_zone == "title":
        centrality += 5.0
    elif mention_zone == "lead":
        centrality += 4.0
    elif mention_zone == "body":
        centrality += 2.0

    if num_companies == 1:
        centrality += 3.0
    elif num_companies <= 3:
        centrality += 1.5
    else:
        centrality -= 1.0

    centrality = max(0.0, min(10.0, centrality))

    is_company_direct = bool(
        target_company and
        company_match_score >= 5.0 and
        centrality >= 4.0
    )

    return {
        "company_mention_zone": mention_zone,
        "company_centrality_score": round(centrality, 3),
        "is_company_direct": is_company_direct
    }

# =========================
# 4. 원인 키워드 / 이벤트 타입
# =========================
def extract_cause_signals(title: str, body: str):
    text = f"{title} {body}"
    cause_keywords = set()
    event_type_scores = Counter()

    for event_type, keywords in CAUSE_EVENT_RULES.items():
        hit = 0
        for kw in keywords:
            c = text.count(kw)
            if c > 0:
                cause_keywords.add(kw)
                hit += c
        if hit > 0:
            event_type_scores[event_type] += hit

    return {
        "cause_keywords": sorted(cause_keywords),
        "cause_event_type_candidates": [k for k, _ in event_type_scores.most_common(3)],
        "cause_keyword_score": int(sum(event_type_scores.values()))
    }

# =========================
# 5. 기사 유형 분류
# =========================
def classify_article_type(row: dict, company_info: dict, cause_info: dict):
    title = row.get("title", "")
    body = row.get("body", "")
    small_code_nm = row.get("small_code_nm", "")
    text = f"{title} {body}"

    is_noise = (
        any(k in title for k in NOISE_KEYWORDS)
        or small_code_nm in {"핫이슈", "영화", "방송/TV", "사회일반"}
    )

    price_recap_hits = contains_any(text, PRICE_RECAP_KEYWORDS)
    macro_hits = contains_any(text, MACRO_POLICY_KEYWORDS)
    industry_hits = contains_any(text, INDUSTRY_NEWS_HINTS)

    is_price_recap = (price_recap_hits >= 2 and cause_info["cause_keyword_score"] == 0)

    if is_noise:
        article_type = "noise_or_human_interest"
    elif is_price_recap:
        article_type = "price_recap_news"
    elif macro_hits >= 2 and not company_info["is_company_direct"]:
        article_type = "macro_policy_news"
    elif company_info["is_company_direct"]:
        article_type = "direct_company_news"
    elif industry_hits >= 1:
        article_type = "industry_news"
    else:
        article_type = "industry_news"

    if company_info["is_company_direct"]:
        market_scope = "company_specific"
    elif article_type == "macro_policy_news":
        market_scope = "market_wide"
    else:
        market_scope = "sector_wide"

    signal_strength = (
        company_info["company_centrality_score"] * 0.45 +
        float(row.get("company_match_score", 0.0)) * 0.35 +
        min(cause_info["cause_keyword_score"], 10) * 0.20
    )

    return {
        "article_type": article_type,
        "market_scope": market_scope,
        "article_signal_strength": round(min(signal_strength, 10.0), 3),
        "is_noise_article": article_type == "noise_or_human_interest",
        "is_price_recap_article": is_price_recap
    }

# =========================
# 6. 매칭용 텍스트
# =========================
def build_article_match_text(row: dict, cause_info: dict, article_type_info: dict):
    lead = " ".join(split_sentences(row.get("body", ""), 2))
    company = row.get("target_company") or ""
    causes = ", ".join(cause_info["cause_keywords"][:8])
    top_industry = row.get("top_industry_final", "UNKNOWN")

    parts = [
        company,
        row.get("title", ""),
        lead,
        f"industry={top_industry}",
        f"article_type={article_type_info['article_type']}",
        f"cause_keywords={causes}"
    ]
    return normalize_space(" | ".join([p for p in parts if p]))

# =========================
# 7. 메인
# =========================
def process_all():
    summary = {
        "total_rows": 0,
        "article_type_counter": Counter(),
        "is_company_direct_true": 0,
        "noise_article_count": 0,
        "price_recap_count": 0,
        "cause_event_counter": Counter(),
        "top_industry_counter": Counter(),
        "high_signal_count": 0
    }

    review_rows = []

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, \
         OUT_JSONL.open("w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, start=1):
            row = json.loads(line)
            summary["total_rows"] += 1

            company_info = compute_company_directness(row)
            cause_info = extract_cause_signals(
                title=row.get("title", ""),
                body=row.get("body", "")
            )
            article_type_info = classify_article_type(row, company_info, cause_info)
            article_match_text = build_article_match_text(row, cause_info, article_type_info)

            out_row = {
                **row,
                **company_info,
                **cause_info,
                **article_type_info,
                "article_match_text": article_match_text
            }

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            summary["article_type_counter"][out_row["article_type"]] += 1
            summary["top_industry_counter"][out_row.get("top_industry_final", "UNKNOWN")] += 1

            if out_row["is_company_direct"]:
                summary["is_company_direct_true"] += 1
            if out_row["is_noise_article"]:
                summary["noise_article_count"] += 1
            if out_row["is_price_recap_article"]:
                summary["price_recap_count"] += 1
            if out_row["article_signal_strength"] >= 6.0:
                summary["high_signal_count"] += 1

            for ev in out_row["cause_event_type_candidates"]:
                summary["cause_event_counter"][ev] += 1

            if len(review_rows) < 400:
                review_rows.append({
                    "article_id": out_row.get("article_id"),
                    "small_code_nm": out_row.get("small_code_nm"),
                    "title": out_row.get("title"),
                    "target_company": out_row.get("target_company"),
                    "top_industry_final": out_row.get("top_industry_final"),
                    "article_type": out_row.get("article_type"),
                    "is_company_direct": out_row.get("is_company_direct"),
                    "company_match_score": out_row.get("company_match_score"),
                    "company_centrality_score": out_row.get("company_centrality_score"),
                    "company_mention_zone": out_row.get("company_mention_zone"),
                    "cause_event_type_candidates": " | ".join(out_row.get("cause_event_type_candidates", [])),
                    "cause_keywords": " | ".join(out_row.get("cause_keywords", [])[:10]),
                    "article_signal_strength": out_row.get("article_signal_strength"),
                    "is_noise_article": out_row.get("is_noise_article"),
                    "is_price_recap_article": out_row.get("is_price_recap_article"),
                    "url": out_row.get("url")
                })

    if review_rows:
        fieldnames = list(review_rows[0].keys())
        with OUT_REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_rows)

    summary_dump = {
        **summary,
        "article_type_counter": dict(summary["article_type_counter"]),
        "cause_event_counter": dict(summary["cause_event_counter"]),
        "top_industry_counter": dict(summary["top_industry_counter"]),
    }

    with OUT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary_dump, f, ensure_ascii=False, indent=2)

    print("=== STEP1-D DONE ===")
    print(json.dumps(summary_dump, ensure_ascii=False, indent=2))
    print("saved:", OUT_JSONL)
    print("saved:", OUT_REVIEW_CSV)
    print("saved:", OUT_SUMMARY_JSON)

if __name__ == "__main__":
    process_all()