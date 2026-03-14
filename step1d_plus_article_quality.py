# step1d_plus_article_quality.py

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

OUT_JSONL = OUT_DIR / "structured_articles_step1d_plus.jsonl"
OUT_REVIEW_CSV = OUT_DIR / "structured_articles_step1d_plus_review.csv"
OUT_SUMMARY_JSON = OUT_DIR / "structured_articles_step1d_plus_summary.json"

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

RECOVERABLE_NEUTRAL = {
    "기업 경영", "경제일반", "비즈니스", "중견/중소기업", "집중기획"
}

DISCARD_CATEGORY_NAMES = {
    "영화", "방송/TV", "핫이슈", "사회일반", "아파트/분양", "국제정치"
}

# cause keyword / event type -> industry 매핑
CAUSE_INDUSTRY_HINTS = {
    "BIO": {
        "keywords": {"FDA", "임상", "치료제", "신약", "바이오", "제약", "허가", "승인", "의약품", "백신"},
        "event_types": {"approval_regulation", "earnings_positive", "earnings_negative"}
    },
    "SEMI": {
        "keywords": {"HBM", "DRAM", "낸드", "반도체", "파운드리", "웨이퍼", "칩", "후공정", "패키징"},
        "event_types": {"investment_capex", "contract_order", "earnings_positive"}
    },
    "CULTURE": {
        "keywords": {"아이돌", "컴백", "투어", "음원", "앨범", "드라마", "영화", "배우", "예능", "엔터"},
        "event_types": {"artist_activity", "rumor_or_expectation"}
    },
    "DEFENSE": {
        "keywords": {"방산", "무기", "국방", "유도탄", "전투기", "장갑차", "군수", "수주"},
        "event_types": {"contract_order", "policy_sentiment"}
    },
    "BANK_INSURANCE": {
        "keywords": {"은행", "보험", "손보", "생보", "예대마진", "대출", "금융", "자산운용"},
        "event_types": {"policy_sentiment", "earnings_positive", "earnings_negative"}
    },
    "BEAUTY": {
        "keywords": {"화장품", "K뷰티", "스킨케어", "코스메틱", "뷰티", "면세"},
        "event_types": {"overseas_expansion", "earnings_positive"}
    },
    "MOBILITY": {
        "keywords": {"자동차", "전기차", "배터리", "완성차", "모빌리티", "타이어", "자율주행"},
        "event_types": {"investment_capex", "policy_sentiment", "contract_order"}
    },
    "FOOD": {
        "keywords": {"식품", "라면", "음료", "제과", "푸드", "외식", "프랜차이즈"},
        "event_types": {"overseas_expansion", "earnings_positive"}
    },
    "CHEMICAL": {
        "keywords": {"화학", "정유", "석유화학", "LNG", "정제마진", "가스", "원유"},
        "event_types": {"policy_sentiment", "earnings_positive", "earnings_negative"}
    },
    "IT_AI": {
        "keywords": {"AI", "인공지능", "클라우드", "플랫폼", "SaaS", "데이터", "로봇", "소프트웨어"},
        "event_types": {"investment_capex", "policy_sentiment", "rumor_or_expectation"}
    }
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

def contains_any_count(text: str, keywords: set) -> int:
    return sum(1 for kw in keywords if kw in text)

# =========================
# 3. 회사 중심성
# =========================
def detect_company_mention_zone(company_candidate_rows: list):
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
    mention_zone = detect_company_mention_zone(company_rows)

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

    price_recap_hits = contains_any_count(text, PRICE_RECAP_KEYWORDS)
    macro_hits = contains_any_count(text, MACRO_POLICY_KEYWORDS)
    industry_hits = contains_any_count(text, INDUSTRY_NEWS_HINTS)

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
# 6. UNKNOWN 버킷 분류
# =========================
def classify_unknown_bucket(row: dict, company_info: dict, cause_info: dict, article_type_info: dict):
    top_industry_final = row.get("top_industry_final", "UNKNOWN")
    small_code_nm = row.get("small_code_nm", "")
    target_company = row.get("target_company")
    company_match_score = float(row.get("company_match_score", 0.0))
    cause_keyword_score = int(cause_info.get("cause_keyword_score", 0))
    is_company_direct = bool(company_info.get("is_company_direct", False))

    if top_industry_final != "UNKNOWN":
        return {
            "unknown_bucket": "known",
            "recoverable_reason": "already_known"
        }

    if small_code_nm in DISCARD_CATEGORY_NAMES:
        return {
            "unknown_bucket": "discard_unknown",
            "recoverable_reason": "discard_category"
        }

    if article_type_info["is_noise_article"]:
        return {
            "unknown_bucket": "discard_unknown",
            "recoverable_reason": "noise_article"
        }

    if small_code_nm in RECOVERABLE_NEUTRAL:
        if target_company and company_match_score >= 5.0:
            return {
                "unknown_bucket": "recoverable_unknown",
                "recoverable_reason": "neutral_category_with_company_match"
            }
        if is_company_direct:
            return {
                "unknown_bucket": "recoverable_unknown",
                "recoverable_reason": "neutral_category_company_direct"
            }
        if cause_keyword_score >= 1:
            return {
                "unknown_bucket": "recoverable_unknown",
                "recoverable_reason": "neutral_category_with_cause_keyword"
            }

    if target_company and company_match_score >= 6.0:
        return {
            "unknown_bucket": "recoverable_unknown",
            "recoverable_reason": "strong_company_match"
        }

    if cause_keyword_score >= 2 and not article_type_info["is_price_recap_article"]:
        return {
            "unknown_bucket": "recoverable_unknown",
            "recoverable_reason": "cause_signal_recoverable"
        }

    return {
        "unknown_bucket": "discard_unknown",
        "recoverable_reason": "weak_signal_unknown"
    }

# =========================
# 7. cause-based 산업 복구
# =========================
def recover_industry_from_cause(row: dict, cause_info: dict):
    if row.get("top_industry_final", "UNKNOWN") != "UNKNOWN":
        return {
            "top_industry_recovered": row.get("top_industry_final"),
            "industry_recovery_source": "already_known",
            "industry_recovery_score": 0
        }

    text = f"{row.get('title', '')} {row.get('body', '')}"
    cause_keywords = set(cause_info.get("cause_keywords", []))
    cause_event_types = set(cause_info.get("cause_event_type_candidates", []))

    scores = []

    for industry, rule in CAUSE_INDUSTRY_HINTS.items():
        score = 0

        kw_hits = sum(1 for kw in rule["keywords"] if kw in text or kw in cause_keywords)
        ev_hits = sum(1 for ev in rule["event_types"] if ev in cause_event_types)

        score += kw_hits * 2
        score += ev_hits * 2

        if score > 0:
            scores.append((industry, score))

    scores.sort(key=lambda x: (-x[1], x[0]))

    if scores and scores[0][1] >= 4:
        return {
            "top_industry_recovered": scores[0][0],
            "industry_recovery_source": "cause_based_recovery",
            "industry_recovery_score": scores[0][1]
        }

    return {
        "top_industry_recovered": "UNKNOWN",
        "industry_recovery_source": "not_recovered",
        "industry_recovery_score": 0
    }

# =========================
# 8. 매칭용 텍스트
# =========================
def build_article_match_text(row: dict, cause_info: dict, article_type_info: dict, recovered_industry: str):
    lead = " ".join(split_sentences(row.get("body", ""), 2))
    company = row.get("target_company") or ""
    causes = ", ".join(cause_info["cause_keywords"][:8])
    industry = recovered_industry if recovered_industry != "UNKNOWN" else row.get("top_industry_final", "UNKNOWN")

    parts = [
        company,
        row.get("title", ""),
        lead,
        f"industry={industry}",
        f"article_type={article_type_info['article_type']}",
        f"cause_keywords={causes}"
    ]
    return normalize_space(" | ".join([p for p in parts if p]))

# =========================
# 9. 매칭 후보 점수
# =========================
def compute_candidate_quality_score(row: dict, company_info: dict, cause_info: dict, article_type_info: dict, recovered_info: dict):
    score = 0

    if company_info["is_company_direct"]:
        score += 3
    if company_info["company_mention_zone"] == "title":
        score += 3
    elif company_info["company_mention_zone"] == "lead":
        score += 2
    elif company_info["company_mention_zone"] == "body":
        score += 1

    if cause_info["cause_keyword_score"] >= 1:
        score += 2
    if cause_info["cause_keyword_score"] >= 3:
        score += 1

    if not article_type_info["is_noise_article"]:
        score += 1
    if not article_type_info["is_price_recap_article"]:
        score += 1

    if row.get("small_code_nm") in RECOVERABLE_NEUTRAL:
        score += 1

    if recovered_info["industry_recovery_source"] == "cause_based_recovery":
        score += 2

    return score

# =========================
# 10. 메인
# =========================
def process_all():
    summary = {
        "total_rows": 0,
        "article_type_counter": Counter(),
        "unknown_bucket_counter": Counter(),
        "recoverable_reason_counter": Counter(),
        "industry_recovery_source_counter": Counter(),
        "cause_event_counter": Counter(),
        "high_signal_count": 0,
        "high_quality_candidate_count": 0,
        "recovered_industry_counter": Counter(),
    }

    review_rows = []

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, \
         OUT_JSONL.open("w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, start=1):
            row = json.loads(line)
            summary["total_rows"] += 1

            company_info = compute_company_directness(row)
            cause_info = extract_cause_signals(row.get("title", ""), row.get("body", ""))
            article_type_info = classify_article_type(row, company_info, cause_info)
            unknown_info = classify_unknown_bucket(row, company_info, cause_info, article_type_info)
            recovered_info = recover_industry_from_cause(row, cause_info)

            final_industry_for_matching = row.get("top_industry_final", "UNKNOWN")
            if final_industry_for_matching == "UNKNOWN" and recovered_info["top_industry_recovered"] != "UNKNOWN":
                final_industry_for_matching = recovered_info["top_industry_recovered"]

            article_match_text = build_article_match_text(
                row=row,
                cause_info=cause_info,
                article_type_info=article_type_info,
                recovered_industry=final_industry_for_matching
            )

            candidate_quality_score = compute_candidate_quality_score(
                row=row,
                company_info=company_info,
                cause_info=cause_info,
                article_type_info=article_type_info,
                recovered_info=recovered_info
            )

            is_match_candidate = bool(
                unknown_info["unknown_bucket"] != "discard_unknown"
                and article_type_info["is_noise_article"] is False
                and (
                    candidate_quality_score >= 6
                    or company_info["is_company_direct"]
                    or cause_info["cause_keyword_score"] >= 2
                )
            )

            out_row = {
                **row,
                **company_info,
                **cause_info,
                **article_type_info,
                **unknown_info,
                **recovered_info,
                "final_industry_for_matching": final_industry_for_matching,
                "candidate_quality_score": candidate_quality_score,
                "is_match_candidate": is_match_candidate,
                "article_match_text": article_match_text
            }

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            summary["article_type_counter"][out_row["article_type"]] += 1
            summary["unknown_bucket_counter"][out_row["unknown_bucket"]] += 1
            summary["recoverable_reason_counter"][out_row["recoverable_reason"]] += 1
            summary["industry_recovery_source_counter"][out_row["industry_recovery_source"]] += 1
            summary["recovered_industry_counter"][final_industry_for_matching] += 1

            if out_row["article_signal_strength"] >= 6.0:
                summary["high_signal_count"] += 1
            if out_row["is_match_candidate"]:
                summary["high_quality_candidate_count"] += 1

            for ev in out_row["cause_event_type_candidates"]:
                summary["cause_event_counter"][ev] += 1

            if len(review_rows) < 500:
                review_rows.append({
                    "article_id": out_row.get("article_id"),
                    "small_code_nm": out_row.get("small_code_nm"),
                    "title": out_row.get("title"),
                    "target_company": out_row.get("target_company"),
                    "top_industry_final": out_row.get("top_industry_final"),
                    "unknown_bucket": out_row.get("unknown_bucket"),
                    "recoverable_reason": out_row.get("recoverable_reason"),
                    "top_industry_recovered": out_row.get("top_industry_recovered"),
                    "industry_recovery_source": out_row.get("industry_recovery_source"),
                    "final_industry_for_matching": out_row.get("final_industry_for_matching"),
                    "article_type": out_row.get("article_type"),
                    "is_company_direct": out_row.get("is_company_direct"),
                    "cause_event_type_candidates": " | ".join(out_row.get("cause_event_type_candidates", [])),
                    "candidate_quality_score": out_row.get("candidate_quality_score"),
                    "is_match_candidate": out_row.get("is_match_candidate"),
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
        "unknown_bucket_counter": dict(summary["unknown_bucket_counter"]),
        "recoverable_reason_counter": dict(summary["recoverable_reason_counter"]),
        "industry_recovery_source_counter": dict(summary["industry_recovery_source_counter"]),
        "cause_event_counter": dict(summary["cause_event_counter"]),
        "recovered_industry_counter": dict(summary["recovered_industry_counter"]),
    }

    with OUT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary_dump, f, ensure_ascii=False, indent=2)

    print("=== STEP1-D PLUS DONE ===")
    print(json.dumps(summary_dump, ensure_ascii=False, indent=2))
    print("saved:", OUT_JSONL)
    print("saved:", OUT_REVIEW_CSV)
    print("saved:", OUT_SUMMARY_JSON)

if __name__ == "__main__":
    process_all()