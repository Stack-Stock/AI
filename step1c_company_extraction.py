# step1c_company_extraction.py

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# =========================
# 0. 경로 설정
# =========================
PROJECT_ROOT = Path(r"C:\Users\SSAFY\Desktop\seohyung\Stack&Stock\projects\article_embedding_V2.0")
RAW_ROOT = Path(r"C:\Users\SSAFY\Desktop\seohyung\Stack&Stock\2025")
COMPANY_DICT_CSV = PROJECT_ROOT / "company_dictionary.csv"

OUT_DIR = PROJECT_ROOT / "outputs_step1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "structured_articles_step1c.jsonl"
OUT_REVIEW_CSV = OUT_DIR / "structured_articles_step1c_review.csv"
OUT_SUMMARY_JSON = OUT_DIR / "structured_articles_step1c_summary.json"

MAX_FILES = None   # 전체 돌릴 때 None

# =========================
# 1. 산업군 규칙
# =========================
INDUSTRY_RULES = {
    "FOOD": {
        "small_code_ids": {"MK100306", "MK100312", "MK101604"},
        "keywords": {"식품", "음식료", "외식", "프랜차이즈", "푸드", "음료", "커피", "주류", "간편식", "라면", "제과"}
    },
    "MOBILITY": {
        "small_code_ids": {"MK100308", "MK300103", "MK500309"},
        "keywords": {"자동차", "전기차", "모빌리티", "배터리", "자율주행", "완성차", "차량", "타이어", "이차전지"}
    },
    "IT_AI": {
        "small_code_ids": {
            "MK100404", "MK100405", "MK101501", "MK101503", "MK101504",
            "MK101506", "MK101507", "MK101508", "MK101509", "MK101510"
        },
        "keywords": {"AI", "인공지능", "LLM", "클라우드", "소프트웨어", "플랫폼", "데이터", "SaaS", "생성형AI", "챗봇", "로봇"}
    },
    "BIO": {
        "small_code_ids": {"MK100407", "MK101402", "MK101403", "MK400109"},
        "keywords": {"제약", "바이오", "신약", "임상", "의료", "치료제", "백신", "FDA", "항암", "헬스케어", "의약품"}
    },
    "SEMI": {
        "small_code_ids": {"MK100305", "MK100404"},
        "keywords": {"반도체", "메모리", "HBM", "파운드리", "칩", "후공정", "웨이퍼", "DRAM", "낸드", "패키징"}
    },
    "BEAUTY": {
        "small_code_ids": {"MK101605", "MK101107", "MK500311"},
        "keywords": {"뷰티", "화장품", "미용", "스킨케어", "메이크업", "K뷰티", "에스테틱", "코스메틱"}
    },
    "DEFENSE": {
        "small_code_ids": {"MK100319", "MK100702", "MK100710"},
        "keywords": {"방산", "국방", "무기", "군수", "미사일", "전투기", "장갑차", "무인기", "유도탄"}
    },
    "BANK_INSURANCE": {
        "small_code_ids": {"MK100201", "MK100203", "MK100204"},
        "keywords": {"은행", "보험", "금융", "금리", "대출", "예대마진", "손해보험", "생명보험", "증권", "여신", "자산운용"}
    },
    "CHEMICAL": {
        "small_code_ids": {"MK100304", "MK100316", "MK100510", "MK101006"},
        "keywords": {"화학", "에너지", "정유", "원유", "석유화학", "가스", "LNG", "원자재", "나프타", "정제마진"}
    },
    "CULTURE": {
        "small_code_ids": {"MK101102", "MK101104", "MK101105", "MK101108"},
        "keywords": {"연예", "드라마", "영화", "방송", "콘텐츠", "OTT", "음원", "아이돌", "배우", "엔터", "예능", "컴백", "투어"}
    }
}

INDUSTRY_PRIORITY = [
    "SEMI", "BIO", "DEFENSE", "BANK_INSURANCE", "CHEMICAL",
    "MOBILITY", "BEAUTY", "FOOD", "IT_AI", "CULTURE"
]

EXCLUDE_SMALL_CODE_IDS = {
    "MK101301", "MK101302", "MK101303", "MK101304", "MK101305", "MK101306",
    "MK101320", "MK101321", "MK101322", "MK101323",
    "MK900101", "MK900102", "MK900103", "MK990001",
    "MK981900", "MK989900"
}

EXCLUDE_SMALL_CODE_NAMES = {
    "사설", "오피니언", "칼럼", "기자24시", "카툰", "사고/알림",
    "광고기사", "중복기사", "보도자료", "광고", "전면광고"
}

NEUTRAL_CATEGORY_NAMES = {
    "기업 경영", "중견/중소기업", "경제일반", "비즈니스", "집중기획", "CEO 라운지"
}

# =========================
# 2. 유틸
# =========================
def safe_str(x):
    return "" if x is None else str(x).strip()

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def strip_html(text: str) -> str:
    text = text or ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&amp;", "&")
    return normalize_space(text)

def clean_body(text: str) -> str:
    text = text or ""
    text = re.sub(r"\[[^\]]*기자[^\]]*\]", " ", text)
    text = re.sub(r"\[본 기사[^\]]*\]", " ", text)
    text = re.sub(r"무단전재 및 재배포 금지", " ", text)
    text = re.sub(r"저작권자.*", " ", text)
    return normalize_space(text)

def parse_date(date_text: str) -> str:
    s = safe_str(date_text)
    patterns = [
        r"(20\d{2})-(\d{2})-(\d{2})",
        r"(20\d{2})\.(\d{2})\.(\d{2})",
        r"(20\d{2})/(\d{2})/(\d{2})",
        r"(20\d{2})(\d{2})(\d{2})",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""

def parse_month(date_text: str) -> str:
    d = parse_date(date_text)
    return d[:7] if d else "UNKNOWN"

def split_sentences(text: str, max_sentences: int = 2):
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!다요])\s+", text)
    parts = [normalize_space(x) for x in parts if normalize_space(x)]
    return parts[:max_sentences]

def read_json_file(path: Path):
    encodings = ["utf-8", "utf-8-sig", "cp949"]
    last_error = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except Exception as e:
            last_error = e
    raise last_error

def iter_json_files(root: Path):
    for month_dir in sorted(root.glob("*")):
        if month_dir.is_dir():
            yield from month_dir.rglob("*.json")

# =========================
# 3. 원본 기사 파싱
# =========================
def parse_article_obj(obj: dict) -> dict:
    article = obj.get("article", {}) or {}
    article_body = obj.get("article_body", {}) or {}
    categories = obj.get("categories", []) or []
    cat0 = categories[0] if categories else {}

    title = normalize_space(safe_str(article.get("title")))
    body = clean_body(strip_html(safe_str(article_body.get("body"))))
    published_at = safe_str(article.get("reg_dt") or article.get("service_daytime"))
    article_id = safe_str(article.get("article_id"))
    url = safe_str(obj.get("article_url"))

    return {
        "article_id": article_id,
        "title": title,
        "body": body,
        "small_code_id": safe_str(cat0.get("small_code_id")),
        "small_code_nm": safe_str(cat0.get("small_code_nm")),
        "middle_code_nm": safe_str(cat0.get("middle_code_nm")),
        "published_at": published_at,
        "published_date": parse_date(published_at),
        "month": parse_month(published_at),
        "url": url,
    }

# =========================
# 4. 산업군 분류
# =========================
def classify_industries(article: dict):
    sid = article["small_code_id"]
    snm = article["small_code_nm"]
    title = article["title"]
    body = article["body"]

    if sid in EXCLUDE_SMALL_CODE_IDS or snm in EXCLUDE_SMALL_CODE_NAMES:
        return [], {}

    scored = []

    for industry, rule in INDUSTRY_RULES.items():
        score = 0
        reason = {
            "category_hit": False,
            "title_hits": [],
            "body_hits": []
        }

        if sid in rule["small_code_ids"]:
            score += 5
            reason["category_hit"] = True

        title_hits = [kw for kw in rule["keywords"] if kw in title]
        body_hits = [kw for kw in rule["keywords"] if kw in body]

        score += min(len(title_hits), 3) * 2
        score += min(len(body_hits), 4) * 1

        reason["title_hits"] = title_hits[:5]
        reason["body_hits"] = body_hits[:5]

        if score >= 4:
            scored.append((industry, score, reason))

    scored.sort(
        key=lambda x: (
            -x[1],
            INDUSTRY_PRIORITY.index(x[0]) if x[0] in INDUSTRY_PRIORITY else 999
        )
    )

    industries = [x[0] for x in scored]
    reasons = {x[0]: {"score": x[1], **x[2]} for x in scored}
    return industries, reasons

def pick_top_industry(industries):
    if not industries:
        return "UNKNOWN"
    for industry in INDUSTRY_PRIORITY:
        if industry in industries:
            return industry
    return industries[0]

# =========================
# 5. 회사 사전 로드
# =========================
def load_company_dictionary(csv_path: Path):
    company_dict_by_industry = defaultdict(dict)
    company_to_industry = {}
    alias_entries = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            industry = safe_str(row.get("industry"))
            company_name = safe_str(row.get("company_name"))
            aliases_raw = safe_str(row.get("aliases"))

            if not industry or not company_name:
                continue

            aliases = [a.strip() for a in aliases_raw.split("|") if a.strip()]
            if company_name not in aliases:
                aliases.append(company_name)

            # 짧은 alias 최소 필터
            aliases = [a for a in aliases if len(a) >= 2]
            aliases = sorted(set(aliases), key=len, reverse=True)

            company_dict_by_industry[industry][company_name] = aliases
            company_to_industry[company_name] = industry

            for alias in aliases:
                alias_entries.append({
                    "alias": alias,
                    "company_name": company_name,
                    "industry": industry
                })

    # 긴 alias 우선 검색
    alias_entries.sort(key=lambda x: len(x["alias"]), reverse=True)
    return dict(company_dict_by_industry), company_to_industry, alias_entries

# =========================
# 6. 회사 후보 추출
# =========================
def count_occurrences(text: str, sub: str) -> int:
    if not text or not sub:
        return 0
    return text.count(sub)

def extract_company_candidates(article: dict, alias_entries):
    title = article["title"]
    body = article["body"]
    lead = " ".join(split_sentences(body, 2))

    rows = []

    for entry in alias_entries:
        alias = entry["alias"]
        company_name = entry["company_name"]
        industry = entry["industry"]

        title_hits = count_occurrences(title, alias)
        lead_hits = count_occurrences(lead, alias)
        body_hits = count_occurrences(body, alias)

        total_hits = title_hits + lead_hits + body_hits
        if total_hits == 0:
            continue

        # 점수: 제목 > 리드 > 본문
        score = (
            title_hits * 5.0 +
            lead_hits * 2.0 +
            body_hits * 1.0 +
            min(len(alias), 10) * 0.05
        )

        rows.append({
            "company_name": company_name,
            "industry": industry,
            "matched_alias": alias,
            "title_hits": title_hits,
            "lead_hits": lead_hits,
            "body_hits": body_hits,
            "score": round(score, 3)
        })

    # 같은 회사가 여러 alias로 잡히면 최고 점수만 남김
    best_by_company = {}
    for row in rows:
        cname = row["company_name"]
        if cname not in best_by_company or row["score"] > best_by_company[cname]["score"]:
            best_by_company[cname] = row

    dedup_rows = list(best_by_company.values())
    dedup_rows.sort(
        key=lambda x: (-x["score"], -x["title_hits"], -x["lead_hits"], -x["body_hits"], x["company_name"])
    )

    candidates = [r["company_name"] for r in dedup_rows[:5]]
    target_company = dedup_rows[0]["company_name"] if dedup_rows else None
    company_match_score = dedup_rows[0]["score"] if dedup_rows else 0.0

    if dedup_rows:
        top = dedup_rows[0]
        if top["title_hits"] > 0:
            reason = "title_alias_match"
        elif top["lead_hits"] > 0:
            reason = "lead_alias_match"
        else:
            reason = "body_alias_match"
    else:
        reason = "no_company_match"

    return {
        "company_candidates": candidates,
        "company_candidate_rows": dedup_rows[:5],
        "target_company": target_company,
        "company_match_score": round(company_match_score, 3),
        "company_match_reason": reason
    }

# =========================
# 7. 산업 최종 보정
# =========================
def decide_final_industry(article: dict, rule_top_industry: str, company_info: dict, company_to_industry: dict):
    small_code_nm = article["small_code_nm"]
    target_company = company_info["target_company"]

    if target_company:
        company_industry = company_to_industry.get(target_company, "UNKNOWN")
    else:
        company_industry = "UNKNOWN"

    # 강한 회사 매칭이면 회사 산업 우선
    if target_company and company_info["company_match_score"] >= 5.0:
        if rule_top_industry == "UNKNOWN":
            return company_industry, "company_alias_override_unknown"
        if small_code_nm in NEUTRAL_CATEGORY_NAMES:
            return company_industry, "company_alias_override_neutral_category"
        if company_industry != "UNKNOWN" and rule_top_industry != company_industry:
            return company_industry, "company_alias_override_conflict"

    # 기본은 rule 유지
    return rule_top_industry, "rule_based"

# =========================
# 8. 메인
# =========================
def process_all():
    company_dict_by_industry, company_to_industry, alias_entries = load_company_dictionary(COMPANY_DICT_CSV)

    summary = {
        "total_files_seen": 0,
        "parsed_files": 0,
        "kept_articles": 0,
        "skipped_json_error": 0,
        "skipped_empty_text": 0,
        "skipped_excluded": 0,
        "rule_unknown_count": 0,
        "company_match_count": 0,
        "company_override_count": 0,
        "final_unknown_count": 0,
        "final_top_industry_counter": Counter(),
        "company_match_reason_counter": Counter(),
        "industry_source_counter": Counter(),
        "unknown_samples": []
    }

    review_rows = []

    with OUT_JSONL.open("w", encoding="utf-8") as jout:
        for idx, p in enumerate(iter_json_files(RAW_ROOT), start=1):
            if MAX_FILES is not None and idx > MAX_FILES:
                break

            summary["total_files_seen"] += 1

            if idx % 100 == 0:
                print(f"[progress] {idx} files seen... current={p}")

            try:
                obj = read_json_file(p)
                summary["parsed_files"] += 1
            except Exception as e:
                summary["skipped_json_error"] += 1
                if summary["skipped_json_error"] <= 20:
                    print(f"[json_error] {p} | {e}")
                continue

            article = parse_article_obj(obj)
            if not article["title"] and not article["body"]:
                summary["skipped_empty_text"] += 1
                continue

            if article["small_code_id"] in EXCLUDE_SMALL_CODE_IDS or article["small_code_nm"] in EXCLUDE_SMALL_CODE_NAMES:
                summary["skipped_excluded"] += 1
                continue

            industries, industry_reasons = classify_industries(article)
            top_industry_rule = pick_top_industry(industries)
            if top_industry_rule == "UNKNOWN":
                summary["rule_unknown_count"] += 1

            company_info = extract_company_candidates(article, alias_entries)
            if company_info["target_company"]:
                summary["company_match_count"] += 1
            summary["company_match_reason_counter"][company_info["company_match_reason"]] += 1

            top_industry_final, industry_source = decide_final_industry(
                article=article,
                rule_top_industry=top_industry_rule,
                company_info=company_info,
                company_to_industry=company_to_industry
            )

            if industry_source != "rule_based":
                summary["company_override_count"] += 1

            if top_industry_final == "UNKNOWN":
                summary["final_unknown_count"] += 1
                if len(summary["unknown_samples"]) < 30:
                    summary["unknown_samples"].append({
                        "article_id": article["article_id"],
                        "small_code_nm": article["small_code_nm"],
                        "title": article["title"][:80]
                    })

            # 이번 단계에서는 final_unknown도 일단 저장은 함
            record = {
                **article,
                "industries_rule": industries,
                "top_industry_rule": top_industry_rule,
                "industry_reasons_rule": industry_reasons,
                "company_candidates": company_info["company_candidates"],
                "company_candidate_rows": company_info["company_candidate_rows"],
                "target_company": company_info["target_company"],
                "company_match_score": company_info["company_match_score"],
                "company_match_reason": company_info["company_match_reason"],
                "top_industry_final": top_industry_final,
                "industry_source": industry_source,
                "_source_path": str(p)
            }

            jout.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary["kept_articles"] += 1

            summary["final_top_industry_counter"][top_industry_final] += 1
            summary["industry_source_counter"][industry_source] += 1

            if len(review_rows) < 400:
                review_rows.append({
                    "article_id": record["article_id"],
                    "published_date": record["published_date"],
                    "small_code_nm": record["small_code_nm"],
                    "title": record["title"],
                    "top_industry_rule": record["top_industry_rule"],
                    "target_company": record["target_company"],
                    "company_match_score": record["company_match_score"],
                    "company_match_reason": record["company_match_reason"],
                    "top_industry_final": record["top_industry_final"],
                    "industry_source": record["industry_source"],
                    "company_candidates": " | ".join(record["company_candidates"]),
                    "url": record["url"]
                })

    if review_rows:
        fieldnames = list(review_rows[0].keys())
        with OUT_REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_rows)

    summary_dump = {
        **summary,
        "final_top_industry_counter": dict(summary["final_top_industry_counter"]),
        "company_match_reason_counter": dict(summary["company_match_reason_counter"]),
        "industry_source_counter": dict(summary["industry_source_counter"]),
    }

    with OUT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary_dump, f, ensure_ascii=False, indent=2)

    print("=== STEP1-C DONE ===")
    print(json.dumps(summary_dump, ensure_ascii=False, indent=2))
    print("saved:", OUT_JSONL)
    print("saved:", OUT_REVIEW_CSV)
    print("saved:", OUT_SUMMARY_JSON)

if __name__ == "__main__":
    process_all()