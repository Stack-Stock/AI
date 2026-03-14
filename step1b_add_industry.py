# step1b_add_industry.py

import csv
import json
import re
from collections import Counter
from pathlib import Path

# =========================
# 0. 경로 설정
# =========================
RAW_ROOT = Path(r"C:\Users\SSAFY\Desktop\seohyung\Stack&Stock\2025")
OUT_DIR = Path("./outputs_step1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "structured_articles_step1b.jsonl"
OUT_REVIEW_CSV = OUT_DIR / "structured_articles_step1b_review.csv"
OUT_SUMMARY_JSON = OUT_DIR / "structured_articles_step1b_summary.json"

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

# =========================
# 2. 공통 유틸
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
# 5. 메인
# =========================
def process_all():
    summary = {
        "total_files_seen": 0,
        "parsed_files": 0,
        "kept_articles": 0,
        "skipped_json_error": 0,
        "skipped_empty_text": 0,
        "skipped_no_industry": 0,
        "top_industry_counter": Counter(),
        "multi_industry_counter": 0,
        "unknown_small_code_samples": []
    }

    review_rows = []

    files = iter_json_files(RAW_ROOT)

    with OUT_JSONL.open("w", encoding="utf-8") as jout:
        for idx, p in enumerate(files, start=1):
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

            industries, industry_reasons = classify_industries(article)
            top_industry = pick_top_industry(industries)

            if not industries:
                summary["skipped_no_industry"] += 1
                if len(summary["unknown_small_code_samples"]) < 30:
                    summary["unknown_small_code_samples"].append({
                        "article_id": article["article_id"],
                        "small_code_id": article["small_code_id"],
                        "small_code_nm": article["small_code_nm"],
                        "title": article["title"][:80]
                    })
                continue

            if len(industries) >= 2:
                summary["multi_industry_counter"] += 1

            summary["top_industry_counter"][top_industry] += 1

            record = {
                **article,
                "industries": industries,
                "top_industry": top_industry,
                "industry_reasons": industry_reasons,
                "_source_path": str(p)
            }

            jout.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary["kept_articles"] += 1

            if len(review_rows) < 300:
                review_rows.append({
                    "article_id": record["article_id"],
                    "published_date": record["published_date"],
                    "small_code_id": record["small_code_id"],
                    "small_code_nm": record["small_code_nm"],
                    "industries": " | ".join(record["industries"]),
                    "top_industry": record["top_industry"],
                    "title": record["title"],
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
        "top_industry_counter": dict(summary["top_industry_counter"])
    }

    with OUT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary_dump, f, ensure_ascii=False, indent=2)

    print("=== STEP1-B DONE ===")
    print(json.dumps(summary_dump, ensure_ascii=False, indent=2))
    print("saved:", OUT_JSONL)
    print("saved:", OUT_REVIEW_CSV)
    print("saved:", OUT_SUMMARY_JSON)

if __name__ == "__main__":
    process_all()