import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RAW_ROOT = Path(r"C:\Users\SSAFY\Desktop\seohyung\Stack&Stock\2025")
COMPANY_DICT_CSV = Path("./company_dictionary.csv")

OUT_DIR = Path("./outputs_step1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "structured_articles_step1.jsonl"
OUT_REVIEW_CSV = OUT_DIR / "structured_articles_step1_review.csv"
OUT_SUMMARY_JSON = OUT_DIR / "structured_articles_step1_summary.json"

MAX_FILES = None # 전체 돌릴 땐 None 으로 바꾸기

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

# 임시 최소 버전
def process_all():
    summary = {
        "total_files_seen": 0,
        "parsed_files": 0,
        "kept_articles": 0,
        "skipped_json_error": 0,
        "skipped_empty_text": 0,
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

            # STEP1 최소 출력 먼저
            record = {
                **article,
                "_source_path": str(p)
            }
            jout.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary["kept_articles"] += 1

            if len(review_rows) < 200:
                review_rows.append(record)

    if review_rows:
        fieldnames = list(review_rows[0].keys())
        with OUT_REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_rows)

    with OUT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== STEP1 MINIMAL DONE ===")
    print(summary)
    print("saved:", OUT_JSONL)
    print("saved:", OUT_REVIEW_CSV)
    print("saved:", OUT_SUMMARY_JSON)

if __name__ == "__main__":
    process_all()