import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# =========================
# 0. 경로 설정
# =========================
INPUT_HIGH = Path("./outputs_step1/match_candidates_high_confidence.jsonl")
INPUT_REVIEW = Path("./outputs_step1/match_candidates_review_needed.jsonl")

OUT_DIR = Path("./outputs_step3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "article_embedding_source.csv"

# =========================
# 1. 텍스트 정규화 유틸
# =========================
SPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"<[^>]+>")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
COPYRIGHT_RE = re.compile(r"무단전재.*?금지|재배포.*?금지|ⓒ\s*.*")
REPORTER_RE = re.compile(
    r"\[[^\]]*기자[^\]]*\]"
    r"|\([^\)]*기자[^\)]*\)"
    r"|[가-힣A-Za-z\s]{1,20}\s*기자"
    r"|[가-힣A-Za-z\s]{1,20}\s*특파원"
)

# 한국어 문장 분리용 간단 패턴
SENT_SPLIT_RE = re.compile(
    r"(?<=[\.\!\?。！？])\s+"
    r"|(?<=다\.)\s+"
    r"|(?<=요\.)\s+"
    r"|(?<=니다\.)\s+"
)

def norm_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(norm_text(x) for x in v if x is not None)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)

    s = str(v)
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    s = HTML_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def clean_body(text: Any) -> str:
    text = norm_text(text)
    if not text:
        return ""

    text = EMAIL_RE.sub(" ", text)
    text = REPORTER_RE.sub(" ", text)
    text = COPYRIGHT_RE.sub(" ", text)

    # 불필요한 메타 패턴 제거
    text = re.sub(r"\[[^\]]+\s*=\s*[^\]]+\]", " ", text)
    text = re.sub(r"\([^\)]*특파원[^\)]*\)", " ", text)
    text = re.sub(r"[가-힣A-Za-z\s]*=\s*[가-힣A-Za-z0-9._-]+@[A-Za-z0-9._-]+", " ", text)

    text = SPACE_RE.sub(" ", text).strip()
    return text


def ensure_list_str(v: Any) -> List[str]:
    if v is None:
        return []

    if isinstance(v, list):
        out = []
        for x in v:
            sx = norm_text(x)
            if sx:
                out.append(sx)
        return out

    s = norm_text(v)
    if not s:
        return []

    # JSON 문자열 list 처리
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [norm_text(x) for x in parsed if norm_text(x)]
        except Exception:
            pass

    # 일반 구분자 split
    return [x.strip() for x in re.split(r"[,/|;]+", s) if x.strip()]


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================
# 2. 본문 요약
# =========================
def split_sentences(text: str) -> List[str]:
    text = clean_body(text)
    if not text:
        return []

    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

    # 분리가 잘 안 되면 길이 기준 fallback
    if len(sents) <= 1 and len(text) > 120:
        chunks = re.split(r"(?<=\.)\s+|(?<=다\.)\s+", text)
        sents = [c.strip() for c in chunks if c.strip()]

    return sents


def take_summary_sentences(text: Any, max_sentences: int = 3, max_chars: int = 500) -> str:
    text = clean_body(text)
    if not text:
        return ""

    sents = split_sentences(text)
    if not sents:
        return text[:max_chars].strip()

    picked = []
    total = 0

    for sent in sents:
        if len(picked) >= max_sentences:
            break
        if total + len(sent) > max_chars and picked:
            break
        picked.append(sent)
        total += len(sent) + 1

    summary = " ".join(picked).strip()
    if not summary:
        summary = text[:max_chars].strip()

    return summary


# =========================
# 3. 임베딩용 article_match_text 생성
# =========================
def build_article_match_text(row: Dict[str, Any]) -> str:
    title = norm_text(row.get("title"))
    body = clean_body(row.get("body"))
    body_summary = take_summary_sentences(body, max_sentences=2, max_chars=260)

    industry = norm_text(row.get("final_industry_for_matching"))
    article_type = norm_text(row.get("article_type"))
    market_scope = norm_text(row.get("market_scope"))
    target_company = norm_text(row.get("target_company"))

    company_candidates = uniq_keep_order(ensure_list_str(row.get("company_candidates")))
    cause_keywords = uniq_keep_order(ensure_list_str(row.get("cause_keywords")))

    meta_parts = []

    if target_company:
        meta_parts.append(f"target_company={target_company}")
    elif company_candidates:
        meta_parts.append(f"company_candidates={', '.join(company_candidates[:5])}")

    if industry:
        meta_parts.append(f"industry={industry}")

    if article_type:
        meta_parts.append(f"article_type={article_type}")

    if market_scope:
        meta_parts.append(f"market_scope={market_scope}")

    if cause_keywords:
        meta_parts.append(f"cause_keywords={', '.join(cause_keywords[:8])}")

    parts = []
    if title:
        parts.append(title)
    if body_summary:
        parts.append(body_summary)
    parts.extend(meta_parts)

    return " | ".join([p for p in parts if p])


# =========================
# 4. JSONL 로드
# =========================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    if not path.exists():
        print(f"[WARN] file not found: {path}")
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception as e:
                print(f"[WARN] json parse failed: {path} line={line_no} err={e}")

    print(f"[LOAD] {path} -> {len(rows)} rows")
    return rows


# =========================
# 5. 레코드 변환
# =========================
def transform_rows(rows: List[Dict[str, Any]], source_group: str) -> List[Dict[str, Any]]:
    out = []

    for row in rows:
        article_id = norm_text(row.get("article_id"))
        title = norm_text(row.get("title"))
        body = clean_body(row.get("body"))
        body_summary = take_summary_sentences(body, max_sentences=3, max_chars=500)

        published_at = norm_text(row.get("published_at"))
        published_date = norm_text(row.get("published_date"))
        url = norm_text(row.get("url"))

        final_industry = norm_text(row.get("final_industry_for_matching"))
        article_type = norm_text(row.get("article_type"))
        market_scope = norm_text(row.get("market_scope"))
        target_company = norm_text(row.get("target_company"))

        cause_keywords = uniq_keep_order(ensure_list_str(row.get("cause_keywords")))
        company_candidates = uniq_keep_order(ensure_list_str(row.get("company_candidates")))

        article_signal_strength = row.get("article_signal_strength")
        candidate_quality_score = row.get("candidate_quality_score")
        is_match_candidate = row.get("is_match_candidate")
        unknown_bucket = norm_text(row.get("unknown_bucket"))
        recoverable_reason = norm_text(row.get("recoverable_reason"))

        article_match_text = build_article_match_text(row)

        out.append(
            {
                "article_id": article_id,
                "source_group": source_group,
                "title": title,
                "body": body,
                "body_summary": body_summary,
                "published_at": published_at,
                "published_date": published_date,
                "url": url,
                "final_industry_for_matching": final_industry,
                "article_type": article_type,
                "market_scope": market_scope,
                "target_company": target_company,
                "company_candidates": json.dumps(company_candidates, ensure_ascii=False),
                "cause_keywords": json.dumps(cause_keywords, ensure_ascii=False),
                "article_signal_strength": article_signal_strength,
                "candidate_quality_score": candidate_quality_score,
                "is_match_candidate": is_match_candidate,
                "unknown_bucket": unknown_bucket,
                "recoverable_reason": recoverable_reason,
                "article_match_text": article_match_text,
            }
        )

    return out


# =========================
# 6. 메인
# =========================
def main():
    high_rows = load_jsonl(INPUT_HIGH)
    review_rows = load_jsonl(INPUT_REVIEW)

    records = []
    records.extend(transform_rows(high_rows, "high_confidence"))
    records.extend(transform_rows(review_rows, "review_needed"))

    if not records:
        print("[ERROR] no rows loaded")
        return

    df = pd.DataFrame(records)

    # 동일 article_id 중 high_confidence 우선
    priority_map = {"high_confidence": 0, "review_needed": 1}
    df["source_priority"] = df["source_group"].map(priority_map).fillna(9)

    df = (
        df.sort_values(["article_id", "source_priority"])
          .drop_duplicates(subset=["article_id"], keep="first")
          .drop(columns=["source_priority"])
          .reset_index(drop=True)
    )

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n=== DONE ===")
    print(f"saved: {OUT_CSV}")
    print(f"rows: {len(df)}")
    print(f"high_confidence_loaded: {len(high_rows)}")
    print(f"review_needed_loaded: {len(review_rows)}")

    print("\n[Sample]")
    sample_cols = [
        "article_id",
        "source_group",
        "title",
        "body_summary",
        "final_industry_for_matching",
        "article_type",
        "article_match_text",
    ]
    print(df[sample_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()