import os
import json
import time
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
from anthropic import Anthropic

# =========================
# 0. 경로 / 설정
# =========================
STEP4_DIR = Path("./outputs_step4")
STEP5_DIR = Path("./outputs_step5")
STEP5_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CASE_DATASET = STEP4_DIR / "case_selection_final.csv"
INPUT_STOCK_UNIVERSE = STEP5_DIR / "stock_universe_top30.csv"

OUT_CSV = STEP5_DIR / "game_case_generated.csv"
OUT_JSONL = STEP5_DIR / "game_case_generated.jsonl"
OUT_INSERT_READY = STEP5_DIR / "game_case_insert_ready.csv"
OUT_ERROR_LOG = STEP5_DIR / "game_case_generation_errors.csv"

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

# 전체 생성이면 None
MAX_CASES = None

SLEEP_SEC = 0.8
RETRY_COUNT = 3
RETRY_SLEEP_SEC = 3

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# up_down_json 생성 설정
MAX_GAME_ABS = 0.12
SCALE_BASE = 0.12
NOISE_STD = 0.012
NOISE_CLIP = 0.03

SECTOR_BIAS_PROB = 0.65
SECTOR_PEER_MIN = 2
SECTOR_PEER_MAX = 4

MARKET_BIAS_PROB = 0.35
MARKET_BIAS_MAG = 0.006

NON_EVENT_RELATIVE_CAP = 0.78

REPLACE_COMPANY_WITH = "해당 기업"

# 과생성 탐지용
SUSPICIOUS_PATTERNS = [
    r"\d+조원", r"\d+억원", r"\d+만대", r"\d+%", r"\d+개년",
    r"연방정부", r"주정부", r"장기 공급 계약", r"수주 계약",
    r"하이브리드", r"전기차 옵션", r"연비", r"점유율"
]

# 직접 방향 표현 금지 검사
FORBIDDEN_DIRECTION_PATTERNS = [
    r"급등", r"급락", r"상승", r"하락", r"오를", r"내릴",
    r"올랐", r"내렸", r"상한가", r"하한가",
    r"\d+(\.\d+)?\s*%", r"수익률", r"등락률"
]

# =========================
# 1. 유틸
# =========================
def safe_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def compact_text(s: str) -> str:
    return " ".join(str(s).split()).strip()


def json_dumps_ensure(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def parse_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        if lines and lines[0].strip().lower() == "json":
            lines = lines[1:]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def sanitize_phone_text(phone: str, company: str) -> str:
    phone = compact_text(phone)
    company = safe_str(company)
    if company and company in phone:
        phone = phone.replace(company, REPLACE_COMPANY_WITH)
    return phone


def sanitize_no_company_text(text: str, company: str) -> str:
    text = compact_text(text)
    company = safe_str(company)
    if company and company in text:
        text = text.replace(company, REPLACE_COMPANY_WITH)
    return text


def infer_direction_label(real_return: float) -> str:
    if real_return > 0:
        return "상승"
    if real_return < 0:
        return "하락"
    return "보합"


def scale_game_return(real_return: float, max_game_abs: float = MAX_GAME_ABS, scale: float = SCALE_BASE) -> float:
    return float(max_game_abs * math.tanh(real_return / scale))


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sample_noise(cap_abs: float, mean_shift: float = 0.0) -> float:
    for _ in range(100):
        r = random.gauss(mean_shift, NOISE_STD)
        r = clip(r, -NOISE_CLIP, NOISE_CLIP)
        if abs(r) <= cap_abs:
            return float(r)
    return float(clip(mean_shift, -cap_abs, cap_abs))


def normalize_key(company: str, ticker: str) -> str:
    return f"{safe_str(company)}|{safe_str(ticker)}"


def ensure_column(df: pd.DataFrame, col: str, default: Any = "") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df


def looks_hallucinated(text: str, allowed_text: str = "") -> bool:
    text = safe_str(text)
    allowed_text = safe_str(allowed_text)

    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text):
            if not re.search(pattern, allowed_text):
                return True
    return False


def contains_forbidden_direction_words(text: str) -> bool:
    text = safe_str(text)
    for pattern in FORBIDDEN_DIRECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def strip_direction_leakage(text: str) -> str:
    text = safe_str(text)

    # 퍼센트 / 수익률 류 제거
    text = re.sub(r"\d+(\.\d+)?\s*%", "", text)
    text = re.sub(r"(수익률|등락률)", "", text)

    # 직접 방향 표현 제거
    text = re.sub(r"(급등|급락|상승세|하락세|상승|하락|상한가|하한가)", "", text)
    text = re.sub(r"(오를|내릴|올랐|내렸)", "", text)

    # 가격/주가 직접 표현 약화
    text = re.sub(r"주가", "흐름", text)

    # 과한 숫자 힌트 일부 완화
    text = re.sub(r"\b\d+(,\d{3})+\b", "", text)

    return compact_text(text)


def sanitize_visible_text(text: str, company: str = "") -> str:
    text = compact_text(text)
    text = strip_direction_leakage(text)
    return text


def validate_generated_fields(story: str, phone: str, tv: str, newspaper: str, company: str, allowed_basis: str) -> List[str]:
    problems = []

    if not story:
        problems.append("empty_story")
    if not phone:
        problems.append("empty_phone")
    if not tv:
        problems.append("empty_tv")
    if not newspaper:
        problems.append("empty_newspaper")

    if company and company in phone:
        problems.append("company_in_phone")

    if company and company in tv:
        problems.append("company_in_tv")

    if company and company not in newspaper:
        problems.append("company_missing_in_newspaper")

    if contains_forbidden_direction_words(phone):
        problems.append("forbidden_direction_in_phone")
    if contains_forbidden_direction_words(tv):
        problems.append("forbidden_direction_in_tv")
    if contains_forbidden_direction_words(newspaper):
        problems.append("forbidden_direction_in_newspaper")

    if looks_hallucinated(tv, allowed_basis):
        problems.append("hallucination_in_tv")
    if looks_hallucinated(newspaper, allowed_basis):
        problems.append("hallucination_in_newspaper")

    return problems


# =========================
# 2. Claude 프롬프트
# =========================
SYSTEM_PROMPT = """
너는 뉴스 기반 투자 판단 게임의 케이스 작가다.

입력으로 특정 종목 이벤트와 기사 정보가 주어진다.
출력은 반드시 JSON object 하나만 반환한다.

절대 규칙:
1. 반드시 입력으로 제공된 정보만 사용하라.
2. 입력에 없는 신규 사실을 추가하지 마라.
3. 특히 다음 정보는 입력에 명시되지 않으면 절대 쓰지 마라:
   - 구체적인 계약 내용, 고객사, 정부기관명, 정책명
   - 투자 금액, 매출액, 영업이익, 점유율, 수량, 퍼센트
   - 제품 세부 사양, 출시 일정, 수주 규모, 지역 확장 계획
   - 기사 본문에 없는 원인/배경/전망
4. 정보가 부족하면 더 일반적이고 보수적으로 써라.
5. 추정, 상상, 업계 통념 보강, 흔한 기사 문구 보충을 하지 마라.
6. 제목과 본문 요약에 없는 고유명사나 수치를 새로 만들지 마라.

출력 목표:
- 같은 사건을 정보 밀도가 다른 3개의 텍스트(phone, tv, newspaper)로 재구성한다.
- 사용자는 이 텍스트만 보고 투자 판단을 고민해야 한다.

핵심 제약:
1. phone, tv, newspaper에는 주가 상승/하락 결과를 직접 쓰지 마라.
2. phone, tv, newspaper에는 상승률/하락률, 퍼센트, 급등/급락, 상한가/하한가 같은 표현을 절대 쓰지 마라.
3. phone은 가장 불완전하고 노이즈가 섞인 텍스트여야 하며 소문이나 떠도는 이야기처럼 작성한다.
4. phone에는 종목명이나 회사명을 직접 쓰지 마라.
5. phone은 산업, 테마, 정책, 업종 분위기 정도만 흐리게 언급하고, "ㅋㅋ", "ㄷㄷ", "?" 같은 인터넷 표현을 자연스럽게 사용한다.
6. tv는 종목명이나 회사명을 직접 쓰지 마라.
7. tv는 산업/테마/정책/업종이 드러나지만, 사용자가 바로 정답을 확신할 수 있을 정도로 직접적이면 안 된다.
8. tv는 기사 핵심을 일부 반영하되, 세부 배경이나 결정적 단서는 풀지 말고 긍정과 부정이 섞인 애매한 힌트 수준으로 남겨라.
9. newspaper에는 반드시 기업명 또는 회사명을 직접 포함하라.
10. newspaper는 가장 상세한 정보여야 하며, phone/tv보다 훨씬 더 많은 힌트를 줘도 된다.
11. newspaper는 기사에 나온 사업 내용, 정책 맥락, 실적/수요/경쟁/산업 영향 등을 충분히 설명할 수 있다.
12. newspaper는 사용자가 방향을 어느 정도 유추할 수 있을 만큼 확실한 힌트를 제공하되, 직접적인 가격 결과나 수익률은 쓰지 마라.
13. 세 텍스트는 서로 정보 밀도가 분명히 달라야 한다.
14. 사실관계는 일관되어야 한다.
15. story와 reason에는 결과 방향이나 수치, 기사 기반 이유를 더 분명하게 반영해도 된다.
16. story는 케이스 두 줄 요약이다. 너무 길지 않게 작성한다.
17. reason은 운영자용 설명이다. 이 사건이 왜 그런 결과로 이어졌는지 기사 기반으로 비교적 명확하게 써도 된다.
18. 출력은 설명 없이 JSON object 하나만 반환한다.

출력 형식:
{
  "story": "...",
  "phone": "...",
  "tv": "...",
  "newspaper": "...",
  "reason": "..."
}
""".strip()


def build_generation_input(row: pd.Series) -> Dict[str, Any]:
    clean_title = strip_direction_leakage(safe_str(row.get("article_title")))
    clean_summary = strip_direction_leakage(safe_str(row.get("article_body_summary")))
    clean_reason_seed = strip_direction_leakage(safe_str(row.get("reason_seed")))
    clean_keywords = strip_direction_leakage(safe_str(row.get("cause_keywords")))

    return {
        "company": safe_str(row.get("company")),
        "ticker": safe_str(row.get("ticker")),
        "industry": safe_str(row.get("industry")),
        "event_date": safe_str(row.get("event_date")),
        "event_type": safe_str(row.get("event_type")),
        "volume_ratio": safe_float(row.get("volume_ratio")),
        "event_strength": safe_float(row.get("event_strength")),
        "article_title": clean_title,
        "article_body_summary": clean_summary,
        "cause_keywords": clean_keywords,
        "article_type": safe_str(row.get("article_type")),
        "market_scope": safe_str(row.get("market_scope")),
        "article_url": safe_str(row.get("article_url")),
        "reason_seed": clean_reason_seed,
    }


def build_user_prompt(payload: Dict[str, Any]) -> str:
    return f"""
다음 사건을 기반으로 투자 판단 게임용 케이스를 생성하라.

입력 데이터:
{json.dumps(payload, ensure_ascii=False, indent=2)}

반드시 지킬 규칙:
- 입력 데이터에 있는 정보만 사용하라.
- article_title, article_body_summary, cause_keywords에 없는 구체 사실은 새로 만들지 마라.
- 입력에 없는 수치, 정책명, 기관명, 계약명, 고객사명, 제품 사양, 지역 확장 계획을 추가하지 마라.
- 불확실하면 더 일반적으로 써라.
- 기사 요약에 없는 배경 설명을 상식으로 보충하지 마라.

작성 지침:
- phone: 1~2문장, 매우 짧고 불완전해야 한다.
- phone: 종목명/회사명을 직접 쓰지 마라.
- phone: 산업/테마/정책/업종 분위기 정도만 언급하고, 소문·잡담·찌라시 같은 톤을 허용한다.
- tv: 2~4문장, 중간 길이, 정제된 정보 요약
- tv: 종목명/회사명을 직접 쓰지 마라.
- tv: 산업/테마/정책/업종은 드러나지만, 사용자가 바로 방향을 확신할 수 없도록 애매한 힌트 수준으로 작성하라.
- newspaper: 4~7문장, 가장 상세, 기사 기반 맥락 충분히 설명
- newspaper: 반드시 기업명/회사명을 직접 포함하라.
- newspaper: phone/tv보다 훨씬 더 많은 힌트를 주고, 사용자가 방향을 어느 정도 유추할 수 있을 만큼 확실하게 작성하라.
- story: 케이스 제목/요약
- reason: 운영자용 설명. 기사 기반 원인과 결과 방향을 비교적 명확하게 설명 가능
- phone/tv/newspaper에는 직접적인 수익률/상승률/하락률/급등/급락/상승/하락 표현 금지
- story/reason에는 입력의 실제 결과 방향과 수익률 정보를 참고하지 말고, 기사 기반 사건 요약 중심으로 작성하라

금지 사항:
- phone/tv/newspaper에는 직접적인 수익률/상승률/하락률/급등/급락/상승/하락 표현 금지
- phone/tv/newspaper에는 입력에 없는 수치나 세부 정보 추가 금지
- 과장된 전망, 추정, 새로운 원인 추가 금지

반드시 JSON object 하나만 반환하라.
""".strip()


# =========================
# 3. Claude 호출
# =========================
def call_claude_json(client: Anthropic, model: str, user_prompt: str) -> Dict[str, Any]:
    last_err: Optional[Exception] = None

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            text_blocks = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    text_blocks.append(block.text)

            raw_text = "\n".join(text_blocks).strip()
            return parse_json_from_text(raw_text)

        except Exception as e:
            last_err = e
            print(f"[WARN] Claude call failed attempt={attempt}/{RETRY_COUNT}: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_SLEEP_SEC)

    raise RuntimeError(f"Claude generation failed after retries: {last_err}")


# =========================
# 4. stock universe 로드 / 매핑
# =========================
def load_stock_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"stock universe file not found: {path}\n"
            f"Required columns: stock_id, company, ticker, industry"
        )

    df = pd.read_csv(path).copy()

    required = ["stock_id", "company", "ticker", "industry"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"stock universe missing columns: {missing}")

    df["stock_id"] = pd.to_numeric(df["stock_id"], errors="coerce")
    df = df.dropna(subset=["stock_id"]).copy()
    df["stock_id"] = df["stock_id"].astype(int)

    df["company"] = df["company"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["industry"] = df["industry"].astype(str).str.strip()

    return df.reset_index(drop=True)


def attach_stock_id(case_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    stock_df = stock_df.copy()
    stock_df["join_key"] = stock_df.apply(
        lambda r: normalize_key(r["company"], r["ticker"]), axis=1
    )

    case_df = case_df.copy()
    case_df["join_key"] = case_df.apply(
        lambda r: normalize_key(r.get("company"), r.get("ticker")), axis=1
    )

    mapped = stock_df[["join_key", "stock_id"]].rename(columns={"stock_id": "mapped_stock_id"})
    merged = case_df.merge(mapped, on="join_key", how="left")

    if "stock_id" in merged.columns:
        merged["stock_id"] = pd.to_numeric(merged["stock_id"], errors="coerce")
        merged["stock_id"] = merged["stock_id"].fillna(merged["mapped_stock_id"])
    else:
        merged["stock_id"] = merged["mapped_stock_id"]

    merged = merged.drop(columns=["join_key", "mapped_stock_id"])
    return merged


# =========================
# 5. article_json / up_down_json
# =========================
def build_article_json(row: pd.Series) -> str:
    article_obj = {
        "article_id": safe_str(row.get("article_id")),
        "title": safe_str(row.get("article_title")),
        "url": safe_str(row.get("article_url")),
        "published_at": safe_str(row.get("article_published_at")),
        "article_type": safe_str(row.get("article_type")),
        "market_scope": safe_str(row.get("market_scope")),
        "final_score": safe_float(row.get("final_score")),
    }
    return json_dumps_ensure([article_obj])


def build_up_down_json(row: pd.Series, stock_df: pd.DataFrame) -> str:
    event_stock_id = row.get("stock_id")
    event_stock_id = None if pd.isna(event_stock_id) else int(event_stock_id)
    event_company = safe_str(row.get("company"))
    event_ticker = safe_str(row.get("ticker"))
    event_industry = safe_str(row.get("industry"))
    real_return = safe_float(row.get("real_return"))

    event_game_return = scale_game_return(real_return)
    event_abs_cap = max(0.01, abs(event_game_return) * NON_EVENT_RELATIVE_CAP)

    market_bias = 0.0
    if random.random() < MARKET_BIAS_PROB:
        market_bias = random.choice([-MARKET_BIAS_MAG, MARKET_BIAS_MAG])

    peer_df = stock_df[
        (stock_df["industry"] == event_industry)
        & ~(
            (stock_df["company"] == event_company)
            & (stock_df["ticker"] == event_ticker)
        )
    ].copy()

    sector_peer_ids = []
    peer_count = len(peer_df)

    if peer_count > 0 and random.random() < SECTOR_BIAS_PROB:
        lower = min(SECTOR_PEER_MIN, peer_count)
        upper = min(SECTOR_PEER_MAX, peer_count)
        k = random.randint(lower, upper)
        sector_peer_ids = random.sample(peer_df["stock_id"].tolist(), k=k)

    results: List[Dict[str, Any]] = []

    for _, srow in stock_df.iterrows():
        sid = int(srow["stock_id"])
        company = safe_str(srow["company"])
        ticker = safe_str(srow["ticker"])
        industry = safe_str(srow["industry"])

        if event_stock_id is not None and sid == event_stock_id:
            game_return = event_game_return
        elif sid in sector_peer_ids:
            peer_mag = min(abs(event_game_return) * random.uniform(0.25, 0.55), event_abs_cap)
            peer_sign = 1 if event_game_return >= 0 else -1
            peer_noise = random.gauss(0, 0.004)
            game_return = peer_sign * peer_mag + peer_noise + market_bias
            game_return = clip(game_return, -event_abs_cap, event_abs_cap)
        else:
            game_return = sample_noise(event_abs_cap, mean_shift=market_bias)

        results.append({
            "stock_id": sid,
            "company": company,
            "ticker": ticker,
            "industry": industry,
            "real_return": float(real_return) if (event_stock_id is not None and sid == event_stock_id) else None,
            "game_return": round(float(game_return), 4),
            "is_event_stock": 1 if (event_stock_id is not None and sid == event_stock_id) else 0,
        })

    random.shuffle(results)
    return json_dumps_ensure(results)


# =========================
# 6. 입력 데이터 보정
# =========================
def normalize_case_dataset(case_df: pd.DataFrame) -> pd.DataFrame:
    case_df = case_df.copy()

    if "real_return" not in case_df.columns:
        if "return_1d" in case_df.columns:
            case_df["real_return"] = pd.to_numeric(case_df["return_1d"], errors="coerce").fillna(0)
        else:
            case_df["real_return"] = 0.0

    if "article_title" not in case_df.columns:
        if "title" in case_df.columns:
            case_df["article_title"] = case_df["title"]
        else:
            case_df["article_title"] = ""

    if "article_url" not in case_df.columns:
        if "url" in case_df.columns:
            case_df["article_url"] = case_df["url"]
        else:
            case_df["article_url"] = ""

    if "article_published_at" not in case_df.columns:
        if "published_at" in case_df.columns:
            case_df["article_published_at"] = case_df["published_at"]
        else:
            case_df["article_published_at"] = ""

    for col, default in [
        ("article_body_summary", ""),
        ("body", ""),
        ("reason_seed", ""),
        ("cause_keywords", ""),
        ("case_id", ""),
        ("event_id", ""),
        ("company", ""),
        ("ticker", ""),
        ("industry", ""),
        ("event_date", ""),
        ("event_type", ""),
        ("volume_ratio", 0.0),
        ("event_strength", 0.0),
        ("article_id", ""),
        ("final_score", 0.0),
        ("article_type", ""),
        ("market_scope", ""),
    ]:
        ensure_column(case_df, col, default)

    case_df["real_return"] = pd.to_numeric(case_df["real_return"], errors="coerce").fillna(0)
    case_df["volume_ratio"] = pd.to_numeric(case_df["volume_ratio"], errors="coerce").fillna(0)
    case_df["event_strength"] = pd.to_numeric(case_df["event_strength"], errors="coerce").fillna(0)
    case_df["final_score"] = pd.to_numeric(case_df["final_score"], errors="coerce").fillna(0)

    case_df["event_date"] = pd.to_datetime(case_df["event_date"], errors="coerce")
    case_df["event_date"] = case_df["event_date"].dt.strftime("%Y-%m-%d")
    case_df["event_date"] = case_df["event_date"].fillna("")

    if "case_id" not in case_df.columns or case_df["case_id"].astype(str).str.strip().eq("").all():
        case_df["case_id"] = [f"CASE_{i+1:04d}" for i in range(len(case_df))]

    return case_df


# =========================
# 7. reason / fallback 생성
# =========================
def build_reason_from_input(row: pd.Series) -> str:
    company = safe_str(row.get("company"))
    event_date = safe_str(row.get("event_date"))
    article_title = safe_str(row.get("article_title"))
    article_body_summary = safe_str(row.get("article_body_summary"))
    real_return = safe_float(row.get("real_return"))

    direction = infer_direction_label(real_return)

    base = (
        f"{company}의 {event_date} 이벤트는 기사 제목 "
        f"'{article_title}' 및 기사 요약을 근거로 연결된 케이스다. "
        f"실제 결과 방향은 {direction}이다."
    )

    if article_body_summary:
        return base + f" 기사 요약 기준 핵심 내용은 다음과 같다: {article_body_summary}"
    return base


def build_story_fallback(row: pd.Series) -> str:
    company = safe_str(row.get("company"))
    article_title = strip_direction_leakage(safe_str(row.get("article_title")))
    event_type = safe_str(row.get("event_type"))

    if article_title:
        return compact_text(f"{company} 관련 이슈: {article_title}")
    return compact_text(f"{company} {event_type} 케이스")


def build_phone_fallback(row: pd.Series) -> str:
    industry = safe_str(row.get("industry"))
    cause_keywords = strip_direction_leakage(safe_str(row.get("cause_keywords")))
    article_type = safe_str(row.get("article_type"))

    parts = []
    if industry:
        parts.append(f"{industry} 쪽 분위기 좀 묘한데")
    if cause_keywords:
        parts.append(f"{cause_keywords} 얘기 계속 돌더라")
    elif article_type:
        parts.append(f"{article_type} 느낌 기사도 보이고")
    else:
        parts.append("관련 얘기 슬슬 도는 듯 ㅋㅋ")

    text = " ".join(parts)
    return compact_text(text[:120])


def build_tv_fallback(row: pd.Series) -> str:
    industry = safe_str(row.get("industry"))
    article_title = strip_direction_leakage(safe_str(row.get("article_title")))
    summary = strip_direction_leakage(safe_str(row.get("article_body_summary")))

    basis = article_title or summary or "관련 이슈가 포착됐다."
    basis = basis.replace(safe_str(row.get("company")), REPLACE_COMPANY_WITH)

    text = f"{industry} 업종 관련 이슈가 시장에서 주목받고 있습니다. {basis}"
    return compact_text(text[:220])


def build_newspaper_fallback(row: pd.Series) -> str:
    company = safe_str(row.get("company"))
    article_title = strip_direction_leakage(safe_str(row.get("article_title")))
    summary = strip_direction_leakage(safe_str(row.get("article_body_summary")))
    market_scope = safe_str(row.get("market_scope"))
    article_type = safe_str(row.get("article_type"))

    sentences = []
    if company and article_title:
        sentences.append(f"{company} 관련 기사에서는 '{article_title}'라는 이슈가 다뤄졌다.")
    elif company:
        sentences.append(f"{company} 관련 이슈가 기사에서 다뤄졌다.")

    if summary:
        sentences.append(summary)

    extra = []
    if market_scope:
        extra.append(f"시장 범위는 {market_scope}로 분류됐다")
    if article_type:
        extra.append(f"기사 유형은 {article_type}로 정리됐다")

    if extra:
        sentences.append(". ".join(extra) + ".")

    if not sentences:
        sentences.append(f"{company} 관련 이슈가 시장에서 참고할 정보로 제시됐다.")

    return compact_text(" ".join(sentences))


def generate_fallback_case(row: pd.Series, reason_override: Optional[str] = None) -> Dict[str, str]:
    company = safe_str(row.get("company"))

    story = build_story_fallback(row)
    phone = sanitize_phone_text(sanitize_visible_text(build_phone_fallback(row), company), company)
    tv = sanitize_no_company_text(sanitize_visible_text(build_tv_fallback(row), company), company)
    newspaper = sanitize_visible_text(build_newspaper_fallback(row), company)
    reason = compact_text(reason_override) if reason_override else build_reason_from_input(row)

    if company and company not in newspaper:
        newspaper = compact_text(f"{company} 관련 이슈가 기사에서 언급됐다. {newspaper}")

    return {
        "story": story,
        "phone": phone,
        "tv": tv,
        "newspaper": newspaper,
        "reason": reason,
    }


# =========================
# 8. 메인
# =========================
def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 환경변수 설정 필요")

    if not INPUT_CASE_DATASET.exists():
        raise FileNotFoundError(f"input not found: {INPUT_CASE_DATASET}")

    print("[LOAD] case dataset...")
    case_df = pd.read_csv(INPUT_CASE_DATASET).copy()
    case_df = normalize_case_dataset(case_df)

    print("[LOAD] stock universe...")
    stock_df = load_stock_universe(INPUT_STOCK_UNIVERSE)

    case_df = attach_stock_id(case_df, stock_df)

    print("[DEBUG] case_df columns after attach_stock_id:")
    print(case_df.columns.tolist())
    print("[DEBUG] stock_id null count:", case_df["stock_id"].isna().sum())

    total_input_rows = len(case_df)

    if MAX_CASES is not None:
        case_df = case_df.head(MAX_CASES).copy()

    print(f"[INFO] rows to generate: {len(case_df)}")
    print(f"[INFO] stock universe rows: {len(stock_df)}")

    expected_case_cols = ["company", "ticker", "industry", "event_date", "event_type", "real_return", "article_id"]
    missing_case_cols = [c for c in expected_case_cols if c not in case_df.columns]
    if missing_case_cols:
        raise ValueError(f"case_dataset missing required columns: {missing_case_cols}")

    client = Anthropic(api_key=api_key)

    results = []
    errors = []

    for idx, row in case_df.iterrows():
        case_uid = safe_str(row.get("case_id")) or f"CASE_{idx+1:04d}"
        company = safe_str(row.get("company"))
        stock_id_value = row.get("stock_id")

        print(f"[GENERATE] {idx + 1}/{len(case_df)} case={case_uid} company={company}")

        allowed_basis = " ".join([
            strip_direction_leakage(safe_str(row.get("article_title"))),
            strip_direction_leakage(safe_str(row.get("article_body_summary"))),
            strip_direction_leakage(safe_str(row.get("cause_keywords"))),
            strip_direction_leakage(safe_str(row.get("reason_seed"))),
        ])

        generation_mode = "claude"
        validation_issues: List[str] = []

        if pd.isna(stock_id_value):
            generation_mode = "fallback_unmapped_stock"
            errors.append({
                "case_uid": case_uid,
                "event_id": safe_str(row.get("event_id")),
                "company": company,
                "ticker": safe_str(row.get("ticker")),
                "article_id": safe_str(row.get("article_id")),
                "error": "unmapped_stock_id",
            })
            gen = generate_fallback_case(row, reason_override="stock_universe 매핑 실패로 fallback 케이스를 생성했다.")
        else:
            payload = build_generation_input(row)
            prompt = build_user_prompt(payload)

            try:
                raw_gen = call_claude_json(client, CLAUDE_MODEL, prompt)

                story = compact_text(safe_str(raw_gen.get("story")))
                phone = sanitize_phone_text(
                    sanitize_visible_text(safe_str(raw_gen.get("phone")), company),
                    company
                )
                tv = sanitize_no_company_text(
                    sanitize_visible_text(safe_str(raw_gen.get("tv")), company),
                    company
                )
                newspaper = sanitize_visible_text(
                    safe_str(raw_gen.get("newspaper")),
                    company
                )
                reason = compact_text(safe_str(raw_gen.get("reason"))) or build_reason_from_input(row)

                validation_issues = validate_generated_fields(
                    story=story,
                    phone=phone,
                    tv=tv,
                    newspaper=newspaper,
                    company=company,
                    allowed_basis=allowed_basis,
                )

                if validation_issues:
                    generation_mode = "fallback_after_validation"
                    errors.append({
                        "case_uid": case_uid,
                        "event_id": safe_str(row.get("event_id")),
                        "company": company,
                        "ticker": safe_str(row.get("ticker")),
                        "article_id": safe_str(row.get("article_id")),
                        "error": "|".join(validation_issues),
                    })
                    gen = generate_fallback_case(
                        row,
                        reason_override=f"Claude 출력 검증 실패({', '.join(validation_issues)})로 fallback 케이스를 생성했다."
                    )
                else:
                    gen = {
                        "story": story,
                        "phone": phone,
                        "tv": tv,
                        "newspaper": newspaper,
                        "reason": reason,
                    }

            except Exception as e:
                generation_mode = "fallback_after_exception"
                errors.append({
                    "case_uid": case_uid,
                    "event_id": safe_str(row.get("event_id")),
                    "company": company,
                    "ticker": safe_str(row.get("ticker")),
                    "article_id": safe_str(row.get("article_id")),
                    "error": str(e),
                })
                gen = generate_fallback_case(
                    row,
                    reason_override=f"Claude 호출 실패로 fallback 케이스를 생성했다. 원인: {e}"
                )

        article_json = build_article_json(row)
        up_down_json = build_up_down_json(row, stock_df)

        result_row = {
            "case_uid": case_uid,
            "event_id": safe_str(row.get("event_id")),
            "company": company,
            "ticker": safe_str(row.get("ticker")),
            "industry": safe_str(row.get("industry")),
            "event_date": safe_str(row.get("event_date")),
            "event_type": safe_str(row.get("event_type")),
            "real_return": safe_float(row.get("real_return")),
            "stock_id": None if pd.isna(stock_id_value) else int(stock_id_value),
            "article_id": safe_str(row.get("article_id")),
            "article_title": safe_str(row.get("article_title")),
            "article_url": safe_str(row.get("article_url")),
            "article_published_at": safe_str(row.get("article_published_at")),
            "final_score": safe_float(row.get("final_score")),
            "article_type": safe_str(row.get("article_type")),
            "market_scope": safe_str(row.get("market_scope")),
            "reason_seed": safe_str(row.get("reason_seed")),
            "article_body_summary": safe_str(row.get("article_body_summary")),

            "story": compact_text(gen.get("story", "")),
            "phone": compact_text(gen.get("phone", "")),
            "tv": compact_text(gen.get("tv", "")),
            "newspaper": compact_text(gen.get("newspaper", "")),
            "article_json": article_json,
            "reason": compact_text(gen.get("reason", "")),
            "up_down_json": up_down_json,

            "generation_mode": generation_mode,
            "validation_issues": "|".join(validation_issues) if validation_issues else "",
        }

        results.append(result_row)
        time.sleep(SLEEP_SEC)

    result_df = pd.DataFrame(results)
    error_df = pd.DataFrame(errors)

    result_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_CSV}")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for _, r in result_df.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"saved: {OUT_JSONL}")

    insert_cols = [
        "story",
        "phone",
        "tv",
        "newspaper",
        "article_json",
        "reason",
        "up_down_json",
        "stock_id",
    ]

    if len(result_df) > 0:
        insert_ready_df = result_df[insert_cols].copy()
    else:
        insert_ready_df = pd.DataFrame(columns=insert_cols)

    insert_ready_df.to_csv(OUT_INSERT_READY, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_INSERT_READY}")

    error_df.to_csv(OUT_ERROR_LOG, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_ERROR_LOG}")

    print("\n=== DONE ===")
    print(f"input_rows_before_max_cases: {total_input_rows}")
    print(f"generated_rows: {len(result_df)}")
    print(f"error_rows: {len(error_df)}")

    if MAX_CASES is None:
        print(f"[CHECK] input vs output match: {total_input_rows} == {len(result_df)}")
    else:
        print(f"[CHECK] truncated by MAX_CASES, output rows: {len(result_df)}")

    if len(result_df):
        sample_cols = ["case_uid", "company", "generation_mode", "story", "phone", "reason"]
        print("\n[SAMPLE]")
        print(result_df[sample_cols].head(3).to_string(index=False))

        print("\n[CHECK] stock_id null rows in result:", result_df["stock_id"].isna().sum())
        print("[CHECK] generation_mode counts:")
        print(result_df["generation_mode"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()