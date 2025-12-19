# evaluate.py
import time
from typing import Optional, List, Dict, Any
import requests
from requests import Session
from tqdm import tqdm
from utils import build_message, FileLogger

def _to_req(item: Dict[str, Any]) -> Dict[str, Any]:
    """utils.build_message() 결과에서 system/user를 추출해 서버 스키마에 맞게 변환"""
    msgs = build_message(item)
    sys_txt, usr_txt = "", ""
    for m in msgs:
        if m.get("role") == "system":
            sys_txt = m.get("content", "") or ""
        elif m.get("role") == "user":
            usr_txt = m.get("content", "") or ""
    return {
        "text": usr_txt,
        "system_prompt": sys_txt,
    }

def _brief_id(item: Dict[str, Any]) -> str:
    """로그 식별자: 가능한 경우 SMILES 코드/엔드포인트 카테고리 등 간단 요약"""
    try:
        smiles = item.get("compound_info", {}).get("smiles_code")
        cat = item.get("data_meta_info", {}).get("endpoint_category")
        if smiles and cat:
            return f"SMILES={smiles} | CAT={cat}"
        if smiles:
            return f"SMILES={smiles}"
        if cat:
            return f"CAT={cat}"
    except Exception:
        pass
    return ""

def _post_json(session: Session, url: str, payload: dict, timeout_s: float) -> requests.Response:
    """
    POST JSON 헬퍼.
    timeout은 (connect, read) 튜플로 주는 게 안전합니다. ex) (10, 300)
    """
    connect_t = min(10.0, timeout_s)  # 연결은 짧게
    read_t = timeout_s                # 응답 대기는 길게
    return session.post(url, json=payload, timeout=(connect_t, read_t))

def run_batched(
    data: List[Dict[str, Any]],
    base_url: str,
    path: str = "/generate_batch",
    batch_size: int = 32,
    logger: Optional[FileLogger] = None,
    retries: int = 2,
    initial_delay: float = 1.0,
    request_timeout: float = 300.0,
) -> List[Optional[str]]:
    """
    requests 기반 배치 호출:
      - /generate_batch 로 batch_size개씩 전송
      - 실패 시 지수 백오프 재시도
      - 최종 실패 구간은 /generate 로 항목별 폴백
      - 모든 응답을 전부 로깅
    """
    url_batch = base_url.rstrip("/") + path
    url_single = base_url.rstrip("/") + "/generate"
    results: List[Optional[str]] = [None] * len(data)

    with requests.Session() as session:
        headers = {"Content-Type": "application/json"}
        session.headers.update(headers)

        for start in tqdm(range(0, len(data), batch_size), desc=f"배치 추론(batch={batch_size})"):
            end = min(start + batch_size, len(data))
            chunk = data[start:end]
            payload = {"items": [_to_req(x) for x in chunk]}

            delay = initial_delay
            ok = False
            for attempt in range(retries + 1):
                try:
                    resp = _post_json(session, url_batch, payload, timeout_s=request_timeout)
                    resp.raise_for_status()
                    out = resp.json()  # 서버가 list[str] 반환
                    if not isinstance(out, list):
                        raise RuntimeError(f"Unexpected response type: {type(out)}")
                    if len(out) != len(chunk):
                        raise RuntimeError(f"Size mismatch: got {len(out)} for batch {len(chunk)}")

                    # === 결과 매핑 + 전체 로깅 ===
                    for i, text in enumerate(out):
                        idx = start + i
                        results[idx] = text
                        if logger:
                            ident = _brief_id(chunk[i])
                            head = f"MODEL_RESPONSE[{idx}]"
                            if ident:
                                head += f" ({ident})"
                            logger.log(f"{head}:\n{text}")
                    ok = True
                    break
                except Exception as e:
                    if logger:
                        logger.log(f"[WARN] batch {start}:{end} attempt {attempt+1}/{retries+1} failed: {repr(e)}")
                    if attempt < retries:
                        time.sleep(delay); delay *= 2

            if ok:
                continue

            # ---- 배치 최종 실패 → 단건 폴백 ----
            if logger:
                logger.log(f"[FALLBACK] falling back to per-item for indices {start}:{end}")
            for j, item in enumerate(chunk):
                req = _to_req(item)
                single_delay = initial_delay
                single_ok = False
                for s_try in range(retries + 1):
                    try:
                        r = _post_json(session, url_single, req, timeout_s=request_timeout)
                        r.raise_for_status()
                        text = r.text  # /generate는 str 반환
                        idx = start + j
                        results[idx] = text
                        if logger:
                            ident = _brief_id(item)
                            head = f"MODEL_RESPONSE[{idx}]"
                            if ident:
                                head += f" ({ident})"
                            logger.log(f"{head}:\n{text}")
                        single_ok = True
                        break
                    except Exception as e2:
                        if logger:
                            logger.log(f"[WARN] single {start+j} attempt {s_try+1}/{retries+1} failed: {repr(e2)}")
                        if s_try < retries:
                            time.sleep(single_delay); single_delay *= 2
                if not single_ok and logger:
                    logger.log(f"[ERROR] item {start+j} failed permanently.")

    return results
