import re
from typing import Iterable, Mapping, Any, Tuple, List

# === 최소 헬퍼 ===
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[\s\-]+', '', s)
    s = re.sub(r'[^\w가-힣]+', '', s)
    return s

def _pred_from_answer(raw: str) -> int | None:
    """<answer> 태그들에서 0/1 예측 산출. 모호/없음이면 None."""
    toks = {_norm(m) for m in _ANS_RE.findall(str(raw))}
    if not toks:
        return None
    # 동의어 정리
    if "nontoxic" in toks or "nontoxicity" in toks:
        toks.add("비독성")
    if "toxic" in toks:
        toks.add("독성")
    has_pos = ("독성" in toks)
    has_neg = ("비독성" in toks)
    if has_pos and has_neg:
        return None  # 모호
    if has_pos:
        return 1
    if has_neg:
        return 0
    return None

# === 정확도만 계산 ===
def evaluate_accuracy_only(
    data: Iterable[Mapping[str, Any]],
    answer_key: str = "answer",
    label_key: str = "LABEL",
    count_missing_as_wrong: bool = False,
) -> float:

    correct = 0
    denom = 0

    for it in data:
        y = it.get(label_key)
        if y not in (0, 1):
            continue  # 라벨 없는 샘플 제외

        p = _pred_from_answer(it.get(answer_key, ""))

        if p is None:
            if count_missing_as_wrong:
                denom += 1   # 오답으로 카운트
            continue

        denom += 1
        if int(y) == int(p):
            correct += 1

    return (correct / denom) if denom > 0 else 0.0