import os
import sys
import re
import json
import time
from datetime import datetime, timezone  # ← 추가

# ====== 유틸 ======
def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def mkdir_p(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_list_to_json(lst, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lst, f, ensure_ascii=False, indent=2)

class FileLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        mkdir_p(os.path.dirname(log_path))
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")  # truncate
    def log(self, msg: str, also_print: bool = True):
        line = f"[{utc_ts()}] {msg}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if also_print:
            print(line, flush=True)

# ====== 메시지 구성 ======
def build_message(item):
    system = (
        "당신은 분자 독성 예측에 특화된 화학정보학/독성학 전문가입니다.\n\n"
        "사용자는 RDkit을 이용해 분자에 대한 feature들을 추출해냈고, 각 feature별로 독성에 미치는 SHAP Value를 계산한 다음 상위값 세개를 추출했습니다.(SHAP Value가 제공되지는 않습니다.)\n"
        "아래는 그 이후 사용자가 제공하는 데이터들입니다.\n\n"
        "입력(사용자가 제공):\n"
        "- Cell Line \n- Bio Assay Name\n- SMILES \n- Feature NL (상위 feature 3개가 독성에 어떤 영향을 끼치는지에 대한 설명문)\n"
        "- Feature Descript (상위 feature 3개에 대해 각각 상관관계가 높은 feature 3개 (총 9개)에 대한 설명문)\n\n"
        "수행 과업(Tasks):\n"
        "SMILES 구조 분석\n"
        "- 고리(방향족/지방족), 헤테로원자, 전하 중심, 명백한 반응성 모티프(에폭사이드/아지리딘/마이클 수용체/니트로/알데하이드/아실할라이드 등), "
        "H-결합 공여/수용기 등을 SMILES에서 직접 관찰 가능한 범위 내에서만 기술.\n\n"
        "- Cell Type, Cell Line, Assay Name 특징 분석 및 SMILES와 연결\n\n"
        "- Feature의 특성 파악 및 독성 여부와의 상관관계 추론\n\n"
        "종합 판단(최종 결론)\n"
        "- (1) SMILES 모티프, (2) Cell line + Assay 맥락 (3) 상위 feature의 특성 파악\n"
        "세가지 정보를 통합하여 주어진 분자식의 독성 여부를 판단\n"
        "- 이진 결론을 명확히: toxic vs nontoxic.\n\n"
        "출력 규칙(반드시 준수):\n"
        "- 최종 답변 본문은 반드시 영어로 작성.\n"
        "- 마지막 줄에 아래 중 하나만 단독 표기:\n"
        "<answer>독성</answer>\n"
        "<answer>비독성</answer>\n\n"
        "tool call을 사용하지 마세요."
    )

    user_prompt = (
        f"SMILES: {item['compound_info']['smiles_code']}\n"
        f"Cell Line: {item['compound_info']['toxicity']['cell_line']}\n"
        f"Bio Assay Name: {item['data_meta_info']['endpoint_category']}\n"
        f"Feature NL: {item['compound_info']['feature_NL']}\n"
        f"Feature Descript: {item['compound_info']['feature_descript']}\n\n"
        f"{item['cot_info']['cot_instruction']}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]