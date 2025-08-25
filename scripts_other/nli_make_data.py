import json
from pathlib import Path

# ───────── 1) 경로 설정 ─────────
BASE = Path("/workspace/nli_2025/datasets")
ORIG_DIR = BASE / "original"
CONV_DIR = BASE / "converted_v2"
TRAIN_IN = ORIG_DIR / "train.json"
DEV_IN   = ORIG_DIR / "dev.json"
TRAIN_OUT = CONV_DIR / "train.json"
DEV_OUT   = CONV_DIR / "dev.json"

# ───────── 2) 관계 + 접속사 리스트 정의 ─────────
RELATIONS = {
    "역접": ["그러나", "그렇지만", "하지만"], #["그러나", "그렇지만", "하지만", "그런데", "오히려", "그럼에도 불구하고", "다만"],
    "순접": ["그래서", "그러므로", "따라서"], #["그러므로", "그래서", "따라서", "그리하여", "즉", "왜냐하면"],
    "양립": ["그리고", "또한", "게다가"], #["그리고", "또", "또한", "및", "더불어", "게다가", "나아가"],
}

# ───────── 3) 함수 ─────────
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def ensure_dir(path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def make_prompt(front: str, back: str, relation: str, conj: str) -> str:
    return (
        f"문장관계는 {front} {conj} {back} 처럼, {conj} 접속사를 사용한 문장이 어울립니다. "
        f"{conj} 접속사는 '{relation}'입니다. 따라서, 정답은 ###{relation}###입니다."
    )

def augment(records):
    new_data = []
    for ex in records:
        ex_id  = ex["id"]
        front  = ex["input"]["front"]
        back   = ex["input"]["back"]
        origin = ex.get("output")

        # 원본 라벨 없거나 사전에 없는 라벨이면 건너뜀
        if not origin:
            print(f"[SKIP] {ex_id}: origin(label) 없음")
            continue
        origin = origin.strip()
        if origin not in RELATIONS:
            print(f"[SKIP] {ex_id}: unknown label '{origin}'")
            continue

        # ★ 핵심: origin과 동일한 관계에 대해서만 증강
        for conj in RELATIONS[origin]:
            new_ex = {
                "id": f"{ex_id}::{origin}::{conj}",
                "input": {"front": front, "back": back},
                "origin": origin,  # 원본 라벨 보존
                "output": make_prompt(front, back, origin, conj),
            }
            new_data.append(new_ex)
            print(f"[ADD] {new_ex['id']}")
    return new_data

def save_json(path: Path, data):
    ensure_dir(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ───────── 4) 실행 ─────────
if __name__ == "__main__":
    train_data = load_json(TRAIN_IN)
    dev_data   = load_json(DEV_IN)

    train_aug = augment(train_data)
    dev_aug   = augment(dev_data)

    save_json(TRAIN_OUT, train_aug)
    save_json(DEV_OUT, dev_aug)
