"""5강 실습(QLoRA / VERA / Vision Adapter)에서 공통으로 쓰는 유틸리티.

원본 데이터셋은 HuggingFace의 `Tutoruslabs/ChartQA_KOR`(gated, test split 2,000개)입니다.
벤치마크용으로 공개되어 train split이 없으므로, `1. VLM 학습 데이터 준비.ipynb`가
`prepare_chartqa_kor()`로 test split을 train/val로 나눈 뒤 `push_dataset_to_hub()`로
학생들이 공유할 HuggingFace Hub 데이터셋 repo(`HF_DATASET_REPO_ID`)에 업로드합니다.
이후 노트북들은 `download_dataset_from_hub()`로 그 repo에서 데이터를 내려받아 사용합니다.

접근하려면 `huggingface-cli login`으로 로그인하고, 데이터셋 페이지에서 이용약관에
동의해야 합니다: https://huggingface.co/datasets/Tutoruslabs/ChartQA_KOR

QLoRA·Vision Adapter 노트북은 RunPod GPU 환경 실행을 전제로 하고,
VERA 노트북은 가중치 학습이 없어 로컬 Ollama(Qwen3-VL)만으로 실행됩니다.
"""

import io
import json
import re
from pathlib import Path

import ollama
from PIL import Image
from tqdm.auto import tqdm

DATASET_ID = "Tutoruslabs/ChartQA_KOR"

# "-instruct" 태그를 씀: 기본 "qwen3-vl:4b"(thinking 변형)는 Ollama가 아직
# think=False를 지켜주지 않아, 답변이 비어있고 reasoning만 잔뜩 나오는 경우가 있음.
VLM_MODEL = "qwen3-vl:4b-instruct"

ANSWER_INSTRUCTION = "질문에 대한 답만 아주 간결하게 말하세요. 숫자면 숫자만, 단답이면 단답만 답하고 설명은 덧붙이지 마세요."


def build_conversation(question: str, answer: str) -> list[dict]:
    instruction = f"<image>\n{question}\n{ANSWER_INSTRUCTION}"
    return [
        {"from": "human", "value": instruction},
        {"from": "gpt", "value": answer},
    ]


def _write_split(data, split_dir: Path) -> None:
    """HF `datasets` split(image/question/answer 컬럼)을 로컬 images/ + annotations.jsonl로 저장.

    ChartQA_KOR에는 answer가 비어있는(null) 레코드가 소수 섞여 있어 건너뜁니다.
    """
    img_dir = split_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for ex in tqdm(data, desc=f"{split_dir.name} 저장 중"):
        question, answer = ex["question"], ex["answer"]
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue

        fname = f"{len(records):04d}.png"
        ex["image"].convert("RGB").save(img_dir / fname)
        records.append(
            {
                "image": f"images/{fname}",
                "question": question,
                "answer": answer,
                "conversations": build_conversation(question, answer),
            }
        )

    with open(split_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare_chartqa_kor(root: str, val_ratio: float = 0.2, seed: int = 0) -> None:
    """Tutoruslabs/ChartQA_KOR(test split)을 다운로드해 train/val로 나눠 저장.

    이 데이터셋은 벤치마크(평가)용으로 공개되어 test split(2,000개)만 존재합니다.
    실습 목적으로 이를 train/val로 임의 분할합니다. 이후 `push_dataset_to_hub()`로
    HuggingFace Hub에 업로드해 다른 노트북들과 공유합니다.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="test")
    split_ds = ds.train_test_split(test_size=val_ratio, seed=seed)

    root_path = Path(root)
    _write_split(split_ds["train"], root_path / "train")
    _write_split(split_ds["test"], root_path / "val")


def push_dataset_to_hub(root: str, repo_id: str, private: bool = True, token: str | None = None) -> None:
    """`root`(data/train, data/val)의 annotations.jsonl + images를 HuggingFace Hub 데이터셋으로 업로드.

    train/val 두 개의 split을 가진 DatasetDict로 업로드됩니다.
    """
    from datasets import Dataset, DatasetDict
    from datasets import Image as HFImage

    root_path = Path(root)
    dataset_dict = {}
    for split_name in ["train", "val"]:
        records = load_jsonl(root_path / split_name / "annotations.jsonl")
        split = Dataset.from_dict(
            {
                "image": [str(root_path / split_name / r["image"]) for r in records],
                "question": [r["question"] for r in records],
                "answer": [r["answer"] for r in records],
            }
        ).cast_column("image", HFImage())
        dataset_dict[split_name] = split

    DatasetDict(dataset_dict).push_to_hub(repo_id, private=private, token=token)


def download_dataset_from_hub(root: str, repo_id: str, token: str | None = None) -> None:
    """`push_dataset_to_hub()`로 업로드된 데이터셋(train/val split)을 내려받아 로컬에 저장.

    split별로 `annotations.jsonl`이 이미 있으면 (허브 재다운로드 + 로컬 이미지 재인코딩을
    건너뛰고) 그대로 둡니다 — 매번 실행할 때마다 처음부터 다시 쓰는 걸 방지.
    """
    root_path = Path(root)
    splits_to_fetch = [s for s in ["train", "val"] if not (root_path / s / "annotations.jsonl").exists()]
    if not splits_to_fetch:
        print(f"'{root_path}'에 이미 다운로드되어 있어 건너뜁니다.")
        return

    from datasets import load_dataset

    ds = load_dataset(repo_id, token=token)
    for split_name in splits_to_fetch:
        _write_split(ds[split_name], root_path / split_name)


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_chat_messages(record: dict, image_root: str) -> list[dict]:
    """레코드를 Qwen-VL류 processor의 chat template 입력 형태로 변환."""
    image_path = str(Path(image_root) / record["image"])
    human, gpt = record["conversations"]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": human["value"].replace("<image>\n", "")},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": gpt["value"]}]},
    ]


def count_trainable_params(model) -> tuple[int, int]:
    """(trainable, total) 파라미터 수를 반환."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?%?$")


def _to_float(text: str) -> float | None:
    text = text.strip().replace(",", "")
    if not _NUMBER_RE.match(text):
        return None
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def is_numeric_answer(text: str) -> bool:
    return _to_float(text) is not None


def judge_answer(answer: str, gold: str, max_relative_change: float = 0.05) -> bool:
    """ChartQA 공식 relaxed accuracy 방식: 숫자 답은 5% 오차까지 정답으로 인정하고,
    그 외(텍스트) 답은 정규화 후 완전 일치할 때만 정답으로 판정."""
    pred_value = _to_float(answer)
    gold_value = _to_float(gold)
    if pred_value is not None and gold_value is not None:
        if gold_value == 0:
            return pred_value == 0
        return abs(pred_value - gold_value) / abs(gold_value) <= max_relative_change
    return answer.strip().lower() == gold.strip().lower()


# ---------------------------------------------------------------------------
# VERA 실습용: Ollama(Qwen3-VL) 호출 헬퍼 — 가중치 학습 없이 텍스트만 오간다.
# ---------------------------------------------------------------------------

def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ask_image(img: Image.Image, prompt: str, model: str = VLM_MODEL, json_format: bool = False, **options) -> str:
    """이미지 1장 + 프롬프트를 Ollama VLM에 보내고 응답 텍스트를 반환."""
    options.setdefault("num_predict", 512)
    options.setdefault("temperature", 0.2)
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt + "\n\n(답변은 반드시 한국어로만, 최종 답변만 작성해.)",
                "images": [_pil_to_png_bytes(img)],
            }
        ],
        format="json" if json_format else None,
        think=False,
        options=options,
    )
    return response["message"]["content"]


def ask_image_json(img: Image.Image, prompt: str, model: str = VLM_MODEL, **options) -> dict:
    """`ask_image(..., json_format=True)` 응답을 dict로 파싱해서 반환."""
    text = ask_image(img, prompt, model=model, json_format=True, **options)
    return json.loads(text)


def ask_text(prompt: str, model: str = VLM_MODEL, **options) -> str:
    """이미지 없이 텍스트만 주고받는 호출 — VERA의 '판단 기준 재작성' 단계에서 사용."""
    options.setdefault("num_predict", 512)
    options.setdefault("temperature", 0.4)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options=options,
    )
    return response["message"]["content"].strip()
