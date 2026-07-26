"""5강 실습(QLoRA / VERA / Vision Adapter)에서 공통으로 쓰는 유틸리티.

QLoRA·Vision Adapter 노트북은 RunPod GPU 환경 실행을 전제로 하고,
VERA 노트북은 가중치 학습이 없어 로컬 Ollama(Qwen3-VL)만으로 실행됩니다.
"""

import io
import json
import random
from pathlib import Path

import ollama
from PIL import Image, ImageDraw

IMG_SIZE = 224
PRODUCT_MARGIN = 30
SCRATCH_LEN = 40

VLM_MODEL = "qwen3-vl:4b"

CORNERS = ["좌측 상단", "우측 상단", "좌측 하단", "우측 하단"]


def _draw_product(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle(
        [PRODUCT_MARGIN, PRODUCT_MARGIN, IMG_SIZE - PRODUCT_MARGIN, IMG_SIZE - PRODUCT_MARGIN],
        outline="black",
        width=3,
        fill="#d9d9d9",
    )


def _corner_point(name: str) -> tuple[int, int]:
    if name == "좌측 상단":
        return PRODUCT_MARGIN + 5, PRODUCT_MARGIN + 5
    if name == "우측 상단":
        return IMG_SIZE - PRODUCT_MARGIN - 5, PRODUCT_MARGIN + 5
    if name == "좌측 하단":
        return PRODUCT_MARGIN + 5, IMG_SIZE - PRODUCT_MARGIN - 5
    return IMG_SIZE - PRODUCT_MARGIN - 5, IMG_SIZE - PRODUCT_MARGIN - 5


def make_sample_image(has_defect: bool, corner: str | None, seed: int) -> Image.Image:
    """모서리에 스크래치가 있는/없는 장난감 '제품' 이미지를 생성."""
    rng = random.Random(seed)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    _draw_product(draw)

    if has_defect:
        cx, cy = _corner_point(corner)
        jx, jy = rng.randint(-8, 8), rng.randint(-8, 8)
        draw.line(
            [(cx, cy), (cx + jx + SCRATCH_LEN // 2, cy + jy + SCRATCH_LEN // 2)],
            fill="red",
            width=3,
        )

    return img


def build_conversation(has_defect: bool, corner: str | None) -> list[dict]:
    instruction = "<image>\n이 제품 사진에서 불량 부위를 찾아줘."
    if has_defect:
        answer = f"{corner} 모서리에 스크래치가 있습니다."
    else:
        answer = "불량이 발견되지 않았습니다. 정상 제품입니다."
    return [
        {"from": "human", "value": instruction},
        {"from": "gpt", "value": answer},
    ]


def make_toy_defect_dataset(root: str, n_per_split: dict[str, int], seed: int = 0) -> None:
    """모서리 스크래치 유무를 라벨로 하는 장난감 '제품 불량 검사' 데이터셋 생성.

    각 split(train/val)마다 절반은 정상, 절반은 불량(4개 모서리 중 하나)으로 구성.
    """
    root_path = Path(root)
    rng = random.Random(seed)

    for split, n in n_per_split.items():
        img_dir = root_path / split / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for i in range(n):
            has_defect = i % 2 == 0
            corner = rng.choice(CORNERS) if has_defect else None
            img = make_sample_image(has_defect, corner, seed=seed * 10_000 + i)

            fname = f"{i:04d}.png"
            img.save(img_dir / fname)

            records.append(
                {
                    "image": f"images/{fname}",
                    "label": {"has_defect": has_defect, "corner": corner},
                    "conversations": build_conversation(has_defect, corner),
                }
            )

        rng.shuffle(records)
        with open(root_path / split / "annotations.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_chat_messages(record: dict, image_root: str) -> list[dict]:
    """레코드를 Qwen2-VL류 processor의 chat template 입력 형태로 변환."""
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


def judge_answer(answer: str, label: dict) -> bool:
    """생성된 답변이 정답 라벨과 맞는지 아주 단순한 키워드 매칭으로 판정."""
    if label["has_defect"]:
        return "스크래치" in answer and label["corner"] in answer
    return ("불량" in answer and "없" in answer) or "정상" in answer


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
