"""노트북 전반에서 재사용하는 Multi-Vector RAG 헬퍼 (OpenAI + Chroma)."""

import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VLM_MODEL = "gpt-4o-mini"                 # 이미지를 이해하는 Vision-Language 모델 (요약/QA 공용)
EMBED_MODEL = "text-embedding-3-small"    # 요약 텍스트를 벡터로 바꾸는 임베딩 모델

_client = OpenAI()


def _image_content(image_path):
    ext = Path(image_path).suffix.lstrip(".").lower() or "png"
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}}


def _chat(images, prompt, model, options):
    content = [{"type": "text", "text": prompt}]
    for image_path in images or []:
        content.append(_image_content(image_path))

    options = {"temperature": 0.2, **options}  # 답변 일관성을 위해 낮은 온도를 기본값으로 사용
    response = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        **options,
    )
    return response.choices[0].message.content.strip()


def summarize_image(image_path, instruction, model=VLM_MODEL, **options):
    """이미지(표/차트/사진 등) 한 장을 짧은 텍스트로 요약한다. 검색(임베딩) 전용으로 쓰인다."""
    return _chat([image_path], instruction, model, options)


def ask_vlm(prompt, image_paths=None, model=VLM_MODEL, **options):
    """이미지 0장 이상 + 프롬프트를 VLM에 보내고 답변 텍스트를 반환한다 (최종 답변 생성용)."""
    return _chat(image_paths, prompt, model, options)


def embed_text(text, model=EMBED_MODEL):
    """텍스트를 벡터(list[float])로 변환한다. Chroma 컬렉션에 넣을 임베딩을 만들 때 사용."""
    return _client.embeddings.create(model=model, input=text).data[0].embedding


def show(image_path, width=320):
    """노트북 셀에서 이미지를 바로 미리 보여준다."""
    from IPython.display import Image as IPyImage, display

    display(IPyImage(filename=str(image_path), width=width))
