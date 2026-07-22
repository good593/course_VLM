"""노트북 전반에서 재사용하는 Ollama(Qwen3-VL) 호출 헬퍼."""

import json

import ollama

DEFAULT_MODEL = "qwen3-vl:4b"


def ask(image_path, prompt, model=DEFAULT_MODEL, json_format=False, **options):
    """이미지 한 장 + 프롬프트를 VLM에 보내고 응답 텍스트를 반환한다."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
        format="json" if json_format else None,
        options=options or None,
    )
    return response["message"]["content"]


def ask_json(image_path, prompt, model=DEFAULT_MODEL, **options):
    """`ask(..., json_format=True)` 응답을 dict로 파싱해서 반환한다."""
    text = ask(image_path, prompt, model=model, json_format=True, **options)
    return json.loads(text)


def show(image_path, width=360):
    """노트북 셀에서 이미지를 바로 미리 보여준다."""
    from IPython.display import Image as IPyImage, display

    display(IPyImage(filename=str(image_path), width=width))
