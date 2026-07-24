"""노트북 전반에서 재사용하는 Ollama(Qwen3-VL) 호출 헬퍼."""

import json

import ollama

DEFAULT_MODEL = "qwen3-vl:4b"


def ask(image_path, prompt, model=DEFAULT_MODEL, json_format=False, think=False, **options):
    """이미지 한 장 + 프롬프트를 VLM에 보내고 응답 텍스트를 반환한다.

    qwen3-vl은 think=False를 넘겨도 내부적으로 <think> 추론을 계속 이어가는
    경우가 있다. 이미지가 복잡해 추론이 길어지면 num_predict/num_ctx 한도에
    걸려 최종 답변(content)이 끝내 비어 있는 채로 끝날 수 있어서, 대기 시간을
    무한정 늘리는 대신 num_predict 기본값을 두고 content가 비면 thinking
    내용이라도 반환한다.
    """
    options.setdefault("num_predict", 2048)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
        format="json" if json_format else None,
        think=think,
        options=options,
    )
    message = response["message"]
    content = message["content"]
    if not content.strip() and message.get("thinking"):
        thinking = message["thinking"].removeprefix("<think>").strip()
        if json_format:
            # qwen3-vl이 답변을 content가 아닌 thinking에 그대로 적어버릴 때가 있다.
            # json_format 모드에서는 설명 문구를 덧붙이면 json.loads가 깨지므로 그대로 반환한다.
            content = thinking
        else:
            content = (
                "이미지가 복잡해 추론이 길어지면 num_predict/num_ctx 한도에 걸려 최종 답변(content)이 끝내 비어 있는 채로 끝났습니다. 아래 thinking 내용이라도 참고하세요.\n\n"
                + thinking
            )
    return content


def ask_json(image_path, prompt, model=DEFAULT_MODEL, **options):
    """`ask(..., json_format=True)` 응답을 dict로 파싱해서 반환한다."""
    text = ask(image_path, prompt, model=model, json_format=True, **options)
    return json.loads(text)


def show(image_path, width=360):
    """노트북 셀에서 이미지를 바로 미리 보여준다."""
    from IPython.display import Image as IPyImage, display

    display(IPyImage(filename=str(image_path), width=width))
