"""노트북 전반에서 재사용하는 영상 처리 + Ollama(Qwen3-VL) 호출 헬퍼."""

import json
from pathlib import Path

import cv2
import ollama

VLM_MODEL = "qwen3-vl:4b"
EMBED_MODEL = "qwen3-embedding:0.6b"


# ---------------------------------------------------------------------------
# 1. 영상 메타데이터 / 프레임 추출 (OpenCV)
# ---------------------------------------------------------------------------

def get_video_info(video_path):
    """fps, 총 프레임 수, 재생 시간(초), 해상도를 dict로 반환한다."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps if fps else 0.0,
        "width": width,
        "height": height,
    }


def _resize_max_width(frame, max_width):
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    return frame


def extract_frames_uniform(video_path, n_frames=8, max_width=512):
    """영상을 n_frames개로 균등 분할해 프레임을 뽑는다.

    반환값: [(timestamp_sec, frame_bgr), ...]
    """
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    results = []
    for i in range(n_frames):
        # 양 끝(0초, 마지막 프레임)이 아니라 구간의 "가운데" 지점을 뽑아 대표성을 높인다.
        target_frame = int((i + 0.5) / n_frames * info["frame_count"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok:
            continue
        timestamp = target_frame / info["fps"] if info["fps"] else 0.0
        results.append((timestamp, _resize_max_width(frame, max_width)))
    cap.release()
    return results


def extract_frames_fixed_fps(video_path, sample_fps=1.0, max_width=512):
    """고정 간격(예: 1초에 1장)으로 프레임을 뽑는다."""
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    step_sec = 1.0 / sample_fps
    results = []
    t = 0.0
    while t < info["duration_sec"]:
        frame_idx = int(t * info["fps"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        results.append((t, _resize_max_width(frame, max_width)))
        t += step_sec
    cap.release()
    return results


def extract_frames_range(video_path, start_sec, end_sec, n_frames=3, max_width=512):
    """[start_sec, end_sec] 구간 안에서만 균등 샘플링한다 (Video RAG의 클립 단위 추출용)."""
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    results = []
    for i in range(n_frames):
        t = start_sec + (i + 0.5) / n_frames * (end_sec - start_sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * info["fps"]))
        ok, frame = cap.read()
        if ok:
            results.append((t, _resize_max_width(frame, max_width)))
    cap.release()
    return results


def extract_frames_scene_change(video_path, threshold=30.0, min_gap_sec=0.5, max_width=512):
    """직전에 뽑은 프레임과의 히스토그램 차이가 threshold를 넘을 때만 프레임을 뽑는다.

    장면(컷) 전환 지점을 포착하는 것이 목적이라 균등 샘플링과 달리 프레임 수가
    영상 내용에 따라 들쭉날쭉하다.
    """
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    results = []
    prev_hist = None
    last_picked_sec = -min_gap_sec
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / info["fps"] if info["fps"] else 0.0
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is None:
            results.append((t, _resize_max_width(frame, max_width)))
            last_picked_sec = t
        else:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > threshold and (t - last_picked_sec) >= min_gap_sec:
                results.append((t, _resize_max_width(frame, max_width)))
                last_picked_sec = t
        prev_hist = hist
        frame_idx += 1
    cap.release()
    return results


def save_frames(frames, out_dir, prefix="frame"):
    """(timestamp, frame) 리스트를 jpg로 저장하고 저장된 경로 리스트를 반환한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (t, frame) in enumerate(frames):
        path = out_dir / f"{prefix}_{i:03d}_{t:06.2f}s.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def show_frames(frames, max_cols=4, figsize_per=(3, 2)):
    """(timestamp, frame) 리스트를 matplotlib grid로 미리 보여준다."""
    import matplotlib.pyplot as plt

    n = len(frames)
    cols = min(max_cols, n) or 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, (t, frame) in zip(axes, frames):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{t:.2f}s")
        ax.axis("off")
    for ax in axes[len(frames):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. VLM 호출 (Ollama, 여러 프레임을 한 메시지에 담아 전달)
# ---------------------------------------------------------------------------

def _frames_to_png_bytes(frames):
    payload = []
    for _, frame in frames:
        ok, buf = cv2.imencode(".png", frame)
        payload.append(buf.tobytes())
    return payload


def _build_prompt_with_timestamps(prompt, frames):
    timeline = ", ".join(f"{i+1}번={t:.1f}초" for i, (t, _) in enumerate(frames))
    return f"{prompt}\n\n[프레임 순서와 시간] {timeline}"


_KOREAN_ONLY_INSTRUCTION = (
    "\n\n(주의: 생각 과정을 출력하지 말고 최종 답변만 작성해. 답변은 반드시 한국어로만 써.)"
)


def _looks_like_leaked_reasoning(text):
    """think=False에도 모델이 영어 사고 과정을 그대로 답변에 흘린 경우를 휴리스틱으로 감지한다."""
    head = text.strip()[:20].lower()
    if head.startswith(("okay", "got it", "let's", "let me", "i need", "so,", "alright", "first,")):
        return True
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return len(text) > 50 and latin > hangul * 2


def ask_video(frames, prompt, model=VLM_MODEL, json_format=False, with_timestamps=True,
              max_retries=4, **options):
    """프레임 시퀀스 + 프롬프트를 VLM에 보내고 응답 텍스트를 반환한다.

    간혹 사고 과정(영어)이 그대로 답변에 흘러나오는 경우가 있어, 그런 응답이면
    최대 `max_retries`번 다시 요청한다.
    """
    options.setdefault("num_predict", 2048)
    # 프레임이 여러 장이면 이미지 토큰만으로 기본 컨텍스트(4096)를 넘기기 쉬워
    # 넉넉하게 잡아둔다 (프레임 수가 아주 많으면 이마저도 늘려야 할 수 있음).
    options.setdefault("num_ctx", 16384)
    options.setdefault("temperature", 0.3)
    base_prompt = _build_prompt_with_timestamps(prompt, frames) if with_timestamps else prompt
    full_prompt = base_prompt + _KOREAN_ONLY_INSTRUCTION
    images = _frames_to_png_bytes(frames)

    content = ""
    thinking_fallback = ""
    for _ in range(max_retries + 1):
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": full_prompt,
                "images": images,
            }],
            format="json" if json_format else None,
            think=False,
            options=options,
        )
        message = response["message"]
        content = message["content"]
        # think=False로도 답변이 비어있고 생각 과정만 나올 때가 있다. 이 경우 재시도하되,
        # 재시도가 모두 실패할 때를 대비해 생각 과정을 최후의 대안으로만 남겨둔다.
        if not thinking_fallback and message.get("thinking"):
            thinking_fallback = message["thinking"].removeprefix("<think>").strip()
        if content.strip() and not _looks_like_leaked_reasoning(content):
            return content
    return content.strip() or thinking_fallback


def ask_video_json(frames, prompt, model=VLM_MODEL, **options):
    """`ask_video(..., json_format=True)` 응답을 dict/list로 파싱해서 반환한다."""
    text = ask_video(frames, prompt, model=model, json_format=True, **options)
    return json.loads(text)


# ---------------------------------------------------------------------------
# 3. 텍스트 임베딩 (Video RAG용, Ollama 로컬 임베딩 모델)
# ---------------------------------------------------------------------------

def embed_text(text, model=EMBED_MODEL):
    """텍스트를 벡터(list[float])로 변환한다."""
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]
