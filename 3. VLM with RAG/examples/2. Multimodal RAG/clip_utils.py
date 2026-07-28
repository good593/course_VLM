"""'4. 이미지 임베딩 직접 검색(CLIP).ipynb' 전용 헬퍼.

CLIP은 이미지 인코더와 텍스트 인코더가 같은 벡터 공간을 공유하도록 학습된
모델이라, 캡션(요약) 없이도 이미지를 곧바로 임베딩해 텍스트 질의로 검색할 수
있다. 다만 텍스트 인코더는 다국어 버전(`clip-ViT-B-32-multilingual-v1`)을 써야
한국어 질의가 제대로 매칭된다 (원본 CLIP 텍스트 인코더는 영어 전용).
"""

from functools import lru_cache

from PIL import Image
from sentence_transformers import SentenceTransformer

IMAGE_MODEL_NAME = "clip-ViT-B-32"
TEXT_MODEL_NAME = "clip-ViT-B-32-multilingual-v1"  # 한국어 등 50+ 언어 지원


@lru_cache(maxsize=1)
def _image_model():
    return SentenceTransformer(IMAGE_MODEL_NAME)


@lru_cache(maxsize=1)
def _text_model():
    return SentenceTransformer(TEXT_MODEL_NAME)


def embed_images(image_paths):
    """이미지 경로 리스트 -> 벡터(list[list[float]]). 캡션 없이 이미지 자체를 임베딩."""
    images = [Image.open(p) for p in image_paths]
    return _image_model().encode(images, convert_to_numpy=True).tolist()


def embed_texts(texts):
    """텍스트(한국어 포함) 리스트 -> 벡터(list[list[float]]). 이미지 임베딩과 같은 공간."""
    return _text_model().encode(list(texts), convert_to_numpy=True).tolist()
