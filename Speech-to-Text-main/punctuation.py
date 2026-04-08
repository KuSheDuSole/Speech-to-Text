from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)
class PunctuationRestorer:
    def __init__(self, use_gpu: bool = True):
        try:
            from deepmultilingualpunctuation import PunctuationModel
            import torch
        except ImportError:
            raise ImportError(
                "Установи: pip install deepmultilingualpunctuation"
            )

        import torch
        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        print(f"✏️  Загружаем модель пунктуации на {device}...")
        self._model = PunctuationModel()
        print("✅ Модель пунктуации загружена.")

    def restore(self, text: str) -> str:
        if not text or not text.strip():
            return text

        try:
            result = self._model.restore_punctuation(text.strip())
            return _post_process(result)
        except Exception as e:
            logger.warning(f"Ошибка пунктуации: {e}. Возвращаем оригинал.")
            return text

    def restore_segments(self, segments: list[dict]) -> list[dict]:
        for seg in segments:
            original = seg.get("text", "")
            if original.strip():
                seg["text"] = self.restore(original)
        return segments


def _post_process(text: str) -> str:
    text = re.sub(r'\s+([\.,:;!?])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    text = _capitalize_sentences(text)
    return text


def _capitalize_sentences(text: str) -> str:
    if not text:
        return text

    result = []
    capitalize_next = True

    for i, ch in enumerate(text):
        if capitalize_next and ch.isalpha():
            result.append(ch.upper())
            capitalize_next = False
        else:
            result.append(ch)
            if ch in '.!?' and i + 1 < len(text) and text[i + 1] == ' ':
                capitalize_next = True

    return ''.join(result)
