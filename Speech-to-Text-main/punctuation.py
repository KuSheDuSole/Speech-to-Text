"""
punctuation.py — восстановление пунктуации через deepmultilingualpunctuation.

Модель: oliverguhr/fullstop-punctuation-multilang-large
  - XLM-RoBERTa large, обучена на 4 языках включая русский
  - Лицензия: MIT
  - Размер: ~2.2 GB (скачивается один раз в кэш HuggingFace)

Установка: pip install deepmultilingualpunctuation
"""

from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PunctuationRestorer:
    """
    Восстанавливает пунктуацию и заглавные буквы в тексте.

    Параметры
    ----------
    use_gpu : использовать GPU если доступен
    """

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
        # deepmultilingualpunctuation сам управляет устройством через transformers
        self._model = PunctuationModel()
        print("✅ Модель пунктуации загружена.")

    def restore(self, text: str) -> str:
        """
        Принимает текст без пунктуации, возвращает текст с пунктуацией.
        Пустая строка возвращается без изменений.
        """
        if not text or not text.strip():
            return text

        try:
            result = self._model.restore_punctuation(text.strip())
            return _post_process(result)
        except Exception as e:
            logger.warning(f"Ошибка пунктуации: {e}. Возвращаем оригинал.")
            return text

    def restore_segments(self, segments: list[dict]) -> list[dict]:
        """
        Обрабатывает список сегментов (dict с ключом 'text').
        Меняет поле 'text' на месте, возвращает тот же список.
        """
        for seg in segments:
            original = seg.get("text", "")
            if original.strip():
                seg["text"] = self.restore(original)
        return segments


def _post_process(text: str) -> str:
    """Убираем артефакты после модели пунктуации."""
    # пробел перед знаком препинания
    text = re.sub(r'\s+([\.,:;!?])', r'\1', text)
    # двойные пробелы
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    # заглавная буква в начале и после . ! ?
    text = _capitalize_sentences(text)
    return text


def _capitalize_sentences(text: str) -> str:
    """Заглавная буква в начале текста и после . ! ?"""
    if not text:
        return text

    result = []
    capitalize_next = True  # первая буква всегда заглавная

    for i, ch in enumerate(text):
        if capitalize_next and ch.isalpha():
            result.append(ch.upper())
            capitalize_next = False
        else:
            result.append(ch)
            # после . ! ? и пробела — следующая буква заглавная
            if ch in '.!?' and i + 1 < len(text) and text[i + 1] == ' ':
                capitalize_next = True

    return ''.join(result)
