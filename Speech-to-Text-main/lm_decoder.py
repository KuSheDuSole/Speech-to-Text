"""
lm_decoder.py — Beam search декодер с автоматическим определением KenLM.

Логика:
  - Если KenLM установлен И передан lm_path → полноценная LM (лучшее качество)
  - Если KenLM не установлен или lm_path не передан → beam search без LM
  - В обоих случаях поддерживаются hotwords

На Linux:  pip install https://github.com/kpu/kenlm/archive/master.zip
На Windows: KenLM недоступен без компиляции, используем beam search

Лицензия pyctcdecode: Apache 2.0
Лицензия KenLM: LGPL-2.1 (допускает коммерческое использование)
"""

from __future__ import annotations
import logging
import numpy as np
import torch
from typing import Optional

logger = logging.getLogger(__name__)


def _get_vocab(processor) -> list[str]:
    """Достаём список токенов из tokenizer-а, отсортированный по id."""
    vocab_dict: dict[str, int] = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda kv: kv[1])
    return [tok for tok, _ in sorted_vocab]


def _kenlm_available() -> bool:
    """Проверяем доступен ли KenLM."""
    try:
        import kenlm  # noqa: F401
        return True
    except ImportError:
        return False


class LMDecoder:
    """
    Beam search декодер с опциональной языковой моделью KenLM.

    Параметры
    ----------
    processor      : AutoProcessor (wav2vec2)
    lm_path        : путь к .arpa или .bin файлу KenLM (только Linux)
                     None → beam search без LM
    alpha          : вес LM (0.3–0.8), используется только с KenLM
    beta           : бонус вставки слова (0.5–2.0), используется только с KenLM
    beam_width     : ширина луча (50–200)
    hotwords       : список важных слов с повышенным весом
                     например: ["ЕГЭ", "КубГУ", "ФКТиПМ"]
    hotword_weight : вес hotwords (5.0–20.0)
    """

    def __init__(
        self,
        processor,
        lm_path: Optional[str] = None,
        alpha: float = 0.5,
        beta: float = 1.5,
        beam_width: int = 100,
        hotwords: Optional[list[str]] = None,
        hotword_weight: float = 10.0,
    ):
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError:
            raise ImportError("Установи pyctcdecode: pip install pyctcdecode")

        self.beam_width = beam_width
        self.hotwords = [w.lower() for w in hotwords] if hotwords else []
        self.hotword_weight = hotword_weight
        self.has_lm = False

        labels = _get_vocab(processor)

        # Пробуем загрузить KenLM если передан путь
        if lm_path:
            import os
            if not os.path.isfile(lm_path):
                logger.warning(f"Файл LM не найден: {lm_path} — работаем без LM")
                lm_path = None
            elif not _kenlm_available():
                logger.warning(
                    "KenLM не установлен.\n"
                    "На Ubuntu: pip install https://github.com/kpu/kenlm/archive/master.zip\n"
                    "Работаем без LM."
                )
                lm_path = None

        if lm_path:
            self._decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=lm_path,
                alpha=alpha,
                beta=beta,
            )
            self.has_lm = True
            print(f"✅ LMDecoder: KenLM загружен из {lm_path}")
            print(f"   alpha={alpha}, beta={beta}, beam_width={beam_width}")
        else:
            self._decoder = build_ctcdecoder(labels=labels)
            print(f"✅ LMDecoder: beam search без KenLM, beam_width={beam_width}")

        if self.hotwords:
            print(f"   hotwords: {self.hotwords}")

    def decode(self, logits: np.ndarray) -> str:
        """
        Декодирует один массив логитов (shape: [T, vocab_size]).
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().float().numpy()

        text = self._decoder.decode(
            logits,
            beam_width=self.beam_width,
            hotwords=self.hotwords if self.hotwords else None,
            hotword_weight=self.hotword_weight,
        )
        return text.strip()
