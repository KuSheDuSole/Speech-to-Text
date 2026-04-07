"""
lm_decoder.py — Beam search декодер для wav2vec2.

Использует pyctcdecode для beam search декодирования с поддержкой hotwords.
KenLM не используется — чистый beam search без языковой модели.

Лицензия pyctcdecode: Apache 2.0
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


class LMDecoder:
    """
    Beam search декодер с поддержкой hotwords.

    Параметры
    ----------
    processor      : AutoProcessor (wav2vec2)
    beam_width     : ширина луча (50–200)
    hotwords       : список важных слов с повышенным весом
                     например: ["ЕГЭ", "КубГУ", "ФКТиПМ"]
    hotword_weight : вес hotwords (5.0–20.0)
    """

    def __init__(
        self,
        processor,
        beam_width: int = 100,
        hotwords: Optional[list[str]] = None,
        hotword_weight: float = 10.0
    ):
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError:
            raise ImportError("Установи pyctcdecode: pip install pyctcdecode")

        self.beam_width = beam_width
        self.hotwords = [w.lower() for w in hotwords] if hotwords else []
        self.hotword_weight = hotword_weight

        labels = _get_vocab(processor)
        self._decoder = build_ctcdecoder(labels=labels)

        print(f"✅ LMDecoder: beam search, beam_width={beam_width}")
        if self.hotwords:
            print(f"   hotwords ({len(self.hotwords)}): {self.hotwords}")

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