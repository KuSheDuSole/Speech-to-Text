import argparse
import math
from collections import defaultdict
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC
from diarization_silera_ecapa import diarize
import textwrap
import re
import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hotwords — добавляй сюда аббревиатуры и термины своей предметной области
# ---------------------------------------------------------------------------
DEFAULT_HOTWORDS = [
    "егэ", "огэ", "кубгу", "фктипм", "фкт", "макрос",
    "краснодар", "университет", "институт", "факультет", "интернетцентр", "ран",
]


# ---------------------------------------------------------------------------
# Текстовая нормализация
# ---------------------------------------------------------------------------

def prepare_for_evaluation(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<unk>', '', text)
    text = re.sub(r'[^а-яё0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Слияние соседних сегментов одного спикера
# ---------------------------------------------------------------------------

def merge_adjacent_segments(segments):
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.get("speaker") == last.get("speaker") and abs(seg["start"] - last["end"]) < 1.2:
            last["end"] = seg["end"]
            last["text"] = (last.get("text", "") + " " + seg.get("text", "")).strip()
        else:
            merged.append(seg.copy())
    return merged


# ---------------------------------------------------------------------------
# Основной pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    audio_path: str,
    model_path: str,
    out_path: str = None,
    min_speakers: int = 1,
    max_speakers: int = 4,
    sample_rate: int = 16000,
    device: str = None,
    # LM параметры
    use_lm: bool = True,
    #lm_path: str = None,
    lm_path: str = os.path.expanduser("~/Tom_D/CoursePaper/lm/ru_3gram.bin"),
    lm_alpha: float = 0.5,
    lm_beta: float = 1.5,
    lm_beam_width: int = 100,
    hotwords: list[str] = None,
    hotword_weight: float = 10.0,
    # Пунктуация
    use_punctuation: bool = True,
):
    """
    Параметры
    ----------
    use_lm         : True → beam search декодирование (лучше greedy)
    lm_path        : путь к .arpa/.bin файлу KenLM (только Linux/Ubuntu)
                     None → beam search без LM
    lm_alpha       : вес LM (0.3–0.8), только с KenLM
    lm_beta        : бонус вставки слова (0.5–2.0), только с KenLM
    lm_beam_width  : ширина луча (50–200)
    hotwords       : список важных слов (аббревиатуры, термины)
    hotword_weight : вес hotwords (5–20)
    use_punctuation: True → восстанавливать пунктуацию после ASR
    """

    # ── устройство ────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ℹ️  Используем устройство: {device}")

    # ── диаризация ────────────────────────────────────────────────────────
    print(f"\n🔎 Запускаем диаризацию для файла: {audio_path}")
    segments = diarize(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)

    if not segments:
        print("⚠️  Диаризация вернула 0 сегментов. Завершаю.")
        return {}

    segments = sorted(segments, key=lambda x: x["start"])

    # ── загрузка ASR модели ───────────────────────────────────────────────
    print(f"\n🔧 Загружаем ASR модель из: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()

    # ── инициализация LM-декодера ─────────────────────────────────────────
    lm_decoder = None
    if use_lm:
        print(f"\n📖 Инициализируем beam search декодер...")
        try:
            from lm_decoder import LMDecoder
            hw = hotwords if hotwords is not None else DEFAULT_HOTWORDS
            lm_decoder = LMDecoder(
                processor=processor,
                lm_path=lm_path,
                alpha=lm_alpha,
                beta=lm_beta,
                beam_width=lm_beam_width,
                hotwords=hw,
                hotword_weight=hotword_weight,
            )
        except ImportError as e:
            print(f"⚠️  {e}\nПродолжаем с greedy декодированием.")

    # ── инициализация пунктуации ──────────────────────────────────────────
    punct_restorer = None
    if use_punctuation:
        try:
            from punctuation import PunctuationRestorer
            punct_restorer = PunctuationRestorer(use_gpu=(device == "cuda"))
        except ImportError as e:
            print(f"⚠️  {e}\nПродолжаем без пунктуации.")

    # ── загрузка аудио ────────────────────────────────────────────────────
    wav, sr = librosa.load(audio_path, sr=sample_rate)
    wav = wav.astype(np.float32)

    # ── распознавание сегментов ───────────────────────────────────────────
    print("\n🎙 Распознавание сегментов:")
    results = []

    for seg in tqdm(segments, desc="segments", unit="seg"):
        start_s = seg["start"]
        end_s   = seg["end"]
        start_idx = max(0, int(math.floor(start_s * sr)))
        end_idx   = min(len(wav), int(math.ceil(end_s * sr)))
        audio_seg = wav[start_idx:end_idx]

        if len(audio_seg) == 0:
            text = ""
        else:
            inputs = processor(audio_seg, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                logits = model(input_values).logits  # [1, T, vocab]

            if lm_decoder is not None:
                # Beam search
                logits_np = logits[0].cpu().float().numpy()
                text = lm_decoder.decode(logits_np)
                text = prepare_for_evaluation(text)
            else:
                # Greedy (оригинальное поведение)
                pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                try:
                    text = processor.decode(pred_ids, skip_special_tokens=True)
                except Exception:
                    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
                text = prepare_for_evaluation(text)

        results.append({
            "start":   start_s,
            "end":     end_s,
            "speaker": seg.get("speaker", "speaker_0"),
            "text":    text,
        })

    # ── слияние соседних сегментов ────────────────────────────────────────
    merged = merge_adjacent_segments(results)

    # ── восстановление пунктуации ─────────────────────────────────────────
    if punct_restorer is not None:
        print("\n✏️  Восстанавливаем пунктуацию...")
        merged = punct_restorer.restore_segments(merged)

    # ── сборка диалога ────────────────────────────────────────────────────
    speaker_dialog = defaultdict(list)
    for seg in merged:
        sp = seg["speaker"].upper()
        t  = seg["text"].strip()
        if t:
            speaker_dialog[sp].append(t)

    lines = ["📝 Итоговый диалог:\n"]
    for seg in merged:
        sp = seg["speaker"].upper()
        t  = seg["text"].strip()
        if not t:
            continue
        lines.append(f"{sp}:")
        for line in textwrap.fill(t, width=70).split("\n"):
            lines.append("    " + line)
        lines.append("")

    transcript_text = "\n".join(lines)

    print("\n\n📄 Итоговый протокол:\n")
    print(transcript_text)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"\n💾 Транскрипт сохранён в: {out_path}")

    return {
        "segments":   merged,
        "by_speaker": dict(speaker_dialog),
        "transcript": transcript_text,
    }


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def format_time(t_seconds: float) -> str:
    ms = int((t_seconds - int(t_seconds)) * 1000)
    s  = int(t_seconds) % 60
    m  = (int(t_seconds) // 60) % 60
    h  = int(t_seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ASR + Diarization + Beam Search + Punctuation")

    parser.add_argument("--audio", "-a",
                        default= os.path.expanduser('~/Tom_D/CoursePaper/my_recorded_waw/record_20251209_211331.wav'))
    parser.add_argument("--model", "-m",
                        default=os.path.expanduser('~/Tom_D/CoursePaper/model/wav2vec2_finetuned_subset_002'))
    parser.add_argument("--out", "-o",
                        default=os.path.expanduser('~/Tom_D/CoursePaper/final_texts/transcript.txt'))
    parser.add_argument("--min_speakers", type=int, default=1)
    parser.add_argument("--max_speakers", type=int, default=4)

    # LM / beam search
    parser.add_argument("--no_lm", action="store_true",
                        help="Отключить beam search, использовать greedy")
    parser.add_argument("--lm_path", default=os.path.expanduser('~/Tom_D/CoursePaper/lm/ru_3gram.bin'),
                        help="Путь к .arpa/.bin файлу KenLM (только Linux)")
    parser.add_argument("--lm_alpha", type=float, default=0.2,
                        help="Вес LM (default=0.5, только с KenLM)")
    parser.add_argument("--lm_beta", type=float, default=1.0,
                        help="Бонус вставки слова (default=1.5, только с KenLM)")
    parser.add_argument("--lm_beam", type=int, default=100,
                        help="Ширина луча beam search (default=100)")
    parser.add_argument("--hotwords", nargs="*", default=None,
                        help="Список hotwords через пробел: --hotwords ЕГЭ КубГУ ФКТиПМ")
    parser.add_argument("--hotword_weight", type=float, default=10.0,
                        help="Вес hotwords (default=10.0)")

    # Пунктуация
    parser.add_argument("--no_punct", action="store_true",
                        help="Отключить восстановление пунктуации")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        audio_path      = args.audio,
        model_path      = args.model,
        out_path        = args.out,
        min_speakers    = args.min_speakers,
        max_speakers    = args.max_speakers,
        use_lm          = not args.no_lm,
        lm_path         = args.lm_path,
        lm_alpha        = args.lm_alpha,
        lm_beta         = args.lm_beta,
        lm_beam_width   = args.lm_beam,
        hotwords        = args.hotwords,
        hotword_weight  = args.hotword_weight,
        use_punctuation = not args.no_punct,
    )