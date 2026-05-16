import argparse
import math
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC
from diarization_silera_ecapa import diarize
import textwrap
import re
import logging

from speech_analysis import analyze_speech, format_analysis
from emotion_analysis import EmotionAnalyzer, format_emotion


logger = logging.getLogger(__name__)

DEFAULT_HOTWORDS = [
    "егэ", "огэ", "кубгу", "фктипм", "фкт", "макрос",
    "краснодар", "университет", "институт", "факультет",
]


def prepare_for_evaluation(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<unk>', '', text)
    text = re.sub(r'[^а-яё0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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


def run_pipeline(
        audio_path: str,
        model_path: str,
        out_path: str = None,
        min_speakers: int = 1,
        max_speakers: int = 4,
        sample_rate: int = 16000,
        device: str = None,
        use_lm: bool = True,
        lm_beam_width: int = 100,
        hotwords: list[str] = None,
        hotword_weight: float = 10.0,
        use_punctuation: bool = True,
        use_emotion: bool = True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ℹ️  Используем устройство: {device}")

    print(f"\n🔎 Запускаем диаризацию для файла: {audio_path}")
    segments = diarize(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)

    if not segments:
        print("⚠️  Диаризация вернула 0 сегментов. Завершаю.")
        return {}

    segments = sorted(segments, key=lambda x: x["start"])

    print(f"\n🔧 Загружаем ASR модель из: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path).to(device)
    model.eval()

    lm_decoder = None
    if use_lm:
        print(f"\n📖 Инициализируем beam search декодер...")
        try:
            from lm_decoder import LMDecoder
            hw = hotwords if hotwords is not None else DEFAULT_HOTWORDS
            lm_decoder = LMDecoder(
                processor=processor,
                beam_width=lm_beam_width,
                hotwords=hw,
                hotword_weight=hotword_weight,
            )
        except ImportError as e:
            print(f"⚠️  {e}\nПродолжаем с greedy декодированием.")

    punct_restorer = None
    if use_punctuation:
        try:
            from punctuation import PunctuationRestorer
            punct_restorer = PunctuationRestorer(use_gpu=(device == "cuda"))
        except ImportError as e:
            print(f"⚠️  {e}\nПродолжаем без пунктуации.")

    wav, sr = librosa.load(audio_path, sr=sample_rate)
    wav = wav.astype(np.float32)

    print("\n🎙 Распознавание сегментов:")
    results = []
    logits_list = []

    for seg in tqdm(segments, desc="segments", unit="seg"):
        start_s = seg["start"]
        end_s = seg["end"]
        start_idx = max(0, int(math.floor(start_s * sr)))
        end_idx = min(len(wav), int(math.ceil(end_s * sr)))
        audio_seg = wav[start_idx:end_idx]

        if len(audio_seg) == 0:
            text = ""
        else:
            inputs = processor(audio_seg, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                logits = model(input_values).logits

            logits_np = logits[0].cpu().float().numpy()
            logits_list.append(logits_np)

            if lm_decoder is not None:
                text = lm_decoder.decode(logits_np)
                text = prepare_for_evaluation(text)
            else:
                pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]
                try:
                    text = processor.decode(pred_ids, skip_special_tokens=True)
                except Exception:
                    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
                text = prepare_for_evaluation(text)

        results.append({
            "start": start_s,
            "end": end_s,
            "speaker": seg.get("speaker", "speaker_0"),
            "text": text,
        })

    merged = merge_adjacent_segments(results)

    if punct_restorer is not None:
        print("\n✏️  Восстанавливаем пунктуацию...")
        merged = punct_restorer.restore_segments(merged)

    print("\n📊 Выполняем объективный анализ речи...")
    speech_analysis_data = analyze_speech(merged, wav, sr, logits_list)
    speech_analysis_text = format_analysis(speech_analysis_data)

    emotion_text = ""
    if use_emotion:
        print("🎭 Анализируем эмоции...")
        try:
            ea = EmotionAnalyzer(device=device)
            emotion_stats = ea.analyze_segments(merged, wav, sr)
            emotion_text = format_emotion(emotion_stats)
        except Exception as e:
            print(f"⚠️ Ошибка анализа эмоций: {e}")

    lines = ["📝 Итоговый диалог:\n"]
    for seg in merged:
        sp = seg["speaker"].upper()
        t = seg["text"].strip()
        if not t:
            continue
        lines.append(f"{sp}:")
        for line in textwrap.fill(t, width=70).split("\n"):
            lines.append("    " + line)
        lines.append("")

    transcript_main = "\n".join(lines)

    full_report = (
            transcript_main +
            "\n" + speech_analysis_text +
            ("\n" + emotion_text if emotion_text else "")
    )

    print("\n\n📄 Итоговый протокол:\n")
    print(full_report)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n💾 Полный отчет сохранён в: {out_path}")

    return {
        "segments": merged,
        "transcript": full_report,
        "analysis": speech_analysis_data,
    }


def format_time(t_seconds: float) -> str:
    ms = int((t_seconds - int(t_seconds)) * 1000)
    s = int(t_seconds) % 60
    m = (int(t_seconds) // 60) % 60
    h = int(t_seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_args():
    parser = argparse.ArgumentParser(description="ASR + Diarization + Beam Search + Punctuation + Analysis")

    parser.add_argument("--audio", "-a", required=True)
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--out", "-o", default="transcript.txt")
    parser.add_argument("--min_speakers", type=int, default=1)
    parser.add_argument("--max_speakers", type=int, default=4)

    parser.add_argument("--no_lm", action="store_true")
    parser.add_argument("--lm_beam", type=int, default=100)
    parser.add_argument("--hotwords", nargs="*", default=None)
    parser.add_argument("--hotword_weight", type=float, default=10.0)

    parser.add_argument("--no_punct", action="store_true")
    parser.add_argument("--no_emotion", action="store_true", help="Отключить анализ эмоций")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        audio_path=args.audio,
        model_path=args.model,
        out_path=args.out,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        use_lm=not args.no_lm,
        lm_beam_width=args.lm_beam,
        hotwords=args.hotwords,
        hotword_weight=args.hotword_weight,
        use_punctuation=not args.no_punct,
        use_emotion=not args.no_emotion,
    )