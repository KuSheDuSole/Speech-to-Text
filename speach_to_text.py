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
            last_text = last.get("text", "")
            seg_text = seg.get("text", "")
            last["text"] = (last_text + " " + seg_text).strip()
        else:
            merged.append(seg.copy())
    return merged

def run_pipeline(audio_path: str, model_path: str, out_path: str = None,
                 min_speakers: int = 1, max_speakers: int = 4,
                 sample_rate: int = 16000, device: str = None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ℹ️ Используем устройство: {device}")

    print(f"\n🔎 Запускаем диаризацию для файла: {audio_path}")
    segments = diarize(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)

    if not segments:
        print("⚠️ Диаризация вернула 0 сегментов. Завершаю.")
        return {}

    segments = sorted(segments, key=lambda x: x["start"])

    print(f"\n🔧 Загружаем ASR модель из: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path).to(device)
    model.eval()

    wav, sr = librosa.load(audio_path, sr=sample_rate)
    wav = wav.astype(np.float32)

    print("\n🎙 Распознавание сегментов:")
    results = []
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

            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]

            try:
                text = processor.decode(pred_ids, skip_special_tokens=True)
            except Exception:
                text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

            text = prepare_for_evaluation(text)

        seg_out = {
            "start": start_s,
            "end": end_s,
            "speaker": seg.get("speaker", "speaker_0"),
            "text": text
        }
        results.append(seg_out)

    merged = merge_adjacent_segments(results)

    speaker_dialog = defaultdict(list)

    for seg in merged:
        sp = seg["speaker"].upper()
        text = seg["text"].strip()
        if text:
            speaker_dialog[sp].append(text)

    lines = []
    lines.append("📝 Итоговый диалог:\n")

    for seg in merged:
        sp = seg["speaker"].upper()
        t = seg["text"].strip()
        if not t:
            continue

        lines.append(f"{sp}:")
        wrapped = textwrap.fill(t, width=70)
        for line in wrapped.split("\n"):
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
        "segments": merged,
        "by_speaker": dict(speaker_dialog),
        "transcript": transcript_text
    }


def format_time(t_seconds: float) -> str:
    ms = int((t_seconds - int(t_seconds)) * 1000)
    s = int(t_seconds) % 60
    m = (int(t_seconds) // 60) % 60
    h = int(t_seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Diarization + ASR pipeline")
    parser.add_argument("--audio", "-a", default="D:/CoursePaper/my_recorded_waw/record_20251209_211331.wav",
                        help="Path to audio file (wav)")
    parser.add_argument("--model", "-m", default="./model/wav2vec2_finetuned_subset_002",
                        help="Path to pretrained wav2vec2 model (folder with config/tokenizer)")
    parser.add_argument("--out", "-o", default="D:/CoursePaper/final_texts/transcript.txt",
                        help="Output transcript file (optional)")
    parser.add_argument("--min_speakers", type=int, default=1, help="Minimum speakers (default=2)")
    parser.add_argument("--max_speakers", type=int, default=4, help="Maximum speakers (default=4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.audio, args.model, args.out, args.min_speakers, args.max_speakers)
