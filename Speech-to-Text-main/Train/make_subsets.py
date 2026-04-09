"""
make_subsets.py — подготовка сабсетов из датасета GOLOS для обучения wav2vec2.

Структура GOLOS:
    golos_raw/
        train/
            manifest.jsonl          # все транскрипции
            crowd/9/*.wav           # аудио из crowd9
        0/*.wav                     # аудио из crowd0
"""

import json
import os
import re
import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import Dataset
import gc


MANIFEST_PATH = Path("D:/CoursePaper/golos_raw/test/crowd/manifest.jsonl")
OUTPUT_DIR    = Path("D:/CoursePaper/all_datasets/golos_tests")

AUDIO_DIRS = [
    Path("D:/CoursePaper/golos_raw/test/crowd/files")
]

RECORDS_PER_SUBSET = 2000
MIN_DURATION = 1.0
MAX_DURATION = 15.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^а-яё0-9\s]', '', text)
    return text.strip()


def find_audio(filename: str) -> Path | None:
    """Ищет .wav файл по имени в известных папках."""
    name = Path(filename).name  # берём только имя файла
    for audio_dir in AUDIO_DIRS:
        candidate = audio_dir / name
        if candidate.exists():
            return candidate
    return None


print("=== ПОДГОТОВКА САБСЕТОВ GOLOS ===")
print(f"Манифест: {MANIFEST_PATH}")
print(f"Выход: {OUTPUT_DIR}")
print(f"Записей на сабсет: {RECORDS_PER_SUBSET}")

current_subset = []
subset_counter = 1
total_processed = 0
total_skipped = 0

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f):
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        duration = item.get("duration", 0)
        if duration < MIN_DURATION or duration > MAX_DURATION:
            total_skipped += 1
            continue

        text = normalize_text(item.get("text", ""))
        if not text:
            total_skipped += 1
            continue

        audio_path = find_audio(item["audio_filepath"])
        if audio_path is None:
            total_skipped += 1
            continue

        try:
            audio_array, sr = sf.read(str(audio_path), dtype="float32")
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            if len(audio_array) == 0:
                total_skipped += 1
                continue
        except Exception:
            total_skipped += 1
            continue

        current_subset.append({
            "audio_array":  audio_array,
            "sampling_rate": sr,
            "text":          text,
            "duration":      duration,
        })
        total_processed += 1

        if len(current_subset) >= RECORDS_PER_SUBSET:
            subset_path = OUTPUT_DIR / f"golos_subset_{subset_counter:03d}"
            Dataset.from_list(current_subset).save_to_disk(str(subset_path))

            stats = {
                "subset_id":           subset_counter,
                "records_count":       len(current_subset),
                "total_duration_hours": sum(x["duration"] for x in current_subset) / 3600,
                "avg_duration":        float(np.mean([x["duration"] for x in current_subset])),
            }
            with open(OUTPUT_DIR / f"subset_{subset_counter:03d}_stats.json", "w") as sf_out:
                json.dump(stats, sf_out, ensure_ascii=False, indent=2)

            print(f"✅ Сохранён сабсет {subset_counter:03d}: {len(current_subset)} записей "
                  f"({stats['total_duration_hours']:.2f} ч)")

            current_subset = []
            subset_counter += 1
            gc.collect()

        if line_num % 5000 == 0 and line_num > 0:
            print(f"  Обработано строк манифеста: {line_num}, "
                  f"принято: {total_processed}, пропущено: {total_skipped}")

if current_subset:
    subset_path = OUTPUT_DIR / f"golos_subset_{subset_counter:03d}"
    Dataset.from_list(current_subset).save_to_disk(str(subset_path))
    print(f"✅ Сохранён последний сабсет {subset_counter:03d}: {len(current_subset)} записей")

stats_final = {
    "total_processed": total_processed,
    "total_skipped":   total_skipped,
    "total_subsets":   subset_counter,
}
with open(OUTPUT_DIR / "dataset_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats_final, f, ensure_ascii=False, indent=2)

print(f"\n🎉 ГОТОВО!")
print(f"   Принято:    {total_processed}")
print(f"   Пропущено:  {total_skipped}")
print(f"   Сабсетов:   {subset_counter}")
