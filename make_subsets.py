from datasets import load_dataset, Audio, Dataset
import os
import json
import re
import numpy as np
import gc

print("=== РАЗБИЕНИЕ ДАТАСЕТА НА ЧАСТИ ===")

print("📥 Загружаем датасет...")
dataset = load_dataset("Sh1man/common_voice_21_rus", split="train")
print(f"✅ Загружено: {len(dataset)} записей")

print("🎵 Преобразуем аудио в 16kHz...")
dataset = dataset.cast_column("mp3", Audio(sampling_rate=16000))

def normalize_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^а-яё0-9\s\.,!?\-]', '', text)
    text = re.sub(r'\s*([\.,!?])\s*', r'\1 ', text)
    return text.strip()


def prepare_for_training(batch_item):
    try:
        audio_array = batch_item['mp3']['array']
        sampling_rate = batch_item['mp3']['sampling_rate']

        if batch_item['json'] and 'text' in batch_item['json']:
            raw_text = batch_item['json']['text']
            normalized_text = normalize_text(raw_text)
        else:
            normalized_text = ""

        if len(audio_array) == 0 or not normalized_text:
            return None

        duration = len(audio_array) / sampling_rate
        if duration < 1.0 or duration > 15.0:
            return None

        return {
            'audio_array': audio_array,
            'sampling_rate': sampling_rate,
            'text': normalized_text,
            'duration': duration,
            'utterance_id': batch_item['json']['id'] if batch_item['json'] and 'id' in batch_item['json'] else '',
            'original_text': raw_text
        }
    except Exception as e:
        return None


print("🔧 Начинаем обработку данных...")

output_dir = "/mnt/d/CoursePaper/all_datasets/fully_prepared_subsets"
os.makedirs(output_dir, exist_ok=True)

batch_size = 200
records_per_subset = 1000

current_subset_data = []
subset_counter = 1
total_processed = 0
total_filtered = 0

for i in range(0, len(dataset), batch_size):
    end_idx = min(i + batch_size, len(dataset))
    batch = dataset.select(range(i, end_idx))

    print(f"Обрабатываем батч {i // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}...")

    for j in range(len(batch)):
        result = prepare_for_training(batch[j])

        if result is not None:
            current_subset_data.append(result)
            total_processed += 1

            if len(current_subset_data) >= records_per_subset:
                subset = Dataset.from_list(current_subset_data)
                subset_path = os.path.join(output_dir, f"wav2vec2_ready_subset_{subset_counter:03d}")
                subset.save_to_disk(subset_path)

                durations = [item['duration'] for item in current_subset_data]
                stats = {
                    "subset_id": subset_counter,
                    "records_count": len(subset),
                    "total_duration_hours": sum(durations) / 3600,
                    "avg_duration": float(np.mean(durations))
                }

                with open(os.path.join(output_dir, f"subset_{subset_counter:03d}_stats.json"), 'w') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)

                print(f"✅ Сохранен поддатасет {subset_counter:03d}: {len(subset)} записей")

                current_subset_data = []
                subset_counter += 1
        else:
            total_filtered += 1

    gc.collect()

if current_subset_data:
    subset = Dataset.from_list(current_subset_data)
    subset_path = os.path.join(output_dir, f"wav2vec2_ready_subset_{subset_counter:03d}")
    subset.save_to_disk(subset_path)
    print(f"✅ Сохранен последний поддатасет {subset_counter:03d}: {len(subset)} записей")

final_stats = {
    "total_original": len(dataset),
    "total_processed": total_processed,
    "total_filtered": total_filtered,
    "total_subsets": subset_counter,
    "audio_sample_rate": 16000
}

with open(os.path.join(output_dir, "dataset_stats.json"), 'w', encoding='utf-8') as f:
    json.dump(final_stats, f, ensure_ascii=False, indent=2)

print(f"\n🎉 ПОТОЧНАЯ ПОДГОТОВКА ЗАВЕРШЕНА!")
print(f"📊 Итоговая статистика:")
print(f"   - Обработано: {total_processed}")
print(f"   - Отфильтровано: {total_filtered}")
print(f"   - Поддатасетов: {subset_counter}")