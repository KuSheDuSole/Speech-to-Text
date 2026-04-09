"""
train.py — дообучение wav2vec2 на сабсетах GOLOS.
"""

import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

SUBSET_PATH = "D:/CoursePaper/all_datasets/golos_subsets/golos_subset_003"
MODEL_NAME  = "D:/CoursePaper/model/wav2vec2_golos_002"  # стартуем с лучшей модели

subset_number = Path(SUBSET_PATH).name.split("_")[-1]
OUTPUT_DIR    = f"D:/CoursePaper/model/wav2vec2_golos_{subset_number}"

EPOCHS            = 2
BATCH_SIZE        = 4
ACCUMULATION_STEPS = 2
LEARNING_RATE     = 2e-5
FREEZE_LAYERS     = 8

@dataclass
class DataCollatorCTCWithPadding:
    processor: object
    padding: bool = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_values = [torch.tensor(f["input_values"]) for f in features]
        labels       = [torch.tensor(f["labels"])       for f in features]

        input_values = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )
        labels = labels.masked_fill(
            labels == self.processor.tokenizer.pad_token_id, -100
        )
        return {"input_values": input_values, "labels": labels}


print(f"🎯 Сабсет: {subset_number}")
print(f"📂 Модель: {MODEL_NAME}")
print(f"💾 Сохраняем в: {OUTPUT_DIR}")

print("\n🔍 GPU:")
print(f"   CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = "cuda"
else:
    print("   ❌ CPU режим")
    device = "cpu"

print(f"\n📂 Загружаем сабсет...")
dataset = load_from_disk(SUBSET_PATH)
print(f"   → {len(dataset)} записей")

print(f"\n🔧 Загружаем модель...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCTC.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("✅ Модель загружена!")

print(f"\n🧊 Замораживаем CNN + слои трансформера 0-{FREEZE_LAYERS-1}...")
model.freeze_feature_encoder()

for i in range(FREEZE_LAYERS):
    for param in model.wav2vec2.encoder.layers[i].parameters():
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"📊 Обучаемых: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

def prepare_batch(batch):
    inputs = processor(
        np.array(batch["audio_array"], dtype=np.float32),
        sampling_rate=batch["sampling_rate"],
        return_tensors="pt",
    )
    labels = processor(text=batch["text"], return_tensors="pt").input_ids
    return {
        "input_values": inputs.input_values[0],
        "labels":       labels[0],
    }

print("\n📦 Препроцессинг данных...")
dataset = dataset.map(prepare_batch, remove_columns=dataset.column_names)
print(f"   → готово {len(dataset)} записей")

data_collator = DataCollatorCTCWithPadding(processor=processor)

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = ACCUMULATION_STEPS,
    fp16                        = True,
    learning_rate               = LEARNING_RATE,
    num_train_epochs            = EPOCHS,
    save_strategy               = "no",
    logging_steps               = 50,
    logging_first_step          = True,
    report_to                   = "none",
    dataloader_num_workers      = 0,
    remove_unused_columns       = False,
    gradient_checkpointing      = True,
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = dataset,
    data_collator = data_collator,
)

print(f"\n🚀 Обучение на {device.upper()}!")
print(f"📊 Эффективный batch: {BATCH_SIZE} × {ACCUMULATION_STEPS} = {BATCH_SIZE * ACCUMULATION_STEPS}")
print(f"📊 Шагов: ~{len(dataset) // (BATCH_SIZE * ACCUMULATION_STEPS) * EPOCHS}")

try:
    trainer.train()
    print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    print(f"💾 Модель сохранена: {OUTPUT_DIR}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    raise
