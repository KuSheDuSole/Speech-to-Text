import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class DataCollatorCTCWithPadding:
    processor: any
    padding: bool = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_values = [torch.tensor(feature["input_values"]) for feature in features]
        labels = [torch.tensor(feature["labels"]) for feature in features]

        input_values = torch.nn.utils.rnn.pad_sequence(
            input_values,
            batch_first=True,
            padding_value=0.0
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )

        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        batch = {
            "input_values": input_values,
            "labels": labels
        }

        return batch


#SUBSET_PATH = "D:/CoursePaper/all_datasets/fully_prepared_subsets/wav2vec2_ready_subset_001"
SUBSET_PATH = "D:/CoursePaper/all_datasets/fully_prepared_subsets/wav2vec2_ready_subset_031"

#MODEL_NAME = "bond005/wav2vec2-large-ru-golos"
MODEL_NAME = "D:/CoursePaper/model/wav2vec2_finetuned_subset_004"


subset_number = Path(SUBSET_PATH).name.split("_")[-1]
#OUTPUT_DIR = f"./model/wav2vec2_finetuned_subset_{subset_number}"
OUTPUT_DIR = f"./model/wav2vec2_finetuned_subset_002"

EPOCHS = 4
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2

print(f"🎯 Обучение поддатасета: {subset_number}")
print(f"💾 Сохраняем в: {OUTPUT_DIR}")

print("🔍 Проверяем доступность GPU...")
print(f"   CUDA доступен: {torch.cuda.is_available()}")
print(f"   Количество GPU: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("   ❌ CUDA недоступен, используем CPU")
    device = "cpu"


print("📂 Загружаем сабсет...")
dataset = load_from_disk(SUBSET_PATH)
print(f"   → загружено {len(dataset)} записей")


print(f"🔧 Загружаем модель {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCTC.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("✅ Модель загружена!")
print("🧊 Замораживаем слои...")
model.freeze_feature_encoder()

for i in range(16):
    for param in model.wav2vec2.encoder.layers[i].parameters():
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Обучаемых параметров: {trainable_params:,}/{total_params:,} ({trainable_params / total_params * 100:.1f}%)")


def prepare_batch(batch):
    audio_array = np.array(batch["audio_array"], dtype=np.float32)
    text = batch["text"]
    sampling_rate = batch["sampling_rate"]

    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )

    labels = processor(
        text=text,
        return_tensors="pt"
    ).input_ids

    return {
        "input_values": inputs.input_values[0],
        "labels": labels[0]
    }


print("📦 Преобразуем данные...")
dataset = dataset.map(prepare_batch, remove_columns=dataset.column_names)
print(f"   → готово {len(dataset)} записей")


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    fp16=True,
    learning_rate=1e-4,
    num_train_epochs=EPOCHS,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print(f"\n🚀 Начинаем обучение на {device.upper()}!")
print(f"📊 Batch size: {BATCH_SIZE} × {ACCUMULATION_STEPS} = {BATCH_SIZE * ACCUMULATION_STEPS}")

try:
    trainer.train()
    print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    print(f"💾 Модель сохранена в: {OUTPUT_DIR}")
except Exception as e:
    print(f"❌ Ошибка обучения: {e}")