import torch
from transformers import AutoProcessor, AutoModelForCTC
from datasets import load_from_disk
import evaluate
from tqdm import tqdm
import  re


def prepare_for_evaluation(text):
    text = text.lower()
    text = re.sub(r'<unk>', '', text)
    text = re.sub(r'[^а-яё0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def evaluate_model():

    #MODEL_PATH = "D:/CoursePaper/model/wav2vec2_finetuned_subset_002"
    MODEL_PATH = "D:/CoursePaper/model/wav2vec2_golos_002"

    TEST_DATA_PATH = "D:/CoursePaper/all_datasets/golos_tests/golos_subset_002"
    NUM_TEST_SAMPLES = 1000

    print("🔍 Загружаем модель и данные...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForCTC.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Модель загружена на {device}")

    try:
        test_dataset = load_from_disk(TEST_DATA_PATH)
        test_dataset = test_dataset.select(range(min(NUM_TEST_SAMPLES, len(test_dataset))))
        print(f"✅ Загружено {len(test_dataset)} тестовых примеров")
    except:
        print("❌ Не удалось загрузить тестовые данные, проверьте существование тестового датасета")
        return

    print("\n🎯 БЫСТРАЯ ПРОВЕРКА (5 примеров):")
    print("=" * 60)

    correct_predictions = 0
    for i in range(min(5, len(test_dataset))):
        audio_array = test_dataset[i]["audio_array"]
        true_text = test_dataset[i]["text"]

        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_text = processor.batch_decode(predicted_ids)[0]

        is_correct = (prepare_for_evaluation(true_text) == prepare_for_evaluation(predicted_text))
        if is_correct:
            correct_predictions += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} Пример {i + 1}:")
        print(f"   Истинный:    '{prepare_for_evaluation(true_text)}'")
        print(f"   Предсказанный: '{prepare_for_evaluation(predicted_text)}'")
        print(f"   Совпадение: {is_correct}")
        print("-" * 50)

    print(f"📊 Точность на 5 примерах: {correct_predictions}/5 ({correct_predictions / 5 * 100:.1f}%)")

    print(f"\n📊 ВЫЧИСЛЕНИЕ WER на {len(test_dataset)} примерах...")

    wer_metric = evaluate.load("wer")
    predictions = []
    references = []

    for i in tqdm(range(len(test_dataset))):
        try:
            audio_array = test_dataset[i]["audio_array"]
            true_text = prepare_for_evaluation(test_dataset[i]["text"])

            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_text = processor.batch_decode(predicted_ids)[0]

            predictions.append(prepare_for_evaluation(predicted_text))
            references.append(true_text)
        except Exception as e:
            print(f"❌ Ошибка в примере {i}: {e}")

    wer = wer_metric.compute(predictions=predictions, references=references)

    print(f"\n🎯 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Количество примеров: {len(predictions)}")
    print(f"   WER: {wer:.4f} ({wer * 100:.2f}%)")
    print(f"   Точность (Accuracy): {(1 - wer):.4f} ({(1 - wer) * 100:.2f}%)")

    shift = 50
    print(f"\n🔍 ДОПОЛНИТЕЛЬНЫЕ ПРИМЕРЫ:")
    for i in range(min(shift, len(predictions) - 12), min(12 + shift, len(predictions)), 3):
        print(f"   {i + 1}. Истинный: '{references[i]}'")
        print(f"      Предсказанный: '{predictions[i]}'")
        print()


if __name__ == "__main__":
    evaluate_model()
