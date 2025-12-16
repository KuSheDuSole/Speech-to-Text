import os
import sys
import time
import subprocess
from pathlib import Path

RECORD_DIR = Path("D:/CoursePaper/my_recorded_waw")
MODEL_DIR = Path("./model/wav2vec2_finetuned_subset_002")
OUTPUT_FILE = Path("D:/CoursePaper/final_texts/transcript.txt")

RECORD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def wait_key(msg="Нажмите Enter..."):
    input(msg)


def get_last_recorded_file():
    wav_files = list(RECORD_DIR.glob("record_*.wav"))
    if not wav_files:
        return None
    return max(wav_files, key=lambda p: p.stat().st_mtime)


def run_recording():
    clear_console()
    print("🎙 Запуск модуля записи...")
    time.sleep(0.5)

    subprocess.run([sys.executable, "recording_waw.py"])
    print("\n✔ Запись завершена.")

    last_file = get_last_recorded_file()
    if last_file:
        print(f"Последний файл: {last_file}")
    else:
        print("❌ Не найден файл записи.")

    wait_key()


def run_processing(file_path):
    clear_console()
    print(f"🧠 Запуск обработки файла:\n{file_path}")
    time.sleep(0.5)

    subprocess.run([
        sys.executable,
        "speach_to_text.py",
        "--audio", str(file_path),
        "--model", str(MODEL_DIR),
        "--out", str(OUTPUT_FILE),
        "--min_speakers", "1",
        "--max_speakers", "4"
    ])

    print("\n✔ Обработка завершена.")
    print(f"Текст сохранён в:\n{OUTPUT_FILE}")
    wait_key()


def choose_file_and_process():
    clear_console()
    print("📁 Выбор файла для обработки:")
    print(f"(файлы ищутся в {RECORD_DIR})\n")

    files = list(RECORD_DIR.glob("*.wav"))
    if not files:
        print("❌ Нет wav-файлов.")
        wait_key()
        return

    for i, f in enumerate(files):
        print(f"{i+1}. {f.name}")

    try:
        num = int(input("\nВведите номер файла: "))
        file_path = files[num - 1]
    except:
        print("❌ Некорректный ввод")
        wait_key()
        return

    run_processing(file_path)


def menu_loop():
    while True:
        clear_console()
        print("=" * 60)
        print("🎛  ГЛАВНОЕ МЕНЮ — Speech-to-Text (ASR + DIARIZATION)")
        print("=" * 60)
        print("1 — 🎤 Записать аудио")
        print("2 — 🤖 Обработать последнее записанное аудио")
        print("3 — 📁 Выбрать файл вручную и обработать")
        print("0 — 🚪 Выход")
        print("-" * 60)

        choice = input("Ваш выбор: ").strip()

        if choice == "1":
            run_recording()

        elif choice == "2":
            last_file = get_last_recorded_file()
            if last_file:
                run_processing(last_file)
            else:
                print("❌ Нет записанных файлов.")
                wait_key()

        elif choice == "3":
            choose_file_and_process()

        elif choice == "0":
            print("👋 Выход.")
            break

        else:
            print("❌ Неизвестная команда.")
            wait_key()


if __name__ == "__main__":
    menu_loop()