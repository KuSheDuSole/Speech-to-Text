import os
import sys
import time
import subprocess
from pathlib import Path
from config import (
    RECORD_DIR, MODEL_DIR, OUTPUT_FILE,
    MIN_SPEAKERS, MAX_SPEAKERS,
    USE_LM, LM_PATH, LM_ALPHA, LM_BETA, LM_BEAM_WIDTH,
    USE_PUNCTUATION, DEFAULT_HOTWORDS, HOTWORD_WEIGHT,
    MONOLOGUE_STD_THRESHOLD,
)

# ── Состояние сессии (переопределяет config.py до перезапуска) ─────────────────
_hotwords      = list(DEFAULT_HOTWORDS)   # текущий список hotwords
_speaker_mode  = "auto"                   # "auto" | "1" | "2" | "3" | "4"

MAX_HOTWORDS = 25


# ── Утилиты ────────────────────────────────────────────────────────────────────
def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

def wait_key(msg="Нажмите Enter..."):
    input(msg)

def get_last_recorded_file():
    wav_files = list(RECORD_DIR.glob("record_*.wav"))
    if not wav_files:
        return None
    return max(wav_files, key=lambda p: p.stat().st_mtime)

def get_speaker_args():
    """Возвращает min/max_speakers на основе текущего _speaker_mode."""
    if _speaker_mode == "auto":
        return MIN_SPEAKERS, MAX_SPEAKERS
    n = int(_speaker_mode)
    return n, n


# ── Запуск записи ──────────────────────────────────────────────────────────────
def run_recording():
    clear_console()
    print("🎙 Запуск модуля записи...")
    time.sleep(0.5)
    script = Path(__file__).parent / "recording_waw.py"
    subprocess.run([sys.executable, str(script)])
    print("\n✔ Запись завершена.")
    last_file = get_last_recorded_file()
    if last_file:
        print(f"Последний файл: {last_file}")
    else:
        print("❌ Не найден файл записи.")
    wait_key()


# ── Запуск обработки ───────────────────────────────────────────────────────────
def run_processing(file_path):
    clear_console()
    print(f"🧠 Запуск обработки файла:\n{file_path}")
    min_sp, max_sp = get_speaker_args()
    print(f"👥 Режим спикеров: {_speaker_mode} (min={min_sp}, max={max_sp})")
    print(f"🔑 Hotwords ({len(_hotwords)}): {_hotwords if _hotwords else '—'}")
    time.sleep(0.5)

    script = Path(__file__).parent / "speach_to_text.py"

    cmd = [
        sys.executable, str(script),
        "--audio",          str(file_path),
        "--model",          str(MODEL_DIR),
        "--out",            str(OUTPUT_FILE),
        "--min_speakers",   str(min_sp),
        "--max_speakers",   str(max_sp),
        "--lm_beam",        str(LM_BEAM_WIDTH),
        "--hotword_weight", str(HOTWORD_WEIGHT),
    ]

    # LM
    lm_file = Path(LM_PATH) if LM_PATH else None
    if USE_LM and lm_file and lm_file.exists():
        cmd += ["--lm_path", str(lm_file),
                "--lm_alpha", str(LM_ALPHA),
                "--lm_beta",  str(LM_BETA)]
    else:
        cmd += ["--no_lm"]

    if not USE_PUNCTUATION:
        cmd += ["--no_punct"]

    if _hotwords:
        cmd += ["--hotwords"] + _hotwords

    subprocess.run(cmd)
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
    except Exception:
        print("❌ Некорректный ввод")
        wait_key()
        return
    run_processing(file_path)


# ── Меню управления Hotwords ───────────────────────────────────────────────────
def menu_hotwords():
    while True:
        clear_console()
        print("=" * 50)
        print("🔑  УПРАВЛЕНИЕ HOTWORDS")
        print("=" * 50)
        if _hotwords:
            print(f"\nТекущий список ({len(_hotwords)}/{MAX_HOTWORDS}):")
            for i, w in enumerate(_hotwords, 1):
                print(f"  {i:2}. {w}")
        else:
            print("\n  Список пуст.")
        print()
        print("1 — ➕ Добавить слово")
        print("2 — ➖ Удалить слово")
        print("3 — 🗑  Очистить весь список")
        print("0 — ◀  Назад")
        print("-" * 50)

        choice = input("Ваш выбор: ").strip()

        if choice == "1":
            if len(_hotwords) >= MAX_HOTWORDS:
                print(f"❌ Достигнут лимит {MAX_HOTWORDS} слов.")
                wait_key()
                continue
            word = input("Введите слово: ").strip()
            if not word:
                print("❌ Пустое слово.")
                wait_key()
                continue
            # Нормализация — приводим к нижнему регистру
            word = word.lower()
            if word in _hotwords:
                print(f"⚠️  Слово '{word}' уже есть в списке.")
            else:
                _hotwords.append(word)
                print(f"✅ Добавлено: '{word}'")
            wait_key()

        elif choice == "2":
            if not _hotwords:
                print("❌ Список пуст.")
                wait_key()
                continue
            word = input("Введите слово для удаления: ").strip().lower()
            if word in _hotwords:
                _hotwords.remove(word)
                print(f"✅ Удалено: '{word}'")
            else:
                print(f"❌ Слово '{word}' не найдено.")
            wait_key()

        elif choice == "3":
            confirm = input("Очистить весь список? (y/N): ").strip().lower()
            if confirm == "y":
                _hotwords.clear()
                print("✅ Список очищен.")
                wait_key()

        elif choice == "0":
            break


# ── Меню настроек спикеров ─────────────────────────────────────────────────────
def menu_speakers():
    global _speaker_mode
    while True:
        clear_console()
        print("=" * 50)
        print("👥  НАСТРОЙКИ СПИКЕРОВ")
        print("=" * 50)
        print(f"\nТекущий режим: {_speaker_mode}")
        print(f"(порог монолога: std < {MONOLOGUE_STD_THRESHOLD})\n")
        print("1 — 🤖 Авто (определять автоматически)")
        print("2 — 👤 1 спикер (монолог)")
        print("3 — 👥 2 спикера")
        print("4 — 👥 3 спикера")
        print("5 — 👥 4 спикера")
        print("0 — ◀  Назад")
        print("-" * 50)

        choice = input("Ваш выбор: ").strip()

        mapping = {"1": "auto", "2": "1", "3": "2", "4": "3", "5": "4"}
        if choice in mapping:
            _speaker_mode = mapping[choice]
            labels = {
                "auto": "Авто — количество определяется автоматически",
                "1":    "1 спикер — монолог",
                "2":    "2 спикера",
                "3":    "3 спикера",
                "4":    "4 спикера",
            }
            print(f"✅ Установлено: {labels[_speaker_mode]}")
            wait_key()
            break
        elif choice == "0":
            break


# ── Главное меню ───────────────────────────────────────────────────────────────
def menu_loop():
    while True:
        clear_console()
        min_sp, max_sp = get_speaker_args()
        speaker_label = (
            "авто" if _speaker_mode == "auto"
            else f"{_speaker_mode} чел."
        )
        hotwords_label = f"{len(_hotwords)} сл." if _hotwords else "нет"

        print("=" * 60)
        print("🎛  ГЛАВНОЕ МЕНЮ — Speech-to-Text (ASR + DIARIZATION)")
        print("=" * 60)
        print("1 — 🎤 Записать аудио")
        print("2 — 🤖 Обработать последнее записанное аудио")
        print("3 — 📁 Выбрать файл вручную и обработать")
        print("4 — 🔑 Управление hotwords"
              f"  [{hotwords_label}]")
        print("5 — 👥 Настройки спикеров"
              f"      [{speaker_label}]")
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
        elif choice == "4":
            menu_hotwords()
        elif choice == "5":
            menu_speakers()
        elif choice == "0":
            print("👋 Выход.")
            break
        else:
            print("❌ Неизвестная команда.")
            wait_key()


if __name__ == "__main__":
    menu_loop()