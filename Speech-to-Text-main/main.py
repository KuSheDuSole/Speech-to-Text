import os
import sys
import subprocess
from pathlib import Path

from config import (
    RECORD_DIR, MODEL_DIR, OUTPUT_FILE, SILERO_DIR,
    MIN_SPEAKERS, MAX_SPEAKERS, SAMPLE_RATE,
    USE_LM, LM_PATH, LM_ALPHA, LM_BETA, LM_BEAM_WIDTH,
    USE_PUNCTUATION, DEFAULT_HOTWORDS, HOTWORD_WEIGHT,
    MONOLOGUE_STD_THRESHOLD,
)

_hotwords     = list(DEFAULT_HOTWORDS)
_speaker_mode = "auto"
MAX_HOTWORDS  = 25


def clear():
    os.system("cls" if os.name == "nt" else "clear")

def pause(msg="Нажмите Enter..."):
    input(msg)

def get_last_wav():
    files = list(RECORD_DIR.glob("record_*.wav"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def speaker_args():
    if _speaker_mode == "auto":
        return MIN_SPEAKERS, MAX_SPEAKERS
    n = int(_speaker_mode)
    return n, n

def build_cmd(script: Path, audio: Path) -> list:
    min_sp, max_sp = speaker_args()
    cmd = [
        sys.executable, str(script),
        "--audio", str(audio), "--model", str(MODEL_DIR),
        "--out",   str(OUTPUT_FILE),
        "--min_speakers", str(min_sp), "--max_speakers", str(max_sp),
        "--lm_beam", str(LM_BEAM_WIDTH), "--hotword_weight", str(HOTWORD_WEIGHT),
    ]
    lm = Path(LM_PATH) if LM_PATH else None
    if USE_LM and lm and lm.exists():
        cmd += ["--lm_path", str(lm), "--lm_alpha", str(LM_ALPHA), "--lm_beta", str(LM_BETA)]
    else:
        cmd += ["--no_lm"]
    if not USE_PUNCTUATION:
        cmd += ["--no_punct"]
    if _hotwords:
        cmd += ["--hotwords"] + _hotwords
    return cmd


def do_record():
    clear()
    print("🎙 Запуск модуля записи...")
    subprocess.run([sys.executable, str(Path(__file__).parent / "recording_waw.py")])
    f = get_last_wav()
    print(f"\n✔ Запись завершена.")
    print(f"Последний файл: {f}" if f else "❌ Файл не найден.")
    pause()


def do_process(path: Path):
    clear()
    print(f"🧠 Обработка: {path}")
    print(f"👥 Спикеры: {_speaker_mode}  |  🔑 Hotwords: {len(_hotwords)}")
    subprocess.run(build_cmd(Path(__file__).parent / "speach_to_text.py", path))
    print(f"\n✔ Готово. Текст сохранён: {OUTPUT_FILE}")
    pause()

def do_choose():
    clear()
    files = list(RECORD_DIR.glob("*.wav"))
    if not files:
        print("❌ Нет wav-файлов.")
        pause()
        return
    print(f"📁 Файлы в {RECORD_DIR}:\n")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    try:
        do_process(files[int(input("\nНомер файла: ")) - 1])
    except Exception:
        print("❌ Некорректный ввод")
        pause()


def menu_hotwords():
    while True:
        clear()
        print("=" * 50)
        print(f"🔑  HOTWORDS  ({len(_hotwords)}/{MAX_HOTWORDS})")
        print("=" * 50)
        if _hotwords:
            for i, w in enumerate(_hotwords, 1):
                print(f"  {i:2}. {w}")
        else:
            print("  Список пуст.")
        print("\n1 — Добавить  2 — Удалить  3 — Очистить  0 — Назад")

        c = input("→ ").strip()
        if c == "1":
            if len(_hotwords) >= MAX_HOTWORDS:
                print(f"❌ Лимит {MAX_HOTWORDS} слов.")
            else:
                w = input("Слово: ").strip().lower()
                if w and w not in _hotwords:
                    _hotwords.append(w)
                    print(f"✅ Добавлено: '{w}'")
                elif w in _hotwords:
                    print("⚠️  Уже есть.")
            pause()
        elif c == "2":
            w = input("Удалить слово: ").strip().lower()
            if w in _hotwords:
                _hotwords.remove(w)
                print(f"✅ Удалено: '{w}'")
            else:
                print("❌ Не найдено.")
            pause()
        elif c == "3":
            if input("Очистить? (y/N): ").strip().lower() == "y":
                _hotwords.clear()
                print("✅ Очищено.")
                pause()
        elif c == "0":
            break

def menu_speakers():
    global _speaker_mode
    opts   = {"1": "auto", "2": "1", "3": "2", "4": "3", "5": "4"}
    labels = {"auto": "Авто", "1": "1 спикер", "2": "2 спикера",
              "3": "3 спикера", "4": "4 спикера"}
    clear()
    print("=" * 50)
    print(f"👥  СПИКЕРЫ  [сейчас: {_speaker_mode}]")
    print(f"    порог монолога: std < {MONOLOGUE_STD_THRESHOLD}")
    print("=" * 50)
    print("1 — Авто  2 — 1 чел  3 — 2 чел  4 — 3 чел  5 — 4 чел  0 — Назад")
    c = input("→ ").strip()
    if c in opts:
        _speaker_mode = opts[c]
        print(f"✅ {labels[_speaker_mode]}")
        pause()


def do_live():
    from live_transcription import run as live_run
    min_sp, max_sp = speaker_args()
    live_run(
        model_dir       = MODEL_DIR,
        record_dir      = RECORD_DIR,
        output_file     = OUTPUT_FILE,
        silero_dir      = SILERO_DIR,
        sample_rate     = SAMPLE_RATE,
        min_speakers    = min_sp,
        max_speakers    = max_sp,
        use_punctuation = USE_PUNCTUATION,
        use_emotion     = True,
        lm_beam_width   = LM_BEAM_WIDTH,
        hotwords        = _hotwords or None,
        hotword_weight  = HOTWORD_WEIGHT,
    )


def main():
    while True:
        clear()
        sp_lbl = "авто" if _speaker_mode == "auto" else f"{_speaker_mode} чел."
        hw_lbl = f"{len(_hotwords)} сл." if _hotwords else "нет"

        print("=" * 60)
        print("🎛  Speech-to-Text  (ASR + Diarization)")
        print("=" * 60)
        print(f"1 — 🎤 Записать аудио")
        print(f"2 — 🤖 Обработать последнее аудио")
        print(f"3 — 📁 Выбрать файл")
        print(f"4 — 🔑 Hotwords                [{hw_lbl}]")
        print(f"5 — 👥 Спикеры                 [{sp_lbl}]")
        print(f"6 — 🔴 Live транскрипция")
        print(f"0 — 🚪 Выход")
        print("-" * 60)

        c = input("→ ").strip()

        if c == "1":
            do_record()
        elif c == "2":
            f = get_last_wav()
            if f:
                do_process(f)
            else:
                print("❌ Нет записанных файлов.")
                pause()
        elif c == "3":
            do_choose()
        elif c == "4":
            menu_hotwords()
        elif c == "5":
            menu_speakers()
        elif c == "6":
            do_live()
        elif c == "0":
            print("👋 Выход.")
            break
        else:
            print("❌ Неизвестная команда.")
            pause()


if __name__ == "__main__":
    main()