import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
import os
import threading
import sys
import queue

class KeyListener(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.events = queue.Queue()

    def run(self):
        while True:
            key = sys.stdin.readline().rstrip("\n")
            self.events.put(key)

    def get_event(self):
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

def record_with_controls(sample_rate=16000):
    print("=" * 60)
    print("🎙️  ПРОГРАММА ЗАПИСИ С ПАУЗОЙ (SPACE) И СОХРАНЕНИЕМ (ENTER)")
    print("=" * 60)

    #save_dir = input("Введите директорию для сохранения (Enter = текущая): ").strip()
    save_dir = os.path.expanduser("~/Tom_D/CoursePaper/my_recorded_waw")

    if save_dir == "":
        save_dir = os.getcwd()

    if not os.path.exists(save_dir):
        print("❌ Такой директории нет!")
        return

    print(f"📁 Файлы будут сохраняться в: {save_dir}")

    print("\nДоступные устройства записи:")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"  {i}: {d['name']}")

    try:
        device_id = int(input(f"\nВыберите устройство (Enter=0): ") or "0")
    except:
        device_id = 0

    paused = False
    chunks = []

    listener = KeyListener()
    listener.start()

    print("\n=== УПРАВЛЕНИЕ ===")
    print("Пробел — пауза/продолжить")
    print("Enter — завершить и сохранить")
    print("====================\n")

    def callback(indata, frames, time, status):
        nonlocal paused
        if not paused:
            chunks.append(indata.copy())

    print("🎤 Запись началась. Говорите...")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=callback,
        device=device_id
    ):
        while True:
            evt = listener.get_event()
            if evt is None:
                sd.sleep(100)
                continue

            if evt == "":
                print("\n⏹️ Завершение записи...")
                break

            if evt == " ":
                paused = not paused
                if paused:
                    print("⏸️ Паузa")
                else:
                    print("▶ Продолжение")

    if len(chunks) == 0:
        print("❌ Нет данных для сохранения")
        return

    audio = np.vstack(chunks).flatten()

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"record_{timestamp}.wav")

    sf.write(filename, audio, sample_rate)

    print(f"\n✅ ФАЙЛ СОХРАНЕН: {filename}")
    print(f"   Длительность: {len(audio) / sample_rate:.2f} сек")
    print("   Готово!")


if __name__ == "__main__":
    record_with_controls()