from __future__ import annotations
import math, re, textwrap, threading, tempfile, queue, difflib
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

LINE_WIDTH = 70


def _merge_texts(history: str, current: str) -> str:
    if not history:
        return current
    h_tail = history[-60:].lower()
    c_head = current[:60].lower()

    s = difflib.SequenceMatcher(None, h_tail, c_head)
    match = s.find_longest_match(0, len(h_tail), 0, len(c_head))

    if match.size > 5:
        overlap_end_in_current = match.b + match.size
        return current[overlap_end_in_current:]

    h_words = history.split()
    c_words = current.split()
    if h_words and c_words:
        for i in range(min(len(h_words), 5), 0, -1):
            if h_words[-i:] == c_words[:i]:
                return " ".join(c_words[i:])

    return current


class LiveDisplay:
    def __init__(self, lw: int = LINE_WIDTH):
        self.lw = lw
        self.full_text = ""  # Вся накопленная строка
        self.cur_line = ""
        self.started = False

    def update(self, new_chunk: str):
        if not new_chunk.strip():
            return

        added_text = _merge_texts(self.full_text, new_chunk)
        added_text = added_text.strip()

        if not added_text:
            return

        if self.full_text and not self.full_text.endswith(" "):
            self.full_text += " "
        self.full_text += added_text

        words = added_text.split()
        for w in words:
            if not self.started:
                self.cur_line = "🎤 " + w
                self.started = True
            else:
                if len(self.cur_line) + 1 + len(w) <= self.lw:
                    self.cur_line += " " + w
                else:
                    print(f"\r{self.cur_line:<{self.lw}}", flush=True)
                    print()
                    self.cur_line = "    " + w

            print(f"\r{self.cur_line:<{self.lw}}", end="", flush=True)

    def finish(self):
        if self.cur_line:
            print(flush=True)


def run(
        model_dir: Path, record_dir: Path, output_file: Path, silero_dir: Path,
        sample_rate: int = 16000, min_speakers: int = 1, max_speakers: int = 4,
        use_punctuation: bool = True, use_emotion: bool = True,
        lm_beam_width: int = 100, hotwords: list[str] | None = None,
        hotword_weight: float = 10.0,
):
    print("=" * 60)
    print("🎙  LIVE ТРАНСКРИПЦИЯ (FUZZY MATCHING MODE)")
    print("=" * 60)
    print("Enter — стоп  |  Ctrl+C — отмена")
    print("-" * 60)

    SR = sample_rate
    CHUNK_SAMP = int(4.0 * SR)
    HOP_SAMP = int(1.5 * SR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir=str(silero_dir), model="silero_vad",
            source="local", force_reload=False)
        get_speech_ts = vad_utils[0]
    except Exception as e:
        print(f"❌ VAD error: {e}");
        return

    from transformers import AutoProcessor, AutoModelForCTC
    processor = AutoProcessor.from_pretrained(str(model_dir))
    asr = AutoModelForCTC.from_pretrained(str(model_dir)).to(device)
    asr.eval()

    def _decode(audio_np):
        inp = processor(audio_np, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = asr(inp.input_values.to(device)).logits
        ids = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        text = processor.decode(ids, skip_special_tokens=True)
        text = text.lower().replace('<unk>', '')
        text = re.sub(r'[^а-яё0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    aq = queue.Queue()
    full_audio = []
    ring = deque(maxlen=CHUNK_SAMP)
    stop = threading.Event()
    disp = LiveDisplay()

    def _cb(indata, *_):
        d = indata[:, 0].copy().astype(np.float32)
        aq.put(d)

    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()

    import sounddevice as sd
    with sd.InputStream(samplerate=SR, channels=1, callback=_cb):
        print("\n✅ Слушаю...\n")
        new_samples = 0
        while not stop.is_set():
            try:
                while True:
                    data = aq.get_nowait()
                    full_audio.append(data);
                    new_samples += len(data)
                    for s in data: ring.append(s)
            except queue.Empty:
                pass

            if len(ring) == CHUNK_SAMP and new_samples >= HOP_SAMP:
                audio = np.array(ring)
                if len(get_speech_ts(torch.tensor(audio), vad_model)) > 0:
                    txt = _decode(audio)
                    if txt:
                        disp.update(txt)
                new_samples = 0
            __import__('time').sleep(0.05)

    disp.finish()

    if full_audio:
        wav = np.concatenate(full_audio)
        path = record_dir / f"live_{datetime.now().strftime('%H%M%S')}.wav"
        sf.write(path, wav, SR)
        print(f"\n📊 Запись сохранена. Запуск финальной обработки...")
        from speach_to_text import run_pipeline
        run_pipeline(audio_path=str(path), model_path=str(model_dir), out_path=str(output_file),
                     min_speakers=min_speakers, max_speakers=max_speakers,
                     use_punctuation=use_punctuation, use_emotion=use_emotion)