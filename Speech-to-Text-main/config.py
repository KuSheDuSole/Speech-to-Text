"""
config.py — централизованная конфигурация путей и параметров.

Измени BASE_DIR под своё окружение:
  - Linux/Ubuntu: Path.home() / "Tom_D/CoursePaper"
  - Windows:      Path("D:/CoursePaper")
"""

from pathlib import Path
import platform

# ── Определяем базовую директорию автоматически ────────────────────────────
if platform.system() == "Windows":
    BASE_DIR = Path("D:/CoursePaper")
else:
    BASE_DIR = Path.home() / "Tom_D/CoursePaper"

# ── Пути к папкам ──────────────────────────────────────────────────────────
RECORD_DIR   = BASE_DIR / "my_recorded_waw"
MODEL_DIR    = BASE_DIR / "model" / "wav2vec2_finetuned_subset_002"
OUTPUT_DIR   = BASE_DIR / "final_texts"
OUTPUT_FILE  = OUTPUT_DIR / "transcript.txt"
SILERO_DIR   = BASE_DIR / "silero_vad" / "silero-vad-master"
LM_PATH      = BASE_DIR / "lm" / "ru_3gram.bin"  # None если не используется

# ── ASR параметры ──────────────────────────────────────────────────────────
MIN_SPEAKERS    = 1
MAX_SPEAKERS    = 4
SAMPLE_RATE     = 16000

# ── Диаризация ─────────────────────────────────────────────────────────────
MONOLOGUE_STD_THRESHOLD = 0.135  # порог определения монолога

# ── Beam search / LM ───────────────────────────────────────────────────────
USE_LM          = False   # True = использовать KenLM если lm_path задан
LM_ALPHA        = 0.5
LM_BETA         = 1.5
LM_BEAM_WIDTH   = 100
HOTWORD_WEIGHT  = 10.0
DEFAULT_HOTWORDS = [
    "егэ", "огэ", "кубгу", "фктипм", "фкт", "макрос",
    "краснодар", "университет", "институт", "факультет",
]

# ── Пунктуация ─────────────────────────────────────────────────────────────
USE_PUNCTUATION = True

# ── Создаём нужные папки если не существуют ────────────────────────────────
RECORD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
