#!/bin/bash
# install_mac.sh — установка всех зависимостей для macOS
#
# Использование:
#   chmod +x install_mac.sh
#   ./install_mac.sh
#
# Поддерживается:
#   - Apple Silicon (M1/M2/M3) — ускорение через MPS
#   - Intel Mac               — CPU режим

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

ok()   { echo -e "${GREEN}✅  $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️   $1${NC}"; }
err()  { echo -e "${RED}❌  $1${NC}"; exit 1; }
info() { echo -e "${CYAN}ℹ️   $1${NC}"; }
step() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }

echo -e "${CYAN}"
echo "========================================================"
echo "  Speech-to-Text — Установка зависимостей (macOS)"
echo "========================================================"
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
info "Директория проекта: $SCRIPT_DIR"

# Определяем архитектуру
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    info "Apple Silicon обнаружен — будет использоваться MPS ускорение"
    IS_APPLE_SILICON=true
else
    info "Intel Mac — будет использоваться CPU режим"
    IS_APPLE_SILICON=false
fi

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 1 — Homebrew
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 1/8 — Проверка Homebrew"

if command -v brew &>/dev/null; then
    ok "Homebrew найден: $(brew --version | head -1)"
else
    warn "Homebrew не найден — устанавливаем..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Добавляем brew в PATH для Apple Silicon
    if [ "$IS_APPLE_SILICON" = true ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    ok "Homebrew установлен"
fi

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 2 — Python 3.11 + portaudio
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 2/8 — Python 3.11 + системные зависимости"

# portaudio нужен для sounddevice (запись микрофона)
if brew list portaudio &>/dev/null; then
    ok "portaudio уже установлен"
else
    brew install portaudio
    ok "portaudio установлен"
fi

# Python 3.11
if command -v python3.11 &>/dev/null; then
    ok "Python 3.11 найден: $(python3.11 --version)"
    PYTHON=python3.11
else
    info "Устанавливаем Python 3.11 через Homebrew..."
    brew install python@3.11
    PYTHON=$(brew --prefix python@3.11)/bin/python3.11
    ok "Python 3.11 установлен"
fi

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 3 — Виртуальное окружение
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 3/8 — Виртуальное окружение"

if [ -d "$VENV_DIR" ]; then
    warn "venv уже существует: $VENV_DIR"
    read -p "Пересоздать? (y/N): " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
        ok "venv пересоздан"
    else
        ok "Используем существующий venv"
    fi
else
    $PYTHON -m venv "$VENV_DIR"
    ok "venv создан: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
ok "pip обновлён"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 4 — PyTorch
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 4/8 — PyTorch"

if [ "$IS_APPLE_SILICON" = true ]; then
    info "Устанавливаем PyTorch с поддержкой MPS (Apple Silicon)..."
    # Для Apple Silicon — стандартный pip, MPS включён по умолчанию
    pip install torch torchaudio -q
    ok "PyTorch установлен (MPS доступен)"
else
    info "Устанавливаем PyTorch CPU версию (Intel Mac)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    ok "PyTorch CPU установлен"
fi

# Проверяем доступность ускорителя
python -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Apple GPU) доступен')
elif torch.cuda.is_available():
    print('✅ CUDA доступна')
else:
    print('ℹ️  CPU режим')
"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 5 — HuggingFace + зависимости
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 5/8 — HuggingFace + зависимости"

# ВАЖНО: huggingface_hub==0.23.4 — иначе конфликт со SpeechBrain 1.0.3
pip install \
    huggingface_hub==0.23.4 \
    transformers==4.40.0 \
    tokenizers==0.19.1 \
    datasets==2.19.0 \
    evaluate \
    jiwer \
    -q
ok "HuggingFace стек установлен"

pip install \
    librosa \
    sounddevice \
    soundfile \
    scikit-learn \
    numpy \
    scipy \
    tqdm \
    hyperpyyaml \
    sentencepiece \
    packaging \
    joblib \
    -q
ok "Аудио и ML зависимости установлены"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 6 — SpeechBrain
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 6/8 — SpeechBrain 1.0.3"

pip install speechbrain==1.0.3 -q
ok "SpeechBrain 1.0.3 установлен"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 7 — pyctcdecode + пунктуация
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 7/8 — pyctcdecode + пунктуация"

pip install pyctcdecode -q
ok "pyctcdecode установлен"

pip install deepmultilingualpunctuation -q
ok "deepmultilingualpunctuation установлен"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 8 — Патч SpeechBrain + папки
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 8/8 — Патч SpeechBrain + папки проекта"

if [ -f "$SCRIPT_DIR/patch_speechbrain.py" ]; then
    python "$SCRIPT_DIR/patch_speechbrain.py"
else
    PATCH_FILE="$VENV_DIR/lib/python3.11/site-packages/speechbrain/utils/torch_audio_backend.py"
    if [ -f "$PATCH_FILE" ]; then
        if grep -q "list_audio_backends()" "$PATCH_FILE" && ! grep -q "hasattr(torchaudio" "$PATCH_FILE"; then
            sed -i '' 's/available_backends = torchaudio\.list_audio_backends()/if hasattr(torchaudio, "list_audio_backends"):\n            available_backends = torchaudio.list_audio_backends()\n        else:\n            available_backends = []/g' "$PATCH_FILE"
            ok "Патч SpeechBrain применён"
        else
            ok "Патч уже применён или не нужен"
        fi
    else
        warn "Файл патча не найден — запусти вручную: python patch_speechbrain.py"
    fi
fi

# Создаём папки
for folder in "model" "my_recorded_waw" "final_texts" "silero_vad" "lm"; do
    dir="$SCRIPT_DIR/$folder"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        ok "Создана: $dir"
    else
        info "Уже существует: $dir"
    fi
done

# ══════════════════════════════════════════════════════════════════════════════
# ИТОГ
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Установка завершена!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo ""
ok "Beam search декодирование готово к работе"
echo ""
info "Активация окружения:"
echo "    source $VENV_DIR/bin/activate"
echo ""
info "Не забудь:"
echo "    1. Скачать Silero VAD в папку silero_vad/"
echo "       https://github.com/snakers4/silero-vad/archive/refs/heads/master.zip"
echo "    2. Скопировать модель wav2vec2 в папку model/"
echo "    3. Обновить BASE_DIR в config.py"
echo ""
info "Запуск:"
echo "    python main.py"
