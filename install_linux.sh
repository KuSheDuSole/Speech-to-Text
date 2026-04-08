#!/bin/bash
# install_linux.sh — установка всех зависимостей для Ubuntu
#
# Использование:
#   chmod +x install_linux.sh
#   ./install_linux.sh
#
# Что делает:
#   1. Проверяет Python 3.11
#   2. Создаёт venv
#   3. Ставит PyTorch с нужной CUDA
#   4. Ставит все зависимости проекта
#   5. Ставит pyctcdecode + deepmultilingualpunctuation
#   6. Патчит speechbrain
#   7. Запускает проверку

set -e

# ── цвета ──────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

ok()   { echo -e "${GREEN}✅  $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️   $1${NC}"; }
err()  { echo -e "${RED}❌  $1${NC}"; exit 1; }
info() { echo -e "${CYAN}ℹ️   $1${NC}"; }
step() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }

echo -e "${CYAN}"
echo "========================================================"
echo "  Speech-to-Text — Установка зависимостей (Ubuntu)"
echo "========================================================"
echo -e "${NC}"

# ── рабочая директория — там где лежит скрипт ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
info "Директория проекта: $SCRIPT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 1 — Python 3.11
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 1/8 — Проверка Python 3.11"

if command -v python3.11 &>/dev/null; then
    PY_VER=$(python3.11 --version)
    ok "Найден $PY_VER"
    PYTHON=python3.11
else
    warn "Python 3.11 не найден, устанавливаем..."
    sudo apt-get update -qq
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    ok "Python 3.11 установлен"
    PYTHON=python3.11
fi

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 2 — Виртуальное окружение
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 2/8 — Виртуальное окружение"

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

# Активируем
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
ok "pip обновлён: $(pip --version)"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 3 — Определяем CUDA и ставим PyTorch
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 3/8 — PyTorch + torchaudio"

# Проверяем реально ли работает GPU (не просто установлен nvidia-smi)
GPU_OK=false
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ] && [[ "$GPU_NAME" != *"failed"* ]] && [[ "$GPU_NAME" != *"error"* ]]; then
        GPU_OK=true
        DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        info "GPU: $GPU_NAME"
        info "Драйвер NVIDIA: $DRIVER_VER"
    fi
fi

if [ "$GPU_OK" = true ]; then
    DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
    if [ "$DRIVER_MAJOR" -ge 570 ] 2>/dev/null; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        CUDA_TAG="cu128"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        CUDA_TAG="cu126"
    fi
    info "Выбран индекс: $CUDA_TAG (драйвер $DRIVER_VER)"
else
    info "GPU/CUDA не обнаружен — устанавливаем CPU версию PyTorch."
    info "Скрипт НЕ трогает драйверы и НЕ устанавливает CUDA."
    info "Всё будет работать корректно, просто медленнее (~в 10 раз)."
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    CUDA_TAG="cpu"
fi

# Устанавливаем PyTorch — пробуем 2.9.0, при неудаче берём последнюю доступную
info "Устанавливаем PyTorch (индекс: $CUDA_TAG)..."
if pip install torch==2.9.0 torchaudio==2.9.0 --index-url "$TORCH_INDEX" -q 2>/dev/null; then
    ok "PyTorch 2.9.0+${CUDA_TAG} установлен"
else
    warn "torch==2.9.0 недоступен для $CUDA_TAG — берём последнюю версию..."
    pip install torch torchaudio --index-url "$TORCH_INDEX" -q
    TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    ok "PyTorch ${TORCH_VER} установлен"
fi

# Проверяем CUDA доступность
python -c "import torch; cuda=torch.cuda.is_available(); print(f'CUDA доступен: {cuda}' + (f', GPU: {torch.cuda.get_device_name(0)}' if cuda else ''))"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 4 — HuggingFace + зависимости (фиксированные версии!)
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 4/8 — HuggingFace + основные зависимости"

# ВАЖНО: huggingface_hub==0.23.4 — иначе конфликт со speechbrain 1.0.3
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
# ШАГ 5 — SpeechBrain
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 5/7 — SpeechBrain 1.0.3"

pip install speechbrain==1.0.3 -q
ok "SpeechBrain 1.0.3 установлен"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 6 — pyctcdecode (beam search декодер)
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 6/7 — pyctcdecode"

pip install pyctcdecode -q
ok "pyctcdecode установлен"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 7 — Пунктуация
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 7/7 — Пунктуация (deepmultilingualpunctuation)"

pip install deepmultilingualpunctuation -q
ok "deepmultilingualpunctuation установлен"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 8 — Патч SpeechBrain
# ══════════════════════════════════════════════════════════════════════════════
step "Патч SpeechBrain"

if [ -f "$SCRIPT_DIR/patch_speechbrain.py" ]; then
    python "$SCRIPT_DIR/patch_speechbrain.py"
else
    # Патчим вручную если скрипта нет
    PATCH_FILE="$VENV_DIR/lib/python3.11/site-packages/speechbrain/utils/torch_audio_backend.py"
    if [ -f "$PATCH_FILE" ]; then
        if grep -q "list_audio_backends()" "$PATCH_FILE" && ! grep -q "hasattr(torchaudio" "$PATCH_FILE"; then
            sed -i 's/available_backends = torchaudio\.list_audio_backends()/if hasattr(torchaudio, "list_audio_backends"):\n            available_backends = torchaudio.list_audio_backends()\n        else:\n            available_backends = []/g' "$PATCH_FILE"
            ok "Патч SpeechBrain применён"
        else
            ok "Патч уже применён или не нужен"
        fi
    else
        warn "Файл патча не найден: $PATCH_FILE"
        warn "Запусти вручную: python patch_speechbrain.py"
    fi
fi

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
info "Активация окружения в будущем:"
echo "    source $VENV_DIR/bin/activate"
echo ""
info "Проверка зависимостей:"
echo "    python check_versions.py"
echo ""
info "Запуск:"
echo "    python main.py"