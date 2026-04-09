# install_windows.ps1 — установка всех зависимостей для Windows
#
# Использование (в PowerShell от имени обычного пользователя):
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   .\install_windows.ps1
#
# Что делает:
#   1. Проверяет Python 3.11
#   2. Создаёт venv
#   3. Ставит PyTorch с нужной CUDA (или CPU если GPU нет)
#   4. Ставит все зависимости проекта
#   5. Ставит SpeechBrain 1.0.3
#   6. Ставит pyctcdecode + deepmultilingualpunctuation
#   7. Патчит SpeechBrain
#   8. Создаёт нужные папки
#   9. Проверяет установку

$ErrorActionPreference = "Stop"

# ── Цвета ──────────────────────────────────────────────────────────────────────
function ok($msg)   { Write-Host "✅  $msg" -ForegroundColor Green }
function warn($msg) { Write-Host "⚠️   $msg" -ForegroundColor Yellow }
function err($msg)  { Write-Host "❌  $msg" -ForegroundColor Red; exit 1 }
function info($msg) { Write-Host "ℹ️   $msg" -ForegroundColor Cyan }
function step($msg) { Write-Host "`n━━━ $msg ━━━" -ForegroundColor Cyan }

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Speech-to-Text — Установка зависимостей (Windows)"    -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR   = Join-Path $SCRIPT_DIR "venv"
info "Директория проекта: $SCRIPT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 1 — Python 3.11
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 1/9 — Проверка Python 3.11"

# Ищем Python 3.11 в стандартных местах
$PythonExe = $null
$candidates = @(
    "C:\Python311\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe",
    (Get-Command python -ErrorAction SilentlyContinue)?.Source
)
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) {
        $ver = & $c --version 2>&1
        if ($ver -match "3\.11") {
            $PythonExe = $c
            ok "Найден: $ver ($c)"
            break
        }
    }
}

if (-not $PythonExe) {
    err "Python 3.11 не найден!`nСкачай с: https://www.python.org/downloads/release/python-31110/`nУстанови в C:\Python311 и перезапусти скрипт."
}

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 2 — Виртуальное окружение
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 2/9 — Виртуальное окружение"

if (Test-Path $VENV_DIR) {
    warn "venv уже существует: $VENV_DIR"
    $answer = Read-Host "Пересоздать? (y/N)"
    if ($answer -eq "y") {
        Remove-Item -Recurse -Force $VENV_DIR
        & $PythonExe -m venv $VENV_DIR
        ok "venv пересоздан"
    } else {
        ok "Используем существующий venv"
    }
} else {
    & $PythonExe -m venv $VENV_DIR
    ok "venv создан: $VENV_DIR"
}

$PIP    = Join-Path $VENV_DIR "Scripts\pip.exe"
$PYTHON = Join-Path $VENV_DIR "Scripts\python.exe"

& $PIP install --upgrade pip -q
ok "pip обновлён"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 3 — PyTorch (автоопределение CUDA)
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 3/9 — PyTorch + torchaudio"

# Скрипт НИКОГДА не устанавливает драйверы CUDA — только выбирает нужные wheels
$TorchIndex = $null
$CudaTag    = "cpu"

$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    try {
        $smiOut = & nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>$null
        if ($smiOut) {
            $parts      = $smiOut.Split(",")
            $driverVer  = $parts[0].Trim()
            $gpuName    = $parts[1].Trim()
            $driverMajor = [int]($driverVer.Split(".")[0])
            info "GPU: $gpuName (драйвер $driverVer)"

            if ($driverMajor -ge 570) {
                $TorchIndex = "https://download.pytorch.org/whl/cu128"
                $CudaTag    = "cu128"
            } elseif ($driverMajor -ge 525) {
                $TorchIndex = "https://download.pytorch.org/whl/cu126"
                $CudaTag    = "cu126"
            } else {
                $TorchIndex = "https://download.pytorch.org/whl/cu121"
                $CudaTag    = "cu121"
            }
            info "Выбран индекс: $CudaTag"
        }
    } catch {
        warn "Не удалось определить версию драйвера"
    }
}

if (-not $TorchIndex) {
    info "GPU/CUDA не обнаружен — устанавливаем CPU версию PyTorch."
    info "Скрипт НЕ трогает драйверы и НЕ устанавливает CUDA."
    $TorchIndex = "https://download.pytorch.org/whl/cpu"
    $CudaTag    = "cpu"
}

info "Устанавливаем PyTorch 2.9.0+$CudaTag ..."
try {
    & $PIP install "torch==2.9.0" "torchaudio==2.9.0" --index-url $TorchIndex -q
    ok "PyTorch 2.9.0+$CudaTag установлен"
} catch {
    warn "torch==2.9.0 недоступен — берём последнюю версию..."
    & $PIP install torch torchaudio --index-url $TorchIndex -q
    ok "PyTorch (последняя) установлен"
}

# Проверяем CUDA
& $PYTHON -c "import torch; cuda=torch.cuda.is_available(); print(f'CUDA: {cuda}' + (f', GPU: {torch.cuda.get_device_name(0)}' if cuda else ''))"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 4 — HuggingFace (фиксированные версии!)
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 4/9 — HuggingFace (фиксированные версии)"

# ВАЖНО: huggingface_hub==0.23.4 — иначе конфликт со SpeechBrain 1.0.3
& $PIP install `
    "huggingface_hub==0.23.4" `
    "transformers==4.40.0" `
    "tokenizers==0.19.1" `
    "datasets==2.19.0" `
    "evaluate" `
    "jiwer" `
    -q
ok "HuggingFace стек установлен"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 5 — Аудио и ML зависимости
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 5/9 — Аудио и ML зависимости"

& $PIP install `
    librosa `
    sounddevice `
    soundfile `
    scikit-learn `
    numpy `
    scipy `
    tqdm `
    hyperpyyaml `
    sentencepiece `
    packaging `
    joblib `
    -q
ok "Аудио и ML зависимости установлены"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 6 — SpeechBrain 1.0.3
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 6/9 — SpeechBrain 1.0.3"

& $PIP install "speechbrain==1.0.3" -q
ok "SpeechBrain 1.0.3 установлен"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 7 — pyctcdecode + пунктуация
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 7/9 — pyctcdecode + пунктуация"

& $PIP install pyctcdecode -q
ok "pyctcdecode установлен"

& $PIP install deepmultilingualpunctuation -q
ok "deepmultilingualpunctuation установлен"

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 8 — Патч SpeechBrain
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 8/9 — Патч SpeechBrain"

$PatchScript = Join-Path $SCRIPT_DIR "patch_speechbrain.py"
if (Test-Path $PatchScript) {
    & $PYTHON $PatchScript
} else {
    # Патчим вручную
    $PatchFile = Join-Path $VENV_DIR "Lib\site-packages\speechbrain\utils\torch_audio_backend.py"
    if (Test-Path $PatchFile) {
        $content = Get-Content $PatchFile -Raw -Encoding UTF8
        $old = "available_backends = torchaudio.list_audio_backends()"
        $new = @"
if hasattr(torchaudio, "list_audio_backends"):
            available_backends = torchaudio.list_audio_backends()
        else:
            available_backends = []
"@
        if ($content -match [regex]::Escape($old)) {
            $content = $content.Replace($old, $new)
            Set-Content $PatchFile -Value $content -Encoding UTF8
            ok "Патч SpeechBrain применён"
        } elseif ($content -match "hasattr") {
            ok "Патч уже применён"
        } else {
            warn "Не удалось применить патч автоматически"
            warn "Запусти вручную: python patch_speechbrain.py"
        }
    } else {
        warn "Файл SpeechBrain не найден — возможно другая версия"
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 9 — Создание папок
# ══════════════════════════════════════════════════════════════════════════════
step "Шаг 9/9 — Создание папок проекта"

# Папки создаются рядом со скриптом (в папке проекта)
$folders = @(
    (Join-Path $SCRIPT_DIR "model"),
    (Join-Path $SCRIPT_DIR "my_recorded_waw"),
    (Join-Path $SCRIPT_DIR "final_texts"),
    (Join-Path $SCRIPT_DIR "silero_vad"),
    (Join-Path $SCRIPT_DIR "lm")
)

foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        ok "Создана: $folder"
    } else {
        info "Уже существует: $folder"
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# ИТОГ
# ══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Установка завершена!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

info "Активация окружения:"
Write-Host "    $VENV_DIR\Scripts\Activate.ps1"
Write-Host ""
info "Проверка зависимостей:"
Write-Host "    python check_versions.py"
Write-Host ""
info "Не забудь:"
Write-Host "    1. Скачать Silero VAD в папку silero_vad\"
Write-Host "       https://github.com/snakers4/silero-vad/archive/refs/heads/master.zip"
Write-Host "    2. Скопировать модель wav2vec2 в папку model\"
Write-Host "    3. Обновить пути в config.py если нужно"
Write-Host ""
info "Запуск:"
Write-Host "    python main.py"
