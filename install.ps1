# install.ps1 — автоматическая установка всех зависимостей
# Запуск: D:\CoursePaper\venv\Scripts\Activate.ps1 ; .\install.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Speech-to-Text — Установка зависимостей" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Проверяем что активирован нужный venv
$pythonVersion = python --version 2>&1
Write-Host "`nPython: $pythonVersion"
if ($pythonVersion -notmatch "3\.11") {
    Write-Host "❌ Нужен Python 3.11! Активируй правильный venv." -ForegroundColor Red
    Write-Host "   D:\CoursePaper\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n📦 Шаг 1/5 — PyTorch + torchaudio (CUDA 12.8)..." -ForegroundColor Yellow
pip install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Ошибка установки PyTorch" -ForegroundColor Red; exit 1 }

Write-Host "`n📦 Шаг 2/5 — HuggingFace (фиксированные версии)..." -ForegroundColor Yellow
pip install huggingface_hub==0.23.4 transformers==4.40.0 tokenizers==0.19.1
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Ошибка установки HuggingFace" -ForegroundColor Red; exit 1 }

Write-Host "`n📦 Шаг 3/5 — SpeechBrain 1.0.3..." -ForegroundColor Yellow
pip install speechbrain==1.0.3
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Ошибка установки SpeechBrain" -ForegroundColor Red; exit 1 }

Write-Host "`n📦 Шаг 4/5 — Datasets, аудио, ML..." -ForegroundColor Yellow
pip install datasets==2.19.0 evaluate jiwer librosa sounddevice soundfile scikit-learn numpy tqdm scipy hyperpyyaml sentencepiece packaging joblib
if ($LASTEXITCODE -ne 0) { Write-Host "❌ Ошибка установки зависимостей" -ForegroundColor Red; exit 1 }

Write-Host "`n📦 Шаг 5/5 — Патч SpeechBrain..." -ForegroundColor Yellow
python patch_speechbrain.py
if ($LASTEXITCODE -ne 0) { Write-Host "⚠️  Патч не применён, сделай вручную (см. SETUP.md)" -ForegroundColor Yellow }

Write-Host "`n✅ Установка завершена!" -ForegroundColor Green
Write-Host "Запусти проверку: python check_versions.py" -ForegroundColor Cyan
