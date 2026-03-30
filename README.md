Speech-to-Text — Инструкция по установке
Требования к системе
- Windows 10/11 64-bit
- NVIDIA GPU (протестировано на RTX 3050, CUDA 13.x)
- Python 3.11.x (не 3.12+, не 3.14 — только 3.11!)
- ~10 GB свободного места (модели + зависимости)
---
Шаг 1 — Установка Python 3.11
Скачай установщик вручную в браузере:
```
https://www.python.org/downloads/release/python-31110/
```
Файл: `python-3.11.10-amd64.exe`
При установке:
- ✅ Выбери Customize installation
- ✅ Укажи путь: `C:\Python311`
- ❌ НЕ добавляй в PATH если уже есть другой Python
---
Шаг 2 — Создание виртуального окружения
```powershell
# Создать venv на Python 3.11
C:\Python311\python.exe -m venv D:\CoursePaper\venv

# Активировать
D:\CoursePaper\venv\Scripts\Activate.ps1
```
---
Шаг 3 — Установка зависимостей
Запусти скрипт `install.ps1` (см. ниже) или выполни команды вручную:
```powershell
# 1. PyTorch с CUDA 12.8 (работает на драйверах CUDA 13.x)
pip install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2. HuggingFace — ВАЖНО: фиксированные версии, иначе конфликт со SpeechBrain
pip install huggingface_hub==0.23.4 transformers==4.40.0 tokenizers==0.19.1

# 3. Datasets и оценка
pip install datasets==2.19.0 evaluate jiwer

# 4. SpeechBrain — ВАЖНО: именно 1.0.3
pip install speechbrain==1.0.3

# 5. Аудио и ML
pip install librosa sounddevice soundfile scikit-learn numpy tqdm scipy

# 6. Утилиты
pip install hyperpyyaml sentencepiece packaging joblib
```
---
Шаг 4 — Патч SpeechBrain (обязательно!)
SpeechBrain 1.0.3 вызывает удалённую функцию `torchaudio.list_audio_backends()`.
Нужно исправить один файл вручную.
Открой файл:
```
D:\CoursePaper\venv\Lib\site-packages\speechbrain\utils\torch_audio_backend.py
```
Найди блок `elif torchaudio_major >= 2 and torchaudio_minor >= 1:` и убедись что он выглядит точно так (только пробелы, никаких табов):
```python
    elif torchaudio_major >= 2 and torchaudio_minor >= 1:
        if hasattr(torchaudio, "list_audio_backends"):
            available_backends = torchaudio.list_audio_backends()
        else:
            available_backends = []
        if len(available_backends) == 0:
            logger.warning(
                "SpeechBrain could not find any working torchaudio backend. Audio files may fail to load. Follow this link for instructions and troubleshooting: https://speechbrain.readthedocs.io/en/latest/audioloading.html"
            )
```
Или запусти автоматический патч:
```powershell
python patch_speechbrain.py
```
---
Шаг 5 — Загрузка Silero VAD (офлайн)
GitHub может быть недоступен из Python из-за SSL. Скачай архив вручную в браузере:
```
https://github.com/snakers4/silero-vad/archive/refs/heads/master.zip
```
Распакуй в:
```
D:\CoursePaper\silero_vad\silero-vad-master\
```
Убедись что файл существует:
```
D:\CoursePaper\silero_vad\silero-vad-master\hubconf.py
```
В файле `diarization_silera_ecapa.py` строки загрузки VAD должны быть:
```python
model_vad, utils = torch.hub.load(
    repo_or_dir='D:/CoursePaper/silero_vad/silero-vad-master',
    model='silero_vad',
    source='local',
    force_reload=False
)
```
---
Шаг 6 — Проверка установки
```powershell
python check_versions.py
```
Ожидаемый вывод (все строки зелёные):
```
✅ Python        3.11.x
✅ torch         2.9.0+cu128
✅ torchaudio    2.9.0+cu128
✅ CUDA          доступен (RTX 3050)
✅ speechbrain   1.0.3
✅ huggingface_hub 0.23.4
✅ transformers  4.40.x
✅ datasets      2.19.x
✅ librosa       OK
✅ sounddevice   OK
✅ scikit-learn  OK
✅ SpeechBrain патч  применён
```
---
Известные предупреждения (не ошибки, игнорировать)
```
SpeechBrain could not find any working torchaudio backend.
```
→ Не влияет на работу, аудио загружается через librosa.
```
torch.cuda.amp.custom_fwd is deprecated
```
→ FutureWarning из SpeechBrain, не влияет на результат.
```
torch.backends.cudnn.allow_tf32 will be deprecated after Pytorch 2.9
```
→ UserWarning из PyTorch, не влияет на работу.
---
Структура проекта
```
D:\CoursePaper\
├── venv\                          # виртуальное окружение Python 3.11
├── silero_vad\
│   └── silero-vad-master\         # Silero VAD (скачан вручную)
│       └── hubconf.py
├── model\
│   └── wav2vec2_finetuned_subset_002\  # дообученная ASR модель
└── Speech-to-Text-main\
    ├── main.py
    ├── speach_to_text.py
    ├── diarization_silera_ecapa.py
    ├── recording_waw.py
    ├── train.py
    ├── make_subsets.py
    ├── tests.py
    ├── install.ps1                # скрипт установки
```
