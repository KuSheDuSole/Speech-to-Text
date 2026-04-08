## Speech-to-Text — Инструкция по установке
Проверено на следующих системных параметрах
- Windows 10/11 64-bit
- NVIDIA GPU (протестировано на RTX 3050, CUDA 13.x)
- Python 3.11.x (для работы проекта важно исполь только 3.11)
- ~10 GB свободного места (модели + зависимости)

Версии зависимостей:
- Python           3.11.x
- torch            2.9.0+cu128
- torchaudio       2.9.0+cu128
- speechbrain      1.0.3
- huggingface_hub  0.23.4
- transformers     4.40.x
- datasets         2.19.x
- librosa            x
- sounddevice        x
- scikit-learn       x
---
### Шаг 1 — Установка виртуального окружения и зависимостей
Запусти скрипт `install_linux.sh`

---
### Шаг 2 — Загрузка Silero VAD (вручную)
GitHub может быть недоступен из Python из-за SSL. Скачай архив вручную в браузере:
```
https://github.com/snakers4/silero-vad/archive/refs/heads/master.zip
```
Распакуй в:
```
Папка_проекта\silero_vad
```
Убедись что файл существует:
```
D:\CoursePaper\silero_vad\silero-vad-master\hubconf.py
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
Итоговая структура проекта
```
Папка_проекта\
├── final_texts                    # папка в которой лежит выходная транскрипция
├── venv\                          # виртуальное окружение Python 3.11
├── silero_vad\
│   └── silero-vad-master\         # Silero VAD (скачан вручную)
│       └── hubconf.py
├── model\
│   └── wav2vec2_finetuned_subset_002\    # дообученная ASR модель
│
├── install_linux_sh                # скрипт установки виртуального окружения и зависимостей
│
├── my_recorded_waw\                # папка с записанными аудиофайлами
│
└── Speech-to-Text-main\
    ├── Train\
    │   ├── make_subsets.py
    │   ├── tests.py
    │   └── train.py
    ├── config.py                  # файл конфигурации
    ├── diarization_silera_ecapa.py
    ├── lm_decoder.py
    ├── main.py                    # точка входа в программу
    ├── punctuation.py
    ├── recording_waw.py
    └── speach_to_text.py


```
