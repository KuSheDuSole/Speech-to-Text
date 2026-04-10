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
Запусти скрипт `install_mac.sh`, он создаст виртуальное окружениие и скачает необходммые зависимости

### macOS
 
```bash
chmod +x install_mac.sh
./install_mac.sh
```
 
Homebrew будет установлен автоматически если его нет.  
На **Apple Silicon (M1/M2/M3)** — ускорение через MPS.  
На **Intel Mac** — CPU режим.

Если хочешь запускать из терминала, тогда активируй окружение через
```
source Папка_проекта/venv/bin/activate
```

Если хочешь запускать из PyCharm, то поставь окружение в настройках `Files -> Settings -> Python -> Interpreter 
-> Add Interpreter -> Add local Interpreter -> Select existing -> В Python path выбери Папка_проекта/venv/bin/python`
В строчке Python Interpreter должен быть показан `Python 3.11`

---
### Шаг 2 — Загрузка Silero VAD (вручную)
GitHub может быть недоступен из Python из-за SSL. Скачай архив вручную в браузере:

<https://github.com/snakers4/silero-vad/archive/refs/heads/master.zip>

Распакуй в:
```
Папка_проекта\silero_vad
```
Убедись что файл существует:
```
Папка_проекта\silero_vad\silero-vad-master\hubconf.py
```
---
### Шаг 3 - загрузка модели ASR (вручную)
Необходимо скачать модель автоматического распознавания речи по ссылке

[Ссылка на модель](https://drive.google.com/drive/folders/1gK7TcRhNwesh33L2-M-qf7sUexWIQ31-?usp=sharing)

Скачай модель waw2vec2_golos_002 и распакуй в папку `model`
Папка по итогу должна выглядеть так:
```
Папка_проекта\model\waw2vec2_golos_002
```
---
### Шаг 4 - Установка пути к проекту в файле конфигурации
Необходимо изменить BASE_DIR в файле `Папка_проекта\Speech-to-Text-main\config.py`

```
if platform.system() == "Windows":
    BASE_DIR = Path("D:/CoursePaper")    # для Windows
else:
    BASE_DIR = Path.home() / "Tom_D/CoursePaper"    # для macOS
```
---
### Запуск
Запуск осуществялется либо посредством терминала с активным окружением через команду
```
python Папка_проекта\Speech-to-Text-main\main.py
```
Либо через PyCharm простым запуском скрипта `main.py` в соответствующей директории (с активным окружением, см. выше)

---
### Известные предупреждения (не ошибки, игнорировать)

- Не влияет на работу, аудио загружается через librosa.
    ```
    SpeechBrain could not find any working torchaudio backend.
    ```
- FutureWarning из SpeechBrain, не влияет на результат.
    ```
    torch.cuda.amp.custom_fwd is deprecated
    ```
- UserWarning из PyTorch, не влияет на работу.
    ```
    torch.backends.cudnn.allow_tf32 will be deprecated after Pytorch 2.9
    ```

---
### Итоговая структура проекта
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
