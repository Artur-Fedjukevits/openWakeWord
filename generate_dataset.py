"""
Генератор датасета для openWakeWord — wake word "Эй Лёня"
==========================================================
Использует Silero TTS (русский) для синтеза позитивных примеров.
Генерирует аугментированные .wav файлы в нужной структуре папок.

Структура выходных данных:
  dataset/
  ├── positive_train/   # ~1500 файлов для обучения
  ├── positive_test/    # ~300 файлов для тестирования
  ├── negative_train/   # похожие фразы (не "Эй Лёня")
  └── negative_test/

Требования:
  pip install torch torchaudio numpy scipy soundfile pydub

Запуск:
  python generate_dataset.py
  python generate_dataset.py --samples 2000 --output ./my_dataset
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import scipy.io.wavfile as wav
from pathlib import Path

# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────
TARGET_PHRASE = "Эй Лёня"

# Варианты произношения (важно для устойчивости модели)
POSITIVE_PHRASES = [
    "Эй Лёня",
    "Эй, Лёня",
    "эй лёня",
    "эй лёня!",
    "Эй Лёня,",
    "Эй Лёня.",
]

# Негативные фразы — похожие по звуку, чтобы снизить ложные срабатывания
NEGATIVE_PHRASES = [
    "Эй Коля",
    "Эй Соня",
    "Эй Толя",
    "привет",
    "алиса",
    "окей гугл",
    "эй сири",
    "слушай",
    "Алёна",
    "Лёня",         # без "Эй" — важно!
    "Эй",           # только первое слово
    "Лёня привет",
    "где Лёня",
    "позови Лёню",
]

SAMPLE_RATE = 16000        # Hz — обязательно для openWakeWord
SILERO_SAMPLE_RATE = 24000 # Silero v3 не поддерживает 16000, синтезируем в 24000
CLIP_DURATION = 1.5  # секунды — длина каждого клипа

# Silero TTS — русские голоса
SILERO_SPEAKERS = [
    "aidar", "baya", "kseniya", "xenia",
    "eugene", "random",
]

# Диапазоны аугментации
SPEED_RANGE = (0.75, 1.3)    # изменение скорости
VOLUME_RANGE = (0.4, 1.2)    # изменение громкости
NOISE_LEVEL_RANGE = (0.001, 0.015)  # уровень белого шума


# ─────────────────────────────────────────────
# Загрузка Silero TTS
# ─────────────────────────────────────────────
def load_silero_model():
    """Загружает Silero TTS модель для русского языка."""
    print("📥 Загрузка Silero TTS (русский)...")
    try:
        # Новый API Silero v3+ возвращает только модель (без example_text)
        result = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v3_1_ru",
            trust_repo=True,
        )
        # Обрабатываем оба варианта возврата
        if isinstance(result, tuple):
            model = result[0]
        else:
            model = result

        # eval() может отсутствовать в некоторых версиях — не критично
        if hasattr(model, "eval"):
            model.eval()

        print("✅ Silero TTS загружен успешно")
        return model
    except Exception as e:
        print(f"❌ Ошибка загрузки Silero: {e}")
        print("   Убедись, что установлены: pip install torch torchaudio omegaconf")
        sys.exit(1)


# ─────────────────────────────────────────────
# Синтез речи
# ─────────────────────────────────────────────
def synthesize(model, text: str, speaker: str) -> np.ndarray:
    """
    Синтезирует текст в аудио через Silero TTS.
    Silero v3 поддерживает только 8000/24000/48000 Hz,
    поэтому синтезируем в 24000 и ресемплируем в 16000.
    Возвращает numpy array (float32).
    """
    import torchaudio

    with torch.no_grad():
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=SILERO_SAMPLE_RATE,
            put_accent=True,
            put_yo=True,
        )

    # Resample 24000 -> 16000
    resampler = torchaudio.transforms.Resample(
        orig_freq=SILERO_SAMPLE_RATE,
        new_freq=SAMPLE_RATE,
    )
    audio_16k = resampler(audio.unsqueeze(0)).squeeze(0)
    return audio_16k.numpy()


# ─────────────────────────────────────────────
# Аугментации
# ─────────────────────────────────────────────
def change_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """Меняет скорость речи (растяжение/сжатие во времени)."""
    indices = np.round(np.arange(0, len(audio), factor)).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices]


def add_white_noise(audio: np.ndarray, noise_level: float) -> np.ndarray:
    """Добавляет белый шум."""
    noise = np.random.normal(0, noise_level, len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def change_volume(audio: np.ndarray, factor: float) -> np.ndarray:
    """Меняет громкость."""
    return np.clip(audio * factor, -1.0, 1.0)


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Приводит клип к фиксированной длине (обрезает или добавляет тишину)."""
    if len(audio) >= target_length:
        # Берём центральный фрагмент с небольшим случайным смещением
        max_offset = len(audio) - target_length
        offset = random.randint(0, max_offset)
        return audio[offset:offset + target_length]
    else:
        # Добавляем тишину с двух сторон
        pad_total = target_length - len(audio)
        pad_left = random.randint(0, pad_total)
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0)


def augment(audio: np.ndarray, aggressive: bool = False) -> np.ndarray:
    """Применяет случайный набор аугментаций к аудио."""
    # Скорость
    speed = random.uniform(*SPEED_RANGE)
    audio = change_speed(audio, speed)

    # Громкость
    volume = random.uniform(*VOLUME_RANGE)
    audio = change_volume(audio, volume)

    # Белый шум (не всегда)
    if random.random() < 0.6 or aggressive:
        noise = random.uniform(*NOISE_LEVEL_RANGE)
        audio = add_white_noise(audio, noise)

    return audio


def to_16bit_pcm(audio: np.ndarray) -> np.ndarray:
    """Конвертирует float32 [-1,1] в int16 PCM."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Сохраняет аудио как WAV файл (16kHz, mono, 16-bit PCM)."""
    target_len = int(CLIP_DURATION * sample_rate)
    audio = pad_or_trim(audio, target_len)
    pcm = to_16bit_pcm(audio)
    wav.write(path, sample_rate, pcm)


# ─────────────────────────────────────────────
# Генерация датасета
# ─────────────────────────────────────────────
def generate_positive_samples(
    model,
    output_dir: str,
    n_train: int = 1500,
    n_test: int = 300,
):
    """Генерирует позитивные примеры (фразы "Эй Лёня")."""
    train_dir = Path(output_dir) / "positive_train"
    test_dir = Path(output_dir) / "positive_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total = n_train + n_test
    print(f"\n🎙️  Генерация {total} позитивных примеров...")

    count = 0
    while count < total:
        speaker = random.choice(SILERO_SPEAKERS)
        phrase = random.choice(POSITIVE_PHRASES)

        try:
            audio = synthesize(model, phrase, speaker)
            audio = augment(audio)

            dest_dir = train_dir if count < n_train else test_dir
            filename = f"positive_{count:05d}_{speaker}.wav"
            save_wav(str(dest_dir / filename), audio)

            count += 1
            if count % 100 == 0:
                print(f"   ✔ {count}/{total} ({speaker}: «{phrase}»)")

        except Exception as e:
            print(f"   ⚠ Пропуск (speaker={speaker}, phrase=«{phrase}»): {e}")

    print(f"✅ Позитивные примеры готовы: {n_train} train, {n_test} test")


def generate_negative_samples(
    model,
    output_dir: str,
    n_train: int = 3000,
    n_test: int = 500,
):
    """Генерирует негативные примеры (всё что НЕ является wake word)."""
    train_dir = Path(output_dir) / "negative_train"
    test_dir = Path(output_dir) / "negative_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total = n_train + n_test
    print(f"\n🚫 Генерация {total} негативных примеров...")

    count = 0
    while count < total:
        speaker = random.choice(SILERO_SPEAKERS)
        phrase = random.choice(NEGATIVE_PHRASES)

        try:
            audio = synthesize(model, phrase, speaker)
            audio = augment(audio, aggressive=True)

            dest_dir = train_dir if count < n_train else test_dir
            filename = f"negative_{count:05d}_{speaker}.wav"
            save_wav(str(dest_dir / filename), audio)

            count += 1
            if count % 200 == 0:
                print(f"   ✔ {count}/{total}")

        except Exception as e:
            print(f"   ⚠ Пропуск (speaker={speaker}, phrase=«{phrase}»): {e}")

    print(f"✅ Негативные примеры готовы: {n_train} train, {n_test} test")


def generate_silence_samples(output_dir: str, n: int = 300):
    """Добавляет примеры тишины и случайного шума как негативные."""
    neg_dir = Path(output_dir) / "negative_train"
    neg_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔇 Генерация {n} примеров тишины/шума...")
    target_len = int(CLIP_DURATION * SAMPLE_RATE)

    for i in range(n):
        if i % 2 == 0:
            # Тишина
            audio = np.zeros(target_len, dtype=np.float32)
        else:
            # Случайный шум
            audio = np.random.normal(0, 0.01, target_len).astype(np.float32)

        filename = f"silence_{i:04d}.wav"
        save_wav(str(neg_dir / filename), audio)

    print(f"✅ Тишина/шум добавлены")


def print_dataset_stats(output_dir: str):
    """Выводит статистику по созданному датасету."""
    print("\n📊 Статистика датасета:")
    total = 0
    for subdir in ["positive_train", "positive_test", "negative_train", "negative_test"]:
        path = Path(output_dir) / subdir
        if path.exists():
            n = len(list(path.glob("*.wav")))
            total += n
            print(f"   {subdir:25s} {n:5d} файлов")
    print(f"   {'ИТОГО':25s} {total:5d} файлов")


# ─────────────────────────────────────────────
# Инструкция по дальнейшим шагам
# ─────────────────────────────────────────────
def print_next_steps(output_dir: str):
    print("""
╔══════════════════════════════════════════════════════════════╗
║              Следующие шаги после генерации                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. ЗАПИШИ РЕАЛЬНЫЕ ПРИМЕРЫ своего голоса (опционально,      ║
║     но очень сильно улучшает качество):                      ║
║                                                              ║
║     python record_samples.py --output dataset/positive_train ║
║     Скажи "Эй Лёня" 30-50 раз с паузами                    ║
║                                                              ║
║  2. СКАЧАЙ фоновые датасеты для негативных примеров:         ║
║     - FMA (музыка): tiny.cc/fma_sample                       ║
║     - FSD50k (шумы): zenodo.org/record/4060432              ║
║     - Common Voice RU: huggingface.co/datasets/             ║
║       mozilla-foundation/common_voice_11_0                   ║
║                                                              ║
║  3. НАСТРОЙ конфиг openWakeWord:                             ║
║                                                              ║
║     target_phrase: ["Эй Лёня"]                              ║
║     positive_data_dir: ./dataset/positive_train              ║
║     background_paths: [./fma, ./audioset]                    ║
║                                                              ║
║  4. ОБУЧИ модель:                                            ║
║                                                              ║
║     cd openWakeWord                                          ║
║     python train.py --config config_ey_lyonya.yml            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Генератор датасета "Эй Лёня" для openWakeWord'
    )
    parser.add_argument(
        "--output", default="./dataset",
        help="Папка для сохранения датасета (default: ./dataset)"
    )
    parser.add_argument(
        "--positive-train", type=int, default=1500,
        help="Количество позитивных примеров для обучения (default: 1500)"
    )
    parser.add_argument(
        "--positive-test", type=int, default=300,
        help="Количество позитивных примеров для теста (default: 300)"
    )
    parser.add_argument(
        "--negative-train", type=int, default=3000,
        help="Количество негативных примеров для обучения (default: 3000)"
    )
    parser.add_argument(
        "--negative-test", type=int, default=500,
        help="Количество негативных примеров для теста (default: 500)"
    )
    parser.add_argument(
        "--skip-negative", action="store_true",
        help="Пропустить генерацию негативных примеров (если уже есть)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Быстрый режим: 200 pos train, 50 pos test, 400 neg train, 100 neg test"
    )
    args = parser.parse_args()

    if args.quick:
        args.positive_train = 200
        args.positive_test = 50
        args.negative_train = 400
        args.negative_test = 100
        print("⚡ Быстрый режим: уменьшенный датасет для проверки пайплайна")

    print("=" * 60)
    print(f'  Генерация датасета для wake word: "{TARGET_PHRASE}"')
    print(f"  Выходная папка: {args.output}")
    print("=" * 60)

    # Устанавливаем seed для воспроизводимости
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Загружаем модель
    model = load_silero_model()

    # Генерируем позитивные примеры
    generate_positive_samples(
        model,
        args.output,
        n_train=args.positive_train,
        n_test=args.positive_test,
    )

    # Генерируем негативные примеры
    if not args.skip_negative:
        generate_negative_samples(
            model,
            args.output,
            n_train=args.negative_train,
            n_test=args.negative_test,
        )
        generate_silence_samples(args.output, n=200)

    # Статистика
    print_dataset_stats(args.output)

    # Подсказки
    print_next_steps(args.output)


if __name__ == "__main__":
    main()
