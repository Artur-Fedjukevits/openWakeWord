"""
Запись реальных примеров голоса для датасета
=============================================
Запускай после generate_dataset.py.
Поможет записать 30-50 произношений "Эй Лёня" с реального микрофона.
Это значительно улучшит качество модели.

Требования:
  pip install pyaudio numpy scipy

Запуск:
  python record_samples.py
  python record_samples.py --output ./dataset/positive_train --count 50
"""

import os
import time
import argparse
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

try:
    import pyaudio
except ImportError:
    print("❌ pyaudio не установлен. Запусти: pip install pyaudio")
    exit(1)

SAMPLE_RATE = 16000
CLIP_DURATION = 2.5       # секунды записи после нажатия Enter
SILENCE_THRESHOLD = 0.01  # порог для определения тишины


def record_clip(stream, duration: float = CLIP_DURATION) -> np.ndarray:
    """Записывает аудио с уже открытого потока."""
    chunk = 1024
    frames = []
    n_chunks = int(SAMPLE_RATE / chunk * duration)
    for _ in range(n_chunks):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    return np.concatenate(frames)


def trim_silence(audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> np.ndarray:
    """Обрезает тишину в начале и конце."""
    float_audio = audio.astype(np.float32) / 32767.0
    non_silent = np.where(np.abs(float_audio) > threshold)[0]
    if len(non_silent) == 0:
        return audio
    start = max(0, non_silent[0] - int(0.05 * SAMPLE_RATE))  # +50ms до
    end = min(len(audio), non_silent[-1] + int(0.1 * SAMPLE_RATE))  # +100ms после
    return audio[start:end]


def pad_to_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Дополняет тишиной до нужной длины."""
    if len(audio) >= target_len:
        return audio[:target_len]
    pad = target_len - len(audio)
    return np.pad(audio, (int(pad * 0.3), pad - int(pad * 0.3)), mode="constant")


def main():
    parser = argparse.ArgumentParser(
        description='Запись реальных примеров "Эй Лёня"'
    )
    parser.add_argument(
        "--output", default="/home/artur/openWakeWord/my_custom_model/ey_lyonya/positive_train",
        help="Папка для сохранения записей"
    )
    parser.add_argument(
        "--count", type=int, default=60,
        help="Сколько примеров записать (default: 60)"
    )
    parser.add_argument(
        "--prefix", default="real",
        help="Префикс имени файла (default: real)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_len = int(2.0 * SAMPLE_RATE)

    # Находим максимальный номер уже существующих файлов с этим префиксом
    existing = [
        int(p.stem.split("_")[-1])
        for p in output_dir.glob(f"{args.prefix}_*.wav")
        if p.stem.split("_")[-1].isdigit()
    ]
    next_index = max(existing) + 1 if existing else 0

    print("=" * 55)
    print('  Запись реальных примеров: "Эй Лёня"')
    print("=" * 55)
    print(f"  Всего нужно записать: {args.count}")
    if next_index > 0:
        print(f"  Уже есть файлов: {next_index} (нумерация с {next_index:03d})")
    print(f"  Папка: {args.output}")
    print()
    print("  Советы для лучшего качества:")
    print("  • Говори естественно, как обращаешься к устройству")
    print("  • Меняй интонацию, темп, громкость от записи к записи")
    print("  • Запишись в разных комнатах / с разного расстояния")
    print("  • Добавь несколько записей с фоновым шумом (ТВ, музыка)")
    print()
    print("  Нажимай Enter перед каждым произношением.")
    print("  Ctrl+C — завершить досрочно.")
    print()

    # PyAudio инициализируется один раз — поток уже открыт к моменту записи
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=1024,
    )

    recorded = 0
    try:
        while recorded < args.count:
            input(f"  [{recorded + 1}/{args.count}] Нажми Enter и скажи «Эй Лёня»...")

            print("  🔴 Запись...", end=" ", flush=True)
            audio = record_clip(stream, CLIP_DURATION)
            audio = trim_silence(audio)
            audio = pad_to_length(audio, target_len)

            # Проверяем, что что-то было сказано
            rms = np.sqrt(np.mean((audio.astype(np.float32) / 32767.0) ** 2))
            if rms < 0.005:
                print("⚠ Слишком тихо! Попробуй ещё раз.")
                continue

            filename = f"{args.prefix}_{next_index + recorded:03d}.wav"
            filepath = output_dir / filename
            wav.write(str(filepath), SAMPLE_RATE, audio)

            print(f"✅ Сохранено: {filename} (громкость: {rms:.3f})")
            recorded += 1

    except KeyboardInterrupt:
        print(f"\n\n  Остановлено. Записано {recorded} примеров.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    print(f"\n✅ Готово! {recorded} реальных записей в: {args.output}")
    print("   Теперь запускай обучение модели.")


if __name__ == "__main__":
    main()
