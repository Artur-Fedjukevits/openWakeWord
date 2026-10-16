"""
Тест модели "Эй Лёня" с микрофоном
====================================
Слушает микрофон в реальном времени и выводит
score когда слышит wake word.

Требования:
  pip install pyaudio openwakeword numpy

Запуск:
  python3 test_wakeword.py
  python3 test_wakeword.py --threshold 0.5  # строже
  python3 test_wakeword.py --threshold 0.3  # мягче (больше срабатываний)
"""

import argparse
import numpy as np
import pyaudio
import openwakeword
from openwakeword.model import Model
import time

SAMPLE_RATE = 16000
CHUNK = 1280  # 80ms фреймы — стандарт openWakeWord


def main():
    parser = argparse.ArgumentParser(description='Тест wake word "Эй Лёня"')
    parser.add_argument(
        "--model", 
        default="/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx",
        help="Путь к .onnx модели"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Порог срабатывания 0.0-1.0 (default: 0.4)"
    )
    parser.add_argument(
        "--show-score", action="store_true",
        help="Показывать score в реальном времени"
    )
    args = parser.parse_args()

    print("=" * 50)
    print('  Тест wake word: "Эй Лёня"')
    print(f"  Модель: {args.model}")
    print(f"  Порог: {args.threshold}")
    print("=" * 50)

    # Загружаем модель
    print("\n📥 Загрузка модели...")
    oww = Model(
        wakeword_models=[args.model],
        inference_framework="onnx",
    )
    print("✅ Модель загружена\n")
    print("🎙️  Слушаю... Скажи «Эй Лёня»")
    print("    Ctrl+C — выход\n")

    # Инициализируем микрофон
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    activations = 0
    last_activation = 0

    try:
        while True:
            # Читаем аудио
            audio = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(audio, dtype=np.int16)

            # Прогоняем через модель
            prediction = oww.predict(audio_np)

            # Получаем score
            model_key = list(prediction.keys())[0]
            score = prediction[model_key]

            # Показываем score если нужно
            if args.show_score and score > 0.1:
                bar = "█" * int(score * 20)
                print(f"\r  Score: {score:.3f} |{bar:<20}|", end="", flush=True)

            # Проверяем порог с debounce 1 сек
            now = time.time()
            if score >= args.threshold and (now - last_activation) > 1.0:
                activations += 1
                last_activation = now
                print(f"\n🔔 АКТИВАЦИЯ #{activations}! Score: {score:.3f}  [{time.strftime('%H:%M:%S')}]")

    except KeyboardInterrupt:
        print(f"\n\n  Остановлено. Всего активаций: {activations}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
