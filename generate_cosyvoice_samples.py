"""
Генерация TTS примеров через Fun-CosyVoice3 сервер
для датасета wake word "Эй Лёня"
=====================================================
Запуск:
  python generate_cosyvoice_samples.py
  python generate_cosyvoice_samples.py --count 100 --output ./dataset/positive_train
"""

import argparse
import random
import sys
import io
import numpy as np
import requests
import scipy.io.wavfile as wav
import scipy.signal
from pathlib import Path

SERVER_URL = "http://192.168.10.118:8000/tts"
SAMPLE_RATE = 16000
TARGET_LEN = int(2.0 * SAMPLE_RATE)   # 2 секунды — как в record_samples.py
SILENCE_THRESHOLD = 0.005
PREFIX = "cosyvoice"

# Варианты текста — пунктуация влияет на интонацию TTS
PHRASES = [
    "Эй Лёня",
    "Эй, Лёня",
    "Эй Лёня!",
    "Эй, Лёня!",
    "Эй Лёня.",
]

# Инструкции — максимальное разнообразие стилей, темпа, эмоций
INSTRUCTIONS = [
    # Нейтральные / спокойные
    "Speak naturally at a calm, even pace.",
    "Speak in a neutral tone, neither fast nor slow.",
    "Speak matter-of-factly, like a casual mention.",

    # Скорость
    "Speak quickly, as if in a hurry.",
    "Speak very slowly and deliberately.",
    "Speak at a brisk pace with clear articulation.",
    "Speak fast with a slight breathlessness.",

    # Энергетика и громкость
    "Speak with high energy and enthusiasm.",
    "Speak loudly, as if calling across a room.",
    "Speak softly and quietly, almost under your breath.",
    "Speak in a half-whisper, like you don't want to be overheard.",
    "Speak with confident, projecting voice.",

    # Эмоции
    "Speak cheerfully, with a light smile in your voice.",
    "Speak warmly and in a friendly manner.",
    "Speak with mild irritation, like you've said this before.",
    "Speak with genuine excitement.",
    "Speak tiredly, as if just woken up.",
    "Speak with surprise, like something just occurred to you.",
    "Speak with worry and slight urgency.",
    "Speak affectionately, like addressing a close friend.",
    "Speak with boredom, flatly.",
    "Speak with impatience, tapping-foot energy.",
    "Speak with gentle sternness, like a mild command.",
    "Speak playfully, almost teasing.",

    # Контекстуальные / ситуативные
    "Speak as if calling from the next room.",
    "Speak as if you just remembered something and want attention.",
    "Speak formally and politely, as if to a colleague.",
    "Speak in a low, calm voice, as if not to disturb others.",
    "Speak with rising intonation, as if it's a question.",
    "Speak as if starting a sentence, with a slight pause after.",

    # Разговорные особенности
    "Speak with a slight Russian accent, naturally.",
    "Speak clearly with precise articulation of each syllable.",
    "Speak with a relaxed, slightly drawn-out delivery.",
    "Speak with varied pitch, making it sound animated.",
]


def trim_silence(audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> np.ndarray:
    float_audio = audio.astype(np.float32) / 32767.0
    non_silent = np.where(np.abs(float_audio) > threshold)[0]
    if len(non_silent) == 0:
        return audio
    start = max(0, non_silent[0] - int(0.05 * SAMPLE_RATE))
    end = min(len(audio), non_silent[-1] + int(0.1 * SAMPLE_RATE))
    return audio[start:end]


def pad_to_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) >= target_len:
        return audio[:target_len]
    pad = target_len - len(audio)
    return np.pad(audio, (int(pad * 0.3), pad - int(pad * 0.3)), mode="constant")


def resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == SAMPLE_RATE:
        return audio
    target_len = int(len(audio) * SAMPLE_RATE / src_rate)
    return scipy.signal.resample(audio, target_len).astype(np.int16)


def fetch_tts(text: str, instruct: str, timeout: int = 30) -> bytes:
    payload = {"text": text, "instruct": f"{instruct}<|endofprompt|>"}
    resp = requests.post(SERVER_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def process_audio(data: bytes) -> np.ndarray:
    """Читает WAV из bytes, приводит к 16kHz mono int16, 2.0s."""
    rate, audio = wav.read(io.BytesIO(data))

    # stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.int16)

    # float32 → int16
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)

    # resample если нужно
    audio = resample_to_16k(audio, rate)

    audio = trim_silence(audio)
    audio = pad_to_length(audio, TARGET_LEN)
    return audio


def get_next_index(output_dir: Path) -> int:
    existing = [
        int(p.stem.split("_")[-1])
        for p in output_dir.glob(f"{PREFIX}_*.wav")
        if p.stem.split("_")[-1].isdigit()
    ]
    return max(existing) + 1 if existing else 0


def main():
    parser = argparse.ArgumentParser(
        description='Генерация TTS примеров через CosyVoice3 для датасета "Эй Лёня"'
    )
    parser.add_argument("--count", type=int, default=75,
                        help="Сколько файлов сгенерировать (default: 75)")
    parser.add_argument("--output", default="./dataset/positive_train",
                        help="Папка для сохранения (default: ./dataset/positive_train)")
    parser.add_argument("--server", default=SERVER_URL,
                        help=f"URL TTS сервера (default: {SERVER_URL})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed для воспроизводимости")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    next_index = get_next_index(output_dir)

    print("=" * 60)
    print('  Генерация CosyVoice3 примеров: "Эй Лёня"')
    print("=" * 60)
    print(f"  Сервер:        {args.server}")
    print(f"  Нужно файлов:  {args.count}")
    print(f"  Папка:         {args.output}")
    print(f"  Нумерация с:   {next_index:03d}")
    print(f"  Инструкций:    {len(INSTRUCTIONS)}, фраз: {len(PHRASES)}")
    print()

    # Проверяем доступность сервера
    try:
        requests.get(args.server.rsplit("/tts", 1)[0], timeout=5)
    except Exception:
        print(f"❌ Сервер недоступен: {args.server}")
        print("   Проверь что Fun-CosyVoice3 запущен и адрес верный.")
        sys.exit(1)

    # Перемешиваем все комбинации и проходим по ним по кругу
    combos = [(p, i) for p in PHRASES for i in INSTRUCTIONS]
    random.shuffle(combos)
    combo_iter = iter(combos * ((args.count // len(combos)) + 2))

    generated = 0
    errors = 0
    max_errors = 10

    while generated < args.count:
        phrase, instruct = next(combo_iter)
        short_instruct = instruct[:45] + "..." if len(instruct) > 45 else instruct

        print(f"  [{generated + 1:3d}/{args.count}] «{phrase}» | {short_instruct}")

        try:
            raw = fetch_tts(phrase, instruct)
            audio = process_audio(raw)

            rms = np.sqrt(np.mean((audio.astype(np.float32) / 32767.0) ** 2))
            if rms < 0.003:
                print(f"           ⚠ Слишком тихо (rms={rms:.4f}), пропускаю")
                errors += 1
                if errors >= max_errors:
                    print(f"❌ Слишком много ошибок подряд ({max_errors}), останавливаюсь.")
                    break
                continue

            filename = f"{PREFIX}_{next_index + generated:03d}.wav"
            wav.write(str(output_dir / filename), SAMPLE_RATE, audio)
            print(f"           ✅ {filename}  (rms={rms:.3f})")
            generated += 1
            errors = 0

        except requests.HTTPError as e:
            print(f"           ❌ HTTP ошибка: {e}")
            errors += 1
        except requests.Timeout:
            print("           ❌ Таймаут сервера")
            errors += 1
        except Exception as e:
            print(f"           ❌ Ошибка: {e}")
            errors += 1

        if errors >= max_errors:
            print(f"❌ Слишком много ошибок подряд ({max_errors}), останавливаюсь.")
            break

    print()
    print(f"✅ Готово! Сгенерировано {generated} файлов в: {args.output}")
    if generated < args.count:
        print(f"   ⚠ Не хватает {args.count - generated} файлов — проверь сервер.")


if __name__ == "__main__":
    main()
