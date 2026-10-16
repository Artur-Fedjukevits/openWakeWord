"""
Генерация примеров "Эй Лёня" клонированным голосом через OmniVoice TTS
======================================================================
Сервер: POST {server}/tts/stream  {"text", "emotion", "speed"}
Ответ:  raw PCM int16 mono, частота в заголовке X-Sample-Rate (24000).

Вывод сразу кладётся туда, откуда train.py реально берёт клипы:
  my_custom_model/ey_lyonya/positive_train  (и часть в positive_test)

Запуск:
  python generate_omnivoice_samples.py                 # 300 шт, 15% в test
  python generate_omnivoice_samples.py --count 500 --test-fraction 0.1
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import requests
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
from math import gcd

SERVER = "http://192.168.10.118:8000"
MODEL_DIR = Path("/home/artur/openWakeWord/my_custom_model/ey_lyonya")
SAMPLE_RATE = 16000
TARGET_LEN = int(2.0 * SAMPLE_RATE)
SILENCE_THRESHOLD = 0.005
PREFIX = "omnivoice"

# Пунктуация и написание меняют интонацию
PHRASES = [
    "Эй Лёня",
    "Эй, Лёня",
    "Эй, Лёня!",
    "Эй Лёня!",
    "Эй, Лёня?",
    "Эй, Лёня.",
    "Эй... Лёня!",
    "Эээй, Лёня!",
    "Эй, Лёнь!",
]

# Пресеты сервера; "" = neutral по умолчанию
EMOTIONS = ["", "neutral", "happy", "sad", "surprise"]

SPEED_RANGE = (0.8, 1.25)

# Допустимая длина речи после обрезки тишины — отсекает обрывки и "галлюцинации"
MIN_SPEECH_SEC = 0.45
MAX_SPEECH_SEC = 1.9


def fetch_pcm(server: str, text: str, emotion: str, speed: float, timeout: int = 60):
    payload = {"text": text, "speed": speed}
    if emotion:
        payload["emotion"] = emotion
    resp = requests.post(f"{server}/tts/stream", json=payload, timeout=timeout)
    resp.raise_for_status()
    rate = int(resp.headers.get("x-sample-rate", 24000))
    audio = np.frombuffer(resp.content, dtype=np.int16)
    return audio, rate


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return audio
    g = gcd(SAMPLE_RATE, rate)
    out = resample_poly(audio.astype(np.float32), SAMPLE_RATE // g, rate // g)
    return np.clip(out, -32768, 32767).astype(np.int16)


def trim_silence(audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> np.ndarray:
    float_audio = audio.astype(np.float32) / 32767.0
    non_silent = np.where(np.abs(float_audio) > threshold)[0]
    if len(non_silent) == 0:
        return audio[:0]
    start = max(0, non_silent[0] - int(0.05 * SAMPLE_RATE))
    end = min(len(audio), non_silent[-1] + int(0.1 * SAMPLE_RATE))
    return audio[start:end]


def pad_to_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) >= target_len:
        return audio[:target_len]
    pad = target_len - len(audio)
    return np.pad(audio, (int(pad * 0.3), pad - int(pad * 0.3)), mode="constant")


def next_index(dirs) -> int:
    idx = [
        int(p.stem.split("_")[-1])
        for d in dirs for p in d.glob(f"{PREFIX}_*.wav")
        if p.stem.split("_")[-1].isdigit()
    ]
    return max(idx) + 1 if idx else 0


def main():
    parser = argparse.ArgumentParser(description='OmniVoice примеры "Эй Лёня"')
    parser.add_argument("--count", type=int, default=300, help="Сколько файлов всего (default: 300)")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                        help="Доля файлов в positive_test (default: 0.15)")
    parser.add_argument("--model-dir", default=str(MODEL_DIR),
                        help=f"Папка модели с positive_train/positive_test (default: {MODEL_DIR})")
    parser.add_argument("--server", default=SERVER, help=f"URL TTS сервера (default: {SERVER})")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    train_dir = Path(args.model_dir) / "positive_train"
    test_dir = Path(args.model_dir) / "positive_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    index = next_index([train_dir, test_dir])

    print("=" * 60)
    print('  OmniVoice: "Эй Лёня"')
    print(f"  Сервер: {args.server}   файлов: {args.count}   test: {args.test_fraction:.0%}")
    print(f"  train → {train_dir}")
    print(f"  test  → {test_dir}")
    print("=" * 60)

    try:
        requests.get(f"{args.server}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"❌ Сервер недоступен: {e}")
        sys.exit(1)

    generated = rejected = errors = 0
    counts = {"train": 0, "test": 0}
    while generated < args.count:
        phrase = random.choice(PHRASES)
        emotion = random.choice(EMOTIONS)
        speed = round(random.uniform(*SPEED_RANGE), 2)
        tag = f"«{phrase}» emo={emotion or '-'} speed={speed}"

        try:
            raw, rate = fetch_pcm(args.server, phrase, emotion, speed)
        except Exception as e:
            errors += 1
            print(f"  ❌ {tag}: {e}")
            if errors >= 10:
                print("❌ 10 ошибок подряд, останавливаюсь.")
                break
            continue
        errors = 0

        speech = trim_silence(to_16k(raw, rate))
        dur = len(speech) / SAMPLE_RATE
        if not (MIN_SPEECH_SEC <= dur <= MAX_SPEECH_SEC):
            rejected += 1
            print(f"  ⚠ {tag}: длина речи {dur:.2f}s — пропуск")
            continue

        audio = pad_to_length(speech, TARGET_LEN)
        split = "test" if random.random() < args.test_fraction else "train"
        out_dir = test_dir if split == "test" else train_dir
        filename = f"{PREFIX}_{index + generated:04d}.wav"
        wav.write(str(out_dir / filename), SAMPLE_RATE, audio)
        generated += 1
        counts[split] += 1
        print(f"  [{generated:4d}/{args.count}] {split:5s} {filename}  {tag}  {dur:.2f}s")

    print()
    print(f"✅ Готово: train +{counts['train']}, test +{counts['test']}, отброшено {rejected}")
    print("   Перед обучением удали старые positive_features_*.npy (или запускай с --overwrite).")


if __name__ == "__main__":
    main()
