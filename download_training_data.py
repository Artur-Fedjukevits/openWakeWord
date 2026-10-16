"""
Скачивание датасетов для обучения openWakeWord
===============================================
Скачивает все необходимые файлы с HuggingFace и других источников.

Что скачивается:
  1. openwakeword_features_ACAV100M_2000_hrs_16bit.npy  (~3.5 ГБ) — главный
  2. validation_set_features.npy                         (~200 МБ)
  3. MIT Room Impulse Responses (mit_rirs/)              (~50 МБ)
  4. FMA small (fma/)                                    (~7.2 ГБ, опционально)
  5. AudioSet samples (audioset_16k/)                    (~1-2 ГБ, опционально)

Итого минимум: ~4 ГБ (без FMA и AudioSet)
Итого полный:  ~12 ГБ

Требования:
  pip install datasets huggingface_hub tqdm requests

Запуск:
  python download_training_data.py             # только обязательные файлы
  python download_training_data.py --full      # + FMA и AudioSet
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("pip install tqdm")
    sys.exit(1)


# ─────────────────────────────────────────────
# Прогресс-бар для скачивания
# ─────────────────────────────────────────────
class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: str, desc: str = ""):
    """Скачивает файл с прогресс-баром."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"   ✅ Уже есть: {dest.name} — пропускаем")
        return

    print(f"   📥 {desc or dest.name}")
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=f"   {dest.name}") as t:
        urllib.request.urlretrieve(url, str(dest), reporthook=t.update_to)
    print(f"   ✅ Сохранено: {dest}")


# ─────────────────────────────────────────────
# 1. Обязательные файлы с HuggingFace
# ─────────────────────────────────────────────
def download_features():
    """Скачивает предвычисленные фичи ACAV100M (~3.5 ГБ)."""
    print("\n📦 [1/4] Предвычисленные признаки ACAV100M (~3.5 ГБ)")
    print("   Это самый важный файл — замена 2000 часов аудио для обучения.")

    base = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main"

    download_file(
        f"{base}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "ACAV100M features (3.5 ГБ) — может занять несколько минут..."
    )
    download_file(
        f"{base}/validation_set_features.npy",
        "validation_set_features.npy",
        "Validation set features (~200 МБ)"
    )


# ─────────────────────────────────────────────
# 2. MIT Room Impulse Responses
# ─────────────────────────────────────────────
def download_rirs():
    """Скачивает RIR датасет для симуляции эха (~50 МБ)."""
    print("\n📦 [2/4] MIT Room Impulse Responses (~50 МБ)")
    print("   Нужны для аугментации (симуляция эха в разных комнатах).")

    try:
        import datasets
        print("   Загрузка через HuggingFace datasets...")
        rir_dir = Path("./mit_rirs")
        rir_dir.mkdir(exist_ok=True)

        if len(list(rir_dir.glob("*.wav"))) > 10:
            print("   ✅ Уже скачано — пропускаем")
            return

        import scipy.io.wavfile as wav
        import numpy as np

        ds = datasets.load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train",
            streaming=True
        )
        count = 0
        for row in ds:
            name = f"rir_{count:04d}.wav"
            audio = (np.array(row["audio"]["array"]) * 32767).astype("int16")
            wav.write(str(rir_dir / name), 16000, audio)
            count += 1
            if count % 50 == 0:
                print(f"   ✔ {count} файлов...")

        print(f"   ✅ Скачано {count} RIR файлов → {rir_dir}/")

    except ImportError:
        print("   ⚠ Установи: pip install datasets")
        print("   Или скачай вручную: huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses")


# ─────────────────────────────────────────────
# 3. AudioSet (фоновые звуки окружения)
# ─────────────────────────────────────────────
def download_audioset():
    """Скачивает часть AudioSet через HuggingFace (~1-2 ГБ)."""
    print("\n📦 [3/4] AudioSet — фоновые звуки (~1-2 ГБ)")
    print("   Нужны как негативные примеры (звуки без 'Эй Лёня').")

    try:
        import datasets
        import scipy.io.wavfile as wav
        import numpy as np

        out_dir = Path("./audioset_16k")
        out_dir.mkdir(exist_ok=True)

        if len(list(out_dir.glob("*.wav"))) > 100:
            print("   ✅ Уже скачано — пропускаем")
            return

        print("   Загрузка AudioSet через HuggingFace (streaming)...")
        ds = datasets.load_dataset(
            "agkphysics/AudioSet",
            "unbalanced",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        count = 0
        target = 2000  # файлов достаточно для начала
        for row in ds:
            if count >= target:
                break
            try:
                name = f"audioset_{count:05d}.wav"
                audio = row["audio"]["array"]
                sr = row["audio"]["sampling_rate"]

                # Resample до 16000 если нужно
                if sr != 16000:
                    import torchaudio
                    import torch
                    t = torch.tensor(audio).unsqueeze(0)
                    t = torchaudio.functional.resample(t, sr, 16000)
                    audio = t.squeeze(0).numpy()

                audio_int16 = (np.clip(audio, -1, 1) * 32767).astype("int16")
                wav.write(str(out_dir / name), 16000, audio_int16)
                count += 1

                if count % 200 == 0:
                    print(f"   ✔ {count}/{target} файлов...")
            except Exception:
                continue

        print(f"   ✅ Скачано {count} файлов AudioSet → {out_dir}/")

    except ImportError:
        print("   ⚠ Установи: pip install datasets")


# ─────────────────────────────────────────────
# 4. FMA (музыка как фон)
# ─────────────────────────────────────────────
def download_fma():
    """Скачивает FMA small (~7.2 ГБ). Только с --full флагом."""
    print("\n📦 [4/4] FMA Small — музыка как фон (~7.2 ГБ)")
    print("   Предупреждение: это большой файл!")

    fma_dir = Path("./fma")
    fma_dir.mkdir(exist_ok=True)

    if len(list(fma_dir.glob("**/*.mp3"))) > 100:
        print("   ✅ Уже скачано — пропускаем")
        return

    print("   Скачиваем FMA small с официального сервера...")
    download_file(
        "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "./fma_small.zip",
        "FMA small (7.2 ГБ)"
    )

    print("   Распаковываем...")
    import zipfile
    with zipfile.ZipFile("./fma_small.zip", "r") as z:
        z.extractall("./fma_raw")

    # Конвертируем MP3 -> WAV 16kHz
    print("   Конвертируем MP3 → WAV 16kHz...")
    try:
        import torchaudio
        import torch
        import numpy as np
        import scipy.io.wavfile as wav

        mp3_files = list(Path("./fma_raw").glob("**/*.mp3"))
        for i, mp3 in enumerate(mp3_files):
            try:
                audio, sr = torchaudio.load(str(mp3))
                if sr != 16000:
                    audio = torchaudio.functional.resample(audio, sr, 16000)
                audio_mono = audio.mean(0).numpy()
                out_path = fma_dir / f"fma_{i:05d}.wav"
                wav.write(str(out_path), 16000, (audio_mono * 32767).astype("int16"))
            except Exception:
                continue
            if i % 500 == 0:
                print(f"   ✔ {i}/{len(mp3_files)}")

        print(f"   ✅ Конвертировано {len(mp3_files)} треков → {fma_dir}/")
    except ImportError:
        print("   ⚠ Установи torchaudio для конвертации MP3")


# ─────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Скачивание датасетов для обучения openWakeWord"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Скачать также FMA (~7 ГБ музыки) — улучшает качество"
    )
    parser.add_argument(
        "--skip-audioset", action="store_true",
        help="Пропустить AudioSet (если уже есть или нет места)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Скачивание данных для обучения openWakeWord")
    print("=" * 60)
    print("  Обязательно (~4 ГБ): ACAV100M features + validation + RIR")
    if args.full:
        print("  Дополнительно (~9 ГБ): AudioSet + FMA")
    print()

    # Всегда скачиваем
    download_features()
    download_rirs()

    # Фоновые звуки
    if not args.skip_audioset:
        download_audioset()

    # FMA только с --full
    if args.full:
        download_fma()

    print("\n" + "=" * 60)
    print("  ✅ Готово! Теперь запускай обучение:")
    print()
    print("  # 1. Клонируй openWakeWord (если ещё не):")
    print("  git clone https://github.com/dscripka/openWakeWord")
    print("  cd openWakeWord")
    print("  pip install -e '.[dev]'")
    print()
    print("  # 2. Скопируй конфиг в папку проекта:")
    print("  cp ../ey_lyonya.yml .")
    print()
    print("  # 3. Запусти обучение:")
    print("  python openwakeword/train.py \\")
    print("      --training_config ey_lyonya.yml \\")
    print("      --train_model")
    print()
    print("  Модель появится в: ./my_custom_model/ey_lyonya.onnx")
    print("=" * 60)


if __name__ == "__main__":
    main()
