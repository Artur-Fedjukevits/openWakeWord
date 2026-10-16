"""
Негативы и фон из длинных записей комнаты (room_household, record_room_background.py)
=====================================================================================
Для каждого 10-секундного клипа:
  - клипы с речью проверяются Parakeet; если в тексте есть «Лёня» или похожее — клип
    выбрасывается целиком (дети могли звать робота);
  - клип режется на куски по 2 с: все куски со звуком (> SOUND_DB) и часть тихих;
  - текущая модель ищет пики: окно 2 с, кончающееся на пике (4 сдвига), — трудный негатив;
  - клипы из rejected_wakeword/ (на них модель насторожилась во время записи) — тоже
    трудные негативы, если в них нет «Лёня».
train/test делятся 10-минутными блоками записи (соседние куски одной сцены не расходятся).
Чистые 10-секундные клипы копируются в каталог фона (для background_paths).

Запуск:
  python3 process_room_recordings.py room_household dataset/v8/room_neg room_household_bg
"""

import random
import re
import sys
from pathlib import Path

import numpy as np
import onnx_asr
import soundfile as sf
from openwakeword.model import Model

SR, CLIP, FRAME = 16000, 32000, 1280
SOUND_DB = -75
SPEECH_DB = -65
N_SILENT = 600
HARD_SCORE = 0.5
TEST_FRACTION = 0.15
MODEL = "/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx"
WAKE = re.compile(r"л[её]н|льон|лион|леон|l[eiy]?on|lion|lyon|lön|лон[яьи]|йон[яь]|люн", re.I)


def db(x):
    return 20 * np.log10(np.sqrt(np.mean((x.astype(np.float32) / 32768) ** 2)) + 1e-9)


def main():
    src, out, bg = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    random.seed(8)
    for s in ("train", "test"):
        (out / s).mkdir(parents=True, exist_ok=True)
    bg.mkdir(parents=True, exist_ok=True)
    asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
    oww = Model(wakeword_models=[MODEL], inference_framework="onnx")

    clips = sorted(src.glob("*.wav"))
    # блок = метка записи + номер клипа // 60 (10 минут)
    def block(p):
        label, idx = p.stem.rsplit("_", 1)
        return f"{label}_{int(idx) // 60}"
    blocks = sorted({block(p) for p in clips})
    test_blocks = set(random.sample(blocks, max(1, round(len(blocks) * TEST_FRACTION))))

    stats = dict(clips=len(clips), dropped_wake=0, sound=0, silent=0, hard=0, flagged=0)
    silent_pool = []
    for k, p in enumerate(clips):
        a, _ = sf.read(p, dtype="int16")
        split = "test" if block(p) in test_blocks else "train"
        if db(a) > SPEECH_DB:
            text = asr.recognize(str(p)).strip()
            if WAKE.search(text):
                stats["dropped_wake"] += 1
                print(f"  drop {p.name}: «{text}»")
                continue
        sf.write(bg / p.name, a, SR, subtype="PCM_16")
        for j in range(0, len(a) - CLIP + 1, CLIP):
            c = a[j:j + CLIP]
            if db(c) > SOUND_DB:
                sf.write(out / split / f"room_{p.stem}_{j // CLIP}.wav", c, SR, subtype="PCM_16")
                stats["sound"] += 1
            else:
                silent_pool.append((split, f"room_{p.stem}_{j // CLIP}.wav", c))
        # пики модели внутри клипа (с 1 с разгона на предыдущем звуке не рассчитываем — тишина)
        oww.reset()
        x = np.concatenate([np.zeros(SR, np.int16), a])
        sc = np.array([float(list(oww.predict(x[i:i + FRAME]).values())[0])
                       for i in range(0, len(x) - FRAME + 1, FRAME)])
        if sc.max() >= HARD_SCORE:
            end = (int(sc.argmax()) + 1) * FRAME - SR
            for s_i, shift in enumerate((0, 0.16, 0.32, 0.48)):
                e = end - int(shift * SR)
                if e - CLIP >= -CLIP // 2 and e <= len(a):
                    c = a[max(0, e - CLIP):e]
                    c = np.pad(c, (CLIP - len(c), 0))
                    sf.write(out / split / f"roomhard_{p.stem}_{s_i}.wav", c, SR, subtype="PCM_16")
                    stats["hard"] += 1
        if k % 300 == 0:
            print(f"{k}/{len(clips)} {stats}", flush=True)

    for split, name, c in random.sample(silent_pool, min(N_SILENT, len(silent_pool))):
        sf.write(out / split / name, c, SR, subtype="PCM_16")
        stats["silent"] += 1

    # клипы, отложенные при записи (модель насторожилась)
    for p in sorted((src / "rejected_wakeword").glob("*.wav")):
        text = asr.recognize(str(p)).strip()
        if WAKE.search(text):
            print(f"  flagged drop {p.name}: «{text}»")
            continue
        a, _ = sf.read(p, dtype="int16")
        oww.reset()
        x = np.concatenate([np.zeros(SR, np.int16), a])
        sc = np.array([float(list(oww.predict(x[i:i + FRAME]).values())[0])
                       for i in range(0, len(x) - FRAME + 1, FRAME)])
        end = (int(sc.argmax()) + 1) * FRAME - SR
        split = "test" if random.random() < TEST_FRACTION else "train"
        for s_i, shift in enumerate((0, 0.16, 0.32, 0.48)):
            e = min(len(a), max(CLIP, end - int(shift * SR)))
            sf.write(out / split / f"roomflag_{p.stem}_{s_i}.wav", a[e - CLIP:e], SR, subtype="PCM_16")
        stats["flagged"] += 1
    print("done", stats)


if __name__ == "__main__":
    main()
