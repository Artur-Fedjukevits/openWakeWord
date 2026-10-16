"""
Негативные примеры из реальной речи в комнате (inmoov_sv_debug)
================================================================
Берёт результат scan_sv_debug.py и режет записи на клипы по 2 с:
  svneg_*  — обычная речь (ТВ, радио, разговоры), без перекрытия
  svhard_* — окна вокруг пика модели там, где она ложно сработала (score >= HARD_SCORE),
             с несколькими сдвигами — самые трудные негативы

Защита от попадания настоящего «Эй Лёня» в негативы:
  - записи, в транскрипте которых есть «Лёня» и похожее, пропускаются целиком;
  - первые PRE_ROLL_SEC каждой записи отбрасываются (voice_detector кладёт туда
    звук до активации — там может быть сама фраза);
  - пик модели внутри этой зоны не считается ложным срабатыванием.

Запись целиком уходит либо в train, либо в test (TEST_FRACTION), чтобы test был честным.

Запуск:
  python3 build_room_negatives.py                                    # v3–v7: в my_custom_model/ey_lyonya
  python3 build_room_negatives.py dataset/sv_debug_scan_2.csv dataset/v8/sv_neg   # v8
"""

import csv
import sys
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path("/home/artur/inmoov_sv_debug")
SCAN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/artur/openWakeWord/dataset/sv_debug_scan.csv")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None   # v8: dataset/v8/sv_neg/{train,test}
MODEL_DIR = Path("/home/artur/openWakeWord/my_custom_model/ey_lyonya")
SR = 16000
CLIP = 2 * SR
PRE_ROLL_SEC = 1.5
HARD_SCORE = 0.5
HARD_SHIFTS_SEC = (-0.32, -0.16, 0.0, 0.16)
MIN_RMS = 100          # int16; тише — почти тишина, её и так хватает
TEST_FRACTION = 0.15


def main():
    random.seed(0)
    rows = list(csv.DictReader(open(SCAN)))
    if OUT:
        train_dir, test_dir = OUT / "train", OUT / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
    else:
        train_dir, test_dir = MODEL_DIR / "negative_train", MODEL_DIR / "negative_test"
    for d in (train_dir, test_dir):
        for old in list(d.glob("svneg_*.wav")) + list(d.glob("svhard_*.wav")):
            old.unlink()

    stats = {"skipped_wake_text": 0, "train_neg": 0, "test_neg": 0, "train_hard": 0, "test_hard": 0,
             "hard_files": 0}
    test_files = []
    for r in rows:
        if r["wake_in_text"] == "1":
            stats["skipped_wake_text"] += 1
            continue
        audio, sr = sf.read(SRC / r["file"], dtype="int16")
        assert sr == SR
        if audio.ndim > 1:
            audio = audio[:, 0]
        split = "test" if random.random() < TEST_FRACTION else "train"
        out = test_dir if split == "test" else train_dir
        if split == "test":
            test_files.append(r["file"])
        stem = Path(r["file"]).stem

        # Обычная речь: клипы по 2 с после зоны pre-roll
        body = audio[int(PRE_ROLL_SEC * SR):]
        for j in range(0, len(body) - CLIP + 1, CLIP):
            clip = body[j:j + CLIP]
            if np.sqrt(np.mean(clip.astype(np.float32) ** 2)) < MIN_RMS:
                continue
            sf.write(out / f"svneg_{stem}_{j // CLIP}.wav", clip, SR, subtype="PCM_16")
            stats[f"{split}_neg"] += 1

        # Ложное срабатывание: окно 2 с, заканчивающееся на кадре с пиком
        peak, peak_sec = float(r["peak"]), float(r["peak_sec"])
        if peak >= HARD_SCORE and peak_sec >= PRE_ROLL_SEC + 0.5:
            stats["hard_files"] += 1
            for k, shift in enumerate(HARD_SHIFTS_SEC):
                end = int((peak_sec + shift) * SR)
                start = end - CLIP
                if start < int(PRE_ROLL_SEC * SR) - CLIP // 2 or end > len(audio):
                    continue
                clip = audio[max(0, start):end]
                clip = np.pad(clip, (CLIP - len(clip), 0))
                sf.write(out / f"svhard_{stem}_{k}.wav", clip, SR, subtype="PCM_16")
                stats[f"{split}_hard"] += 1

    ((OUT or MODEL_DIR) / "sv_debug_test_files.txt").write_text("\n".join(test_files))
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
