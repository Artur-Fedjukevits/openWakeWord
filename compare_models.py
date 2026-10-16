"""
Сравнение моделей "Эй Лёня" в условиях робота
=============================================
Поток кадров по 1280 сэмплов, срабатывание = `patience` кадров подряд >= `threshold`,
пауза между срабатываниями 1.5 с — как в wakeword_node.

  Позитивы: записи в комнате и реальные записи из positive_test (не были в обучении).
  Ложные: записи inmoov_sv_debug, отложенные в test при сборке негативов
          (sv_debug_test_files.txt), и для справки — те, что ушли в train.

Запуск:
  python3 compare_models.py ey_lyonya_v2 ey_lyonya --threshold 0.9 --patience 3
"""

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import soundfile as sf
from openwakeword.model import Model

ROOT = Path("/home/artur/openWakeWord/my_custom_model")
SV = Path("/home/artur/inmoov_sv_debug")
FRAME = 1280
DEBOUNCE_FRAMES = int(1.5 * 16000 / FRAME)


def frame_scores(model, audio):
    return np.array([float(list(model.predict(audio[i:i + FRAME]).values())[0])
                     for i in range(0, len(audio) - FRAME + 1, FRAME)])


def count_fires(scores, threshold, patience):
    run, fires, last = 0, 0, -10 ** 9
    for i, v in enumerate(scores):
        run = run + 1 if v >= threshold else 0
        if run >= patience and i - last >= DEBOUNCE_FRAMES:
            fires, last, run = fires + 1, i, 0
    return fires


def load(f):
    a, _ = sf.read(f, dtype="int16")
    return a[:, 0] if a.ndim > 1 else a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--final", default="ey_lyonya_v8", help="версия, чей final_test_files.txt использовать")
    args = p.parse_args()

    # Финальный test v7: записи робота, разделённые по сценам (build_v7_dataset.py). Их не видели
    # ни обучение, ни выбор чекпоинта v7. room/real — это val v7 (по ним выбирается чекпоинт).
    final = (ROOT / args.final / "final_test_files.txt").read_text().split()
    split = dict(l.split("\t")[1:] for l in (ROOT.parent / "dataset/v7/real_split.tsv").read_text().splitlines())
    test_pos = {
        f"робот, test {args.final[-2:]}": [f for f in final if "/positive/" in f],
        "room (val)": sorted(glob.glob(str(ROOT / "ey_lyonya/positive_test/room_*.wav"))),
        "real (val)": sorted(glob.glob(str(ROOT / "ey_lyonya/positive_test/real_*.wav"))),
    }
    wake_false = {
        f"test {args.final[-2:]}": [f for f in final if "/positive/" not in f],
        "val": sorted(str(ROOT.parent / "dataset/wake_debug_used/false" / n) for n, s in split.items() if s == "val"),
        "train": sorted(str(ROOT.parent / "dataset/wake_debug_used/false" / n) for n, s in split.items()
                           if s == "train" and (ROOT.parent / "dataset/wake_debug_used/false" / n).exists()),
    }
    scan = {r["file"]: r for r in csv.DictReader(open(ROOT.parent / "dataset/sv_debug_scan.csv"))}
    test_sv = set((ROOT / "ey_lyonya/sv_debug_test_files.txt").read_text().split())
    train_sv = [f for f, r in scan.items() if f not in test_sv and r["wake_in_text"] == "0"]
    neg_sets = {"sv_debug test": sorted(test_sv), "sv_debug train": sorted(train_sv)}

    silence = np.zeros(16000, np.int16)
    for name in args.models:
        model = Model(wakeword_models=[str(ROOT / f"{name}.onnx")], inference_framework="onnx")
        print(f"\n=== {name}  (threshold {args.threshold}, patience {args.patience})")
        for label, files in test_pos.items():
            hit = 0
            for f in files:
                model.reset()
                sc = frame_scores(model, np.concatenate([silence, load(f), silence[:8000]]))
                hit += count_fires(sc, args.threshold, args.patience) > 0
            print(f"  узнал {label:22s} {hit:3d}/{len(files)}")
        for label, files in neg_sets.items():
            fires, secs = 0, 0.0
            for f in files:
                a = load(SV / f)[int(1.5 * 16000):]   # без pre-roll, как при сборке негативов
                model.reset()
                fires += count_fires(frame_scores(model, np.concatenate([silence, a])), args.threshold, args.patience)
                secs += len(a) / 16000
            print(f"  ложных на {label:15s} {fires:3d} за {secs / 60:5.1f} мин  ({fires / (secs / 3600):.1f} в час)")
        for label, files in wake_false.items():
            fired = 0
            for f in files:
                model.reset()
                fired += count_fires(frame_scores(model, np.concatenate([silence, load(f)])),
                                     args.threshold, args.patience) > 0
            print(f"  ложные срабатывания робота ({label:8s}): повторилось {fired}/{len(files)}")


if __name__ == "__main__":
    main()
