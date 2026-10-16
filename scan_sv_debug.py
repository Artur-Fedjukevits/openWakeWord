"""
Разбор записей inmoov_sv_debug для поиска негативных примеров
=============================================================
Для каждой записи: транскрипт (Parakeet), пиковый score модели в потоковом
режиме (кадры по 1280 сэмплов, как в wakeword_node) и время пика.
Результат — CSV, по которому потом отбираются негативы.

Запуск:
  python3 scan_sv_debug.py --out dataset/sv_debug_scan.csv
"""

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np
import onnx_asr
import soundfile as sf
from openwakeword.model import Model

FRAME = 1280
# Варианты, которыми Parakeet записывает «Лёня» (кириллица и латиница)
WAKE_RE = re.compile(r"л[её]н|льон|л[её]нь|лион|леон|l[eiy]?on|lion|lyon|lön|лон[яьи]|йон[яь]|люн", re.I)


def stream_scores(model: Model, audio: np.ndarray) -> np.ndarray:
    model.reset()
    audio = np.concatenate([np.zeros(16000, np.int16), audio, np.zeros(8000, np.int16)])
    return np.array([float(list(model.predict(audio[i:i + FRAME]).values())[0])
                     for i in range(0, len(audio) - FRAME + 1, FRAME)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/home/artur/inmoov_sv_debug")
    p.add_argument("--model", default="/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx")
    p.add_argument("--out", default="/home/artur/openWakeWord/dataset/sv_debug_scan.csv")
    p.add_argument("--after", default="", help="только файлы, имя которых > этой строки, напр. 20261003_142527")
    args = p.parse_args()

    asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
    oww = Model(wakeword_models=[args.model], inference_framework="onnx")

    files = sorted(f for f in glob.glob(f"{args.src}/*.wav") if Path(f).name > args.after)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "label", "duration", "peak", "peak_sec", "frames_ge_0.9", "wake_in_text", "text"])
        for k, f in enumerate(files):
            try:
                audio, sr = sf.read(f, dtype="int16")
            except Exception:
                continue          # битый/пустой файл
            if audio.ndim > 1:
                audio = audio[:, 0]
            label = re.sub(r"^\d{8}_\d{6}_|(_-?[\d.]+)?\.wav$", "", Path(f).name)
            sc = stream_scores(oww, audio)
            peak_i = int(sc.argmax())
            # кадр i покрывает звук до (i+1)*80 мс; вычитаем 1 с тишины в начале
            peak_sec = (peak_i + 1) * FRAME / 16000 - 1.0
            try:
                text = asr.recognize(f).strip()
            except Exception as e:
                text = f"<asr error {e}>"
            w.writerow([Path(f).name, label, round(len(audio) / sr, 2), round(float(sc.max()), 3),
                        round(peak_sec, 2), int((sc >= 0.9).sum()), int(bool(WAKE_RE.search(text))), text])
            if k % 100 == 0:
                print(f"{k}/{len(files)}", flush=True)
    print("done", args.out)


if __name__ == "__main__":
    main()
