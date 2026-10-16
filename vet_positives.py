"""
Отсев брака в синтетических позитивах «Эй Лёня» (Parakeet)
==========================================================
Клип отбрасывается, если в транскрипте нет ничего похожего на «Лёня»
(в любом написании Parakeet — кириллица или латиница) или слов больше трёх.
Parakeet и сам часто ошибается на этой фразе, поэтому проверка мягкая:
она ловит пустые, обрезанные и «не те» клипы, а не неточное произношение.

Отброшенные файлы переносятся в <dir>/rejected/, список — в rejected.tsv.

Запуск:
  python3 vet_positives.py dataset/v7/pos_silero
"""

import re
import sys
from pathlib import Path

import onnx_asr

WAKE = re.compile(r"(л|l|ł|й|j|y)[^\s,.!?]{0,2}(о|o|ё|е|e|ю|u|ö|ó|i|и)[^\s,.!?]{0,3}(н|n|ñ|ň)", re.I)


def main():
    src = Path(sys.argv[1])
    rej = src / "rejected"
    rej.mkdir(exist_ok=True)
    asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
    files = sorted(src.glob("*.wav"))
    n_bad = 0
    with open(src / "rejected.tsv", "a") as log:
        for k, f in enumerate(files):
            text = asr.recognize(str(f)).strip()
            if not WAKE.search(text) or len(text.split()) > 3:
                f.rename(rej / f.name)
                log.write(f"{f.name}\t{text}\n")
                n_bad += 1
            if k % 1000 == 0:
                print(f"{k}/{len(files)} rejected {n_bad}", flush=True)
    print(f"done: {len(files)} checked, {n_bad} rejected")


if __name__ == "__main__":
    main()
