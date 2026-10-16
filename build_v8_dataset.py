"""
Раскладка данных для v8 (my_custom_model/ey_lyonya_v8/) — как build_v7_dataset.py, плюс:
  - голос дочери (OmniVoice "nicole"): pos_omni_nicole, neg_omni_nicole;
  - room_neg_train/test — отдельный источник негативов из комнаты: 6 ч записи (room_household)
    и новые sv_debug (dataset/v8/room_neg, dataset/v8/sv_neg — уже разделены на train/test);
  - разделение записей робота общее с v7 (dataset/v7/real_split.tsv) — финальный test сопоставим.

Ниже — описание v7.
Раскладка данных для v7 (my_custom_model/ey_lyonya_v7/)
========================================================
1. Новые записи робота (~/inmoov_wake_debug) переносятся в архив dataset/wake_debug_used/.
2. Записи робота делятся по СЦЕНАМ (соседние ближе 5 минут — одна сцена), а не по файлам:
   несколько «Эй Лёня» подряд или один эпизод ТВ не попадают одновременно в train и test.
     - что уже было в test (wake_debug_test_files.txt v4–v5) — остаётся в test вместе со своей сценой;
     - новые ложные: половина сцен в train, половина отложена; отложенные — пополам в
       val (выбор чекпоинта при обучении) и test (финальная проверка, обучение их не видит);
     - новые позитивы: половина сцен в train, половина в test.
3. Каталоги модели собираются жёсткими ссылками (место на диске не тратится):
     positive_train/test        — синтетика (Silero, OmniVoice, CosyVoice)
     real_positive_train/test   — твой голос: комната, старые записи, записи робота (test = val)
     negative_train/test        — фонетические негативы, речь и фон комнаты, тишина
     real_fp_train/test         — ложные срабатывания робота (test = val), 4 окна по 2 с со сдвигом
   Финальный test-список: ey_lyonya_v7/final_test_files.txt (для compare_models.py).

Скрипт идемпотентный: каталоги модели пересобираются с нуля, архив не трогается.
Запуск:
  python3 build_v7_dataset.py                # только пересборка
  python3 build_v7_dataset.py --take-inbox   # сначала забрать РАЗОБРАННЫЕ записи из ~/inmoov_wake_debug
                                             # (всё в корне считается ложным срабатыванием!)
"""

import datetime
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/home/artur/openWakeWord")
INBOX = Path("/home/artur/inmoov_wake_debug")
ARCHIVE = ROOT / "dataset/wake_debug_used"
OLD = ROOT / "my_custom_model/ey_lyonya"          # данные v4–v5
NEW = ROOT / "my_custom_model/ey_lyonya_v8"
V8 = ROOT / "dataset/v8"
V7 = ROOT / "dataset/v7"
SPLIT_FILE = V7 / "real_split.tsv"
SR, CLIP = 16000, 32000
SCENE_GAP_SEC = 300
SYNTH_TEST_FRACTION = 0.05


def ts(name):
    return datetime.datetime.strptime(name[:15], "%Y%m%d_%H%M%S")


def scenes(files):
    """Группы файлов, соседние по времени ближе SCENE_GAP_SEC."""
    files = sorted(files, key=lambda f: f.name)
    groups, cur = [], []
    for f in files:
        if cur and (ts(f.name) - ts(cur[-1].name)).total_seconds() > SCENE_GAP_SEC:
            groups.append(cur)
            cur = []
        cur.append(f)
    if cur:
        groups.append(cur)
    return groups


def phrase_end_trim(audio, keep_after=0.1):
    """Обрезает всё после конца фразы, оставляя keep_after секунд.
    openWakeWord ставит конец клипа в конец окна модели (запас 0–200 мс), поэтому тишина
    после фразы учит модель срабатывать с задержкой — и на «звук, а потом тишина».
    Конец фразы — последний 20-мс отрезок с огибающей >= 10% пика."""
    x = np.abs(audio.astype(np.float32)) / 32768
    w, hop = 320, 160
    env = np.array([x[i:i + w].max() for i in range(0, max(1, len(x) - w + 1), hop)])
    thr = max(env.max() * 0.1, 0.003)
    idx = np.where(env >= thr)[0]
    if len(idx) == 0:
        return audio
    end = min(len(audio), idx[-1] * hop + w + int(keep_after * SR))
    return audio[:end]


def write_positive(src, dst):
    a, _ = sf.read(src, dtype="int16")
    a = phrase_end_trim(a)
    sf.write(dst, a[-CLIP:], SR, subtype="PCM_16")


def link(src, dst_dir, name=None):
    dst = dst_dir / (name or Path(src).name)
    if not dst.exists():
        os.link(src, dst)


def last_2s(src, dst):
    """Запись робота: 3 с до срабатывания → конец фразы + 0.1 с, последние 2 с."""
    write_positive(src, dst)


def fp_windows(src, dst_dir):
    a, _ = sf.read(src, dtype="int16")
    for k, shift in enumerate((0, 0.16, 0.32, 0.48)):
        end = len(a) - int(shift * SR)
        sf.write(dst_dir / f"realfp_{Path(src).stem}_{k}.wav", a[end - CLIP:end], SR, subtype="PCM_16")


def main():
    random.seed(7)
    # 1. Новые записи → архив
    old_names = {p.name for p in ARCHIVE.glob("*/*.wav")}
    new_false = [ARCHIVE / "false" / f.name for f in sorted(INBOX.glob("*.wav"))]
    new_pos = [ARCHIVE / "positive" / f.name for f in sorted(INBOX.glob("positive/*.wav"))]
    if "--take-inbox" in sys.argv:
        for f in INBOX.glob("*.wav"):
            shutil.move(str(f), ARCHIVE / "false" / f.name)
        for f in INBOX.glob("positive/*.wav"):
            shutil.move(str(f), ARCHIVE / "positive" / f.name)
    elif any(INBOX.glob("*.wav")) or any(INBOX.glob("positive/*.wav")):
        print(f"В {INBOX} есть новые записи — не беру их (нужен --take-inbox после разбора)")
    # «Новые» = не использованные в v4–v5 (ни в обучении, ни в их test-списке)
    used = {Path(p).name for p in (OLD / "wake_debug_test_files.txt").read_text().split()}
    used |= {f.name.split("_", 1)[1].rsplit("_", 2)[0] + ".wav" for f in OLD.glob("negative_*/wakeneg_*.wav")}
    used |= {f.name.split("_", 1)[1] for f in OLD.glob("positive_*/wakepos_*.wav")}
    new_names = {f.name for f in ARCHIVE.glob("*/*.wav")} - used

    # 2. Разделение по сценам. Уже сделанное разделение берётся из SPLIT_FILE, чтобы повторный
    # запуск (например, после догенерации синтетики) не перетасовал записи между train и test.
    old_test = {Path(p).name for p in (OLD / "wake_debug_test_files.txt").read_text().split()}
    saved = {}
    if SPLIT_FILE.exists():
        for line in SPLIT_FILE.read_text().splitlines():
            cls, n, s = line.split("\t")
            saved[(cls, n)] = s
    split = {}
    targets = {"false": {"train": 0.5, "val": 0.25, "test": 0.25}, "positive": {"train": 0.5, "test": 0.5}}
    for cls in ("false", "positive"):
        new_groups = []
        for group in scenes(list((ARCHIVE / cls).glob("*.wav"))):
            names = [f.name for f in group]
            known = [saved[(cls, n)] for n in names if (cls, n) in saved]
            if known:
                s = "test" if "test" in known else known[0]
            elif any(n in old_test for n in names):
                s = "test"
            elif all(n in new_names for n in names):
                new_groups.append(names)
                continue
            else:
                s = "train"      # сцену уже видела v5
            for n in names:
                split[(cls, n)] = s
        # новые сцены — жадно по числу записей, чтобы доли train/val/test были близки к целевым
        random.shuffle(new_groups)
        count = {k: 0 for k in targets[cls]}
        total = sum(len(g) for g in new_groups)
        for names in sorted(new_groups, key=len, reverse=True):
            s = max(count, key=lambda k: targets[cls][k] * total - count[k])
            count[s] += len(names)
            for n in names:
                split[(cls, n)] = s
    # если в сцене смешались старые train и новые файлы — сцена целиком train (её уже видела v5)
    with open(SPLIT_FILE, "w") as fh:
        for (cls, n), s in sorted(split.items()):
            fh.write(f"{cls}\t{n}\t{s}\n")

    # 3. Каталоги модели
    if NEW.exists():
        for d in NEW.iterdir():
            if d.is_dir() and d.name not in ("checkpoints",):
                shutil.rmtree(d)
    dirs = {k: NEW / k for k in ["positive_train", "positive_test", "negative_train", "negative_test",
                                 "real_positive_train", "real_positive_test", "real_fp_train", "real_fp_test",
                                 "room_neg_train", "room_neg_test"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    real_prefixes = ("room_", "real_")
    for split_name in ("train", "test"):
        for f in (OLD / f"positive_{split_name}").glob("*.wav"):
            if f.name.startswith("wakepos_"):
                continue                                   # записи робота пересобираются ниже
            target = dirs[f"real_positive_{split_name}"] if f.name.startswith(real_prefixes) else dirs[f"positive_{split_name}"]
            write_positive(f, target / f.name)
        for f in (OLD / f"negative_{split_name}").glob("*.wav"):
            if not f.name.startswith("wakeneg_"):          # ложные робота теперь отдельный источник
                link(f, dirs[f"negative_{split_name}"])

    final_test = []
    for (cls, n), s in split.items():
        src = ARCHIVE / cls / n
        if cls == "positive":
            if s == "train":
                last_2s(src, dirs["real_positive_train"] / f"wakepos_{Path(n).stem}.wav")
            else:
                final_test.append(str(src))
        else:
            if s == "train":
                fp_windows(src, dirs["real_fp_train"])
            elif s == "val":
                fp_windows(src, dirs["real_fp_test"])
            else:
                final_test.append(str(src))
    (NEW / "final_test_files.txt").write_text("\n".join(sorted(final_test)))

    # Синтетика v7: 5% в test
    for src_dir, kind in [("pos_silero", "positive"), ("pos_omni", "positive"), ("pos_omni_nicole", "positive"),
                          ("neg_silero", "negative"), ("neg_omni", "negative"), ("neg_omni_nicole", "negative"),
                          ("neg_quiet", "negative")]:
        for f in sorted((V7 / src_dir).glob("*.wav")):
            s = "test" if random.random() < SYNTH_TEST_FRACTION else "train"
            if kind == "positive":
                write_positive(f, dirs[f"{kind}_{s}"] / f.name)
            else:
                link(f, dirs[f"{kind}_{s}"])

    # Негативы из комнаты v8 (уже разделены по блокам записи / по файлам sv_debug)
    for src_dir in ("room_neg", "sv_neg"):
        for s in ("train", "test"):
            for f in sorted((V8 / src_dir / s).glob("*.wav")):
                link(f, dirs[f"room_neg_{s}"])

    # Сводка
    c = {}
    for (cls, n), s in split.items():
        c[(cls, s, n in new_names)] = c.get((cls, s, n in new_names), 0) + 1
    print("записи робота (класс, split, новая?):", dict(sorted(c.items())))
    for k, d in dirs.items():
        print(f"  {k:20s} {len(list(d.glob('*.wav'))):6d}")
    print(f"  final test files: {len(final_test)}")


if __name__ == "__main__":
    main()
