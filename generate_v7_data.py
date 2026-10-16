"""
Синтетические позитивы и фонетические негативы для v7
======================================================
Два движка:
  silero — v3_1_ru, голос "random" (новый случайный голос на каждый клип) + 5 штатных,
           темп и высота тона через SSML
  omni   — OmniVoice-сервер: клон голоса Артура (emotion neutral/happy/sad/surprise)
           и голос Насти (emotion "nastja"), скорость, теги [laughter] [sigh] ...

Клипы: 16 кГц mono int16, тишина по краям обрезана (паддинг делает train.py).

Запуск:
  python3 generate_v7_data.py --backend silero --kind pos --count 6000 --seed 1
  python3 generate_v7_data.py --backend omni   --kind neg --count 3000 --seed 3
"""

import argparse
import random
from math import gcd
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
OUT_ROOT = Path("/home/artur/openWakeWord/dataset/v7")
OMNI_URL = "http://192.168.10.118:8000/tts/stream"

# ── Позитивы: только сама фраза, без продолжения (модель срабатывает в конце окна) ──
POS_TEXTS = ["Эй, Лёня", "Эй, Лёня!", "Эй Лёня", "Эй, Лёня?", "Эй, Лёня.", "Эй... Лёня!",
             "Эээй, Лёня!", "Эй-эй, Лёня!", "Эй, Лёнь!", "Эй, Лёня,"]

# ── Негативы по категориям (вес = доля в выборке) ──
# Не включаем «Эй, Леня»: это та же фраза без «ё» (она в target_phrase).
NEG = {
    "похожее начало": (0.18, ["Эй, Лёша", "Эй, Лена", "Эй, Лёха", "Эй, Лёва", "Эй, Лиза", "Эй, Лиля",
                              "Эй, Люба", "Эй, Люда", "Эй, Лера", "Эй, Лёвушка", "Эй, Лёшка", "Эй, Ленка"]),
    "похожее окончание": (0.18, ["Эй, Алёна", "Эй, Тоня", "Эй, Соня", "Эй, Ваня", "Эй, Таня", "Эй, Женя",
                                 "Эй, Сеня", "Эй, Боня", "Эй, Маня", "Эй, Феня", "Эй, Дуня", "Эй, Каня",
                                 "Эй, Герман", "Эй, Коля", "Эй, Толя", "Эй, Оля"]),
    "только первое слово": (0.14, ["Эй, подойди", "Эй, послушай", "Эй, смотри", "Эй, ты", "Эй, стой",
                                   "Эй, давай", "Эй, иди сюда", "Эй, погоди", "Эй, слышь", "Эй, кто там",
                                   "Эй, привет", "Эй, ну хватит", "Эй", "Эх", "Ой", "Ай"]),
    "только имя": (0.10, ["Лёня, иди сюда", "Где Лёня?", "Это Лёня", "Лёня, привет", "Лёня, ты тут?",
                          "Лёня, сколько времени?", "А Лёня где?", "Лёня"]),
    # «Хей/Гей/Ой/Эх, Лёня» не берём: акустически почти та же фраза (Parakeet слышит
    # настоящие «Эй, Лёня» как «Hey, Lunya») — такие негативы бьют по узнаванию
    "близкие обращения": (0.08, ["Эй, Леночка", "Эй, Лёшенька", "Эй, Алёнушка", "Эй, Лёлик",
                                 "Эй, Лёнечкин", "Эй, Ленусик"]),
    "непрерывная речь": (0.14, ["Я видел Лёню вчера", "Позови Лёню", "У Лёни день рождения", "Мы с Лёней гуляли",
                                "Скажи Лёне спасибо", "Это Лёнин телефон", "Ленин был в Москве",
                                "Мне лень вставать", "Эта линия занята", "Леонид придёт позже",
                                "Ладно, давай потом", "Время на компьютере", "Одно дело сказать"]),
    "короткие реплики": (0.18, ["Да", "Нет", "Ага", "Угу", "Ладно", "Ну да", "Вот так", "Привет", "Окей",
                                "Понятно", "Конечно", "Ну ладно", "Правда?", "Ничего", "Давай"]),
}

OMNI_VOICES = ["", "neutral", "happy", "sad", "surprise", "nastja", "nastja"]  # Настя чаще: один голос vs четыре
OMNI_PREFIX_TAGS = ["", "", "", "[laughter] ", "[sigh] ", "[confirmation-en] ", "[surprise-ah] ", "[surprise-oh] "]
OMNI_SUFFIX_TAGS = ["", "", " [question-ah]", " [question-oh]", " [question-ei]", " [question-yi]",
                    " [surprise-wa]", " [surprise-yo]", " [dissatisfaction-hnn]", " [laughter]"]


def trim(audio: np.ndarray, thr: float = 0.006) -> np.ndarray:
    nz = np.where(np.abs(audio.astype(np.float32) / 32768) > thr)[0]
    if len(nz) == 0:
        return audio[:0]
    return audio[max(0, nz[0] - int(0.04 * SR)): nz[-1] + int(0.08 * SR)]


def to_16k(x: np.ndarray, rate: int) -> np.ndarray:
    g = gcd(SR, rate)
    y = resample_poly(x.astype(np.float32), SR // g, rate // g)
    return np.clip(y, -32768, 32767).astype(np.int16)


def neg_text(rng):
    cats = list(NEG)
    cat = rng.choices(cats, weights=[NEG[c][0] for c in cats])[0]
    return cat, rng.choice(NEG[cat][1])


def silero_engine(rng):
    import torch
    torch.set_num_threads(4)
    model = torch.package.PackageImporter(
        "/home/artur/.cache/torch/hub/snakers4_silero-models_master/src/silero/model/v3_1_ru.pt"
    ).load_pickle("tts_models", "model")
    speakers = ["random"] * 8 + ["aidar", "baya", "kseniya", "xenia", "eugene"]

    def synth(text):
        rate = rng.choice(["x-slow", "slow", "medium", "medium", "fast"])  # x-fast — каша
        pitch = rng.choice(["x-low", "low", "medium", "medium", "high", "x-high"])
        ssml = f'<speak><prosody rate="{rate}" pitch="{pitch}">{text}</prosody></speak>'
        a = model.apply_tts(ssml_text=ssml, speaker=rng.choice(speakers), sample_rate=24000)
        return to_16k((a.numpy() * 32767), 24000), f"silero r={rate} p={pitch}"
    return synth


def omni_engine(rng, kind, voices=None):
    def synth(text):
        voice = rng.choice(voices or OMNI_VOICES)
        pre = rng.choice(OMNI_PREFIX_TAGS)
        # позитиву хвост после фразы не нужен: окно модели должно кончаться на «Лёня»
        suf = rng.choice(OMNI_SUFFIX_TAGS) if kind == "neg" else ""
        payload = {"text": pre + text + suf, "speed": round(rng.uniform(0.8, 1.25), 2)}
        if voice:
            payload["emotion"] = voice
        r = requests.post(OMNI_URL, json=payload, timeout=60)
        r.raise_for_status()
        rate = int(r.headers.get("x-sample-rate", 24000))
        return to_16k(np.frombuffer(r.content, dtype=np.int16), rate), f"omni v={voice or '-'} {pre}{suf}"
    return synth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["silero", "omni"], required=True)
    p.add_argument("--kind", choices=["pos", "neg"], required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--voices", default="", help="голоса OmniVoice через запятую, напр. nicole (по умолчанию — все)")
    p.add_argument("--out-suffix", default="", help="суффикс каталога, напр. _nicole → neg_omni_nicole")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = OUT_ROOT / f"{args.kind}_{args.backend}{args.out_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    voices = [v for v in args.voices.split(",") if v] or None
    synth = silero_engine(rng) if args.backend == "silero" else omni_engine(rng, args.kind, voices)
    lo, hi = (0.45, 1.9) if args.kind == "pos" else (0.2, 2.7)

    log = open(out / f"log_seed{args.seed}.tsv", "a")
    made = rejected = errors = 0
    while made < args.count:
        cat, text = ("pos", rng.choice(POS_TEXTS)) if args.kind == "pos" else neg_text(rng)
        try:
            audio, info = synth(text)
        except Exception as e:
            errors += 1
            if errors > 20:
                raise
            continue
        errors = 0
        audio = trim(audio)
        dur = len(audio) / SR
        if not (lo <= dur <= hi):
            rejected += 1
            continue
        name = f"{args.backend}{args.kind}_s{args.seed}_{made:05d}.wav"
        sf.write(out / name, audio, SR, subtype="PCM_16")
        log.write(f"{name}\t{cat}\t{text}\t{info}\t{dur:.2f}\n")
        made += 1
        if made % 500 == 0:
            log.flush()
            print(f"{made}/{args.count} (rejected {rejected})", flush=True)
    print(f"done {made}, rejected {rejected}")


if __name__ == "__main__":
    main()
