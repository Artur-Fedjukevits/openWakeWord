"""
Запись фона комнаты для аугментации wake word "Эй Лёня"
=========================================================
Пишет с микрофона непрерывно и режет на 10-секундные клипы
16 kHz mono int16 — в том же формате, что audioset_16k.
Потом папка добавляется в background_paths в ey_lyonya.yml.

Каждый клип проверяется текущей моделью: если в нём похоже прозвучало
"Эй Лёня" (score >= --reject-score), клип отбрасывается — в фоне
wake word быть не должно.

Запуск:
  python3 record_room_background.py --minutes 5 --label tv
  python3 record_room_background.py --minutes 5 --label talk --device Jabra
  python3 record_room_background.py --list-devices
Ctrl+C — остановить досрочно (уже записанные клипы сохраняются).
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from math import gcd
from pathlib import Path

import numpy as np
import pyaudio
import scipy.io.wavfile as wav
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
CLIP_SEC = 10
CHUNK = 1280  # 80 ms при 16 kHz
DEFAULT_OUTPUT = "/home/artur/openWakeWord/room_background"
DEFAULT_MODEL = "/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx"


@contextmanager
def quiet_stderr():
    """Глушит мусор ALSA/JACK, который PyAudio печатает при инициализации."""
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def find_device(p: pyaudio.PyAudio, name: str):
    if name is None:
        return None
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d["maxInputChannels"] > 0 and name.lower() in d["name"].lower():
            return i
    print(f"❌ Не найден входной девайс, содержащий «{name}». Смотри --list-devices")
    print("   Микрофон, занятый PipeWire (напр. Jabra), доступен только через default —")
    print("   запускай без --device и выбери его: pactl set-default-source <имя>")
    sys.exit(1)


def open_stream(p: pyaudio.PyAudio, device):
    """Пробует 16 kHz mono; если девайс не умеет — его родные параметры + ресемплинг."""
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True,
                        input_device_index=device, frames_per_buffer=CHUNK)
        return stream, SAMPLE_RATE, 1
    except OSError:
        info = p.get_device_info_by_index(device) if device is not None else p.get_default_input_device_info()
        rate = int(info["defaultSampleRate"])
        channels = min(2, int(info["maxInputChannels"]))
        frames = int(CHUNK * rate / SAMPLE_RATE)
        stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                        input_device_index=device, frames_per_buffer=frames)
        return stream, rate, channels


def to_16k_mono(data: bytes, rate: int, channels: int) -> np.ndarray:
    audio = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, rate)
        audio = resample_poly(audio.astype(np.float32), SAMPLE_RATE // g, rate // g)
    return np.clip(audio, -32768, 32767).astype(np.int16)


def level_bar(rms: float, width: int = 20) -> str:
    # -60..0 dBFS → 0..width
    db = 20 * np.log10(max(rms, 1e-6))
    n = int(np.clip((db + 60) / 60 * width, 0, width))
    return "█" * n + " " * (width - n) + f" {db:5.1f} dBFS"


def next_index(out_dir: Path, label: str) -> int:
    idx = [int(p.stem.rsplit("_", 1)[-1]) for p in out_dir.glob(f"{label}_*.wav")
           if p.stem.rsplit("_", 1)[-1].isdigit()]
    return max(idx) + 1 if idx else 0


def main():
    parser = argparse.ArgumentParser(description="Запись фона комнаты для обучения wake word")
    parser.add_argument("--minutes", type=float, default=5, help="Длительность записи (default: 5)")
    parser.add_argument("--label", default="room",
                        help="Префикс файлов, описывающий сцену: tv, talk, music, quiet... (default: room)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Папка (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--device", default=None,
                        help="Часть имени входного девайса, напр. Jabra (default: системный)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Модель для отсева клипов с wake word")
    parser.add_argument("--reject-score", type=float, default=0.3,
                        help="Отбрасывать клип, если score модели >= этого (default: 0.3)")
    parser.add_argument("--no-filter", action="store_true", help="Не проверять клипы моделью")
    parser.add_argument("--list-devices", action="store_true", help="Показать входные девайсы и выйти")
    args = parser.parse_args()

    with quiet_stderr():
        p = pyaudio.PyAudio()

    if args.list_devices:
        default = p.get_default_input_device_info()["index"]
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["maxInputChannels"] > 0:
                mark = "*" if i == default else " "
                print(f" {mark}{i:2d}  {d['name']}  ({d['maxInputChannels']} ch, {int(d['defaultSampleRate'])} Hz)")
        p.terminate()
        return

    oww = None
    if not args.no_filter:
        from openwakeword.model import Model
        oww = Model(wakeword_models=[args.model], inference_framework="onnx")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = next_index(out_dir, args.label)

    device = find_device(p, args.device)
    with quiet_stderr():
        stream, rate, channels = open_stream(p, device)
    dev_name = (p.get_device_info_by_index(device) if device is not None
                else p.get_default_input_device_info())["name"]
    if device is None:
        # default идёт через PipeWire — покажем, какой микрофон за ним стоит
        src = os.popen("pactl get-default-source 2>/dev/null").read().strip()
        if src:
            dev_name += f" → {src}"
    read_frames = int(CHUNK * rate / SAMPLE_RATE)

    total_clips = int(args.minutes * 60 / CLIP_SEC)
    print("=" * 60)
    print("  Запись фона комнаты")
    print(f"  Девайс:  {dev_name} ({rate} Hz, {channels} ch)")
    print(f"  Сцена:   {args.label}   длительность: {args.minutes} мин ({total_clips} клипов по {CLIP_SEC} с)")
    print(f"  Папка:   {out_dir}")
    print("=" * 60)
    print("  Не произноси «Эй Лёня» во время записи. Ctrl+C — стоп.\n")

    saved = rejected = 0
    clip_len = CLIP_SEC * SAMPLE_RATE
    buf = []
    buf_len = 0
    max_score = 0.0
    clipped = 0
    t0 = time.time()

    try:
        while saved + rejected < total_clips:
            chunk = to_16k_mono(stream.read(read_frames, exception_on_overflow=False), rate, channels)
            buf.append(chunk)
            buf_len += len(chunk)
            clipped += int(np.sum(np.abs(chunk) >= 32000))

            if oww is not None:
                score = float(list(oww.predict(chunk).values())[0])
                max_score = max(max_score, score)

            rms = np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))
            elapsed = time.time() - t0
            print(f"\r  [{saved + rejected + 1:3d}/{total_clips}] {elapsed / 60:4.1f} мин  "
                  f"|{level_bar(rms)}|", end="", flush=True)

            if buf_len >= clip_len:
                audio = np.concatenate(buf)
                clip, rest = audio[:clip_len], audio[clip_len:]
                buf, buf_len = [rest], len(rest)

                clip_rms = np.sqrt(np.mean((clip.astype(np.float32) / 32768.0) ** 2))
                reason = None
                if clip_rms < 1e-5:
                    reason = "полная тишина — микрофон не пишет?"
                elif oww is not None and max_score >= args.reject_score:
                    reason = f"похоже на wake word (score {max_score:.2f})"

                if reason:
                    rejected += 1
                    print(f"\n  ⚠ клип отброшен: {reason}")
                    if clip_rms >= 1e-5:
                        # Если «Эй Лёня» никто не говорил, это ложное срабатывание модели —
                        # ценный трудный негатив. Сохраняем отдельно, для ручной проверки.
                        rej_dir = out_dir / "rejected_wakeword"
                        rej_dir.mkdir(exist_ok=True)
                        wav.write(str(rej_dir / f"{args.label}_{index + saved + rejected:04d}_{max_score:.2f}.wav"),
                                  SAMPLE_RATE, clip)
                else:
                    name = f"{args.label}_{index + saved:04d}.wav"
                    wav.write(str(out_dir / name), SAMPLE_RATE, clip)
                    saved += 1
                    if clipped > SAMPLE_RATE // 100:
                        print(f"\n  ⚠ {name}: перегруз ({clipped} сэмплов на пределе) — убавь громкость ТВ или усиление микрофона")
                max_score = 0.0
                clipped = 0

    except KeyboardInterrupt:
        print("\n\n  Остановлено.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    total = len(list(out_dir.glob("*.wav")))
    print(f"\n✅ Сцена «{args.label}»: сохранено {saved} клипов ({saved * CLIP_SEC / 60:.1f} мин), отброшено {rejected}")
    print(f"   Всего в {out_dir}: {total} клипов ({total * CLIP_SEC / 60:.1f} мин)")


if __name__ == "__main__":
    main()
