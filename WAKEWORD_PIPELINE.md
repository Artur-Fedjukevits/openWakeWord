# Wake word «Эй Лёня»: как устроено обучение

Документ для продолжения работы без истории прошлых сессий. Результаты сравнений версий —
в [my_custom_model/model_comparisons.md](my_custom_model/model_comparisons.md).

Состояние на 2026-10-16: **на роботе v8**, порог 0.8, `patience` 2.

## 1. Что сейчас работает на роботе

- Модель: `my_custom_model/ey_lyonya.onnx` + `ey_lyonya.onnx.data` = копия **v8**.
- Запасные: `ey_lyonya_v7_seq1_step44736.onnx` (стояла 15.10 днём), `ey_lyonya_v5.onnx`.
  Все версии `ey_lyonya_v1…v8.onnx` лежат рядом, каждая со своим `.onnx.data`.
- Робот: systemd user-сервис `inmoov.service` (`systemctl --user start|stop inmoov`,
  логи `journalctl --user -u inmoov -f`). Запускает `ros2 launch inmoov_bringup inmoov.launch.py`.
- Настройки wake word — аргументы launch-файла `~/ros2_ws/src/inmoov_bringup/launch/inmoov.launch.py`:
  `wakeword_threshold` = 0.8, `wakeword_patience` = 2, `wakeword_save_dir` = `~/inmoov_wake_debug`.
  После правок launch или нод: `colcon build --packages-select inmoov_bringup inmoov_voice`
  (workspace собран без `--symlink-install`).

### Изменения в ноде робота (`inmoov_voice/openwakeword_node.py`)
- Звук копится и подаётся в модель ровно по 1280 сэмплов (80 мс). `raw_audio` приходит кусками
  по 512, а на неполных кусках `predict()` просто повторяет прошлый score — без буфера
  `patience` считал бы один кадр модели 2–3 раза.
- `patience`: срабатывание только если score ≥ порога N кадров подряд.
- `save_dir` / `save_sec`: при каждом срабатывании сохраняются последние 3 с звука в
  `~/inmoov_wake_debug/YYYYmmdd_HHMMSS_<score>.wav` — так собираются ложные срабатывания.

## 2. Цикл сбора данных и новой версии

1. Робот работает, срабатывания падают в `~/inmoov_wake_debug/`.
2. **Разбор вручную:** свои настоящие «Эй Лёня» (и детей) — в `~/inmoov_wake_debug/positive/`,
   ложные — оставить в корне. Всё, что в корне, сборщик считает ложным срабатыванием.
3. (по желанию) Запись фона: `python3 record_room_background.py --minutes 120 --label <сцена> --output room_household`
   и нарезка: `python3 process_room_recordings.py room_household dataset/v8/room_neg room_household_bg`.
4. (по желанию) Новые записи `~/inmoov_sv_debug` (фразы, которые слышал робот):
   `python3 scan_sv_debug.py --after <последний обработанный файл> --out dataset/sv_debug_scan_N.csv`,
   затем `python3 build_room_negatives.py dataset/sv_debug_scan_N.csv dataset/v8/sv_neg_N`
   (и добавить каталог в `build_v8_dataset.py`, цикл «Негативы из комнаты v8»).
5. Сборка данных: `python3 build_v8_dataset.py --take-inbox`
   (для новой версии — скопировать в `build_v9_dataset.py`, поменять `NEW` на `ey_lyonya_v9`).
6. Конфиг: скопировать `ey_lyonya_v8.yml` → `ey_lyonya_v9.yml`, заменить `ey_lyonya_v8` на `ey_lyonya_v9`.
7. Обучение (~2 ч на CPU; признаки считаются заново, т.к. каталог новый):
   `python3 openwakeword/train.py --training_config ey_lyonya_v9.yml --augment_clips --train_model`
   Выход: `my_custom_model/ey_lyonya_v9.onnx`; модель робота не трогается.
8. Сравнение в условиях робота (поток, порог, patience):
   `python3 compare_models.py ey_lyonya_v8 ey_lyonya_v9 --threshold 0.8 --patience 2 --final ey_lyonya_v9`
9. Если лучше — поставить на робота (см. п. 6 «Подводные камни» про `.onnx.data`) и перезапустить сервис.
10. Дописать результаты в `model_comparisons.md`.

Любой сохранённый чекпоинт можно экспортировать:
`python3 export_checkpoint.py ey_lyonya_v8 seq1_step40789` → `my_custom_model/ey_lyonya_v8_seq1_step40789.onnx`.
Метрики всех чекпоинтов — `my_custom_model/<модель>/checkpoints/metrics.csv`.

## 3. Где лежат данные

| Путь | Что это |
|---|---|
| `dataset/wake_debug_used/{false,positive}/` | архив разобранных записей робота (3 с до срабатывания) |
| `dataset/v7/real_split.tsv` | деление записей робота на train/val/test **по сценам** (общее для v7, v8, дальше). Сохраняется между запусками — не удалять |
| `dataset/v7/pos_*`, `neg_*` | синтетика v7–v8 (Silero, OmniVoice: Артур, Настя, nicole), тихие куски `neg_quiet` |
| `dataset/v8/room_neg/{train,test}` | 6 ч записи комнаты 15.10, нарезка по 2 с + трудные окна |
| `dataset/v8/sv_neg/{train,test}` | речь из `inmoov_sv_debug` 03–14.10 без «Лёня» |
| `dataset/positive_*`, `negative_*` | исходники первых версий (Silero 6 голосов, CosyVoice, записи `real_*`) |
| `dataset/sv_debug_scan*.csv` | транскрипты и score по записям `inmoov_sv_debug` |
| `dataset/rejected/` | отбракованные клипы (CosyVoice с зачитанной инструкцией и т.п.) |
| `room_background/` | 28 мин сцен (tv, talk, music, kitchen, quiet), фон для аугментации |
| `room_household/`, `room_household_bg/` | 6.5 ч записи комнаты 15.10 (исходник / фон без «Лёня») |
| `room_ambient/` | 1 ч почти цифровой тишины 14.10 (Jabra гасит слабый фон) |
| `my_custom_model/ey_lyonya/` | **данные v4–v5. Не удалять:** сборщики v7/v8 берут отсюда старые позитивы/негативы и прежний test-список |
| `my_custom_model/ey_lyonya_v8/` | собранные каталоги v8 (жёсткие ссылки + обрезанные позитивы), признаки `*.npy`, `checkpoints/`, `final_test_files.txt` |
| `my_custom_model/ey_lyonya_v7/run1/` | неудачный первый прогон v7 (до исправления тишины после фразы) |
| `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | 17 ГБ общих негативов (HuggingFace `davidscripka/openwakeword_features`) |
| `/home/artur/validation_set_features.npy` | 11 ч валидации ложных срабатываний (там же, в домашнем каталоге) |
| `audioset_16k/`, `fma/`, `mit_rirs/` | фон и импульсные отклики для аугментации |

### Состав v8
| Каталог | Клипов | Источник |
|---|---|---|
| positive_train / test | 14 897 / 1 024 | Silero случайные голоса, OmniVoice (Артур, Настя, nicole), старые Silero и CosyVoice |
| real_positive_train / test | 243 / 27 | твой голос и детей: комната, старые записи, робот |
| negative_train / test | 15 918 / 1 460 | фонетические негативы (Silero, OmniVoice), фон и речь комнаты прежних версий, тишина |
| real_fp_train / test | 340 / 60 окон | ложные срабатывания робота, 4 окна по 2 с на запись |
| room_neg_train / test | 6 241 / 744 | 6 ч записи комнаты + новые sv_debug |
| финальный test | 31 ложное + 18 «Эй Лёня» | записи робота, их не видят ни обучение, ни выбор чекпоинта |

## 4. Скрипты

| Скрипт | Назначение |
|---|---|
| `generate_v7_data.py` | синтетика: `--backend silero/omni --kind pos/neg --count N --seed S [--voices nicole --out-suffix _nicole]` |
| `vet_positives.py <каталог>` | отсев брака в синтетических позитивах через Parakeet (в `rejected/`) |
| `record_room_background.py` | запись фона с микрофона, 10-с клипы; клипы с score ≥ 0.3 — в `rejected_wakeword/` |
| `process_room_recordings.py` | нарезка длинных записей на негативы и фон, отсев «Лёня» через Parakeet |
| `scan_sv_debug.py` | транскрипт + пик модели по записям `inmoov_sv_debug` → CSV |
| `build_room_negatives.py` | негативы из CSV `scan_sv_debug` |
| `build_v7_dataset.py`, `build_v8_dataset.py` | сборка каталогов модели; `--take-inbox` — забрать разобранные записи робота |
| `compare_models.py` | сравнение моделей в условиях робота на финальном test |
| `export_checkpoint.py` | экспорт чекпоинта в ONNX |
| `record_samples.py` | запись своих «Эй Лёня» с микрофона |
| `test_wakeword.py` | живой тест модели с микрофона (`--show-score`, `--threshold`) |
| старые: `generate_dataset.py` (Silero 6 голосов), `generate_omnivoice_samples.py`, `generate_cosyvoice_samples.py` (сервер уже не CosyVoice), `generate_negative_features.py` (не используется) | |

Синтетика: Silero `v3_1_ru` локально (`~/.cache/torch/hub/snakers4_silero-models_master`), голос `random` —
новый голос на каждый клип. OmniVoice: `POST http://192.168.10.118:8000/tts/stream`
`{"text", "emotion", "speed"}`, ответ — raw PCM 16 бит 24 кГц. `emotion`: neutral/happy/sad/surprise
(клон Артура), `nastja`, `nicole`. Теги в тексте: `[laughter] [sigh] [question-ah] …`.
Не делать негативами «Эй, Леня» (та же фраза) и «Хей/Гей/Ой/Эх, Лёня» (акустически почти позитив).

## 5. Изменения в openWakeWord (`openwakeword/train.py`, `openwakeword/data.py`)

Ошибки исходного кода и исправления:
1. **Батчи без перемешивания + повтор ×8.** `mmap_batch_generator` читал строки подряд (а признаки
   записаны в порядке файлов — батч из одного источника), а 8 воркеров DataLoader выдавали
   одинаковые батчи. Теперь случайные строки, свой генератор у каждого воркера.
2. **Аугментация по батчам в алфавитном порядке.** Фон/эхо/сдвиг тона применяются ко всему батчу
   из 16 клипов — соседние копии получали одинаковые. Списки клипов перемешиваются перед аугментацией.
3. **Сохранялся только 1 раунд аугментации** (`n_total` = число файлов, а не файлов × раундов).
4. **`best_val_fp` не обновлялся** → `target_false_positives_per_hour` ни на что не влиял. Исправлено.
5. **Метрики при пороге 0.5 и FP по кадрам.** Теперь при `eval_threshold`/`eval_patience`,
   ложные на 11-часовом наборе считаются как активации с паузой 1.5 с.
6. **Валидация ложных одним батчем** (480 тыс. окон) съедала 27 ГБ RAM → батчи по 8192.
7. **`--convert_to_tflite` и др. флаги** имели `default="False"` (строка = истина) → конвертация
   запускалась всегда и падала на `onnx_tf`.
8. **`torchaudio.info` удалён в torchaudio 2.10** → замена через soundfile в `data.py`;
   для `torchaudio.load` установлен `torchcodec==0.10.0` в `~/.local`.

Новые возможности:
- **`extra_sources`** в конфиге — отдельные источники с собственной долей в батче:
  `label`, `train_dir`, `test_dir`, `batch_n`, `rounds`, `test_rounds`, `rir` (эхо для train),
  `test_augment: false` (валидация без добавочного шума). Признаки каждого источника считаются
  отдельно (`<имя>_features_{train,test}.npy`), добавление источника не пересчитывает остальные.
- **Все чекпоинты** сохраняются в `<модель>/checkpoints/seqX_stepY.pt` + `metrics.csv`.
- **`checkpoint_selection: targets`**: recall на реальных позитивах ≥ `target_recall`
  (если никто — в пределах `selection_recall_slack` от лучшего) → `val_fp_per_hr` ≤
  `target_false_positives_per_hour` → минимум суммы `*_fp_rate` реальных источников.
- `seed` — для перемешивания клипов.

## 6. Подводные камни

- **`.onnx` + `.onnx.data`:** веса лежат в `.data`, и **имя этого файла записано внутри `.onnx`**.
  Простое копирование/переименование `.onnx` продолжает читать старый `.data`. Установка версии
  на робота (пример для v8):
  ```
  cd my_custom_model && cp ey_lyonya_v8.onnx.data ey_lyonya.onnx.data && python3 -c "
  import onnx; m=onnx.load('ey_lyonya_v8.onnx', load_external_data=False)
  [setattr(e,'value','ey_lyonya.onnx.data') for t in m.graph.initializer for e in t.external_data if e.key=='location']
  onnx.save(m,'ey_lyonya.onnx')"
  ```
  затем `systemctl --user restart inmoov`.
- **Позитивы должны кончаться на фразе.** openWakeWord ставит конец клипа в конец окна (запас 0–0.2 с).
  Тишина после фразы учит модель срабатывать с задержкой и на «звук, потом тишина» — это была
  главная причина плохих v1–v6. Сборщик обрезает все позитивы до конца фразы + 0.1 с. Клипы длиннее
  2 с `train.py` обрезает **с конца** — у записей робота берутся последние 2 с.
- **Кэш признаков:** если `<модель>/positive_features_train.npy` существует, `train.py` не пересчитывает
  основные признаки. После изменения клипов — удалить `*.npy` в каталоге модели (или новая версия = новый каталог).
- `train.py` имеет окончания строк **CRLF** — при правке скриптом сохранять их.
- Записи робота делить **по сценам** (соседние ближе 5 мин), не по файлам — иначе утечка train/test.
- Офлайн-проверка (`compare_models.py`) занижает ложные срабатывания по сравнению с живой работой —
  смотреть на настройки 0.8/2 и на живую статистику.
- Обучение при работающем роботе — с `nice -n 19` (вернуть приоритет без root нельзя). Робот привязан к CPU 0–13.
- `positive_data_dir` в старом `ey_lyonya.yml` — `train.py` его не читает.
- `target_accuracy` не используется вообще.
- Jabra (через PipeWire, системный default source) гасит тихий фон до ~−90 dBFS.

## 7. Кратко по истории

- v1–v2: Silero 6 голосов + немного реальных; v2 — записи в новой комнате, фон комнаты.
- v3: речь из `inmoov_sv_debug` как негативы — почти без эффекта.
- v4–v5: реальные ложные срабатывания робота (`wake_debug`) — заметно меньше ложных. v5 долго стояла на роботе.
- v6: `max_negative_weight` 1700 — хуже v5.
- v7: исправление `train.py` (п. 5), синтетика ~15 тыс. позитивов / ~10 тыс. фонетических негативов,
  разделение по сценам; найдена ошибка с тишиной после фразы. На робот — чекпоинт `seq1_step44736`.
- v8: голос nicole, 6 ч записи комнаты, новые sv_debug, вес 2500, чистая валидация — лучшая
  (финальный test при 0.8/2: 16/18 «Эй Лёня», 1/31 ложных).
