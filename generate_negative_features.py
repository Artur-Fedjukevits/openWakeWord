import openwakeword.utils as utils
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

model_dir = Path('/home/artur/openWakeWord/my_custom_model/ey_lyonya')
F = utils.AudioFeatures(device='cpu')

negative_sources = [
    Path('/home/artur/openWakeWord/dataset/negative_train'),
    Path('/home/artur/audioset_16k'),
    Path('/home/artur/mit_rirs'),
]

for split, sources in [('train', negative_sources), ('test', [Path('/home/artur/openWakeWord/dataset/negative_test')])]:
    clips = []
    for src in sources:
        if src.exists():
            clips += list(src.glob('*.wav'))
            print(f'  {src.name}: {len(list(src.glob("*.wav")))} файлов')

    print(f'Всего негативных для {split}: {len(clips)} файлов')
    
    all_features = []
    skip = 0
    for i, clip in enumerate(clips):
        try:
            sr, audio = wav.read(str(clip))
            if audio.ndim > 1:
                audio = audio.mean(axis=1).astype(np.int16)
            audio = np.array([audio])
            features = F.embed_clips(audio, batch_size=1)
            # Принимаем любую форму — длинные файлы дадут больше строк
            for frame in features[0]:
                all_features.append(frame)  # каждый фрейм (96,) отдельно
        except Exception as e:
            skip += 1
        if i % 500 == 0:
            print(f'  {i}/{len(clips)} ok={len(all_features)} skip={skip}')

    features_arr = np.array(all_features)  # (N, 96)
    out = model_dir / f'negative_features_{split}.npy'
    np.save(str(out), features_arr)
    print(f'Сохранено: {out} shape={features_arr.shape}')

print('Готово!')