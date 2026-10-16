"""
Экспорт сохранённого чекпоинта (my_custom_model/<model>/checkpoints/seqX_stepY.pt) в ONNX
=========================================================================================
Запуск:
  python3 export_checkpoint.py ey_lyonya_v7 seq1_step44736 [имя_onnx]
По умолчанию имя: <model>_<checkpoint>.onnx в my_custom_model/.
"""

import sys

import torch

sys.argv, args = sys.argv[:1], sys.argv[1:]
from openwakeword.train import Model  # noqa: E402

ROOT = "/home/artur/openWakeWord/my_custom_model"
model_name, ckpt = args[0], args[1]
out_name = args[2] if len(args) > 2 else f"{model_name}_{ckpt}"

m = Model(n_classes=1, input_shape=(16, 96), model_type="rnn", layer_dim=64)
m.model.load_state_dict(torch.load(f"{ROOT}/{model_name}/checkpoints/{ckpt}.pt"))
m.model.eval()
torch.onnx.export(m.model.to("cpu"), torch.rand(1, 16, 96), f"{ROOT}/{out_name}.onnx", opset_version=13)
print(f"saved {ROOT}/{out_name}.onnx")
