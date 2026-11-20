import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from models import keras, baseline_cnn
from train_eval import run_experiment

keras.utils.set_random_seed(0)
runs = [
    ("flat_4_8_8",       baseline_cnn(4, 8, 8),   {}),            # ~2.0k params
    ("flat_4_8_8_lr003", baseline_cnn(4, 8, 8),   {"lr": 0.003}),
    ("flat_4_8_16_lr003", baseline_cnn(4, 8, 16), {"lr": 0.003}),
    ("flat_3_6_8",       baseline_cnn(3, 6, 8),   {"lr": 0.003}), # ~1.5k params
]
for name, model, kw in runs:
    r = run_experiment(name, model, **kw)
    print(name, r["params"], r["val_acc"], r["own_acc"], r["avg_ms"], "ms")