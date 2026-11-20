import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from models import keras, baseline_cnn, mobilenet_style, squeezenet_style
from train_eval import run_experiment

keras.utils.set_random_seed(0)

long = {"epochs": 20, "lr": 0.003}
runs = [
    ("initial",          baseline_cnn(),                          {}),
    ("small_flat_4_8_16", baseline_cnn(4, 8, 16),                 {}),
    ("small_flat_drop",  baseline_cnn(4, 8, 16, 0.2),             {}),
    ("gap",              baseline_cnn(gap=True),                  long),
    ("small_gap",        baseline_cnn(4, 8, 16, gap=True),        long),
    ("small_5ep",        baseline_cnn(4, 8, 16),                  {"epochs": 5}),
    ("mobilenet",        mobilenet_style(),                       {}),
    ("squeezenet",       squeezenet_style(),                      {}),
]
for name, model, kw in runs:
    r = run_experiment(name, model, **kw)
    print(name, r["params"], r["val_acc"], r["own_acc"], r["avg_ms"], "ms")