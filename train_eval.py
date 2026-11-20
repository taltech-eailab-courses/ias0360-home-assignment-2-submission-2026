import os, time, csv
import numpy as np
from PIL import Image
from models import keras

HERE = os.path.dirname(os.path.abspath(__file__))
DIGITS_DIR = os.path.join(HERE, "data_own_digits")
RESULTS_CSV = os.path.join(HERE, "results", "results.csv")
MODELS_DIR = os.path.join(HERE, "results", "saved_models")


def load_mnist():
    (xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
    xtr = xtr[..., None].astype("float32") / 255.0
    xte = xte[..., None].astype("float32") / 255.0
    return xtr, ytr, xte, yte


def load_own_digits():
    """digit_0.png ... digit_9.png -> (10,28,28,1) float, labels 0..9."""
    xs, ys = [], []
    for d in range(10):
        img = Image.open(os.path.join(DIGITS_DIR, f"digit_{d}.png")).convert("L")
        a = np.array(img, dtype="float32") / 255.0
        if a.mean() > 0.5:                      # MNIST = white digit on black
            a = 1.0 - a
        ys_, xs_ = np.where(a > 0.2)            # bounding box of the stroke
        if len(ys_):
            a = a[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
        h, w = a.shape
        s = 20.0 / max(h, w)
        small = Image.fromarray((a * 255).astype("uint8")).resize(
            (max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        canvas = np.zeros((28, 28), dtype="float32")
        oy, ox = (28 - small.height) // 2, (28 - small.width) // 2
        canvas[oy:oy + small.height, ox:ox + small.width] = np.array(small) / 255.0
        xs.append(canvas[..., None]); ys.append(d)
    return np.array(xs), np.array(ys)


def run_experiment(name, model, epochs=10, lr=0.001, optimizer="adam",
                   batch_size=128, notes=""):
    keras.utils.set_random_seed(0)
    xtr, ytr, xte, yte = load_mnist()
    xo, yo = load_own_digits()

    opt = {"adam": keras.optimizers.Adam,
           "sgd": keras.optimizers.SGD}[optimizer](learning_rate=lr)
    model.compile(opt, "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(xtr, ytr, epochs=epochs, batch_size=batch_size,
              validation_split=0.1, verbose=2)

    val_acc = model.evaluate(xte, yte, verbose=0)[1]
    own_pred = model.predict(xo, verbose=0).argmax(1)
    own_acc = int((own_pred == yo).sum())

    # inference speed: predict the 10 own digits one by one, total / 10
    model(xo[:1], training=False)                         # warm-up
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for i in range(10):
            model(xo[i:i+1], training=False)
        times.append((time.perf_counter() - t) / 10)
    avg_ms = 1000 * float(np.mean(times))

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.h5")
    model.save(path)
    row = dict(name=name, epochs=epochs, lr=lr, optimizer=optimizer,
               batch_size=batch_size, params=model.count_params(),
               file_kb=round(os.path.getsize(path) / 1024, 1),
               avg_ms=round(avg_ms, 2), val_acc=round(val_acc, 4),
               own_acc=f"{own_acc}/10", own_pred=list(own_pred), notes=notes)

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new: w.writeheader()
        w.writerow(row)
    return row