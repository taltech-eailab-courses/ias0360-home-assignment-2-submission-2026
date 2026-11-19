"""Development runner: reads own TRAIN/VAL only, never own TEST."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models import build_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def dataset(x, y, batch=128, shuffle=False, seed=10):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        data = data.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    data = data.with_options(options)
    return data.batch(batch).prefetch(1)


def accuracy(model, data):
    return float(model.evaluate(data, verbose=0, return_dict=True)["accuracy"])


def export(model, folder, stage):
    # Save architecture and weights, excluding Adam's training-only state.
    inference_model = tf.keras.Sequential.from_config(model.get_config())
    inference_model.set_weights(model.get_weights())
    path = folder / f"{stage}.keras"
    inference_model.save(path)
    lines = []
    inference_model.summary(print_fn=lambda line: lines.append(line))
    (folder / "model_summary.txt").write_text("\n".join(lines) + "\n")
    return {"model": str(path.name), "sha256": digest(path),
            "parameters": model.count_params(),
            "weight_bytes": sum(w.nbytes for w in model.get_weights()),
            "model_file_bytes": path.stat().st_size}


class EpochLog(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        print(f"  MNIST epoch {epoch + 1}: {time.perf_counter() - self.start:.1f}s "
              f"val_accuracy={logs['val_accuracy']:.4f}", flush=True)


class AdaptationLog(tf.keras.callbacks.Callback):
    def __init__(self, mnist_val, own_val, initial_mnist, initial_own, tolerance):
        super().__init__()
        self.mnist_val, self.own_val = mnist_val, own_val
        self.minimum = initial_mnist - tolerance
        self.best_score = (initial_mnist + initial_own) / 2
        self.best_weights = None
        self.best_epoch = 0
        self.rows = []

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        mnist = accuracy(self.model, self.mnist_val)
        own = accuracy(self.model, self.own_val)
        score = (mnist + own) / 2
        eligible = mnist >= self.minimum
        self.rows.append({"epoch": epoch + 1, "mnist_val_accuracy": mnist,
                          "own_val_accuracy": own, "score": score, "eligible": eligible})
        if eligible and score > self.best_score + 1e-8:
            self.best_score, self.best_epoch = score, epoch + 1
            self.best_weights = self.model.get_weights()
        print(f"  Adapt epoch {epoch + 1}: MNIST val={mnist:.4f}, own val={own:.4f}", flush=True)


def run(args):
    from lite import convert, inference_copy, lite_accuracy, tensor_constants

    plan = json.loads((args.root / 'ml/experiments.json').read_text())
    configs = [c for c in plan['experiments'] if not args.only or c['name'] == args.only]
    if not configs:
        raise ValueError(f'Unknown experiment: {args.only}')
    args.output.mkdir(parents=True, exist_ok=True)
    seed = plan['seed']
    (images, labels), _ = tf.keras.datasets.mnist.load_data()
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=10000,
                                        random_state=seed, stratify=labels)
    if args.smoke:
        train_idx, val_idx = train_idx[:1000], val_idx[:100]
    x = images[train_idx].astype('float32')[..., None] / 255
    y = labels[train_idx]
    vx = images[val_idx].astype('float32')[..., None] / 255
    vy = labels[val_idx]
    with np.load(args.root / 'data/own_train.npz') as data:
        own_x, own_y = data['x'], data['y']
    with np.load(args.root / 'data/own_val.npz') as data:
        own_vx, own_vy = data['x'], data['y']
    replay = np.random.default_rng(seed).choice(len(y), 1000, replace=False)
    calibration = np.concatenate([own_x, x[replay[:100]]])
    adapt_x = np.concatenate([np.tile(own_x, (5, 1, 1, 1)), x[replay]])
    adapt_y = np.concatenate([np.tile(own_y, 5), y[replay]])
    mnist_val, own_val = dataset(vx, vy, 256), dataset(own_vx, own_vy, 50)
    all_rows = []
    for config in configs:
        folder = args.output / config['name']
        folder.mkdir(exist_ok=True)
        if (folder / 'development.json').exists():
            raise FileExistsError(f'{folder} already contains a run; choose a new output folder')
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_model(config)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"START {config['name']}: {model.count_params()} parameters", flush=True)
        checkpoint = folder / 'best.weights.h5'
        history = model.fit(dataset(x, y, 128, True, seed), validation_data=mnist_val,
                            epochs=1 if args.smoke else config['mnist_epochs'], verbose=0,
                            callbacks=[EpochLog(), tf.keras.callbacks.ModelCheckpoint(
                                str(checkpoint), monitor='val_accuracy', save_best_only=True,
                                save_weights_only=True)])
        model.load_weights(checkpoint)
        before = export(model, folder, 'mnist')
        before.update(mnist_val_accuracy=accuracy(model, mnist_val),
                      own_val_accuracy=accuracy(model, own_val))
        # These are the three MNIST-only architecture comparisons in the report.
        if config['name'] in ('cnn_baseline', 'mlp', 'lenet'):
            (folder / 'mnist_float32.tflite').write_bytes(convert(inference_copy(model), None, False))
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        callback = AdaptationLog(mnist_val, own_val, before['mnist_val_accuracy'],
                                 before['own_val_accuracy'], config['max_mnist_drop'])
        model.fit(dataset(adapt_x, adapt_y, 32, True, seed),
                  epochs=1 if args.smoke else config['adaptation_epochs'], verbose=0,
                  callbacks=[callback])
        model.set_weights(callback.best_weights)
        after = export(model, folder, 'adapted')
        inference = inference_copy(model)
        rows = []
        for integer in (False, True):
            mode = 'int8' if integer else 'float32'
            path = folder / f'adapted_{mode}.tflite'
            blob = convert(inference, calibration, integer)
            path.write_bytes(blob)
            interpreter = tf.lite.Interpreter(model_content=blob, num_threads=4)
            interpreter.allocate_tensors()
            row = dict(model=config['name'], format=mode, parameters=model.count_params(),
                       tflite_file_bytes=len(blob), constant_tensor_bytes=tensor_constants(blob),
                       mnist_validation_accuracy=lite_accuracy(interpreter, vx, vy),
                       own_validation_accuracy=lite_accuracy(interpreter, own_vx, own_vy),
                       file=str(path.relative_to(args.output)), sha256=digest(path))
            rows.append(row)
            all_rows.append(row)
        write_json(folder / 'development.json', dict(config=config, smoke=args.smoke,
                   mnist=before, adapted=after, adaptation_selected_epoch=callback.best_epoch,
                   mnist_history=history.history, adaptation_history=callback.rows, exports=rows))
        checkpoint.unlink()
    write_json(args.output / 'validation_results.json', all_rows)
    if not args.smoke and not args.only:
        eligible = [r for r in all_rows if r['format'] == 'int8'
                    and r['mnist_validation_accuracy'] >= 0.95
                    and r['own_validation_accuracy'] >= 0.94
                    and r['model'] not in ('mlp', 'lenet')]
        selected = min(eligible, key=lambda r: r['tflite_file_bytes']) if eligible else None
        write_json(args.output / 'selection.json', dict(selected=selected,
                   rule='Smallest INT8 CNN with MNIST VAL >=95% and own VAL >=94%',
                   test_used_for_selection=False))
    print('Done. Training, adaptation and calibration used TRAIN; selection used VAL only.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('runs/experiments'))
    parser.add_argument('--only', help='Run one name from ml/experiments.json')
    parser.add_argument('--smoke', action='store_true', help='Short pipeline check, not report results')
    run(parser.parse_args())
