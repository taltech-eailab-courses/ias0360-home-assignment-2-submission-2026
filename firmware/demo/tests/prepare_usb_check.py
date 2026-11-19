"""Unveränderte VAL-Eingaben und CPU-Referenzausgaben für den USB-Hardwaretest."""
from pathlib import Path
import json
import sys
import numpy as np
import tensorflow as tf

h2, target = map(Path, sys.argv[1:3])
target.mkdir(parents=True, exist_ok=True)
i = tf.lite.Interpreter(model_path=str(h2 / 'models/final_int8.tflite'), num_threads=1, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
i.allocate_tensors()
input_spec, output_spec = i.get_input_details()[0], i.get_output_details()[0]
scale, zero = input_spec['quantization']
dataset = np.load(h2 / 'data/own_val.npz')
records = []
with (target / 'inputs.bin').open('wb') as stream:
    for image, label, name in zip(dataset['x'], dataset['y'], dataset['filenames']):
        q = np.clip(np.round(image / scale) + zero, -128, 127).astype(np.int8)
        i.set_tensor(input_spec['index'], q[None])
        i.invoke()
        scores = i.get_tensor(output_spec['index'])[0]
        stream.write(q.tobytes())
        records.append(dict(name=str(name), label=int(label), expected=int(scores.argmax()), scores=scores.tolist()))
(target / 'reference.json').write_text(json.dumps(records, indent=2) + '\n')
print(f'Prepared {len(records)} validation inputs; CPU correct: {sum(r["label"] == r["expected"] for r in records)}')
