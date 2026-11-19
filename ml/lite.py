"""TFLite export and measurement helpers used in the experiments."""
import numpy as np
import tensorflow as tf

def inference_copy(model):
    copy = tf.keras.Sequential.from_config(model.get_config())
    copy.set_weights(model.get_weights())
    return copy


def convert(model, calibration, integer):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if integer:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: ([x[None, ...]] for x in calibration)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


def lite_accuracy(interpreter, x, y, batch=256):
    prediction = []
    for start in range(0, len(y), batch):
        values = x[start:start + batch]
        detail = interpreter.get_input_details()[0]
        if list(detail["shape"]) != list(values.shape):
            interpreter.resize_tensor_input(detail["index"], values.shape, strict=False)
            interpreter.allocate_tensors()
        detail = interpreter.get_input_details()[0]
        if detail["dtype"] == np.int8:
            scale, zero = detail["quantization"]
            assert scale > 0
            values = np.clip(np.rint(values / scale + zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(detail["index"], values)
        interpreter.invoke()
        output = interpreter.get_output_details()[0]
        prediction.extend(interpreter.get_tensor(output["index"]).argmax(axis=1).tolist())
    return float(np.mean(np.array(prediction) == y))


def tensor_constants(blob):
    from tensorflow.lite.python import schema_py_generated as schema
    model = schema.Model.GetRootAsModel(blob, 0)
    buffers = set()
    for graph in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(graph)
        for index in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(index)
            buffer = tensor.Buffer()
            if model.Buffers(buffer).DataLength():
                buffers.add(buffer)
    return sum(model.Buffers(index).DataLength() for index in buffers)
