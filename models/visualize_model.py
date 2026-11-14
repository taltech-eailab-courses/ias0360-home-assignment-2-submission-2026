import tensorflow as tf
from pathlib import Path
import argparse

from cnn import build_cnn
from mobilenet import build_mobilenet
from squeezenet import build_squeezenet


def main(args) -> None:

    if args.model == "cnn":
        model = build_cnn()
        output_path = "cnn_architecture.png"
    elif args.model == "mobilenet":
        model = build_mobilenet()
        output_path = "mobilenet_architecture.png"
    elif args.model == "squeezenet":
        model = build_squeezenet()
        output_path = "squeezenet_architecture.png"

    tf.keras.utils.plot_model(
        model,
        to_file=output_path,
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        dpi=150,
    )

    print(f"Saved model diagram to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "mobilenet", "squeezenet"], default="cnn", help="Choose the model to run.")
    args = parser.parse_args()
    main(args)