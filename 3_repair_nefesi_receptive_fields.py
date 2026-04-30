"""
Repair receptive-field metadata stored in an existing Nefesi .obj file.

Example:
    python 3_repair_nefesi_receptive_fields.py Proba/nefesi_model.obj
"""
import argparse
import os

import numpy as np
from torchvision import models

for _np_alias, _python_type in (
    ("float", float),
    ("int", int),
    ("bool", bool),
    ("object", object),
    ("str", str),
):
    if _np_alias not in np.__dict__:
        setattr(np, _np_alias, _python_type)

from functions.layer_data2 import rf_stride_pad_to_layer
from functions.network_data2 import NetworkData


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute and repair receptive-field metadata in a saved Nefesi object."
    )
    parser.add_argument("nefesi_obj", help="Path to the saved Nefesi .obj file.")
    parser.add_argument(
        "--output-obj",
        default=None,
        help="Path for the repaired .obj. Defaults to '<input>_fixed_rf.obj'.",
    )
    parser.add_argument(
        "--architecture",
        default="resnet18",
        help="torchvision.models architecture used to recompute layer geometry.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Square model input size used during the original analysis.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output-obj if it already exists.",
    )
    return parser.parse_args()


def default_output_path(obj_path):
    obj_dir = os.path.dirname(os.path.abspath(obj_path))
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    return os.path.join(obj_dir, f"{obj_name}_fixed_rf.obj")


def build_architecture(architecture):
    if not hasattr(models, architecture):
        raise ValueError(f"Unknown torchvision architecture: {architecture}")
    constructor = getattr(models, architecture)
    try:
        return constructor(weights=None)
    except TypeError:
        return constructor(pretrained=False)


def main():
    args = parse_args()
    obj_path = os.path.abspath(args.nefesi_obj)
    output_path = os.path.abspath(args.output_obj or default_output_path(obj_path))

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output object already exists: {output_path}. Pass --overwrite to replace it.")

    network_data = NetworkData.load_from_disk(obj_path)
    model = build_architecture(args.architecture)
    input_shape = (1, 3, args.input_size, args.input_size)

    print(f"Repairing receptive fields using {args.architecture} and input {args.input_size}x{args.input_size}")
    for layer_data in network_data.layers_data:
        num_neurons, kernel, stride, padding = rf_stride_pad_to_layer(
            model,
            layer_data.layer_id,
            input_shape=input_shape,
        )
        existing_neurons = len(layer_data.neurons_data)
        if existing_neurons != num_neurons:
            print(
                f"Warning: {layer_data.layer_id} has {existing_neurons} saved neurons, "
                f"but architecture reports {num_neurons}."
            )

        old_values = (
            layer_data.receptive_field_Kernel,
            layer_data.receptive_field_Stride,
            layer_data.receptive_field_Padding,
        )
        layer_data.receptive_field_Kernel = kernel
        layer_data.receptive_field_Stride = stride
        layer_data.receptive_field_Padding = padding
        print(f"{layer_data.layer_id}: {old_values} -> {(kernel, stride, padding)}")

    network_data.save_to_disk(file_name=output_path, erase_partials=False)
    print(f"Saved repaired object: {output_path}")


if __name__ == "__main__":
    main()
