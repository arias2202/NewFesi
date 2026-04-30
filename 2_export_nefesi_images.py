"""
Export a small set of neuron feature and top-scoring images from a Nefesi object.

Example:
    python 2_export_nefesi_images.py Proba/nefesi_model.obj
"""
import argparse
import os
import shutil
import stat

import numpy as np
from PIL import Image as PILImage
from torchvision import transforms


for _np_alias, _python_type in (
    ("float", float),
    ("int", int),
    ("bool", bool),
    ("object", object),
    ("str", str),
):
    if _np_alias not in np.__dict__:
        setattr(np, _np_alias, _python_type)


from functions.network_data2 import NetworkData


DEFAULT_IMAGES_PER_LAYER = 4
DEFAULT_IMAGE_SIZE = 1000
DEFAULT_INPUT_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a Nefesi .obj file and export neuron features plus "
            "top-scoring image mosaics for a limited number of neurons per layer."
        )
    )
    parser.add_argument("nefesi_obj", help="Path to the saved Nefesi .obj file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Folder to create. Defaults to '<obj folder>/<obj name>_images_4_per_layer'."
        ),
    )
    parser.add_argument(
        "--images-per-layer",
        type=int,
        default=DEFAULT_IMAGES_PER_LAYER,
        help="Number of neuron feature/top-scoring image pairs to export per layer.",
    )
    parser.add_argument(
        "--selection",
        choices=("max-activation", "first"),
        default="max-activation",
        help=(
            "Which neurons to export: highest max activation per layer, or the first "
            "N neuron indexes."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Optional dataset path override. Use this if the path stored in the .obj "
            "has moved."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Output side length in pixels for each exported jpg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output folder if it already exists.",
    )
    return parser.parse_args()


def default_output_dir(obj_path, images_per_layer):
    obj_dir = os.path.dirname(os.path.abspath(obj_path))
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    return os.path.join(obj_dir, f"{obj_name}_images_{images_per_layer}_per_layer")


def prepare_output_dir(path, overwrite=False):
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {path}. Pass --overwrite to replace it."
            )
        def make_writable_and_retry(function, failed_path, _exc_info):
            os.chmod(failed_path, stat.S_IWRITE)
            function(failed_path)

        shutil.rmtree(path, onerror=make_writable_and_retry)
    os.makedirs(path, exist_ok=True)


def safe_layer_name(layer_name):
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in layer_name
    )


def image_from_array(array, size):
    if array is None:
        raise ValueError("Cannot export an empty neuron feature.")
    image_array = np.asarray(array)
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    return PILImage.fromarray(image_array).resize(
        (size, size),
        resample=PILImage.Resampling.NEAREST,
    )


def make_display_preprocess(target_size):
    if target_size is None:
        target_size = (DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)
    if isinstance(target_size, tuple):
        resize_size = target_size[0]
    else:
        resize_size = int(target_size)

    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(resize_size),
        transforms.ToTensor(),
    ])
    return lambda img: [transform(img)]


def neuron_score(neuron):
    activations = getattr(neuron, "activations", None)
    if activations is None or len(activations) == 0:
        return float("-inf")
    return float(np.nan_to_num(activations[0], nan=float("-inf")))


def has_neuron_feature(neuron):
    feature = getattr(neuron, "neuron_feature", None)
    return feature is not None and np.asarray(feature).size > 0


def select_neuron_indexes(network_data, layer_name, limit, selection):
    total_neurons = network_data.get_len_neurons_of_layer(layer_name)
    if selection == "first":
        return list(range(min(limit, total_neurons)))

    scored_neurons = []
    for neuron_idx in range(total_neurons):
        neuron = network_data.get_neuron_of_layer(layer_name, neuron_idx)
        scored_neurons.append((neuron_score(neuron), neuron_idx))
    scored_neurons.sort(reverse=True)
    return [neuron_idx for _, neuron_idx in scored_neurons[:limit]]


def ensure_neuron_feature(neuron, network_data, layer_data):
    if has_neuron_feature(neuron):
        return

    activations = np.asarray(getattr(neuron, "activations", []), dtype=float)
    if activations.size == 0:
        raise ValueError("Cannot compute neuron feature without activations.")

    max_activation = np.max(np.abs(activations))
    if max_activation == 0:
        raise ValueError("Cannot compute neuron feature when all activations are zero.")

    norm_activations = activations / max_activation
    weights_sum = np.sum(norm_activations)
    if weights_sum == 0:
        raise ValueError("Cannot compute neuron feature when normalized activations sum to zero.")

    patches = neuron.get_patches(network_data, layer_data)
    neuron.norm_activations = norm_activations
    neuron.neuron_feature = np.sum(
        patches.reshape(patches.shape[0], -1) * (norm_activations / weights_sum)[:, np.newaxis],
        axis=0,
    ).reshape(patches.shape[1:])


def export_images(network_data, output_dir, images_per_layer, selection, image_size):
    original_preprocessing = network_data.dataset.preprocessing_function
    network_data.dataset.preprocessing_function = make_display_preprocess(network_data.dataset.target_size)

    nf_root = os.path.join(output_dir, "NF")
    top_scoring_root = os.path.join(output_dir, "Top_scoring")
    os.makedirs(nf_root, exist_ok=True)
    os.makedirs(top_scoring_root, exist_ok=True)

    exported = {}
    try:
        for layer_name in network_data.get_layers_name():
            layer_data = network_data.get_layer_by_name(layer_name)
            layer_dir_name = safe_layer_name(layer_name)
            layer_nf_path = os.path.join(nf_root, layer_dir_name)
            layer_top_path = os.path.join(top_scoring_root, layer_dir_name)
            os.makedirs(layer_nf_path, exist_ok=True)
            os.makedirs(layer_top_path, exist_ok=True)

            neuron_indexes = select_neuron_indexes(
                network_data,
                layer_name,
                images_per_layer,
                selection,
            )
            exported[layer_name] = neuron_indexes

            print(f"Exporting {len(neuron_indexes)} images for layer: {layer_name}")
            for rank, neuron_idx in enumerate(neuron_indexes, start=1):
                neuron = network_data.get_neuron_of_layer(layer_name, neuron_idx)
                ensure_neuron_feature(neuron, network_data, layer_data)
                file_prefix = f"rank_{rank:02d}_neuron_{neuron_idx:04d}"

                image_from_array(neuron.neuron_feature, image_size).save(
                    os.path.join(layer_nf_path, f"{file_prefix}.jpg")
                )
                image_from_array(neuron.get_mosaic(network_data, layer_data), image_size).save(
                    os.path.join(layer_top_path, f"{file_prefix}.jpg")
                )
    finally:
        network_data.dataset.preprocessing_function = original_preprocessing

    return exported


def main():
    args = parse_args()
    if args.images_per_layer < 1:
        raise ValueError("--images-per-layer must be at least 1.")
    if args.image_size < 1:
        raise ValueError("--image-size must be at least 1.")

    obj_path = os.path.abspath(args.nefesi_obj)
    output_dir = os.path.abspath(args.output_dir or default_output_dir(obj_path, args.images_per_layer))

    print(f"Loading Nefesi object: {obj_path}")
    network_data = NetworkData.load_from_disk(obj_path)
    if args.dataset_path is not None:
        network_data.dataset.src_dataset = args.dataset_path

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    exported = export_images(
        network_data=network_data,
        output_dir=output_dir,
        images_per_layer=args.images_per_layer,
        selection=args.selection,
        image_size=args.image_size,
    )

    print(f"Done. Exported images to: {output_dir}")
    for layer_name, neuron_indexes in exported.items():
        indexes = ", ".join(str(idx) for idx in neuron_indexes)
        print(f"  {layer_name}: {indexes}")


if __name__ == "__main__":
    main()
