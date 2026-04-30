"""
Run a focused Nefesi analysis for CORnet-S.

This script loads the local CORnet-S checkpoint, unrolls the recurrent CORnet-S
blocks into explicit time-step modules, calculates activations and neuron
features, and exports four NF/top-scoring images per analyzed layer. It does
not calculate color, class, object, part, orientation, or symmetry indexes.
"""
import argparse
import copy
import os
import shutil
import sys
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image as PILImage
from torch import nn
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

import functions.pytorch_integration as DeepF
from functions.image import ImageDataset
from functions.network_data2 import NetworkData


DEFAULT_INPUT_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_OBJECT_NAME = "cornet_s_nefesi_model"
DEFAULT_IMAGES_PER_LAYER = 4
DEFAULT_IMAGE_SIZE = 1000
DEFAULT_CORNET_DIR = "Corenet-s"
DEFAULT_CHECKPOINT = os.path.join(DEFAULT_CORNET_DIR, "latest_checkpoint.pth.tar")


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def forward(self, x):
        return x


class LinearCORblockStep(nn.Module):
    def __init__(self, source_block, time_step):
        super().__init__()
        self.time_step = time_step
        self.conv1 = copy.deepcopy(source_block.conv1)
        self.norm1 = copy.deepcopy(getattr(source_block, f"norm1_{time_step}"))
        self.nonlin1 = copy.deepcopy(source_block.nonlin1)
        self.conv2 = copy.deepcopy(source_block.conv2)
        self.norm2 = copy.deepcopy(getattr(source_block, f"norm2_{time_step}"))
        self.nonlin2 = copy.deepcopy(source_block.nonlin2)
        self.conv3 = copy.deepcopy(source_block.conv3)
        self.norm3 = copy.deepcopy(getattr(source_block, f"norm3_{time_step}"))
        self.nonlin3 = copy.deepcopy(source_block.nonlin3)
        self.output = Identity()

        if time_step == 0:
            self.skip = copy.deepcopy(source_block.skip)
            self.norm_skip = copy.deepcopy(source_block.norm_skip)
            self.conv2.stride = (2, 2)
        else:
            self.skip = None
            self.norm_skip = None
            self.conv2.stride = (1, 1)

    def forward(self, x):
        if self.time_step == 0:
            skip = self.norm_skip(self.skip(x))
        else:
            skip = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlin2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = x + skip
        x = self.nonlin3(x)
        return self.output(x)


class LinearCORblock(nn.Module):
    def __init__(self, source_block):
        super().__init__()
        self.conv_input = copy.deepcopy(source_block.conv_input)
        self.times = source_block.times
        for time_step in range(self.times):
            setattr(self, f"t{time_step}", LinearCORblockStep(source_block, time_step))

    def forward(self, x):
        x = self.conv_input(x)
        for time_step in range(self.times):
            x = getattr(self, f"t{time_step}")(x)
        return x


class LinearCORnetS(nn.Module):
    """
    CORnet-S with recurrent block loops expanded into explicit modules.

    The cloned convolution weights are identical to the source model. BatchNorm
    weights keep CORnet-S' original per-time-step parameters.
    """

    def __init__(self, source_model):
        super().__init__()
        self.V1 = copy.deepcopy(source_model.V1)
        self.V2 = LinearCORblock(source_model.V2)
        self.V4 = LinearCORblock(source_model.V4)
        self.IT = LinearCORblock(source_model.IT)
        self.decoder = nn.Sequential(OrderedDict([
            ("avgpool", copy.deepcopy(source_model.decoder.avgpool)),
            ("flatten", Flatten()),
            ("linear", copy.deepcopy(source_model.decoder.linear)),
            ("output", Identity()),
        ]))

    def forward(self, x):
        x = self.V1(x)
        x = self.V2(x)
        x = self.V4(x)
        x = self.IT(x)
        return self.decoder(x)


def preprocess_imagenet(img, normalize=True):
    transform_list = [
        transforms.Resize(DEFAULT_INPUT_SIZE),
        transforms.CenterCrop(DEFAULT_INPUT_SIZE),
        transforms.ToTensor(),
    ]
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return [transforms.Compose(transform_list)(img)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run activations and NF-only Nefesi analysis for CORnet-S."
    )
    parser.add_argument("save_path", help="Directory where outputs will be saved.")
    parser.add_argument("images_path", help="Path to an ImageFolder-style dataset.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cornet-dir", default=DEFAULT_CORNET_DIR)
    parser.add_argument(
        "--layers",
        default=None,
        help=(
            "Comma-separated layer names. Defaults to real convolution layers at "
            "each unrolled area step: V1.conv2, V2.t0.conv3, V2.t1.conv3, "
            "V4.t0.conv3... IT.t1.conv3."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--images-per-layer", type=int, default=DEFAULT_IMAGES_PER_LAYER)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--selection",
        choices=("max-activation", "first"),
        default="max-activation",
    )
    parser.add_argument(
        "--keep-existing-export-folder",
        action="store_true",
        help="Do not delete an existing exported_images folder before writing.",
    )
    return parser.parse_args()


def strip_module_prefix(state_dict):
    return OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in state_dict.items()
    )


def load_checkpoint_state_dict(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a checkpoint dict, got {type(checkpoint)!r}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    return strip_module_prefix(state_dict)


def import_cornet_s(cornet_dir):
    cornet_dir = os.path.abspath(cornet_dir)
    if cornet_dir not in sys.path:
        sys.path.insert(0, cornet_dir)
    from cornet.cornet_s import CORnet_S

    return CORnet_S


def load_linear_cornet_s(checkpoint_path, cornet_dir, device):
    constructor = import_cornet_s(cornet_dir)
    source_model = constructor()
    source_model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, device))
    source_model.to(device)
    source_model.eval()

    model = LinearCORnetS(source_model)
    model.to(device)
    model.eval()
    return model


def default_layers_to_analyze():
    return (
        ["V1.conv2"]
        + [f"V2.t{time_step}.conv3" for time_step in range(2)]
        + [f"V4.t{time_step}.conv3" for time_step in range(4)]
        + [f"IT.t{time_step}.conv3" for time_step in range(2)]
    )


def get_layers_to_analyze(layers_arg):
    if layers_arg:
        names = [layer.strip() for layer in layers_arg.split(",") if layer.strip()]
    else:
        names = default_layers_to_analyze()
    return [[name, 0] for name in names]


def save_latest(network_data, object_name):
    network_data.save_to_disk(file_name=object_name, erase_partials=True)
    latest_name = object_name if object_name.endswith(".obj") else object_name + ".obj"
    for file_name in os.listdir(network_data.save_path):
        if file_name.endswith(".obj") and file_name != latest_name:
            os.remove(os.path.join(network_data.save_path, file_name))


def reset_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
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
        resize_size = DEFAULT_INPUT_SIZE
    elif isinstance(target_size, tuple):
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


def export_limited_images(
    network_data,
    output_dir,
    images_per_layer,
    selection,
    image_size,
    overwrite=True,
):
    if overwrite:
        reset_directory(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    original_preprocessing = network_data.dataset.preprocessing_function
    network_data.dataset.preprocessing_function = make_display_preprocess(
        network_data.dataset.target_size
    )

    nf_root = os.path.join(output_dir, "NF")
    top_scoring_root = os.path.join(output_dir, "Top_scoring")
    os.makedirs(nf_root, exist_ok=True)
    os.makedirs(top_scoring_root, exist_ok=True)

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
            print(f"Exporting {len(neuron_indexes)} images for layer: {layer_name}")
            for rank, neuron_idx in enumerate(neuron_indexes, start=1):
                neuron = network_data.get_neuron_of_layer(layer_name, neuron_idx)
                file_prefix = f"rank_{rank:02d}_neuron_{neuron_idx:04d}"
                image_from_array(neuron.neuron_feature, image_size).save(
                    os.path.join(layer_nf_path, f"{file_prefix}.jpg")
                )
                image_from_array(neuron.get_mosaic(network_data, layer_data), image_size).save(
                    os.path.join(layer_top_path, f"{file_prefix}.jpg")
                )
    finally:
        network_data.dataset.preprocessing_function = original_preprocessing


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    print("Loading CORnet-S checkpoint")
    model = load_linear_cornet_s(args.checkpoint, args.cornet_dir, args.device)
    deep_model = DeepF.deep_model(model)

    dataset = ImageDataset(
        src_dataset=args.images_path,
        target_size=(DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE),
        preprocessing_function=preprocess_imagenet,
        color_mode="rgb",
    )
    layers_to_analyze = get_layers_to_analyze(args.layers)

    network_data = NetworkData(
        model=deep_model,
        layer_data=layers_to_analyze,
        save_path=args.save_path,
        dataset=dataset,
        default_file_name=args.object_name,
        input_shape=(1, 3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE),
    )

    print("Generating neuron data")
    network_data.generate_neuron_data()
    save_latest(network_data, args.object_name)

    print("Evaluating network activations")
    network_data.eval_network(
        batch_size=args.batch_size,
        verbose=True,
        save_intermediate=False,
    )
    save_latest(network_data, args.object_name)

    print("Calculating neuron features")
    network_data.calculateNF(save_intermediate=False)
    save_latest(network_data, args.object_name)

    export_dir = os.path.join(args.save_path, "exported_images")
    print(f"Exporting NF and top-scoring images to: {export_dir}")
    export_limited_images(
        network_data=network_data,
        output_dir=export_dir,
        images_per_layer=args.images_per_layer,
        selection=args.selection,
        image_size=args.image_size,
        overwrite=not args.keep_existing_export_folder,
    )

    save_latest(network_data, args.object_name)
    print(
        "Done. Latest Nefesi object saved at: "
        f"{os.path.join(args.save_path, args.object_name + '.obj')}"
    )


if __name__ == "__main__":
    main()
