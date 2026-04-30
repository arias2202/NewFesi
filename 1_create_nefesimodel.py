"""
Run a complete PyTorch Nefesi analysis from the command line.
"""
import argparse
import os
import shutil
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image as PILImage
from torch import nn
from torchvision import models, transforms

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
DEFAULT_BATCH_SIZE = 100
DEFAULT_OBJECT_NAME = "nefesi_model"


def preprocess_imagenet(img, normalize=True):
    transform_list = [
        transforms.Resize(DEFAULT_INPUT_SIZE),
        transforms.CenterCrop(DEFAULT_INPUT_SIZE),
        transforms.ToTensor(),
    ]
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
    return [transforms.Compose(transform_list)(img)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Nefesi analysis for a PyTorch model and image folder."
    )
    parser.add_argument("save_path", help="Directory where the Nefesi object and exported images will be saved.")
    parser.add_argument("images_path", help="Path to an ImageFolder-style dataset.")
    parser.add_argument("--model-path", default=None,
                        help="Path to a PyTorch model file or state_dict. If omitted, torchvision pretrained weights are used.")
    parser.add_argument("--architecture", default="resnet18",
                        help="torchvision.models architecture used when model_path is a state_dict.")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer names. Defaults to all Conv2d layers.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--object-name", default=DEFAULT_OBJECT_NAME,
                        help="Output .obj base name. The same file is overwritten after each stage.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--skip-indexes", action="store_true",
                        help="Skip color and class index calculations.")
    parser.add_argument("--skip-exports", action="store_true",
                        help="Skip NF and top-scoring image export.")
    return parser.parse_args()


def build_model(architecture, num_classes):
    if not hasattr(models, architecture):
        raise ValueError(f"Unknown torchvision architecture: {architecture}")
    constructor = getattr(models, architecture)
    try:
        return constructor(weights=None, num_classes=num_classes)
    except TypeError:
        return constructor(pretrained=False, num_classes=num_classes)


def build_pretrained_model(architecture, num_classes):
    if not hasattr(models, architecture):
        raise ValueError(f"Unknown torchvision architecture: {architecture}")

    constructor = getattr(models, architecture)
    weights = None
    weights_class_name = architecture.replace("_", " ").title().replace(" ", "_") + "_Weights"
    if hasattr(models, weights_class_name):
        weights_class = getattr(models, weights_class_name)
        weights = weights_class.DEFAULT

    try:
        model = constructor(weights=weights)
    except TypeError:
        model = constructor(pretrained=True)

    adapt_classifier(model, num_classes)
    return model


def adapt_classifier(model, num_classes):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        if model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            if classifier.out_features != num_classes:
                model.classifier = nn.Linear(classifier.in_features, num_classes)
            return
        if isinstance(classifier, nn.Sequential):
            for index in range(len(classifier) - 1, -1, -1):
                if isinstance(classifier[index], nn.Linear):
                    if classifier[index].out_features != num_classes:
                        classifier[index] = nn.Linear(classifier[index].in_features, num_classes)
                    return

    raise ValueError("Could not adapt classifier for this architecture. Pass --model-path or update adapt_classifier().")


def infer_num_classes(images_path):
    return len([
        entry for entry in os.scandir(images_path)
        if entry.is_dir()
    ])


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
        return checkpoint["model"]
    if isinstance(checkpoint, (OrderedDict, dict)):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], (OrderedDict, dict)):
                return checkpoint[key]
    return checkpoint


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, (OrderedDict, dict)):
        return state_dict
    return OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in state_dict.items()
    )


def load_model(model_path, architecture, num_classes, device):
    if model_path is None:
        model = build_pretrained_model(architecture, num_classes)
        model.to(device)
        model.eval()
        return model

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    checkpoint = unwrap_state_dict(checkpoint)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = strip_module_prefix(checkpoint)
        model = build_model(architecture, num_classes)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_layers_to_analyze(model, layers_arg):
    if layers_arg:
        return [[layer.strip(), 0] for layer in layers_arg.split(",") if layer.strip()]

    layers = [
        [name, 0]
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if not layers:
        raise ValueError("No Conv2d layers found. Pass explicit layer names with --layers.")
    return layers


def save_latest(network_data, object_name):
    network_data.save_to_disk(file_name=object_name, erase_partials=True)
    latest_name = object_name if object_name.endswith(".obj") else object_name + ".obj"
    for file_name in os.listdir(network_data.save_path):
        if file_name.endswith(".obj") and file_name != latest_name:
            os.remove(os.path.join(network_data.save_path, file_name))


def calculate_indexes(network_data, dataset, object_name):
    for layer_name in network_data.get_layers_name():
        layer_data = network_data.get_layer_by_name(layer_name)
        print(f"Calculating color index: {layer_name}")
        for neuron_idx in range(network_data.get_len_neurons_of_layer(layer_name)):
            neuron = network_data.get_neuron_of_layer(layer_name, neuron_idx)
            neuron.color_selectivity_idx_new(network_data, layer_data, dataset)
    save_latest(network_data, object_name)

    for layer_name in network_data.get_layers_name():
        print(f"Calculating class index: {layer_name}")
        for neuron_idx in range(network_data.get_len_neurons_of_layer(layer_name)):
            neuron = network_data.get_neuron_of_layer(layer_name, neuron_idx)
            neuron.class_selectivity_idx()
    save_latest(network_data, object_name)


def reset_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def image_from_float_array(array, size=(1000, 1000)):
    image_array = np.clip(array * 255, 0, 255).astype(np.uint8)
    return PILImage.fromarray(image_array).resize(size, resample=PILImage.Resampling.NEAREST)


def export_neuron_feature_images(network_data, save_path):
    original_dataset = network_data.dataset
    if hasattr(original_dataset, "without_normalization"):
        network_data.dataset = original_dataset.without_normalization()

    nf_root = os.path.join(save_path, "NF")
    top_scoring_root = os.path.join(save_path, "Top_scoring")
    reset_directory(nf_root)
    reset_directory(top_scoring_root)

    try:
        for layer_name in network_data.get_layers_name():
            print(f"Exporting NF and top scoring images: {layer_name}")
            layer_data = network_data.get_layer_by_name(layer_name)
            layer_nf_path = os.path.join(nf_root, layer_name)
            layer_top_path = os.path.join(top_scoring_root, layer_name)
            os.makedirs(layer_nf_path, exist_ok=True)
            os.makedirs(layer_top_path, exist_ok=True)

            for neuron_idx in range(network_data.get_len_neurons_of_layer(layer_name)):
                neuron = network_data.get_neuron_of_layer(layer_name, neuron_idx)
                image_from_float_array(neuron.neuron_feature).save(
                    os.path.join(layer_nf_path, f"{neuron_idx}.jpg")
                )
                image_from_float_array(neuron.get_mosaic(network_data, layer_data)).save(
                    os.path.join(layer_top_path, f"{neuron_idx}.jpg")
                )
    finally:
        network_data.dataset = original_dataset


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    num_classes = infer_num_classes(args.images_path)
    if num_classes == 0:
        raise ValueError(f"No class folders found in images_path: {args.images_path}")

    print("Loading model")
    print(f"Detected {num_classes} classes in {args.images_path}")
    model = load_model(args.model_path, args.architecture, num_classes, args.device)
    deep_model = DeepF.deep_model(model)

    dataset = ImageDataset(
        src_dataset=args.images_path,
        target_size=(DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE),
        preprocessing_function=preprocess_imagenet,
        color_mode="rgb",
    )
    layers_to_analyze = get_layers_to_analyze(model, args.layers)

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
    network_data.eval_network(batch_size=args.batch_size, verbose=True, save_intermediate=False)
    save_latest(network_data, args.object_name)

    print("Calculating neuron features")
    network_data.calculateNF(save_intermediate=False)
    save_latest(network_data, args.object_name)

    if not args.skip_indexes:
        calculate_indexes(network_data, dataset, args.object_name)

    if not args.skip_exports:
        export_neuron_feature_images(network_data, args.save_path)

    save_latest(network_data, args.object_name)
    print(f"Done. Latest Nefesi object saved at: {os.path.join(args.save_path, args.object_name + '.obj')}")


if __name__ == "__main__":
    main()
