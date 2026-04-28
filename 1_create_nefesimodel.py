"""CLI para crear un fichero NeFESI desde un modelo PyTorch.

Flujo recomendado:
1) Cargar una CNN + pesos.
2) Analizar capas convolucionales.
3) Guardar activaciones máximas por neurona.
4) Calcular `neuron_feature` (weighted average de imágenes top-activadoras).
5) Exportar visualizaciones (weighted average + mosaico top scoring).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import models, transforms

from functions.image import ImageDataset
from functions.network_data2 import NetworkData
import interface_DeepFramework.DeepFramework as DeepF


DEFAULT_INPUT_SHAPE = [(1, 3, 224, 224)]


def build_preprocess(normalize: bool) -> transforms.Compose:
    steps = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(steps)


def discover_conv_layers(model: torch.nn.Module) -> List[List[object]]:
    """Devuelve capas objetivo con formato NeFESI: [[layer_name, layer_index], ...]."""
    return [[name, 0] for name, module in model.named_modules() if name and isinstance(module, torch.nn.Conv2d)]


def load_model(model_name: str, weights_path: str, device: torch.device) -> torch.nn.Module:
    if not hasattr(models, model_name):
        raise ValueError(f"Modelo torchvision no soportado: {model_name}")

    model_builder = getattr(models, model_name)
    model = model_builder()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_visual_assets(nefesi_model: NetworkData, out_dir: Path) -> None:
    weighted_dir = out_dir / "NF"
    top_scoring_dir = out_dir / "Top_scoring"
    weighted_dir.mkdir(parents=True, exist_ok=True)
    top_scoring_dir.mkdir(parents=True, exist_ok=True)

    for layer_name in nefesi_model.get_layers_name():
        layer_data = nefesi_model.get_layer_by_name(layer_name)
        layer_weighted = weighted_dir / layer_name
        layer_top = top_scoring_dir / layer_name
        layer_weighted.mkdir(parents=True, exist_ok=True)
        layer_top.mkdir(parents=True, exist_ok=True)

        for neuron_idx in range(nefesi_model.get_len_neurons_of_layer(layer_name)):
            neuron = nefesi_model.get_neuron_of_layer(layer_name, neuron_idx)

            weighted_avg = PILImage.fromarray((neuron.neuron_feature * 255).astype(np.uint8))
            weighted_avg = weighted_avg.resize((1000, 1000), resample=PILImage.Resampling.NEAREST)
            weighted_avg.save(layer_weighted / f"{neuron_idx}.jpg")

            top_mosaic = neuron.get_mosaic(nefesi_model, layer_data)
            top_mosaic_img = PILImage.fromarray((top_mosaic * 255).astype(np.uint8))
            top_mosaic_img = top_mosaic_img.resize((1000, 1000), resample=PILImage.Resampling.NEAREST)
            top_mosaic_img.save(layer_top / f"{neuron_idx}.jpg")


def run_pipeline(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, args.weights, device=device)
    deep_model = DeepF.deep_model(model)

    layers_to_analyze = discover_conv_layers(model)
    if not layers_to_analyze:
        raise RuntimeError("No se encontraron capas convolucionales para analizar.")

    dataset_for_activations = ImageDataset(
        src_dataset=args.dataset,
        target_size=(224, 224),
        preprocessing_function=build_preprocess(normalize=True),
        color_mode="rgb",
    )

    dataset_for_nf = ImageDataset(
        src_dataset=args.dataset,
        target_size=(224, 224),
        preprocessing_function=build_preprocess(normalize=False),
        color_mode="rgb",
    )

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    nefesi_model = NetworkData(
        model=deep_model,
        layer_data=layers_to_analyze,
        save_path=str(save_dir),
        dataset=dataset_for_activations,
        default_file_name=args.run_name,
        input_shape=DEFAULT_INPUT_SHAPE,
    )

    nefesi_model.generate_neuron_data()
    nefesi_model.save_to_disk(f"{args.run_name}_neurons")

    nefesi_model.eval_network(batch_size=args.batch_size)
    nefesi_model.save_to_disk(f"{args.run_name}_activations")

    nefesi_model.dataset = dataset_for_nf
    nefesi_model.calculateNF()
    nefesi_model.dataset = dataset_for_activations
    nefesi_model.save_to_disk(f"{args.run_name}_NF")

    export_visual_assets(nefesi_model, save_dir)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un modelo NeFESI a partir de una CNN de PyTorch + pesos + dataset."
    )
    parser.add_argument("--model", default="resnet18", help="Nombre del modelo en torchvision.models")
    parser.add_argument("--weights", required=True, help="Ruta al state_dict (.pth)")
    parser.add_argument("--dataset", required=True, help="Ruta raíz del dataset (ImageFolder) ")
    parser.add_argument("--output-dir", required=True, help="Directorio de salida para .obj + visualizaciones")
    parser.add_argument("--run-name", default="nefesi_run", help="Prefijo para artefactos serializados")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size para el barrido de activaciones")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
