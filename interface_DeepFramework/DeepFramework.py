"""Abstracción del backend deep learning usado por NeFESI.

Backend soportado:
- `Pytorch_flexible` (por defecto)
- `Pytorch`
- `Keras` (compatibilidad legacy)

Puedes seleccionar backend con variable de entorno `NEFESI_FRAMEWORK`.
"""

from __future__ import annotations

import os


def _resolve_framework() -> str:
    framework = os.getenv("NEFESI_FRAMEWORK", "Pytorch_flexible")
    allowed = {"Keras", "Pytorch", "Pytorch_flexible"}
    if framework not in allowed:
        raise ValueError(
            f"NEFESI_FRAMEWORK inválido: {framework}. Valores soportados: {sorted(allowed)}"
        )
    return framework


Type_Framework = _resolve_framework()

if Type_Framework == "Keras":
    from .keras_functions import DeepModel as ModelType
    from .keras_functions import DataBatchGenerator, get_preprocess_function
    from .keras_functions import _load_multiple_images, _load_single_image
    model_file_extension = "h5"
    channel_type = "channel_last"
elif Type_Framework == "Pytorch":
    from .pytorch_functions import DeepModel as ModelType
    from .pytorch_functions import DataBatchGenerator, get_preprocess_function
    from .pytorch_functions import _load_multiple_images, _load_single_image
    model_file_extension = "pkl"
    channel_type = "channel_first"
else:
    from .pytorch_flex_functions import DeepModel as ModelType
    from .pytorch_flex_functions import DataBatchGenerator, get_preprocess_function
    from .pytorch_flex_functions import _load_multiple_images, _load_single_image
    model_file_extension = "pkl"
    channel_type = "channel_first"


def deep_model(model_name):
    return ModelType(model_name)


def data_batch_generator(preprocessing_function, src_dataset, target_size, batch_size, color_mode):
    return DataBatchGenerator(preprocessing_function, src_dataset, target_size, batch_size, color_mode)


def preprocess_function(model_name):
    return get_preprocess_function(model_name)


def load_multiple_images(
    src_dataset,
    img_list,
    color_mode,
    target_size,
    preprocessing_function=None,
    prep_function=True,
):
    """Carga múltiple de imágenes según el backend configurado."""
    return _load_multiple_images(
        src_dataset,
        img_list,
        color_mode,
        target_size,
        preprocessing_function=preprocessing_function,
        prep_function=prep_function,
    )


def load_single_image(
    src_dataset,
    img_name,
    color_mode,
    target_size,
    preprocessing_function=None,
    prep_function=False,
):
    """Carga una imagen en formato compatible con el backend configurado."""
    return _load_single_image(
        src_dataset,
        img_name,
        color_mode,
        target_size,
        preprocessing_function=preprocessing_function,
        prep_function=prep_function,
    )
