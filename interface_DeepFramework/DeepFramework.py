"""Backend único de NeFESI: PyTorch flexible."""

from .pytorch_flex_functions import DeepModel as ModelType
from .pytorch_flex_functions import DataBatchGenerator, get_preprocess_function
from .pytorch_flex_functions import _load_multiple_images, _load_single_image

Type_Framework = "Pytorch_flexible"
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
    return _load_single_image(
        src_dataset,
        img_name,
        color_mode,
        target_size,
        preprocessing_function=preprocessing_function,
        prep_function=prep_function,
    )
