"""
The vgg model is from: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
"""
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os.path
import numpy as np
import types
from PIL import Image
import functools

# Public integration settings used by the rest of the project.
model_file_extension = 'pkl'
channel_type = "channel_first"

# PyTorch models have no input_shape attribute. The project uses channel-first tensors.
input_shape = [(1, 3, 224, 224)]


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def deep_model(model):
    return DeepModel(model)


def data_batch_generator(preprocessing_function, src_dataset, target_size,
                         batch_size, color_mode):
    return DataBatchGenerator(preprocessing_function, src_dataset, target_size,
                              batch_size, color_mode)


def preprocess_function(model_name):
    return get_preprocess_function(model_name)


def load_multiple_images(src_dataset, img_list, color_mode, target_size, preprocessing_function=None,
                         prep_function=True):
    return _load_multiple_images(src_dataset, img_list, color_mode, target_size,
                                 preprocessing_function=preprocessing_function,
                                 prep_function=prep_function)


def load_single_image(src_dataset, img_name, color_mode, target_size, preprocessing_function=None,
                      prep_function=False):
    return _load_single_image(src_dataset, img_name, color_mode, target_size,
                              preprocessing_function=preprocessing_function,
                              prep_function=prep_function)



class DeepModel():
    """
    Lightweight wrapper that gives PyTorch models the API expected by Nefesi.
    """
    def __init__(self, model):
        if isinstance(model, str):
            self.pytorchmodel = torch.load(model)
            self.name = os.path.basename(model).split(".")[0]
        else:
            self.pytorchmodel = model
            self.name = model.__class__.__name__

        self.gpu_ids = [0]  # o []
        use_cuda = torch.cuda.is_available() and len(self.gpu_ids) > 0
        self.device = torch.device(f'cuda:{self.gpu_ids[0]}' if use_cuda else 'cpu')
        self.pytorchmodel.to(self.device)


        self.pytorchmodel.eval()

        # Some attributes are added when older layer-list code needs them.
        # self.set_layers()

    def set_layers(self):
        """
        PyTorch has no Keras-style layers attribute, so it must be set before using legacy layer-list paths.
        :return:
        """
        self.all_layers = []
        layers_setting_hook = []
        for name, layer in self.pytorchmodel.named_modules():
            print(name)
            if  name != '':
                # add new attributes
                layer.name = name
                layer.get_config = types.MethodType(get_config, layer)
                self.all_layers.append(layer)
                layers_setting_hook.append(LayerAttributes(layer))
        x = [torch.rand(inp).to(self.device) for inp in self.input_shape]
        y = self.pytorchmodel(x)

        for setting_hook in layers_setting_hook:
            setting_hook.remove()

    @property
    def layers(self):
        """
        This function gets the list of layers of the model.
        Some attributes are added for older code paths.
        i.e.: layer.name,
        :return:
        """
        return self.all_layers

    @property
    def input_shape(self):
        return input_shape

    def get_layer(self, layer_name, get_index=False):
        """
        Retrieves a layer based on either its name (unique) or index.
        :return:
        """
        return rgetattr(self.pytorchmodel,layer_name)

        # for index, layer in enumerate(self.layers):
        #     if layer.name == layer_name:
        #         if get_index:
        #             return layer, index
        #         else:
        #             return layer

        raise ValueError('No such layer: ' + layer_name)

    def neurons_of_layer(self, layer_name):
        return self.get_layer(layer_name).output_shape[-1]

    def save(self, model_name):
        torch.save(self.pytorchmodel, model_name)

    def calculate_activations(self, layers_name, model_inputs):
        if not isinstance(layers_name, list):
            layers_name = [layers_name]
        # in case that the inputs are numpy array instead of tensors.

        # model_inputs = [model_in.to(self.device) for model_in in model_inputs]
        # model_inputs = [model_inputs.to(self.device)]
        outputs = [LayerActivations(self.get_layer(layer)) for layer in layers_name]


        #
        model_inputs=model_inputs[0].to(self.device)
        self.pytorchmodel.forward(model_inputs)
        layer_outputs = [output.features for output in outputs]

        # remove the hooks.
        for hook in outputs:
            hook.remove()
        return layer_outputs


class LayerActivations:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        feature_np = output.cpu().detach().numpy()
        if len(feature_np.shape) == 4:
            # channel first to channel last.
            self.features = np.transpose(feature_np, (0, 2, 3, 1))
        elif len(feature_np.shape) == 2:
            # for dense layer
            self.features = feature_np
        else:
            raise Exception("Unexpected shape.")

    def remove(self):
        self.hook.remove()


class LayerAttributes:
    """
    Hook for setting the attributes of layers.
    """
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # add some attributes to layers
        if isinstance(input, tuple):
            layer_input_shape = input[0].size()
        else:
            layer_input_shape = input.size()
        layer_output_shape = output.shape
        module.input_shape = self.transform_shape_format(layer_input_shape)
        module.output_shape = self.transform_shape_format(layer_output_shape)

    def transform_shape_format(self, shape_in_tensor):
        if shape_in_tensor.__len__() == 4:
            shape_in_tuple = (None, shape_in_tensor[2], shape_in_tensor[3], shape_in_tensor[1])
        else:
            shape_in_list = list(shape_in_tensor)
            shape_in_list[0] = None
            shape_in_tuple = tuple(shape_in_list)
        return shape_in_tuple

    def remove(self):
        self.hook.remove()


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        path_split = path.split('/')
        filename = "{}/{}".format(path_split[-2], path_split[-1])
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (filename,))
        return tuple_with_path


class DataBatchGenerator():
    def __init__(self, preprocessing_function, src_dataset, target_size,
                         batch_size, color_mode):
        if color_mode == "grayscale":
            preprocessing_function.transforms.append(
                    transforms.Grayscale(num_output_channels=1))
        dataset = ImageFolderWithPaths(src_dataset, transform=preprocessing_function)
        shuffle = False
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    # the attributes that are called
    @property
    def iterator(self):
        # the batch iterator
        return self.dataloader

    @property
    def samples(self):
        # number of images
        return self.dataloader.dataset.__len__()

    @property
    def batch_index(self):
        # index of batch
        return 0

    @property
    def batch_size(self):
        # batch size
        return self.dataloader.batch_size

    @property
    def filenames(self):
        return self.dataloader.dataset.samples


def get_preprocess_function(model_name):
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # return preprocess

    return _imagenet_vgg_preprocess


def _imagenet_vgg_preprocess(img):
    if type(img) is np.ndarray:
        # color_ivet index will call this after load the images.
        # 'RGB'->'BGR'
        img = img[..., ::-1]
        img = img.astype('float32', copy=False)
        mean = [103.939, 116.779, 123.68]
        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
        return img
    # resize
    img = img.resize((224, 224), Image.NEAREST)
    img = transforms.ToTensor()(img)
    img *= 255.0
    # 'RGB'->'BGR'
    img = img[[2, 1, 0], :, :]
    # normalize
    mean = [103.939, 116.779, 123.68]
    img[0, :, :] -= mean[0]
    img[1, :, :] -= mean[1]
    img[2, :, :] -= mean[2]
    return img


def get_config(self):
    config = {}
    attr_list = ["kernel_size", "stride", "padding"]
    target_attr_list = ["kernel_size", "strides", "padding"]
    for index, attr in enumerate(attr_list):
        if hasattr(self, attr):
            value = getattr(self, attr)
            if isinstance(value, int):
            # maxpooling layer will return int value.
                value = (value, value)
            config[target_attr_list[index]] = value
    return config


def array_to_img(x, data_format='channels_last', scale=True, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance."""
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        return Image.fromarray(x.astype('uint8'), 'RGBA')
    if x.shape[2] == 3:
        return Image.fromarray(x.astype('uint8'), 'RGB')
    if x.shape[2] == 1:
        if np.max(x) > 255:
            return Image.fromarray(x[:, :, 0].astype('int32'), 'I')
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    raise ValueError('Unsupported channel number: %s' % (x.shape[2],))


def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array."""
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x



def _load_multiple_images(src_dataset, img_list, color_mode, target_size, preprocessing_function=None,
                          prep_function=True):
    """Returns a list of images after applying the
     corresponding transformations.
    :param image_names: List of strings, name of the images.
    :param prep_function: Boolean.
    :return: Numpy array that contains the images (1+N dimension where N is the dimension of an image).
    """


    inputs = _load_single_image(src_dataset, img_list[0], color_mode, target_size,
                                 preprocessing_function=preprocessing_function,
                                 prep_function=prep_function)
    for i,img_name in enumerate(img_list[1:]):
        img = _load_single_image(src_dataset, img_name, color_mode, target_size,
                                 preprocessing_function=preprocessing_function,
                                 prep_function=prep_function)
        inputs = [torch.cat((inputs[inpt],imag_part),0)  for inpt,imag_part in enumerate(img)]
    # concatenate
    # imgs_batch = np.array(inputs)

    return inputs


def _load_single_image(src_dataset, img_name, color_mode, target_size, preprocessing_function=None,
                       prep_function=False):
    """Loads an image into PIL format.
    :param img_name: String, name of the image.
    :return: PIL image instance
    """
    grayscale = color_mode == 'grayscale'

    if os.path.isfile(src_dataset + img_name):
        img = Image.open(src_dataset + img_name).convert('RGB')

    else:
        img = Image.open('/data/135-1/datasets/' + img_name).convert('RGB')

    if grayscale:
        # r, g, b = img.split()
        # r = g.point(lambda i: i * 0)
        # g = g.point(lambda i: i * 0)
        # b = b.point(lambda i: i * 0)
        # img=Image.merge('RGB', (r, g, b))

        img = img.convert('L').convert('RGB')


    if preprocessing_function is not None and prep_function:

        img = preprocessing_function(img)
        # transform to numpy
        # img = img.numpy()
        # img = np.transpose(img, (1, 2, 0))
        img = [torch.unsqueeze(i, 0) for i in img]
    else:
        img = np.array(img)


    return img


if __name__ == "__main__":
    model = DeepModel("/home/yixiong/PycharmProjects/nefesi/vgg16_pytorch.pkl")
    pass
