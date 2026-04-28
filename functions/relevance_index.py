import numpy as np
import os

import torch
from functions import read_activations

from PIL import Image
import functools

pattern = r'[0-9]'
import util
from multiprocessing.pool import ThreadPool  # ThreadPool don't have documentation :( But uses threads




def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def relevance_ablation(ablated_neuron):
    def hook(model, input, output):

        output[:,ablated_neuron]=0

    return hook

activation ={}


def get_activation(xy_locations,neuron_num,my_activations):
    def hook(model, input, output, my_activations=my_activations):



        if output.dim()>2:
            for n in range(output.shape()[0]):
                x=xy_locations[n][0]
                y = xy_locations[n][1]
                my_activations[n]=output[n,neuron_num,x,y]

        else:
            my_activations = output[:,neuron_num]
    return my_activations




def add_padding(pil_img,padding=0, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + padding + padding
    new_height = height + padding + padding
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (padding, padding))
    return result




def get_relevance_idxex(neuron_data,neuron_num, current_layer,target_layer,num_neuros_ablated ,Nefesimodel):
    """Returns the class selectivity index value.
    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param labels: Dictionary, key: name class, value: label class.
    :param threshold: Float.
    :param type: type=1 As Ivet defined, type=2 sum of classes that overpass the threshold
    :return: A tuple with: label class and class index value.
    """

    relevance=np.zeros(num_neuros_ablated)
    model=Nefesimodel.model.pytorchmodel
    dataset=Nefesimodel.dataset
    ablated_layer=getattr(model,target_layer)
    my_layer=getattr(model,current_layer)


    image_names = neuron_data.images_id
    locations = neuron_data.xy_locations


    images_color = dataset.load_images(image_names)





    max_activation=neuron_data.activations.astype('float16')[0]
    norm_activations_sum = np.sum(neuron_data.activations.astype('float16') / max_activation)


    for n in range(num_neuros_ablated):

        hook=ablated_layer.register_forward_hook(relevance_ablation(n))

        normal_activations = read_activations.get_activation_from_pos(images_color, Nefesimodel,
                                                                      current_layer,
                                                                      neuron_num, locations)

        ablated_activations_sum = np.sum(normal_activations / max_activation)
        result = 1 - ablated_activations_sum / norm_activations_sum

        hook.remove()
        relevance[n]=result


    return relevance









def get_path_sep(image_name):
    path_sep = os.path.sep
    if image_name.find(os.path.sep) < 0:
        if os.path.sep == '\\':
            path_sep = '/'
        else:
            path_sep = '\\'
    return path_sep
