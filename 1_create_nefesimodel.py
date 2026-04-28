"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""
import cv2
import torch
from torchvision import transforms
import numpy as np
import functools
from functions.network_data2 import NetworkData
import types
import functions.GPUtil as gpu
BATCH_SIZE = 100
from  functions.image import ImageDataset
from functions.read_activations import get_activations
import interface_DeepFramework.DeepFramework as DeepF
from Model_generation.Unet import UNet, CAN
import torchvision.models as models
import functions.GPUtil as gpu
from torch import nn
import gc
import torchvision
import os
from PIL import Image as im



def preproces_Resnet( imgs_hr):

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tnsr=[preprocess(imgs_hr)]

    return tnsr

def preproces_Resnet2( imgs_hr):

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    tnsr = [preprocess(imgs_hr)]

    return tnsr




def main():

    #Here You can find all the funcitons that allow us to visualize and quantify a trained Neural Network.
    # To perform the calculation taking into acount the negative activations instead of positives, uncoment line 53 in the file read_activations


    # Load the Model with your weigths first
    # gpu.assignGPU()
    # folder_dir ="C:/Users/arias/Desktop/Nefesi2022/"
    folder_dir = "/home/guillem/Nefesi2022/"



    model = models.resnet18()
    model.load_state_dict(torch.load("/home/guillem/LoRA/lora-pytorch/weights/training_weights_64.pth"))


    # # model.load_state_dict(torch.load('/home/guillem/Nefesi2022/nefesi/Ablation_final/VGG16_retrained100_4'))
    # l = [[name, module] for name, module in model.named_modules() if not isinstance(module, nn.Sequential)]
    #
    # layers_to_analyze = [x[0] for x in l if isinstance(x[1], nn.Conv2d)]
    #
    # layers_to_analyze = [[x, 0] for x in layers_to_analyze]


    deepmodel = DeepF.deep_model(model)

    Path_images='/data/local/datasets/ImageNetFused/'


    dataset = ImageDataset(src_dataset=Path_images,target_size=(224,224),preprocessing_function=preproces_Resnet,color_mode='rgb')
    dataset2 = ImageDataset(src_dataset=Path_images,target_size=(224,224),preprocessing_function=preproces_Resnet2,color_mode='rgb')



    # layers_interest=[['conv1',0],['layer1',0],['layer2',0],['layer3',0],['layer4',0],['fc',0]]
    #
    #
    #
    # # Path where you will save your results
    # save_path= "Nefesi_models/Resnet18_grey_LoRA/"
    # print(1)
    # Nefesimodel = NetworkData(model=deepmodel, layer_data=layers_interest, save_path=save_path, dataset=dataset,
    #                           default_file_name='Resnet18_grey_LoRA', input_shape=[(1, 3, 224, 224)])
    # print(2)
    # Nefesimodel.generate_neuron_data()
    # print(3)
    # Nefesimodel.save_to_disk("Resnet18_grey_LoRA_neurons")
    #
    #
    # Nefesimodel.eval_network()
    #
    # print('Activation Calculus done!')
    # Nefesimodel.save_to_disk('Resnet18_grey_LoRA_activation')
    #
    #
    #
    # Nefesimodel.dataset=dataset2
    # Nefesimodel.calculateNF()
    # print('NF done!')
    # Nefesimodel.dataset = dataset
    # Nefesimodel.save_to_disk('Resnet18_grey_LoRA_NF')
    #


    # calculate the Color selectivity of each neuron

    Nefesimodel = NetworkData.load_from_disk(
        'Nefesi_models/Resnet18_grey_LoRA/Resnet18_grey_LoRA_NF.obj')
    Nefesimodel.model = deepmodel

    for layer in Nefesimodel.get_layers_name():
        layer_data = Nefesimodel.get_layer_by_name(layer)
        print(layer)
        for n in range(Nefesimodel.get_len_neurons_of_layer(layer)):
            neurona = Nefesimodel.get_neuron_of_layer(layer, n)
            neurona.color_selectivity_idx_new(Nefesimodel, layer_data, dataset)
    Nefesimodel.save_to_disk('color_indx_Resnet18_grey_LoRA')

    for layer in Nefesimodel.get_layers_name():
        layer_data = Nefesimodel.get_layer_by_name(layer)
        print(layer)
        for n in range(Nefesimodel.get_len_neurons_of_layer(layer)):
            neurona = Nefesimodel.get_neuron_of_layer(layer, n)
            neurona.class_selectivity_idx()

    Nefesimodel.save_to_disk('Final_Resnet18_grey_LoRA')

    Nefesimodel._dataset = dataset2

    initial_path = 'Nefesi_models/Resnet18_grey_LoRA/NF/'
    initial_path2 = 'Nefesi_models/Resnet18_grey_LoRA/Top_scoring/'

    try:
        os.makedirs(initial_path)
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(initial_path2)
    except FileExistsError:
        # directory already exists
        pass


    layers = Nefesimodel.get_layers_name()

    for layer in layers:
        layer_path = initial_path + '/' + layer
        layer_path2 = initial_path2 + '/' + layer

        layer_data = Nefesimodel.get_layer_by_name(layer)
        try:
            os.makedirs(layer_path)
        except FileExistsError:
            # directory already exists
            pass

        try:
            os.makedirs(layer_path2)
        except FileExistsError:
            # directory already exists
            pass

        print(layer)
        for n in range(Nefesimodel.get_len_neurons_of_layer(layer)):
            neurona = Nefesimodel.get_neuron_of_layer(layer, n)
            # neurona2= Nefesimodel2.get_neuron_of_layer(layer, n)
            NF = neurona.neuron_feature * 255
            NF = im.fromarray(NF.astype(np.uint8)).resize((1000, 1000), resample=im.Resampling.NEAREST)

            NF.save(layer_path + '/' + str(n) + '.jpg')
            patches=im.fromarray((neurona.get_mosaic(Nefesimodel,layer_data)*255).astype(np.uint8)).resize((1000,1000),resample=im.Resampling.NEAREST)



            NF.save(layer_path+'/'+str(n)+'.jpg')

            patches.save(layer_path2+'/'+str(n)+'.jpg')


if __name__ == '__main__':
    main()