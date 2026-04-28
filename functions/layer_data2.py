import matplotlib.pyplot as plt
import torch
import math
from collections import OrderedDict
from torch import nn
import numpy as np
import functools
import copy
from functions.similarity_index import get_row_of_similarity_index

MIN_PROCESS_TIME_TO_OVERWRITE = 10
activation = {}

def get_activation(name):
    def hook(model, input, output):

        activation[name] = output
    return hook


def get_activation_input(name):
    def hook(model, input, output):

        act=input[0]
        activation[name] = act


    return hook

def my_hook(idx, inputs,output):

    if output.ndim > 3:

        output[1, :, output.shape[2]//2, output.shape[3]//2] = 1
        output[2, :, output.shape[2] // 2, output.shape[3] // 2+1] = 1
        output[3, :, 0, 0] = 1

    return output


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
#
# def gradient_rf_OG(model,input_shape,output_layer,main_input=None,dataset=None):
#
#     # Register Hooks
#     weights = copy.deepcopy(model.state_dict())
#     output = rgetattr(model, output_layer)
#     output.register_forward_hook(get_activation('output'))
#     model.register_forward_hook(get_activation_input('input'))
#
#     # set the model weights and biases to 0
#     model = model.train()
#     for module in model.modules():
#         try:
#             nn.init.constant_(module.weight,  0.05)
#             nn.init.zeros_(module.bias)
#             nn.init.zeros_(module.running_mean)
#             nn.init.ones_(module.running_var)
#         except:
#             pass
#
#         module.eval()
#
#     if main_input != None:
#         inputs = [torch.ones(shap, device='cuda:0', requires_grad=True) for shap in input_shape]
#     else:
#         inputs = torch.ones(input_shape, requires_grad=True).cuda()
#
#
#     # inputs=dataset.preprocessing_function(inputs)
#     out = model(inputs[0])
#     out_activations = activation['output']
#     in_activation = activation['input']
#
#
#     if main_input != None:
#         in_activation = in_activation[0][main_input]
#
#
#     if len(out_activations.shape) > 2:
#
#         grad = torch.zeros(out_activations.shape, requires_grad=True).cuda()
#         grad2 = torch.zeros(out_activations.shape, requires_grad=True).cuda()
#         with torch.no_grad():
#             grad[0, 0, grad.shape[2] // 2, grad.shape[3] // 2] = 1
#             grad2[0, 0, grad.shape[2] // 2, grad.shape[3] // 2] = -1
#             grad2[0, 0, grad2.shape[2] // 2, grad2.shape[3] // 2 + 1] = 1
#
#
#         out_activations.backward(gradient=grad, retain_graph=True)
#
#         gradient_of_input = np.abs(in_activation.grad[0, 0].cpu().data.numpy())
#         gradient_of_input = np.around(gradient_of_input / np.amax(gradient_of_input),decimals=3)
#         Kernel = np.max(np.where(gradient_of_input )) - np.min(np.where(gradient_of_input)) + 1
#
#         out_activations.backward(gradient=grad2, retain_graph=True)
#         gradient_of_input2 = np.abs(in_activation.grad[0, 0].cpu().data.numpy())
#         gradient_of_input2 = np.around(gradient_of_input2 / np.amax(gradient_of_input2),decimals=3)
#         Stride = np.argmax(gradient_of_input2) - np.argmax(gradient_of_input)
#
#         Kernel=Kernel+Stride-1
#
#         Padding=int((Stride*(out_activations.shape[2]-1)-input_shape[0][2]+Kernel)/2)
#
#
#
#
#
#
#
#
#
#     else:
#         Kernel=input_shape[0][2]
#         Stride=0
#         Padding=0
#
#
#     num_neurons= out_activations.shape[1]
#
#     model.load_state_dict(weights)
#
#     return num_neurons,Kernel,Stride,Padding
#


def rf_stride_pad_to_layer(model: nn.Module, layer_name: str, input_shape=(1,3,224,224), device=None):
    """
    Retorna:
      - num_neurons (channels) a la capa
      - kernel_acumulat (receptive field) (k_eff)
      - stride_acumulat (s_eff)
      - padding_acumulat (p_eff) (equivalent)
    Tot per una dimensió (assumint simetria H=W). Si vols, ho podem fer per H i W separat.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"Layer '{layer_name}' not found. Example names: {list(modules.keys())[:20]}")

    # 1) Mesura output shape a la capa amb un hook
    out_info = {}
    def hook(_m, _inp, out):
        out_info["shape"] = tuple(out.shape)

    handle = modules[layer_name].register_forward_hook(hook)
    x = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        model(x)
    handle.remove()

    if "shape" not in out_info:
        raise RuntimeError(f"Layer '{layer_name}' was not executed in forward.")

    out_shape = out_info["shape"]
    num_neurons = out_shape[1] if len(out_shape) >= 2 else out_shape[-1]

    # 2) Càlcul analític RF/stride/padding equivalent fins a layer_name
    j = 1.0
    r = 1.0
    start = 0.5  # centre del primer píxel de l'entrada

    # recorrem mòduls en ordre d'execució amb forward hooks "shallow"
    # per evitar FX, fem un forward i capturem els mòduls visitats
    visited = []

    def record(name):
        def _hk(m, inp, out):
            visited.append((name, m))
        return _hk

    handles = []
    for name, m in model.named_modules():
        # captem només “capas” amb efecte geomètric
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            handles.append(m.register_forward_hook(record(name)))

    with torch.no_grad():
        model(torch.zeros(input_shape, device=device))

    for h in handles:
        h.remove()

    # actualitza fins arribar a layer_name (inclosa)
    for name, m in visited:
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0]
            s = m.stride[0]
            p = m.padding[0]
            d = m.dilation[0]
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if m.stride is not None else k
            s = s if isinstance(s, int) else s[0]
            p = m.padding if isinstance(m.padding, int) else m.padding[0]
            d = 1
        else:
            continue

        # update
        start = start + ((k - 1) / 2 - p) * j
        r = r + (k - 1) * d * j
        j = j * s

        if name == layer_name:
            break
    else:
        raise RuntimeError(f"Layer '{layer_name}' not found in execution trace (visited list).")

    # padding equivalent (per dimensió)
    p_eff = ((r - 1) / 2) - (start - 0.5)

    # arrodonim a int quan toca (hauria de sortir .0)
    k_eff = int(round(r))
    s_eff = int(round(j))
    p_eff = int(round(p_eff))

    return num_neurons, k_eff, s_eff, p_eff




def decoder_rf(model,input_shape,study_layer,main_input=None):

    # Register Hooks
    output_layer='outc'
    input = rgetattr(model, study_layer)
    input.register_forward_hook(get_activation('study_layer'))



    weights = copy.deepcopy(model.state_dict())

    # set the model weights and biases to 0
    for module in model.modules():
        if module == torch.nn.modules.conv.Conv2d:
            try:
                nn.init.constant_(module.weight, 0.005)
                nn.init.zeros_(module.bias)
                nn.init.zeros_(module.running_mean)
                nn.init.ones_(module.running_var)
            except:
                pass
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()

    for n,i in enumerate(input_shape):
        i = list(i)
        i[0] = 4
        input_shape[n] = i



    # input_shape = [[i for i in input_s] for input_s in input_shape]
    if main_input != None:
        inputs = [torch.ones(shap, device='cuda:0', requires_grad=True) for shap in input_shape]
    else:
        inputs =( torch.ones(input_shape, requires_grad=True) ).cuda()

    handle = rgetattr(model, study_layer).register_forward_hook(my_hook)

    handle_out=rgetattr(model, output_layer).register_forward_hook(get_activation('output'))
    out_real = model(*inputs).detach().cpu().numpy()
    out= activation['output'].detach().cpu().numpy()

    handle.remove()

    out1=  np.sum(out[0],0)
    out2 = np.abs(out1-np.sum(out[1],0))
    out3 =  np.abs(out1-np.sum(out[2],0))
    out4 =  np.abs(out1-np.sum(out[3],0))
    out2= out2/np.max(out2)
    out3 = out3  / np.max(out3)
    out4 = out4/ np.max(out4)

    # plt.subplot(2, 2, 4)
    # plt.imshow(out1)
    # plt.subplot(2,2,1)
    # plt.imshow(out2)
    # plt.subplot(2,2,2)
    # plt.imshow(out3)
    # plt.subplot(2,2,3)
    # plt.title(study_layer)
    # plt.imshow(out4)
    # plt.show()


    Kernel = np.max(np.where(out2 > 0.00001)) - np.min(np.where(out2 > 0.00001)) + 1
    Stride = np.argmax(out3 > 0.00001) - np.argmax(out2 > 0.00001)
    Padding = Kernel - np.argmin(out4 > 0.00001)



    num_neurons= activation['study_layer'].shape[1]
    model.load_state_dict(weights)

    return num_neurons, Kernel, Stride, Padding






class LayerData(object):
    """This class contains all the information related with the
    layers already evaluated.

    Arguments:
        layer_id: String, name of the layer (This name is the same
            inside of `keras.models.Model` instance)

    Attributes:
        neurons_data: List of `nefesi.neuron_data.NeuronData` instances.
        similarity_index: Non-symmetric matrix containing the result of
            similarity index for each neuron in this layer. When this index is
            calculated, the size of the matrix is len(filters) x len(filters).
        receptive_field_map: Matrix of integer tuples with size equal
            to map activation shape of this layer. Each position i, j from
            the matrix contains a tuple with four values: row_ini, row_fin,
            col_ini, col_fin. This values represents the window of receptive
            field from the input image that provokes the activation there is
            in the location i, j of the map activation.
        receptive_field_size: Tuple of two integers. Size of receptive field
            of the input image in this layer.
    """

    def __init__(self, layer_name,decod=True):
        self.layer_id = layer_name
        self.neurons_data = None
        self.receptive_field_Kernel = None
        self.receptive_field_Stride = None
        self.receptive_field_Padding = None
        self.decoder = decod

    def get_receptive_field(self,model,input_shape,main_input=None,dataset=None):

        if self.decoder:

            n,K,S,P = decoder_rf(model.pytorchmodel, input_shape, self.layer_id, main_input=main_input)


        else:

            n,K,S,P = rf_stride_pad_to_layer(model.pytorchmodel,self.layer_id,input_shape)

        self.neurons_data = n
        self.receptive_field_Kernel = K
        self.receptive_field_Stride = S
        self.receptive_field_Padding = P


    def set_max_activations(self):
        print(self.layer_id)
        for f in self.neurons_data:

            f.set_max_activations()
            if len(f.images_id) > 1:
                f.mean_activation /= f.images_analyzed
                f.mean_norm_activation = f.mean_activation/f.activations[0]



    def sort_neuron_data(self):
        for neuron in self.neurons_data:
            neuron.sortResults(reduce_data=True)


    def get_similarity_idx(self, model=None, dataset=None, neurons_idx=None, verbose = True):
        """Returns the similarity index matrix for this layer.
        If `neurons_idx` is not None, returns a subset of the similarity
        matrix where `neurons_idx` is the neuron index of the neurons returned
        within that subset.

        :param model: The `keras.models.Model` instance.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.
        :param neurons_idx: List of integer. Neuron indexes in the attribute
            class `filters`.

        :return: Non-symmetric matrix of floats. Each position i, j in the matrix
            corresponds to the distance between the neuron with index i and neuron
            with index j, in the attribute class `filters`.
        """




        if self.similarity_index is not None:
            if neurons_idx is None:
                return self.similarity_index
            else:
                size_new_sim = len(neurons_idx)
                new_sim = np.zeros((size_new_sim, size_new_sim))
                for i in range(size_new_sim):
                    idx1 = neurons_idx[i]
                    for j in range(size_new_sim):
                        new_sim[i, j] = self.similarity_index[idx1, neurons_idx[j]]
                return new_sim
        else:
            size = len(self.neurons_data)
            self.similarity_index = np.zeros((size, size))

            idx = range(size)

            for i in idx:

                sim_idx = get_row_of_similarity_index(self.neurons_data[i],  model, self.layer_id, dataset)
                self.similarity_index[i,:] = sim_idx
                if verbose:
                    print("Similarity "+self.layer_id+' '+str(i)+'/'+str(size))
            return self.similarity_index



