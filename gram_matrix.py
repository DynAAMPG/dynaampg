import torch
import numpy as np
from scipy.ndimage import zoom
from matplotlib import colormaps
import matplotlib
import matplotlib.pyplot as plt
import json
from enum import Enum

from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.functional import normalize

class GRAM_TYPE(Enum):
    STANDARD_GRAM=0,
    HIGH_ORDER_GRAM=1,
    PENALIZED_GRAM=2



def gram(f, triang='upper'):
    gram_matrix = torch.mm(f, f.t())

    if triang=='upper':
        gram_matrix = torch.triu(gram_matrix)
    elif triang=='lower':
        gram_matrix = torch.tril(gram_matrix)

    return gram_matrix

def highorder_gram(f, order=1, triang='upper'):
    f = torch.pow(f, order)
    gram_matrix = torch.mm(f, f.t())
    gram_matrix = torch.pow(gram_matrix, 1/order)

    if triang=='upper':
        gram_matrix = torch.triu(gram_matrix)
    elif triang=='lower':
        gram_matrix = torch.tril(gram_matrix)

    return gram_matrix

def penalized_gram(f, margins, triang='upper'):
    gram_matrix = torch.mm(f, f.t())
    gram_matrix = normalize(gram_matrix, p=2.0, dim = 0)

    gram_matrices = []
    for m in margins:
        gram_matrix = gram_matrix * (2 * torch.pi / m)

        if triang=='upper':
            gram_matrix = torch.triu(gram_matrix)
        elif triang=='lower':
            gram_matrix = torch.tril(gram_matrix)
    
        gram_matrices.append(gram_matrix)

    return gram_matrices


def get_square_tensor(source_tesnor):
    r, c = source_tesnor.size()
    zeros = torch.zeros([c-r, c])
    zeros = zeros.cuda()
    destination_tensor = torch.cat([source_tesnor, zeros], dim=0)

    return destination_tensor


def interpolate_array(array, new_size):

    h, w = array.shape
    h_new, w_new = new_size

    # Calculate the zoom factors for height and width
    zoom_factors = (h_new / h, w_new / w)

    # Use scipy's zoom function for bilinear interpolation
    interpolated_array = zoom(array, zoom_factors, order=1)

    return interpolated_array


def get_tab20_cmap():
    black = (0.0, 0.0, 0.0, 1.)
    colors = [plt.cm.tab20(c) for c in range(1,20)]
    colors.insert(0, black)
    cmap=matplotlib.colors.ListedColormap(colors)

    return cmap

def calculate_gram_matrices(features, triang='upper'):
    processed_gram_matrices = {}
    
    layer_names = list(features.keys())    

    for layer_index, layer_name in enumerate(layer_names):
        new_dim = np.max(list(features[layer_name].size()))
        f = interpolate_array(features[layer_name].cpu().numpy(), (new_dim,new_dim))
        f = torch.tensor(f)

        gram_matrix = torch.mm(f, f.t())

        if triang=='upper':
            gram_matrix = torch.triu(gram_matrix)
        elif triang=='lower':
            gram_matrix = torch.tril(gram_matrix)

        gram_matrix = normalize(gram_matrix, p=2.0, dim = 0)
        gram_matrix = gram_matrix.cpu()

        # dim = np.min(list(features[layer_name].size()))
        # gram_matrix = gram_matrix[0:dim, 0:dim]
                
        processed_gram_matrices[layer_name] = torch.tensor(np.array(gram_matrix))


    return processed_gram_matrices


def plot_features(features):
    num_features = len(features)
    num_rows = 1
    num_cols = num_features
    layer_names = list(features.keys())
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
    
    for f_index, layer_name in enumerate(layer_names):
        ax = axs[f_index]
        ax.set_title(f"Feature Matrix for {layer_name}")
        im = ax.imshow(features[layer_name].cpu().numpy(), cmap='coolwarm')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
    plt.savefig('visualization/features.png')
    plt.show()



def plot_all_gram_matrices(data, fontsize=28, cmap='coolwarm'):
    num_rows = len(data)
    num_cols = len(data[0]['matrix'])

        
    fig, axes = plt.subplots(num_rows, num_cols, figsize=((6 * num_cols) + 2, 6 * num_rows))

    for matrix_index in range(0, num_rows):
        matrix = data[matrix_index]['matrix']
        num_grams = len(matrix)
        layer_names = list(matrix.keys())
        
        for g_index, layer_name in enumerate(layer_names):
            ax = axes[matrix_index, g_index]

            if not data[matrix_index]['deviations'] is None:
                dev = data[matrix_index]['deviations'][g_index]
                ax.text(260,40, f'$\delta_{g_index + 1} = {dev} $', fontsize=fontsize + 2, ha='left', va='top', color='white', bbox = dict(facecolor= 'yellow', alpha = 0.5))
            
            if not data[matrix_index]['tot_dev'] is None and g_index==len(layer_names)-1:
                tot_dev = data[matrix_index]['tot_dev']
                ax.text(650,256, f'$\Delta = {tot_dev} $', fontsize=fontsize + 2, ha='center', va='center', color='black', rotation=90)


            ax.set_title(f"{data[matrix_index]['desc']} $l_{{{g_index + 1}}}$", fontsize=fontsize, y=1.04)
            im = ax.imshow(data[matrix_index]['matrix'][layer_name].cpu().numpy(), cmap=cmap)



            if g_index==0:
                ax.set_ylabel(f"{data[matrix_index]['ylabel']}", rotation=90, fontsize=fontsize)

            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=fontsize) 
            plt.tight_layout()
                
    plt.savefig('visualization/fig_gram_matrices.pdf')
    plt.show()



def get_deviation(mean_grams, std_grams, sample_gram, alphas):
    dev_layerwise = []
    dev_total = 0.0

    for layer_index, layer_name in enumerate(sample_gram.keys()):
        mean = mean_grams[layer_name]
        std = std_grams[layer_name]
        sample = sample_gram[layer_name]
        dim1, dim2 = sample.size()

        diff = torch.sqrt(torch.square(mean - sample))
        diff = torch.sum(diff)
        dev = diff / (dim1 * dim2)

        dev_layerwise.append((alphas[layer_index] * dev.tolist()))

    return dev_layerwise, np.sum(dev_layerwise)




def save_dev_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_dev_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)