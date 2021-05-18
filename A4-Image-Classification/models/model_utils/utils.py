# Utility functions 

# Import packages
import numpy as np # Matrix maths
import pandas as pd
import tensorflow as tf # NN functions
import matplotlib.pyplot as plt # For drawing graph
from tensorflow.keras.models import Sequential # ANN Architecture
from tensorflow.keras.layers import Dense # The layers in the ANN
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import keras
import os
from pathlib import Path

# Function obtained from: https://gist.github.com/craffel/2d727968c3aaebd10359
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

# Function for plotting model performance coeffiecnts
def plot_coefs(coefficients, nclasses, name):
    """
    Plot the coefficients for each label
    
    coefficients: output from clf.coef_
    nclasses: total number of possible classes
    """

    title = {'family': 'serif',
            'color': '#4A5B65',
            'weight': 'normal',
            'size': 22,
            }

    scale = np.max(np.abs(coefficients))

    p = plt.figure(figsize=(20, 10))

    for i in range(nclasses):
        p = plt.subplot(2, int(nclasses/2), i + 1)
        p = plt.imshow(coefficients[i].reshape(28, 28),
                      cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
        p = plt.axis('off')
        p = plt.title('Class %i' % i, fontdict = title)
    plt.savefig(name)

    