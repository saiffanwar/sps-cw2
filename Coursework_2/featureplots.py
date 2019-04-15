from __future__ import print_function # to avoid issues between Python 2 and 3 printing

import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
#from voronoi import plot_voronoi
from utilities import load_data
# show matplotlib figures inline
#%matplotlib inline
# By default we set figures to be 12"x8" on a 110 dots per inch (DPI) screen
# (adjust DPI if you have a high res screen!)
plt.rc('figure', figsize=(12, 8), dpi=110)
plt.rc('font', size=12)


def load_data(train_set_path='data/wine_train.csv',
              train_labels_path='data/wine_train_labels.csv',
              test_set_path='data/wine_test.csv',
              test_labels_path='data/wine_test_labels.csv'):
    """
    Loads the wine dataset. If no arguments are passed it will try to load the data
    from the working directory with the default file names

    Args:
        train_set_path : path to the train set .csv file
        train_labels_path : path to the train labels .csv file
        test_set_path : path to the test set .csv file
        test_labels_path : path to the testlabels .csv file
    Returns:
        (train_set, train_labels, test_set, test_labels), numpy arrays containing the
        training and testing sets, along with the respective class labels
    """

    train_set = np.loadtxt(train_set_path, delimiter=',', #dtype = np.float
    )
    train_labels = np.loadtxt(train_labels_path, delimiter=',', dtype=np.int)
    test_set = np.loadtxt(test_set_path, delimiter=',', dtype=np.float)
    test_labels = np.loadtxt(test_labels_path, delimiter=',', dtype=np.int)

    return train_set, train_labels, test_set, test_labels

train_set, train_labels, test_labels, test_set = load_data()
train_set1 = train_set.astype(np.float)
print(train_set1)
n_features = (13)
fig, ax = plt.subplots(n_features, n_features)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

class_1_colour = r'#3366ff'
class_2_colour = r'#cc3300'
class_3_colour = r'#ffc34d'

class_colours = [class_1_colour, class_2_colour, class_3_colour]


colours = np.zeros_like(train_labels, dtype=np.object)
colours[train_labels == 1] = class_1_colour
colours[train_labels == 2] = class_2_colour
colours[train_labels == 3] = class_3_colour


#############################plots all 13x13######################
def subplots(dataset, n, **kwargs):
    for x in range(0,n):
       for y in range(0,n):
           ax[x,y].scatter(dataset[:,x], dataset[:, y], c=colours)

subplots(train_set1, n_features)

plt.show()
