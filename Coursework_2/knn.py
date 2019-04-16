from __future__ import print_function # to avoid issues between Python 2 and 3 printing

import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
#from voronoi import plot_voronoi
from utilities import load_data

train_set, train_labels, test_labels, test_set = load_data()

class_1_colour = r'#3366ff'
class_2_colour = r'#cc3300'
class_3_colour = r'#ffc34d'

class_colours = [class_1_colour, class_2_colour, class_3_colour]


colours = np.zeros_like(train_labels, dtype=np.object)
colours[train_labels == 1] = class_1_colour
colours[train_labels == 2] = class_2_colour
colours[train_labels == 3] = class_3_colour


#will use feature pair 10 and 13 for now but not yet finalised

#####calculating distance/similarity between desired wine and every other wine
def euclideanDistance(wineNo):
    distance = 0
    features = (9,12)
    wine1=train_set[wineNo-1:wineNo].astype(np.float)
    wine2=train_set[0:125].astype(np.float)
    for x in  features:
            distance += np.power(wine1[:,x]-wine2[:,x], 2)
    d1 = np.sqrt(distance)
    allDistances = np.delete(d1, wineNo-1)
    return allDistances


allDistances = euclideanDistance(4)

#gets k nearest neightbours
def nearestNeighbours(wineNo, k):
    neighbours = []
    distances = euclideanDistance(wineNo)
    for x in range(0,k):
        minimum = np.amin(distances)
        neighbours = np.append(neighbours, minimum)
        index = np.argwhere(distances == minimum)
        distances = np.delete(distances, index)
    return neighbours

neighbours = nearestNeighbours(10, 1)
print(neighbours)
