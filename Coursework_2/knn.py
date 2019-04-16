from __future__ import print_function # to avoid issues between Python 2 and 3 printing

import numpy as np
import matplotlib.pyplot as plt

from statistics import mode
from pprint import pprint
#from voronoi import plot_voronoi
from utilities import load_data

train_set, train_labels, test_set, test_labels = load_data()

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
    wine1=test_set[wineNo-1:wineNo].astype(np.float)
    wine2=train_set[0:125].astype(np.float)
    for x in  features:
            distance += np.power(wine1[:,x]-wine2[:,x], 2)
    allDistances = np.delete(np.sqrt(distance), wineNo-1)
    return allDistances

#gets k nearest neighbours
def nearestNeighbours(wineNo, k):
    neighbours = []
    distances = euclideanDistance(wineNo)
    for x in range(0,k):
        neighbours = np.append(neighbours, np.amin(distances))
        distances = np.delete(distances, (np.argwhere(distances == np.amin(distances))))
    return neighbours

#assigns class to wine in question
def classify(wineNo, k):
    distances = euclideanDistance(wineNo)
    neighbours = nearestNeighbours(wineNo, k)
    classes = []
    for i in range(k):
        iclass = np.asscalar(train_labels[np.argwhere(distances == neighbours.item(i))])
        classes = np.append(classes, iclass)
        wineClass = mode(classes)
    return wineClass

def knn(k):
    predicted = []
    for i in range(1,54):
        predicted = np.append(predicted, classify(i, k))
    return predicted

predicted = knn(1)
np.savetxt('results.csv', predicted, delimiter=',', fmt='%d')

###########ACCURACY#######################

def calculate_accuracy(gt_labels, pred_labels):
    total = pred_labels.size
    totalWrong = 0
    for i in range(0,total):
        if (pred_labels.item(i) != gt_labels.item(i)):
            totalWrong += 1
    accuracy = ((total-totalWrong)/total)*100

    return str(accuracy)


accuracy = calculate_accuracy(test_labels, knn(1))
print(accuracy + '%')
