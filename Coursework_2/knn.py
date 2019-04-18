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


###########################################################
def feature_selection(train_set, train_labels, f, x, y):
    if f == 3:
        selected_features =[9,10,13]
    else:
        selected_features = [x,y]
    return selected_features
###########all functions required for knn###########################
def euclideanDistance(train_set,test_set, wineNo, f, x, y):
    selected_features = np.array(feature_selection(train_set, train_labels, f, x, y))-1
    wine1=test_set[wineNo-1:wineNo].astype(np.float)
    allDistances = []
    for y in range(0,125):
        distance = 0
        for x in selected_features:
            wine2 = train_set[y,x].astype(np.float)
            distance += np.power(wine1[:,x]-wine2, 2)
        allDistances = np.append(allDistances, distance )
    return allDistances


#gets k nearest neighbours
def nearestNeighbours(train_set, test_set, wineNo, k, f, x, y):
    neighbours = []
    distances = euclideanDistance(train_set, test_set, wineNo, f, x, y)
    for x in range(0,k):
        neighbours = np.append(neighbours, np.amin(distances))
        distances = np.delete(distances, (np.argwhere(distances == np.amin(distances))))
    return neighbours

#assigns class to wine in question
def classify(train_set, train_labels, test_set, wineNo, k, f, x, y):
    distances = euclideanDistance(train_set, test_set, wineNo, f, x, y)
    neighbours = nearestNeighbours(train_set, test_set, wineNo, k, f, x, y)
    classes = []
    for i in range(k):
        # if there are multiple neighbours at same distance away then pick first one
        iclass = train_labels[np.argwhere(distances == (neighbours.item(i))).item(0)]
        classes = np.append(classes, iclass)
    (values,counts) = np.unique(classes,return_counts=True)
    ind=np.argmax(counts)
    wineClass = values[ind]

    return wineClass

    ###########ACCURACY#######################

def calculate_accuracy(gt_labels, pred_labels):
    total = pred_labels.size
    totalWrong = 0
    for i in range(0,total):
        if (pred_labels.item(i) != gt_labels.item(i)):
            totalWrong += 1
    accuracy = str(((total-totalWrong)/total)*100)
#    print(accuracy + '%')
    return accuracy

######calculates predictions for all wines of test_set
def knn(train_set, train_labels, test_set, k, x, y):
    f=2
    predictions = []
    for i in range(1,54):
        predictions = np.append(predictions, classify(train_set, train_labels, test_set, i, k, f, x, y))
        accuracy = calculate_accuracy(test_labels, predictions)
    return accuracy, predictions

for x in range(1,14):
    for y in range(1,14):
        accuracy, predictions = knn(train_set, train_labels, test_set, 1, x, y)
        print([x,y])
        print(accuracy)

#np.savetxt('results.csv', predictions, delimiter=',', fmt='%d')
