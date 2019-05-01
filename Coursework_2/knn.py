from __future__ import print_function # to avoid issues between Python 2 and 3 printing

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import mode
from pprint import pprint
#from voronoi import plot_voronoi
from utilities import load_data
from sklearn.decomposition import PCA
train_set, train_labels, test_set, test_labels = load_data()
print(test_labels)

class_1_colour = r'#3366ff'
class_2_colour = r'#cc3300'
class_3_colour = r'#ffc34d'

class_colours = [class_1_colour, class_2_colour, class_3_colour]
classes = np.unique(train_labels)

colours = np.zeros_like(train_labels, dtype=np.object)
colours[train_labels == 1] = class_1_colour
colours[train_labels == 2] = class_2_colour
colours[train_labels == 3] = class_3_colour


###########################################################
def feature_selection(train_set, train_labels, f,):
    if f == 3:
        selected_features =[2,7,10]

    if f == 2:
            selected_features = [7,10]
    if f == 0:
        selected_features = [1]
    return selected_features
###########all functions required for knn###########################
def euclideanDistance(train_set,test_set, wineNo, f):
    selected_features = np.array(feature_selection(train_set, train_labels, f))-1
    wine1=test_set[wineNo-1:wineNo].astype(np.float)
    allDistances = []
    for y in range(0,125):
        distance = 0
        for x in selected_features:
            wine2 = train_set[y,x].astype(np.float)
            distance += np.power(wine1[:,x]-wine2, 2)
        allDistances = np.append(allDistances, distance)
    return allDistances


#gets k nearest neighbours
def nearestNeighbours(train_set, test_set, wineNo, k, f):
    neighbours = []
    distances = euclideanDistance(train_set, test_set, wineNo, f)
    for x in range(0,k):
        neighbours = np.append(neighbours, np.amin(distances))
        distances = np.delete(distances, (np.argwhere(distances == np.amin(distances))))
    return neighbours

#assigns class to wine in question
def classify(train_set, train_labels, test_set, wineNo, k, f):
    distances = euclideanDistance(train_set, test_set, wineNo, f)
    neighbours = nearestNeighbours(train_set, test_set, wineNo, k, f)
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
    accuracy = ((total-totalWrong)/total)*100
#    print(accuracy + '%')
    return accuracy

######calculates predictions for all wines of test_set
def knn(train_set, train_labels, test_set, k):
    f=2
    predictions = []
    for i in range(1,54):
        predictions = np.append(predictions, classify(train_set, train_labels, test_set, i, k, f))
        accuracy = calculate_accuracy(test_labels, predictions)
    return accuracy, predictions

def knn3d(train_set, train_labels, test_set, k):
    f=3
    predictions = []
    for i in range(1,54):
        predictions = np.append(predictions, classify(train_set, train_labels, test_set, i, k, f))
        accuracy = calculate_accuracy(test_labels, predictions)
    return accuracy, predictions


#3d plot
def plot3d():
    fig =plt.figure()
    ax = Axes3D(fig)
    plot = ax.scatter(train_set[:,0], train_set[:,1], train_set[:,6], c=colours)
    #ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 7")
    plt.show()
    return plot

#plot3d()


# def knnaccuracy():
#    accuracies= []
#    features = []
#    for x in range(1,14):
#        for y in range(1,14):
#            for z in range(1,14):
#                accuracy, predictions = knn(train_set, train_labels, test_set, 1, f)
#                accuracies = np.append(accuracies, accuracy)
#
#                print((x,y,z))
#                print(accuracy)
#    maxaccuracy = np.amax(accuracies)
#    print(accuracies)
#    maxfeature = np.argwhere(accuracies == maxaccuracy)
#    #returns index of features that give max accuracy
#    print(maxfeature)
#    return maxfeature, maxaccuracy, accuracies, features

#accuracy, predictions = knn(train_set, train_labels, test_set, 1, f)
#print(accuracy, predictions)

######################PCA##########################################
def pca_model(n_components=2):
    pca = PCA(n_components)
    pca.fit(train_set)
    scipy_train_transformed = pca.transform(train_set)
    scipy_test = pca.transform(test_set)
    for index_colour, b in enumerate(classes):
        index = train_labels == b
        pca_x = scipy_train_transformed[index,0]
        pca_y = scipy_train_transformed[index,1]
        plt.scatter(pca_x,pca_y * -1, c = class_colours[index_colour])
    plt.show()
    return scipy_train_transformed, scipy_test


def knn_pca(train_labels, k):
    f=0
    predictions = []
    train_set, test_set = pca_model(2)
    for i in range(1,54):
        predictions = np.append(predictions, classify(train_set, train_labels, test_set, i, k, f))
        accuracy = calculate_accuracy(test_labels, predictions)
    return accuracy, predictions

accuracy, predictions = knn(train_set, train_labels, test_set, 1)
print(accuracy)
