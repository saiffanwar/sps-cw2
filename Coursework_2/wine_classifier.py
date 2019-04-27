#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
from statistics import mode, StatisticsError
# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

TASKS = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca', 'feature_plots']

train_labels, train_set, test_labels, test_set = load_data()
################################plot features##########################################
train_set, train_labels, test_labels, test_set = load_data()
train_set1 = train_set.astype(np.float)

n_features = (13)
fig, ax = plt.subplots(n_features, n_features)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

class_colours = [CLASS_1_C, CLASS_2_C, CLASS_3_C]
colours = np.zeros_like(train_labels, dtype=np.object)
colours[train_labels == 1] = CLASS_1_C
colours[train_labels == 2] = CLASS_2_C
colours[train_labels == 3] = CLASS_3_C

def subplots(dataset, n, **kwargs):
    for x in range(0,n):
       for y in range(0,n):
           ax[x,y].scatter(dataset[:,x], dataset[:, y], c=colours)
#######################select features between 1-13####################################
def feature_selection(train_set, train_labels, f, **kwargs):
    if f == 3:
        selected_features =[1,7,2]
    else:
        selected_features = [1,7]
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
        allDistances = np.append(allDistances, distance )
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
    try:
        wineClass = mode(classes)
    except StatisticsError:
        wineClass = classify(train_set, train_labels, test_set, wineNo, k-1, f)
    return wineClass


###########ACCURACY#######################

def calculate_accuracy(gt_labels, pred_labels):
    total = pred_labels.size
    totalWrong = 0
    for i in range(0,total):
        if (pred_labels.item(i) != gt_labels.item(i)):
            totalWrong += 1
    accuracy = str(((total-totalWrong)/total)*100)
    print(accuracy + '%')
    return accuracy

######calculates predictions for all wines of test_set
def knn(train_set, train_labels, test_set, k, **kwargs):
    f=2
    predictions = []
    for i in range(1,54):
        predictions = np.append(predictions, classify(train_set, train_labels, test_set, i, k, f))
    accuracy = calculate_accuracy(test_labels, predictions)
    return predictions
#################################################################

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    f=3
    predictions = []
    for i in range(1,54):
        predictions = np.append(predictions, classify(train_set, train_labels, test_set, i, k, f))
    accuracy = calculate_accuracy(test_labels, predictions)
    return predictions


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    def pca_model(n_components=2):
    pca = PCA(n_components)
    return pca
pca = pca_model(2)
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', nargs=1, type=str, help='Running task. Must be one of the following tasks: {}'.format(TASKS))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--f', nargs='?', type=int, default=1, help='Number of features to display for feature_sel')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    task = args.task[0]


    return args, task


if __name__ == '__main__':
    args, task = parse_args() # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if task == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels, args.f)
        print_features(selected_features)
    elif task == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif task == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif task == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif task == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
        #plots
    elif task == 'feature_plots':
        subplots(train_set1, n_features)
        plt.show()
    else:
        raise Exception('Unrecognised task: {}. Possible tasks are: {}'.format(task, TASKS))
