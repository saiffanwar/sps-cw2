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
#plt.scatter(train_set[:,9], train_set[:,12], c=colours)
#plt.show()

#####calculating distance/similarity
def euclideanDistance(wine1):
    distance = 0
    features = (9,12)
    for x in  features:
            distance += np.power(wine1[:,x]-wine2[:,x], 2)
#    print(np.sqrt(distance))
    return np.sqrt(distance)

wine1=train_set[:1].astype(np.float)
wine2=train_set.astype(np.float)
print(wine2)
euclideanDistance(wine1)

#print(train_set[:2])
