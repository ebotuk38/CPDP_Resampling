import math
import operator
import numpy as np
import pandas as pd

class nnfilter():

   def __init__(self):
     self.k = 5

   def euclideanDistance(self, instance1, instance2, length):
    distance = 0
    for x in range(length):
     distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

   def getNeighbors(self, trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) #-1
    for x in range(len(trainingSet)):
        dist = self.euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
      neighbors.append(distances[x][0])
    return neighbors


    #start nn-filter
   def filter(self, X, y):
    trainingSet = X.values
    testInstance = y.values
    #save column name
    coname = list(X.columns.values)
    fdat = pd.DataFrame()
    for z in range(len(testInstance)):
      fildat = self.getNeighbors(trainingSet, testInstance[z], 5)
      fdat= fdat.append(fildat)
    #rename colname
    fdat.columns = fdat.columns[:0].tolist() + coname

    fdat.drop_duplicates(keep=False, inplace=True)
    return fdat

#dataframe.columnName.round(-1)
