import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt      
import random   
import pickle  
from sklearn.cluster import KMeans
import sklearn.cluster.k_means_             
from keras.datasets import mnist
from keras.utils import np_utils      
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation


class VariantRHC:
    def __init__(self,datasetName="MNIST",variantName="KFarthest",loadFromStoredReducedSet=False):
        self.datasetName = datasetName
        self.variantName = variantName
        self.loadFromStoredReducedSet = loadFromStoredReducedSet
        if self.loadFromStoredReducedSet==True:
            self.reducedData, done = self.loadFromPickle()
            if done:
                return
            else:
                print("No dataset found in pickle for your variant and dataset. Generating reduced dataset")
        if variantName == "Centroid":
            if datasetName == "MNIST":
                X_train,X_test,Y_train,Y_test = VariantRHC.getMnistData()
                self.reducedData = self.Centroid(X_train,X_test,Y_train,Y_test)
                
            
    def loadFromPickle(self):
        try:
            dbfile = open('{}_{}.pickle'.format(self.variantName,self.datasetName), 'rb')      
        except:
            return None, False
        db = pickle.load(dbfile)
        return db,True

    @staticmethod
    def getL2NormDistnce(v1,v2):
        distance = np.linalg.norm(v1-v2)
        return distance

    @staticmethod
    def getClusterDataPoints(listOfClusterCenters,dataX):
        uniqueClusterDataX = {}
        for clusterCenter in listOfClusterCenters:
            uniqueClusterDataX[tuple(clusterCenter)] = []
        for dataPoint in dataX:
            centerIntial = listOfClusterCenters[0]
            minimumDistance = 10**10
            for centerPoint in listOfClusterCenters:
                distance = VariantRHC.getL2NormDistnce(centerPoint,dataPoint)
                if distance<minimumDistance:
                    minimumDistance = distance
                    centerIntial = centerPoint
            uniqueClusterDataX[tuple(centerIntial)].append(dataPoint)
        return uniqueClusterDataX

    def getMnistData():
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784).astype('float32') # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
        X_test = X_test.reshape(10000, 784).astype('float32')   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.
        X_train /= 255                        # normalize each value for each pixel for the entire vector for each input
        X_test /= 255
        uniqueClasses = 10
        Y_train = y_train
        Y_test = np_utils.to_categorical(y_test, uniqueClasses)
        return X_train,X_test,Y_train,Y_test

    def Centroid(self,X_train,X_test,Y_train,Y_test):
        imageList = {}
        CondensedSet = []
        for i in range(len(X_train)):
            imageList[tuple(X_train[i])] = Y_train[i]  # image label mapping
            imagesAll = []
            imagesAll.append(X_train)     # images enqueue
        while len(imagesAll)>0:
            cImages = imagesAll.pop(0)      # pop front
            checkForHomogenousLabels = []           # find if homogenous
            for img in cImages:    
                checkForHomogenousLabels.append(imageList[tuple(img)])
            if len(set(checkForHomogenousLabels))==1:
                meanVector = np.zeros(cImages[0].shape)
                for j in cImages:
                    meanVector+=j
                meanVector = meanVector/len(cImages)
                CondensedSet.append([meanVector,checkForHomogenousLabels[0]]) # put mean vector of C into CS 
            else:
                classCentroids = []   # all centroids
                uniqueCluster = {}           # unique
                for iLabel in range(10):   
                    uniqueCluster[iLabel] = []
                for i in cImages:           # separate on basis of labels
                    uniqueCluster[imageList[tuple(i)]].append(i)  
                for i in uniqueCluster.keys(): # find centroids of all classes
                    if uniqueCluster[i]:
                        meanVector = np.zeros(uniqueCluster[i][0].shape)
                        for j in uniqueCluster[i]:
                            meanVector+=j
                        classCentroids.append(meanVector/len(uniqueCluster[i]))
                classCentroids = np.array(classCentroids)
                clusters = KMeans(n_clusters=len(classCentroids), init=classCentroids, n_init=1)
                clusters.fit(np.array(cImages))
                setOfClusters = VariantRHC.getClusterDataPoints(clusters.cluster_centers_,cImages)
                for i in setOfClusters.keys():
                    imagesAll.append(setOfClusters[i])
        return CondensedSet

    def saveAsPickle(self):        
        dbfile = open('{}_{}.pickle'.format(self.variantName,self.datasetName), 'wb') 
        pickle.dump(self.reducedData, dbfile)                      
        dbfile.close() 
