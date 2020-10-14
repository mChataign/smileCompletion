
import pandas as pd
import numpy as np
import tensorflow as tf
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import FunctionalApproach
import plottingTools


class FunctionalModelFwd(FunctionalApproach.FunctionalModel):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestFunctionalModelFwd"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    def buildArchitecture(self):
        
        nbCoordinates = 3
        #Location to interpolate which are common to each day
        
        #Factors values for each day
        self.factorTensor = tf.placeholder(tf.float32, 
                                           shape=[None, self.nbFactors], 
                                           name = "factorTensor")
        #Get the number of days as dynamic shape
        numObs = tf.shape(self.factorTensor)[0]
        
        #Repeat the locations for each day
        self.inputTensor = tf.placeholder(tf.float32, shape=[None, nbCoordinates], name = "inputTensor")
        reshapedInputTensor = tf.reshape(self.inputTensor, 
                                         tf.stack([numObs, -1, nbCoordinates]), 
                                         name = "reshapedInputTensor") #[None, None, 2]
        # reshapedInputTensor = tf.tile(tf.expand_dims(self.inputTensor, 0),
                                      # tf.stack([numObs, 1, 1]))
        
        #Repeat the factors for each location
        surfaceSize = tf.shape(reshapedInputTensor, 
                               name = "surfaceSize")[1]
        reshapedFactorTensor = tf.tile(tf.expand_dims(self.factorTensor, 1),
                                       tf.stack([1, surfaceSize, 1]), 
                                       name = "reshapedFactorTensor")
        
        if self.verbose :
            print(self.inputTensor)
            print(reshapedInputTensor)
        
        #self.factorTensor = tf.Variable(np.ones(shape=(1, self.nbFactors)).astype(np.float32))
        if self.verbose :
            print(self.factorTensor)
            print(reshapedFactorTensor)
        
        #Dynamic Shape of features shoud be [None, surfaceSize, 2 + self.nbFactors] that is [nbDays, surfaceSize, 2 + self.nbFactors]
        features = tf.reshape(tf.concat([reshapedInputTensor, reshapedFactorTensor],2), 
                                        [-1, nbCoordinates + self.nbFactors], 
                                        name = "features")
         # filteredFeatures = tf.boolean_mask(features, 
                                            # tf.logical_not(tf.reduce_any(tf.is_nan(features), axis=1)), 
                                            # name = "filteredFeatures")
         # partitions = tf.cast(tf.logical_not(tf.reduce_any(tf.is_nan(features), axis=1)), tf.int32)
         # filteredFeatures = tf.dynamic_partition(features, 
                                                 # partitions,
                                                 # num_partitions = 2,
                                                 # name = "filteredFeaturesVar")[1]
        filteredFeatures = tf.where(tf.reduce_any(tf.is_nan(features), axis=1), 
                                    tf.zeros_like(features), 
                                    features)
                                    
        #Factors values for each day
        # self.factorTensor = tf.placeholder(tf.float32, 
                                           # shape=[None, self.nbFactors], 
                                           # name = "factorTensor")
        # self.inputTensor = tf.placeholder(tf.float32, shape=[None, nbCoordinates], name = "inputTensor")
        # filteredFeatures = tf.concat([self.factorTensor, self.inputTensor],1 , name = "features")
        # if self.verbose :
            # print(self.inputTensor)
            # print(self.factorTensor)
            # print(filteredFeatures)
        
        
        # if self.verbose :
            # print(features)
            # print(filteredFeatures)
        
        #Build neuronal architecture
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        
        
        
        hidden1 = self.buildDenseLayer(20, 
                                       filteredFeatures,
                                       activation = tf.nn.softplus,
                                       kernelRegularizer = l2_regularizer,
                                       kernelInitializer = he_init)
        if self.verbose :
            print(hidden1)
        hidden2 = self.buildDenseLayer(20, 
                                       hidden1,
                                       activation = tf.nn.softplus,
                                       kernelRegularizer = l2_regularizer,
                                       kernelInitializer = he_init)
        if self.verbose :
            print(hidden2)
        hidden3 = self.buildDenseLayer(1, 
                                       hidden2,
                                       activation = None,
                                       kernelRegularizer = l2_regularizer,
                                       kernelInitializer = he_init)
        if self.verbose :
            print(hidden3)
        
        #Reshape output as (nbDays, surfaceSize)
        self.outputTensor = hidden3#tf.reshape(hidden3, tf.stack([-1, surfaceSize]))
        if self.verbose :
            print(self.outputTensor)
        
        return
    
        
    #Build a tensor that construct a surface from factors values
    def buildReconstructionTensor(self, factorTensor):
        nbCoordinates = 3
        
        #Get the number of days as dynamic shape
        numObs = tf.shape(factorTensor)[0]
        
        #Repeat the locations for each day
        reshapedInputTensor = tf.reshape(self.inputTensor, 
                                         tf.stack([numObs, -1, nbCoordinates]),
                                         name = "reshapedInputTensorVar") 
        # reshapedInputTensor = tf.tile(tf.expand_dims(self.inputTensor, 0),
                                      # tf.stack([numObs, 1, 1]))
        
        #Repeat the factors for each location
        surfaceSize = tf.shape(reshapedInputTensor)[1]
        reshapedFactorTensor = tf.tile(tf.expand_dims(factorTensor, 1),
                                       tf.stack([1, surfaceSize, 1]),
                                       name = "reshapedFactorTensorVar")
        
        if self.verbose :
            print(self.inputTensor)
            print(reshapedInputTensor)
        
        #self.factorTensor = tf.Variable(np.ones(shape=(1, self.nbFactors)).astype(np.float32))
        if self.verbose :
            print(factorTensor)
            print(reshapedFactorTensor)
        
        #Dynamic Shape of features shoud be [None, surfaceSize, 3 + self.nbFactors] that is [nbDays, surfaceSize, 3 + self.nbFactors]
        features = tf.reshape(tf.concat([reshapedInputTensor, reshapedFactorTensor],2), 
                              [-1, nbCoordinates + self.nbFactors],
                              name = "featuresVar")
        self.features = features
        # filteredFeatures = tf.boolean_mask(features, 
                                           # tf.logical_not(tf.reduce_any(tf.is_nan(features), axis=1)),
                                           # name = "filteredFeaturesVar")
        # partitions = tf.cast(tf.logical_not(tf.reduce_any(tf.is_nan(features), axis=1)), tf.int32)
        # filteredFeatures = tf.dynamic_partition(features, 
                                                # partitions,
                                                # num_partitions = 2,
                                                # name = "filteredFeaturesVar")[1]
        filteredFeatures = tf.where(tf.reduce_any(tf.is_nan(features), axis=1), 
                                    tf.zeros_like(features), 
                                    features)
        self.filteredFeatures = filteredFeatures
        if self.verbose :
            print(features)
            print(filteredFeatures)
        
        lastTensor = filteredFeatures
        #Iterate on layers to build the decoder with the same weights as those used for training
        for factory in self.layers :
            lastTensor = factory(lastTensor)
        
        
        reshapedOutputTensor = lastTensor#tf.reshape(lastTensor,[-1, surfaceSize])
        
        self.trainingPred = reshapedOutputTensor
        if self.verbose :
            print(reshapedOutputTensor)
            
        return reshapedOutputTensor
    
    #Extract for each day the volatility value as output values the coordinates as input input values
    def getLocationFromDatasetList(self, dataSet):
        if dataSet[1].ndim > 1 :#historical data
            nbObs = dataSet[1].shape[0]
            nbPoints = dataSet[1].shape[1]
            
            vol = dataSet[0].values if dataSet[0] is not None else dataSet[0]
            
            coordinates = dataSet[1]
            yCoor = np.ravel(coordinates.applymap(lambda x : x[1]))
            xCoor = np.ravel(coordinates.applymap(lambda x : x[0]))
            
            fwd = np.ravel(dataSet[2].values)
            
            l_Feature = np.reshape(np.vstack([xCoor, yCoor, fwd]).T, (nbObs, nbPoints, 3))
        else :#Data for a single day
            nbObs = 1
            nbPoints = dataSet[1].shape[0]
            
            vol = np.expand_dims(dataSet[0].values, 0) if dataSet[0] is not None else dataSet[0]
            
            coordinates = dataSet[1]
            yCoor = np.ravel(coordinates.map(lambda x : x[1]))
            xCoor = np.ravel(coordinates.map(lambda x : x[0]))
            
            fwd = np.ravel(dataSet[2].values)
            
            l_Feature = np.reshape(np.vstack([xCoor, yCoor, fwd]).T, (nbObs, nbPoints, 3))
            
        return l_Feature, vol
    
    def createFeedDictEncoder(self, dataSetList):
        feedDict = {self.inputTensor : np.reshape(dataSetList[0],(-1,3)),
                    self.outputTensorRef : np.reshape(dataSetList[1],(-1,1))}
        return feedDict
    
    
    def createFeedDictDecoder(self, *args):
        feedDict = {self.inputTensor : np.reshape(args[0][0], (-1,3)),
                    self.factorTensor : args[1]}
        return feedDict
    
    def plotInterpolatedSurface(self,
                                valueToInterpolate, 
                                locationToInterpolate,
                                calibratedFactors, 
                                exogenousVariable,
                                colorMapSystem=None, 
                                plotType=None):
        y = list(map( lambda x : x[1] if x is not None else x,
                      locationToInterpolate.values))
        x = list(map( lambda x : x[0] if x is not None else x,
                      locationToInterpolate.values))
        xMax = np.nanmax(x)
        yMax = np.nanmax(y)
        xMin = np.nanmin(x)
        yMin = np.nanmin(y)
        
        xNewValues = np.linspace(xMin,xMax,num=100)
        yNewValues = np.linspace(yMin,yMax,num=100)
        grid = np.meshgrid(xNewValues,yNewValues)
        
        xInterpolated = np.reshape(grid[0],(100 * 100, 1))
        yInterpolated = np.reshape(grid[1],(100 * 100, 1))
        coordinates = np.concatenate([xInterpolated, yInterpolated], axis=1)
        ind = pd.Series(list(zip(xInterpolated,yInterpolated))).rename(locationToInterpolate.name)
        
        
        interpolatedSurface = self.evalSingleDayWithoutCalibrationOnCustomLocation(calibratedFactors, 
                                                                                   [None, ind, exogenousVariable, None])
        
        interpolatedSurfaceDf = pd.Series(interpolatedSurface, index = ind.index)
        
        plottingTools.plotGrid(interpolatedSurfaceDf, 
                               ind,
                               "Interpolated Data with Functional Approach", 
                               colorMapSystem=colorMapSystem, 
                               plotType=plotType)
        
        plottingTools.standardInterp(valueToInterpolate, 
                                     locationToInterpolate,
                                     colorMapSystem=colorMapSystem, 
                                     plotType=plotType)
        return    
    