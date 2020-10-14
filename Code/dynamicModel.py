#Import modules
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pandas import DatetimeIndex
import dask
import scipy
import time
import glob

from functools import partial
from abc import ABCMeta, abstractmethod

import plottingTools
import factorialModel

#Implementation of stationary kalman filter
#Z_t = A Z_{t-1} + B U_t + Noise_1
#Y_t = C Z_t + D U_t + Noise_2
#Z continuous latent variable identified as the factors returned by the encoder
#Y observed output variable identified as the whole volatility surface
#U Exogenous variable identified as the forward swap rate

#Calibration
#Assumes that we observe Z in the training set thanks to the compression then
#A and B are estimated through a simple least square regression.
#C and D are estimated similarly.

#Prediction

#Filtering

#Smoothing

#A posteriori prediction of Z

#Entropy for prediction of Z

#Completion is initialized with the kalman filter prediction
#Then we penalize the objective function with the 1 - probability of calibrated Z given Y_{1:t-1} and U_{1:t}


#This kind of models tries to fit the dynamics of latent variables
#These variables are typically observed during compression step
#Observation (output) variables are never predicted 
#and we let another learning system (autoencoder) recover outputs from latent factors
class dynamicModel:
    def __init__(self,
                 hyperParameters, 
                 modelName = "./bestDynamicModel"):
        self.hyperParameters = hyperParameters
        self.modelName = modelName
        self.encodingModel = None
        
        
    def buildModel(self):
        return
    
    def buildArchitecture(self):
        return
    
    def setEncodingModel(self, encodingModel):
        self.encodingModel = encodingModel
        return
    ################################################################################
    #Model Calibration
    ################################################################################
    
    #Observing variables should allow for easy and direct estimation of models parameters
    def trainWithObservedFactors(self, 
                                 observationVariable, 
                                 exogenousVariables, 
                                 observedLatentVariables):
        raise NotImplementedError("Abstract class")
        return
    
    #Estimate model parameters (A,B,C,D,Q,R) and infer latent variables
    def trainWithoutObservedFactors(self, 
                                    observationVariable, 
                                    exogenousVariables):
        #In that case estimation is based on Blocked Gibbs Sampling Algorithm
        #The idea of these methods is to have an initial guess of z_t through a PCA like method.
        raise NotImplementedError("Abstract class")
        return
    
    #Datasetlist : list containing [observationVariable, coordinestes, exogenousVariables, ...]
    #observationVariable : serie of variables whose latest observation correspond to yesterday
    #exogenousVariables : additional variables with the same shape and chronology as observationVariable
    #Coordinates : serie of coordinates for each point in observationVariable and exogenousVariables
    #observedLatentVariables : Latent factors typically obtained during compression step
    #Return the inferred (or smoothed if observedLatentVariables is provided) serie for (z_t)_{t in [0,T]}
    #and the smoothed serie for (y_t)_{t in [0,T]} 
    def train(self, dataSetList, 
              observedLatentVariables = None):
        if observedLatentVariables is None :
            return self.trainWithoutObservedFactors(dataSetList[0], 
                                                    dataSetList[1])
        else :
            self.trainWithObservedFactors(dataSetList[0], 
                                          dataSetList[1],
                                          observedLatentVariables)
        return
        
        
    ################################################################################
    #Prediction methods
    ################################################################################
    
    #Predict values for latent factors at time t (i.e. z_t)
    #pastObservedVariables is the serie of output variables until time t-1
    #presentExogenousVariables is the serie of output variables until time t
    #pastLatentVariables is the serie of output variables until time t-1
    def predictNextObservation(self, 
                               pastObservedVariables, 
                               presentExogenousVariables, 
                               pastLatentVariables):
        raise NotImplementedError("Abstract class")
        return []
    
    #Return the log density of observing at time t currentLatentVariables as values for latent factors 
    #with respect to an a priori probabily 
    #It can be used to assess how likely is z_t
    #i.e. P(z_t | z_{1:t-1}, y_{1:t-1}, u_{1:t})
    #currentLatentVariables is the postulated value for z_t
    #pastObservedVariables is the serie of output variables until time t-1
    #presentExogenousVariables is the serie of output variables until time t
    #pastLatentVariables is the serie of output variables until time t-1
    def entropyOfInferredValue(self, 
                               currentLatentVariables,
                               pastObservedVariables, 
                               presentExogenousVariables, 
                               pastLatentVariables):
        raise NotImplementedError("Abstract class")
        return 0
    
    #Return a tensor to execute the same task as entropyOfInferredValue
    def buildPenalizationFactoriesAlongHistory(self, 
                                               presentDataSetList, 
                                               presentLatentVariables):
        raise NotImplementedError("Abstract class")
        return 0
    
    
    #Return a tensor to execute the same task as buildPenalizationFactoriesAlongHistory but only for latest date
    def buildPenalizationFactory(self, 
                                 presentDataSetList, 
                                 presentLatentVariables):
        raise NotImplementedError("Abstract class")
        return 0
    
    ################################################################################
    #Filtering methods
    ################################################################################
    
    #Filter values for latent factors at time t (i.e. z_t) : same as predictNextObservation *
    #except that output variables y is observed until t
    #presentObservedVariables is the serie of output variables until time t
    #presentExogenousVariables is the serie of output variables until time t
    #pastLatentVariables is the serie of output variables until time t-1
    def filterNextObservation(self, 
                              presentObservedVariables, 
                              presentExogenousVariables, 
                              pastLatentVariables):
        raise NotImplementedError("Abstract class")
        return  []
    
    #Return the log density of observing at time t currentLatentVariables as latent factors values
    #with respect to an a posteriori probability 
    #It can be used to assess how likely is z_t
    #i.e. P(z_t | z_{1:t-1}, y_{1:t-1}, u_{1:t})
    #currentLatentVariables is the postulated value for z_t
    #presentObservedVariables is the serie of output variables until time t
    #presentExogenousVariables is the serie of output variables until time t
    #pastLatentVariables is the serie of output variables until time t-1
    def entropyOfFilteredValue(self, 
                               currentLatentVariables,
                               presentObservedVariables, 
                               presentExogenousVariables, 
                               pastLatentVariables):
        raise NotImplementedError("Abstract class")
        return 0
    
    
    ################################################################################
    #Smoothing functionalities
    ################################################################################
    
        
    #Return smoothed serie of latent variables (z_t)_{t in [0,T]} and observed variables 
    #pastAndFutureObservedVariables is the serie of output variables in [0,T]
    #pastAndFutureExogenousVariables is the serie of output variables in [0,T]
    #initialGuessLatentVariables is the serie of latent variables obtained before smoothing in [0,T]
    def smoothedLatentVariables(self, 
                                pastAndFutureObservedVariables, 
                                pastAndFutureExogenousVariables, 
                                initialGuessLatentVariables):
        raise NotImplementedError("Abstract class")
        return {}
    
    
    ################################################################################
    #Observation loglikelyhood
    ################################################################################
    
    
    #Return the log likelyhood of observed variables 
    #with respect to some exogenous variables and model parameters (A,B,C,D,Q,R)
    #presentObservedVariables is the serie of output variables until time t
    #presentExogenousVariables is the serie of output variables until time t
    def computeSerieLogLikelyhood(self, 
                                  presentObservedVariables, 
                                  presentExogenousVariables = None):
        raise NotImplementedError("Abstract class")
        return  0
        
    def saveModel(self):
        raise NotImplementedError("Abstract class")
        return
    
    def restoreModel(self):
        raise NotImplementedError("Abstract class")
        return
    
        

    
    
    
    
            