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
import kalmanFilter
import linearModel
from sklearn.linear_model import LinearRegression
import math as m
#import tensorflow_probability as tfp
#tfd = tfp.distributions

#Implementation of stationary extended kalman filter
#Z_t = A Z_{t-1} + B U_t + Noise_1
#Y_t = h(Z_t, U_t) + Noise_2
#Z continuous latent variable identified as the factors returned by the encoder
#Y observed output variable identified as the whole volatility surface
#U Exogenous variable identified as the forward swap rate

#Calibration
#Assumes that we observe Z in the training set thanks to the compression then
#A and B are estimated through a simple least square regression.
#h stands for the decoder and is supposed to be pretrained

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
class ExtendedkalmanFilter(kalmanFilter.kalmanFilter):

    ################################################################################
    #Kalman Initialization
    ################################################################################
    def __init__(self,
                 hyperParameters,
                 modelName = "./bestEKFModel"):
        super().__init__(hyperParameters = hyperParameters,
                         modelName = modelName)
    
    ################################################################################
    #Kalman Calibration
    ################################################################################
    
    #Observing variables should allow for easy and direct estimation of models parameters
    def trainWithObservedFactors(self, 
                                 observationVariable, 
                                 exogenousVariables, 
                                 observedLatentVariables):
                                 
        #Since h function is already learned we only need to compute the residuals 
        #for estimating covarance matrix R
        inputFeatures = exogenousVariables
        if self.encodingModel is None :
            raise NotImplementedError("Extented kalman filter requires a decoding model.")
        dataSetList = [observationVariable, None, exogenousVariables, None]
        prediction = self.encodingModel.evalSingleDayWithoutCalibration(observedLatentVariables.values, dataSetList)
        outputFeatures = observationVariable
        
        residuals = outputFeatures - prediction
        deltaMean = residuals.mean(axis=0)
        self.R = residuals.cov()
        
        #In that case implementation is equivalent to least square linear regression
        #Z_t = A Z_{t-1} + B U_t + Noise_1, Noise_1~N(0,Q)
        inputFeatures2 = pd.concat([observedLatentVariables.shift(), 
                                    exogenousVariables], axis=1).dropna(axis = 0, 
                                                                        how = 'any')
        outputFeatures2 = observedLatentVariables.loc[inputFeatures2.index]
        
        reg2 = LinearRegression(fit_intercept = False)
        reg2.fit(inputFeatures2, outputFeatures2)
        nbLatentVariables = observedLatentVariables.shape[1]
        
        self.B = reg2.coef_[:,nbLatentVariables:]
        self.A = reg2.coef_[:,:nbLatentVariables]
        
        residuals2 = outputFeatures2 - reg2.predict(inputFeatures2)
        epsilonMean = residuals2.mean(axis=0)
        self.Q = residuals2.cov()
        
        self.RInv = np.linalg.inv(self.R)
        return
    
    ################################################################################
    #Kalman Equations
    ################################################################################
    
    #Only update step change with self.C replaced with 
    def updateStep(self, ySpot, 
                   conditionalMu, 
                   conditionalSigma, 
                   uSpot = None):
        #Reshape argument
        
        conditionalSigmaConverted = np.reshape(conditionalSigma,
                                               (len(conditionalMu),len(conditionalMu)))
        
        #update matrixGain
        invConditionalSigma = np.linalg.inv(conditionalSigmaConverted)
        
        #Extended kalman filter replaces Matrix C mu_{t|t-1} with h(mu_{t-1}) and C with H (d D(z)/dz en mu_{t|t-1}) otherwise
        if self.encodingModel is None :
            raise NotImplementedError("Extented kalman filter requires a decoding model.")
        
        
        dataSetList = [ySpot, None, uSpot, None]
        predictedY, H = self.encodingModel.evalSingleDayWithoutCalibrationWithSensi(conditionalMu.astype('float32'), 
                                                                                    dataSetList)
            
            
        
        transposeH = np.transpose(H)
        
        kalmanGainNumerator = np.matmul(conditionalSigmaConverted, transposeH)
        kalmanGainDenominator = np.matmul(np.matmul(H,conditionalSigmaConverted),transposeH) + self.R
        
        kalmanGain = np.matmul(kalmanGainNumerator,np.linalg.inv(kalmanGainDenominator))
        
        residualSpot = ySpot - predictedY
        #update muSerie 
        muSpot = conditionalMu + np.matmul(kalmanGain, residualSpot)
        #update SigmaSerie
        covGain = np.matmul(kalmanGain, H)
        sigmaSpot = np.matmul((np.eye(covGain.shape[0], M = covGain.shape[1]) - covGain),
                              conditionalSigmaConverted)
        
        return {"muSpot" : muSpot.astype(dtype=self.R.values.dtype),
                "sigmaSpot" : np.ravel(sigmaSpot).astype(dtype=self.R.values.dtype),
                "residualSpot" : residualSpot.astype(dtype=self.R.values.dtype), 
                "kalmanGain" : np.ravel(kalmanGain).astype(dtype=self.R.values.dtype)}
        
    
    
    ################################################################################
    #Kalman observation loglikelyhood
    ################################################################################
    
    
    #Return the log likelyhood of observed variables 
    #with respect to some exogenous variables and model parameters (A,B,C,D,Q,R)
    #presentObservedVariables is the serie of output variables until time t
    #presentExogenousVariables is the serie of output variables until time t
    def computeSerieLogLikelyhood(self, 
                                  presentObservedVariables, 
                                  presentExogenousVariables,
                                  priorGuessForLatentVariable):
        marginalLikelyhoodSerie = []
        
        dataSetList = [presentObservedVariables, 
                       None, 
                       presentExogenousVariables, 
                       None]
        forwardResults = self.iterateOnSeries(dataSetList, 
                                              priorGuessForLatentVariable)
        
        for d in presentObservedVariables.index :
            conditionalMu = forwardResults["muConditionalSerie"].loc[d].values.astype(dtype=self.A.dtype)
            conditionalSigma = np.reshape(forwardResults["sigmaConditionalSerie"].loc[d].values,
                                          (conditionalMu.size,conditionalMu.size)).astype(dtype=self.A.dtype)
            dataSetListDay = [x.loc[d] if (x is not None) else x for x in dataSetList]
            smoothedY, H = self.evalSingleDayWithoutCalibrationWithSensi(conditionalMu, dataSetListDay)
            posteriorObservationMu = smoothedY
            posteriorObservationSigma = (np.matmul(np.matmul(H,conditionalSigma), np.transpose(H)) + self.R)
            
            marginalLikelyhood = scipy.stats.norm.logpdf(presentObservedVariables.loc[d], 
                                                         loc=posteriorObservationMu, 
                                                         scale=posteriorObservationSigma)
            marginalLikelyhoodSerie.append(marginalLikelyhood)
            
        return  np.sum(marginalLikelyhoodSerie)
    
    
    
    def saveModel(self):
        return
    
    def restoreModel(self):
        return

