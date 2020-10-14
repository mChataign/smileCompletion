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
import dynamicModel
import linearModel
from sklearn.linear_model import LinearRegression
import math as m
#import tensorflow_probability as tfp
#tfd = tfp.distributions

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
class kalmanFilter(dynamicModel.dynamicModel):

    ################################################################################
    #Kalman Initialization
    ################################################################################
    def __init__(self,
                 hyperParameters,
                 modelName = "./bestDynamicModel"):
        super().__init__(hyperParameters = hyperParameters,
                         modelName = modelName)
        
        #Model parameter with :
        #Z_t = A Z_{t-1} + B U_t + Noise_1, Noise_1~N(0,Q)
        #Y_t = C Z_t + D U_t + Noise_2, Noise_2~N(0,R)
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        
        self.Q = None
        self.R = None
        self.RInv = None
        
        #Temporary variables for forward algorithm
        #p(z_t | y_{1:t}, u_{1:t}) ~ N(z_t | mu_t , Sigma_t)
        self.resetForwardSeries()
        
        #Temporary variables for backward algorithm
        self.resetBackwardSeries()
        
    def buildModel(self):
        return
    
    def buildArchitecture(self):
        return
    
    def resetForwardSeries(self):
        #Series of matrices
        self.muSerie = None
        self.SigmaSerie = None
        self.conditionalMuSerie = None
        self.conditionalSigmaSerie = None
        
        self.residual = None
        self.matrixGain = None
        return
    
    def resetBackwardSeries(self):
        self.conditionalBackwardMuSerie = None
        self.conditionalBackwardSigmaSerie = None
        self.backwardGainMatrix = None
        return
    
    def initialStatePosterior(self, ySeries, uSerie, zSerie):
        #y0 = ySerie.mean(axis=0)
        mean0 = zSerie.mean(axis=0) if zSerie is not None else np.zeros(self.hyperParameters['nbFactors'])
        sigma0 = zSerie.cov() if zSerie is not None else np.eye(len(mean0))
        return mean0 , sigma0
    
    #In case we do not not observe initial values for z,
    #we must estimate initial values for A, B, C, D, Q and R
    def initialValuesForModelParameters(self):
        #Large values for Q and R since we have no confidence with our initial values
        self.Q = 100 * np.diag(np.eye(self.hyperParameters['nbFactors'])).astype(dtype=float) 
        self.R = 100 * np.diag(np.eye(self.hyperParameters['nbFactors'])).astype(dtype=float) 
        self.RInv = np.linalg.inv(self.R)
        
        #Initialise it as a diagonal dominant matrix with positive coefficients, 
        #it ensures we have positive definite matrix
        self.A = (0.01 * np.eye(self.hyperParameters['nbFactors'] ) 
                  + np.eye(self.hyperParameters['nbFactors'])).astype(dtype=float) 
        #Adjust if different order of magnitudes are used
        self.B = np.eye(self.hyperParameters['nbExogenousVariables']).astype(dtype=float) 
        
        self.C = (0.01 * np.eye(self.hyperParameters['nbOutputVariables'] , M = self.hyperParameters['nbFactors']).astype(dtype=float) 
                  + np.eye(self.hyperParameters['nbOutputVariables'] , M = self.hyperParameters['nbFactors'])) 
        self.D = np.eye(self.hyperParameters['nbOutputVariables'] , M = self.hyperParameters['nbExogenousVariables']).astype(dtype=float) 
        
        return
    
    ################################################################################
    #Kalman Calibration
    ################################################################################
    
    #Observing variables should allow for easy and direct estimation of models parameters
    def trainWithObservedFactors(self, 
                                 observationVariable, 
                                 exogenousVariables, 
                                 observedLatentVariables):
        #In that case implementation is equivalent to least square linear regression
        #Y_t = C Z_t + D U_t + Noise_2, Noise_2~N(0,R)
        inputFeatures = exogenousVariables
        outputFeatures = None
        #We take the matrix C from the encoding Model if he can provide it 
        #since the dependancy between Y and Z which has been already learned by the decoder
        self.C =  self.encodingModel.getDecoderCoefficients().T if (self.encodingModel is not None) else None
        if self.C is not None : 
            outputFeatures = observationVariable - np.matmul(observedLatentVariables, self.C.T)
        else :
            inputFeatures = pd.concat([observedLatentVariables, 
                                       exogenousVariables],axis=1)
            outputFeatures = observationVariable
            
        reg = LinearRegression(fit_intercept = False)
        reg.fit(inputFeatures, outputFeatures)
        nbLatentVariables = observedLatentVariables.shape[1]
        
        if self.C is not None :
            self.D = reg.coef_
        else :
            self.D = reg.coef_[:,nbLatentVariables:]
            self.C = reg.coef_[:,:nbLatentVariables]
        
        residuals = outputFeatures - reg.predict(inputFeatures)
        deltaMean = residuals.mean(axis=0)
        self.R = residuals.cov()
        
        #Z_t = A Z_{t-1} + B U_t + Noise_1, Noise_1~N(0,Q)
        inputFeatures2 = pd.concat([observedLatentVariables.shift(), 
                                    exogenousVariables], axis=1).dropna(axis = 0, 
                                                                        how = 'any')
        outputFeatures2 = observedLatentVariables.loc[inputFeatures2.index]
        
        reg2 = LinearRegression(fit_intercept = False)
        reg2.fit(inputFeatures2, outputFeatures2)
        
        self.B = reg2.coef_[:,nbLatentVariables:]
        self.A = reg2.coef_[:,:nbLatentVariables]
        
        residuals2 = outputFeatures2 - reg2.predict(inputFeatures2)
        epsilonMean = residuals2.mean(axis=0)
        self.Q = residuals2.cov()
        
        self.RInv = np.linalg.inv(self.R)
        return
    
    def checkAIsContractive(self):
        eigenValues, eigenVectors = np.linalg.eig(self.A)
        data = pd.Series(np.sort(np.abs(eigenValues)),
                         index = np.arange(len(eigenValues))).rename("Transition matrix eigen values")
        plottingTools.plotSeries([data], title = "Transition matrix eigen values")
        if np.amax(np.abs(eigenValues)) <= 1 :
            print("Transition matrix is contractive")
        else : 
            print("Transition matrix is not contractive")
        return eigenValues
    
    #Estimate model parameters (A,B,C,D,Q,R) and infer latent variables
    def trainWithoutObservedFactors(self, 
                                    observationVariable, 
                                    exogenousVariables):
        #In that case estimation is based on Blocked Gibbs Sampling Algorithm
        #The idea of these methods is to have an initial guess of z_t through a PCA like method.
        raise NotImplementedError("TODO")
        return
    
    ################################################################################
    #Kalman Equations
    ################################################################################
    
    def predictionStep(self, muPrev, sigmaPrev, uSpot = None):
        #Reshape argument
        
        muFormatted = muPrev.values.astype(dtype=self.A.dtype)
        sigmaFormatted = np.reshape(sigmaPrev.values,(muPrev.size,muPrev.size)).astype(dtype=self.A.dtype)
        
        #update self.conditionalMuSerie
        conditionalMu = np.matmul(self.A,muFormatted) 
        if uSpot is not None :#In presence of exogenous variables
            conditionalMu += np.matmul(self.B,uSpot)
        #update self.conditionalSigmaSerie
        conditionalSigma = np.matmul(np.matmul(self.A,sigmaFormatted),np.transpose(self.A)) + self.Q
        
        return {"conditionalMu" : conditionalMu.astype(dtype=self.A.dtype), 
                "conditionalSigma" : np.ravel(conditionalSigma).astype(dtype=self.A.dtype)}
    
    def updateStep(self, ySpot, 
                   conditionalMu, 
                   conditionalSigma, 
                   uSpot = None):
        #Reshape argument
        
        conditionalSigmaConverted = np.reshape(conditionalSigma,
                                               (len(conditionalMu),len(conditionalMu)))
        
        #update matrixGain
        invConditionalSigma = np.linalg.inv(conditionalSigmaConverted)
        
        transposeC = np.transpose(self.C)
        
        kalmanGainNumerator = np.matmul(conditionalSigmaConverted, transposeC)
        kalmanGainDenominator = np.matmul(np.matmul(self.C,conditionalSigmaConverted),transposeC) + self.R
        
        kalmanGain = np.matmul(kalmanGainNumerator,np.linalg.inv(kalmanGainDenominator))
        #kalmanGain = np.matmul(np.linalg.inv(invConditionalSigma + np.matmul(np.matmul(transposeC,self.R),
        #                                                                     self.C)), 
        #                       np.matmul(transposeC, self.RInv))
        
        #update residual
        predictedY = np.matmul(self.C,conditionalMu) 
        if uSpot is not None : 
            predictedY += np.matmul(self.D, uSpot)
        residualSpot = ySpot - predictedY
        #update muSerie 
        muSpot = conditionalMu + np.matmul(kalmanGain, residualSpot)
        #update SigmaSerie
        covGain = np.matmul(kalmanGain, self.C)
        sigmaSpot = np.matmul((np.eye(covGain.shape[0], M = covGain.shape[1]) - covGain),
                              conditionalSigmaConverted)
        
        return {"muSpot" : muSpot.astype(dtype=self.C.dtype),
                "sigmaSpot" : np.ravel(sigmaSpot).astype(dtype=self.C.dtype),
                "residualSpot" : residualSpot.astype(dtype=self.C.dtype), 
                "kalmanGain" : np.ravel(kalmanGain).astype(dtype=self.C.dtype)}
    
    #Implement forward algorithm 
    def iterateOnSeries(self, dataSetList, zSerie):
        #Initilization
        ySerie = dataSetList[0]
        uSerie = dataSetList[2]
        muPrev ,sigmaPrev = self.initialStatePosterior(ySerie, uSerie, zSerie)
        
        muColumns = np.arange(len(muPrev))
        sigmaColumns = np.arange(sigmaPrev.size)
        muSerie = pd.DataFrame(index=ySerie.index, columns = muColumns) #serie of mu_{t|t}
        sigmaSerie = pd.DataFrame(index=ySerie.index, columns = sigmaColumns) #serie of sigma_{t|t}
        muConditionalSerie = pd.DataFrame(index=ySerie.index, columns = muSerie.columns) #serie of mu_{t+1|t}
        sigmaConditionalSerie = pd.DataFrame(index=ySerie.index, columns = sigmaSerie.columns) #serie of sigma_{t+1|t}
        
        #We run forward algorithm on complete series to get mu_{T|T} and Sigma_{T|T}
        for d in ySerie.index :
            conditionalLaw = self.predictionStep(muPrev, 
                                                 sigmaPrev, 
                                                 uSpot = uSerie.loc[d] if uSerie is not None else None)
            posteriorLaw = self.updateStep(ySerie.loc[d],
                                           conditionalLaw['conditionalMu'],
                                           conditionalLaw['conditionalSigma'],
                                           uSpot = uSerie.loc[d] if uSerie is not None else None)
            
            #Save intermediary results for backward pass
            muSerie.loc[d] = posteriorLaw['muSpot']
            muConditionalSerie.loc[d] = conditionalLaw['conditionalMu']
            sigmaSerie.loc[d] = posteriorLaw['sigmaSpot']
            sigmaConditionalSerie.loc[d] = conditionalLaw['conditionalSigma']
            
            #Save a reference to latest results 
            muPrev = muSerie.loc[d]
            sigmaPrev = sigmaSerie.loc[d]
            # print("-------------")
            # print("-------------")
            # print(d)
            # print("-------------")
            # print(muPrev)
            # print("-------------")
            # print(sigmaPrev)
            # print("-------------")
            # print(posteriorLaw['kalmanGain'].shape)
            
            
        return {"muSerie" : muSerie, 
                "sigmaSerie" : sigmaSerie, 
                "muConditionalSerie" : muConditionalSerie, 
                "sigmaConditionalSerie" : sigmaConditionalSerie}
    
    #Backward pass step : used for smoothing
    def updateFromTheFuture(self, 
                            smoothMuPrev, #mu_{t+1|T}
                            smoothSigmaPrev, #sigma_{t+1|T}
                            forwardMuSpot, #mu_{t|t}
                            forwardSigmaSpot, #sigma_{t|t}
                            forwardMuPrev, #mu_{t+1|t}
                            forwardSigmaPrev): #sigma_{t+1|t}
        smoothMuPrevFormatted = smoothMuPrev.values.astype(dtype=self.A.dtype)
        smoothSigmaPrevFormatted = np.reshape(smoothSigmaPrev.values,
                                              (smoothMuPrevFormatted.size,smoothMuPrevFormatted.size)).astype(dtype=self.A.dtype)
        
        forwardMuSpotFormatted = forwardMuSpot.values.astype(dtype=self.A.dtype)
        forwardSigmaSpotFormatted = np.reshape(forwardSigmaSpot.values,
                                               (forwardMuSpotFormatted.size,forwardMuSpotFormatted.size)).astype(dtype=self.A.dtype)
        
        forwardMuPrevFormatted = forwardMuPrev.values.astype(dtype=self.A.dtype)
        forwardSigmaPrevFormatted = np.reshape(forwardSigmaPrev.values,
                                               (forwardMuPrev.size,forwardMuPrev.size)).astype(dtype=self.A.dtype)
                                               
        #J_t = sigma_{t|t} * A^T * sigma_{t+1|t}^-1
        backwardGainMatrix = np.matmul(forwardSigmaSpotFormatted,
                                       np.matmul( np.transpose(self.A), np.linalg.inv(forwardSigmaPrevFormatted))) 
        #mu update : mu_{t|T} = mu_{t|t} + J_t * (mu_{t+1|T}-mu_{t+1|t})
        muDifference = (smoothMuPrevFormatted - forwardMuPrevFormatted)
        backwardMuSerie = forwardMuSpotFormatted + np.matmul(backwardGainMatrix,muDifference)
        #Sigma update : #sigma_{t|T} = sigma_{t|t} + J_t * (sigma_{t+1|T} - sigma_{t+1|T}) * (J_t)^{T}
        sigmaDifference = smoothSigmaPrevFormatted - forwardSigmaPrevFormatted
        backwardSigmaSerie = forwardSigmaSpotFormatted + np.matmul(np.matmul(backwardGainMatrix,sigmaDifference), 
                                                                   np.transpose(backwardGainMatrix))
        return np.ravel(backwardGainMatrix), backwardMuSerie, np.ravel(backwardSigmaSerie)
        
    ################################################################################
    #Kalman Functionalities for prediction
    ################################################################################
    
    #Predict values for latent factors at time t (i.e. z_t)
    #pastObservedVariables is the serie of output variables until time t-1
    #presentExogenousVariables is the serie of output variables until time t
    #pastLatentVariables is the serie of output variables until time t-1
    def predictNextObservation(self, 
                               pastObservedVariables, 
                               presentExogenousVariables, 
                               pastLatentVariables):
        #Run forward pass until t-1
        dataSetList = [pastObservedVariables, 
                       None, 
                       presentExogenousVariables.head(-1) if presentExogenousVariables is not None else None, 
                       None]
        forwardPass = self.iterateOnSeries(dataSetList,
                                           pastLatentVariables)
        #return mu_t|t-1 where mu_t|t-1 = A * mu_{t-1} + B_t u_t
        conditionalLaw = self.predictionStep(forwardPass["muSerie"].iloc[-1],
                                             forwardPass["sigmaSerie"].iloc[-1],
                                             uSpot = presentExogenousVariables.iloc[-1] if presentExogenousVariables is not None else None)
        #In a gaussian assumption
        return conditionalLaw['conditionalMu']
    
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
        #Run forward pass until t-1
        dataSetList = [pastObservedVariables, 
                       None, 
                       presentExogenousVariables.head(-1) if presentExogenousVariables is not None else None, 
                       None]
        forwardPass = self.iterateOnSeries(dataSetList, 
                                           pastLatentVariables)
        #return mu_t|t-1 where mu_t|t-1 = A * mu_{t-1} + B_t u_t
        conditionalLaw = self.predictionStep(forwardPass["muSerie"].iloc[-1],
                                             forwardPass["sigmaSerie"].iloc[-1],
                                             uSpot = presentExogenousVariables.iloc[-1] if presentExogenousVariables is not None else None)
        return  scipy.stats.norm.logpdf(currentLatentVariables, 
                                        loc=conditionalLaw['conditionalMu'], 
                                        scale=conditionalLaw['conditionalSigma'])
    
    def buildSinglePenalization(self, predictivePostoriorMean, 
                                predictivePostoriorSigma,
                                latentVariablesTensor):
        
        meanTensor = tf.constant(predictivePostoriorMean.values,
                                 shape = [predictivePostoriorMean.shape[0],1],
                                 dtype = tf.float32)
        covTensor = tf.constant(predictivePostoriorSigma.values,
                                shape = list([predictivePostoriorMean.shape[0],predictivePostoriorMean.shape[0]]),
                                dtype = tf.float32)
                                
        #Clean version is not possible because tensorflow_probability module is not available
        #dist = tfd.MultivariateNormalFullCovariance(loc=meanTensor, scale=covTensor)
        #return  dist.log_prob(currentLatentVariables,name="entropyOfInferredValue")
        #pi = tf.constant(m.pi)
        #nbMarginals = predictivePostoriorMean.size
        
        centeredFactor = tf.transpose(latentVariablesTensor) - meanTensor
        invCovMatrix = tf.linalg.inv(covTensor)
        exponent = tf.matmul(centeredFactor,tf.matmul(invCovMatrix,centeredFactor), 
                             transpose_a = True, 
                             transpose_b = False)
        logpdf =  - 0.5 * exponent #- tf.pow(2 * pi, nbMarginals / 2.0) - tf.sqrt(tf.linalg.det(covTensor))
        return logpdf
    #Return a list of tensor to execute the same task as entropyOfInferredValue
    #a tensor representing z_t
    def buildPenalizationFactoriesAlongHistory(self, 
                                               presentDataSetList, 
                                               presentLatentVariables):
        #Run forward pass until t-1
        forwardPass = self.iterateOnSeries(presentDataSetList,
                                           presentLatentVariables)
        
        muConditionalSerie = forwardPass["muConditionalSerie"]
        sigmaConditionalSerie = forwardPass["sigmaConditionalSerie"]
        factoryList = []
        for d in presentDataSetList[0].index : 
            conditionalMu = muConditionalSerie.loc[d]
            conditionalSigma = sigmaConditionalSerie.loc[d]
            
            factoryList.append(lambda x : - self.buildSinglePenalization(conditionalMu, conditionalSigma, x) / 10000)
        
        penalizationFactorySerie = pd.Series(factoryList,index = presentDataSetList[0].index)
        
        return penalizationFactorySerie

    #Return a list of tensor to execute the same task as entropyOfInferredValue
    #a tensor representing z_t
    def buildPenalizationFactory(self, 
                                 presentDataSetList,
                                 presentLatentVariables):
        #Run forward pass until t-1
        forwardPass = self.iterateOnSeries(presentDataSetList,
                                           presentLatentVariables)
        
        muConditionalSerie = forwardPass["muConditionalSerie"]
        sigmaConditionalSerie = forwardPass["sigmaConditionalSerie"]
        factory = None
        
        conditionalMu = muConditionalSerie.iloc[-1]
        conditionalSigma = sigmaConditionalSerie.iloc[-1]
        
        factory = (lambda x : - self.buildSinglePenalization(conditionalMu, conditionalSigma, x) / 10000)
        
        return factory
    
    ################################################################################
    #Kalman Functionalities for filtering
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
        #Run forward pass until t-1
        dataSetList = [presentObservedVariables, 
                       None, 
                       presentExogenousVariables, 
                       None]
        forwardPass = self.iterateOnSeries(dataSetList,
                                           pastLatentVariables)
        #return mu_t = mu_t|t-1 + K_t * r_t
        return  forwardPass["muSerie"].iloc[-1]
    
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
        #Run forward pass until t-1
        dataSetList = [presentObservedVariables, 
                       None, 
                       presentExogenousVariables, 
                       None]
        forwardPass = self.iterateOnSeries(dataSetList,
                                           pastLatentVariables)
        return  scipy.stats.norm.logpdf(currentLatentVariables, 
                                        loc=forwardPass['muSerie'].iloc[-1], 
                                        scale=forwardPass['sigmaSerie'].iloc[-1])
    
    
    ################################################################################
    #Kalman Functionalities for smoothing
    ################################################################################
    
        
    #Return smoothed serie of latent variables (z_t)_{t in [0,T]} and observed variables 
    #pastAndFutureObservedVariables is the serie of output variables in [0,T]
    #pastAndFutureExogenousVariables is the serie of output variables in [0,T]
    #initialGuessLatentVariables is the serie of latent variables obtained before smoothing in [0,T]
    def smoothedLatentVariables(self, 
                                pastAndFutureObservedVariables, 
                                pastAndFutureExogenousVariables, 
                                initialGuessLatentVariables):
        self.resetForwardSeries()
        self.resetBackwardSeries()
        
        #First we run forward algorithm on complete series to get mu_{T|T} and Sigma_{T|T}
        dataSetList = [pastAndFutureObservedVariables, 
                       None, 
                       pastAndFutureExogenousVariables, 
                       None]
        forwardPass = self.iterateOnSeries(dataSetList,
                                           initialGuessLatentVariables)
        muSerie = forwardPass["muSerie"] #serie of mu_{t|t}
        sigmaSerie = forwardPass["sigmaSerie"] #serie of sigma_{t|t}
        muConditionalSerie = forwardPass["muConditionalSerie"] #serie of mu_{t+1|t}
        sigmaConditionalSerie = forwardPass["sigmaConditionalSerie"] #serie of sigma_{t+1|t}
        
        
        #Backward Step : From future to past
        backwardMuSerie = pd.DataFrame(index = pastAndFutureObservedVariables.index, 
                                       columns = muSerie.columns)
        backwardSigmaSerie = pd.DataFrame(index = pastAndFutureObservedVariables.index, 
                                          columns = sigmaSerie.columns)
        dPrev = None
        
        backwardGainMatrix = pd.DataFrame(index = pastAndFutureObservedVariables.index)
        for d in pastAndFutureObservedVariables.sort_index(ascending=False).index :
            if d == pastAndFutureObservedVariables.index[-1]:#Last day
                #Initialize with latest values from forward pass
                backwardMuSerie.loc[d] = muSerie.loc[d]
                backwardSigmaSerie.loc[d] = sigmaSerie.loc[d]
            else :
                J, mu, sigma = self.updateFromTheFuture(backwardMuSerie.loc[dPrev], 
                                                        backwardSigmaSerie.loc[dPrev], 
                                                        muSerie.loc[d], 
                                                        sigmaSerie.loc[d], 
                                                        muConditionalSerie.loc[dPrev], 
                                                        sigmaConditionalSerie.loc[dPrev])
                
                backwardGainMatrix.loc[d] = J
                backwardMuSerie.loc[d] = mu
                backwardSigmaSerie.loc[d] = sigma
            dPrev = d
        
        return {"smoothedMu" : backwardMuSerie, "smoothedSigma" : backwardSigmaSerie}
    
    
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
            
            posteriorObservationMu = np.matmul(self.C, conditionalMu)
            posteriorObservationSigma = (np.matmul(np.matmul(self.C,conditionalSigma), np.transpose(self.C)) + self.R)
            
            marginalLikelyhood = scipy.stats.norm.logpdf(presentObservedVariables.loc[d], 
                                                         loc=posteriorObservationMu, 
                                                         scale=posteriorObservationSigma)
            marginalLikelyhoodSerie.append(marginalLikelyhood)
            
        return  np.sum(marginalLikelyhoodSerie)
    
    
    
    def saveModel(self):
        return
    
    def restoreModel(self):
        return

