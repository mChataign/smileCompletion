#Import modules
import os
import pandas as pd
import numpy as np
from pandas import DatetimeIndex
import dask
import scipy
import time
import glob

from functools import partial
from abc import ABCMeta, abstractmethod

import plottingTools

tmpFolder = "./tmp/" #"./../tmp/"
#tmpFolder = os.environ['tmp']







class FactorialModel :
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestFactorialModel"):
        # Hyperparameters
        self.learningRate = learningRate
        self.hyperParameters = hyperParameters
        self.nbUnitsPerLayer = nbUnitsPerLayer
        self.nbFactors = nbFactors
        self.layers = []
        self.nbEncoderLayer = 0
        self.batchSize = -1
        self.lossHolderExponent = (hyperParameters["lossHolderExponent"] if "lossHolderExponent" in hyperParameters else 4)
        
        pathToModels = os.path.join(tmpFolder,'modelSaved')#os.getcwd() + "\\..\\modelSaved\\"#"..\\modelSaved\\"
        if not os.path.isdir(pathToModels):
            os.mkdir(pathToModels)
        self.metaModelName = os.path.normpath(os.path.join(pathToModels,os.path.normpath(modelName) + ".cpkt")).replace("\\","/") 
        self.metaModelNameInit = os.path.normpath(os.path.join(pathToModels,os.path.normpath(modelName) + "Init" + ".cpkt")).replace("\\","/") 
        self.verbose = (('verbose' in hyperParameters) & hyperParameters['verbose'])
        
        if self.verbose :
            print("Create tensor for training")
        self.buildModel()
        
        self.variationMode = False #activate learning with variations instead of 
        #train, diff sur l'ensemble, puis dropna, puis 
        
    #Build the architecture, losses and optimizer.
    def buildModel(self):
        raise NotImplementedError("Abstract Class !")
        return None
    
        
    #######################################################################################################
    #Training functions
    #######################################################################################################
    
    
    
    #Sample Mini-batch
    def generateMiniBatches(self, dataSetList, nbEpoch):
        batchSize = 100
        #return self.selectMiniBatchWithoutReplacement(dataSetList, batchSize)
        return [dataSetList]
    
    
    def splitValidationAndTrainingSet(self, dataSetTrainList):
        #Sample days in validation set
        percentageForValidationSet = self.hyperParameters['validationPercentage'] if ('validationPercentage' in self.hyperParameters) else 0.2
        #validationSetDays =  dataSetTrainList[0].sample(frac=percentageForValidationSet,
        #                                                replace=False).sort_index().index 
        #select Time series ends
        nbValidationSetDays = int(percentageForValidationSet * dataSetTrainList[0].index.size)
        validationSetDays = dataSetTrainList[0].index[-nbValidationSetDays:]
        
        validationDataSetList = [x.loc[validationSetDays].sort_index() if x is not None else None for x in dataSetTrainList]
        trainingDataSetList = [x.drop(validationSetDays).sort_index() if x is not None else None for x in dataSetTrainList]
        return validationDataSetList, trainingDataSetList
    
    #Sample minibatch of size batchSize for a list of dataset 
    def selectMiniBatchWithoutReplacement(self, dataSetList, batchSize):
        miniBatchList = []
        if len(dataSetList)==0:
            return miniBatchList
        
        nbObs = dataSetList[0].shape[0]
        idx = np.arange(nbObs)
        np.random.shuffle(idx)
        nbBatches =  int(np.ceil(nbObs/batchSize))
        
        lastIndex = 0
        for k in range(nbBatches) :
            firstIndex = k * batchSize
            lastIndex = (k+1) * batchSize
            miniBatchIndex = idx[firstIndex:lastIndex]
            
            miniBatch = [x.iloc[miniBatchIndex,:] for x in dataSetList] 
            miniBatchList.append(miniBatch)
        
        #We add observations which were not drawn to use a full epoch
        lastIndex = (nbBatches) * batchSize
        miniBatchIndex = idx[lastIndex:]
        miniBatch = [x.iloc[miniBatchIndex,:] for x in dataSetList] 
        miniBatchList.append(miniBatch)
        
        return miniBatchList
    
    #Sample minibatch of size batchSize for a list of dataset with replacement
    def selectMiniBatchWithReplacement(self, dataSetList, batchSize):
        miniBatchList = []
        if len(dataSetList)==0:
            return miniBatchList
        
        nbObs = dataSetList[0].shape[0]
        nbBatches =  int(np.ceil(nbObs/batchSize))
        
        lastIndex = 0
        for k in range(nbBatches+1) :
            miniBatchIndex =  dataSetList[0].sample(n=batchSize,replace=False).index 
            miniBatch = [x.iloc[miniBatchIndex,:] for x in dataSetList] 
            miniBatchList.append(miniBatch)
        
        return miniBatchList
        
    def train(self, inputTrain, nbEpoch, inputTest = None):
        raise NotImplementedError("Abstract Class !")
        return None
    
    
    
    
    
    #######################################################################################################
    #Evaluation functions
    #######################################################################################################
    
    #Same but with default session 
    def evalModel(self, inputTest):
        raise NotImplementedError("Abstract Class !")
        return None
    
    
    def completeDataTensor(self, 
                           sparseSurfaceList, 
                           initialValueForFactors, 
                           nbCalibrationStep, 
                           *args):
        raise NotImplementedError("Abstract Class !")
        return None 
        
    
    def commonEvalSingleDayWithoutCalibration(self, 
                                              initialValueForFactors,
                                              dataSetList,
                                              computeSensi = False):
        raise NotImplementedError("Abstract Class !")
        return None
        
    
    
    def evalSingleDayWithoutCalibrationWithSensi(self, initialValueForFactors, dataSetList):
        return  self.commonEvalSingleDayWithoutCalibration(initialValueForFactors, 
                                                           dataSetList,
                                                           computeSensi = True)
    
    
    def evalSingleDayWithoutCalibration(self, initialValueForFactors, dataSetList):
        s,_ = self.commonEvalSingleDayWithoutCalibration(initialValueForFactors, dataSetList)
        return s
    
    #Take a full surface in entry, reconstruct it 
    #and return sensitivities between points i.e. the jacobian of D(E(S)) w.r.t S
    def evalInterdependancy(self, fullSurfaceList):
        #Build tensor for reconstruction
        nbObs = 1
        nbPoints = 0
        fullSurface = fullSurfaceList[0].values
        if fullSurface.ndim == 1 :
            nbPoints = fullSurface.shape[0]
        elif fullSurface.ndim == 2 :
            nbObs = fullSurface.shape[0]
            nbPoints = fullSurface.shape[1]
        else :
            raise NotImplementedError("Tensor of rank greater than 2")
        reshapedFullSurface = np.reshape([fullSurface],
                                         (nbObs,nbPoints))
        
        reconstructedSurface, interdependancies = self.commonEvalInterdependancy([reshapedFullSurface])
        
        if fullSurface.ndim == 1 :
            reshapedInterdependancies = np.reshape(interdependancies,(nbPoints,nbPoints))
            reshapedReconstructedSurface = np.reshape(reconstructedSurface,(nbPoints))
        elif fullSurface.ndim == 2 :
            reshapedInterdependancies = np.reshape(interdependancies,(nbObs,nbPoints,nbPoints))
            reshapedReconstructedSurface = np.reshape(reconstructedSurface,(nbObs,nbPoints))
        
        return reshapedReconstructedSurface, reshapedInterdependancies
    
    def commonEvalInterdependancy(self, fullDataSet):
        raise NotImplementedError("Abstract Class !")
        return None
        
    
    #Return None if not supported
    def getDecoderCoefficients(self):
        raise NotImplementedError("Abstract Class !")
        return None
    
    
    def getArbitrageTheta(self, dataSetList, initialFactorValue):
        #Default implementation
        reshapedReconstruction = pd.DataFrame(np.expand_dims(dataSetList[0].values, 0), 
                                              index = [dataSetList[0].name], 
                                              columns = dataSetList[0].index)
        
        return reshapedReconstruction#.rename(sparseSurface.name)

