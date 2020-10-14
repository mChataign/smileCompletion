#Import modules
import os
import pandas as pd
import numpy as np
from pandas import DatetimeIndex
import dask
import scipy
import time
import glob
import torch
import torch.nn as nn
from live_plotter import live_plotter
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from functools import partial
from abc import ABCMeta, abstractmethod

import plottingTools
import pytorchModel
import loadData






class pytorchFwdModel(pytorchModel.pytorchModel) :
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestPyTorchFwdModel"):
        super().__init__(learningRate, hyperParameters, nbUnitsPerLayer, nbFactors, 
                         modelName = modelName)
    
    def buildModel(self):
        self.fe = pytorchModel.Functional_encoder(self.nbFactors + 1) #Neural network architecture
        return
    
    #######################################################################################################
    #Evaluation functions
    #######################################################################################################
    
    def evalBatch(self, batch, code):
        
        batchLogMoneyness = self.getLogMoneyness(batch)
        scaledMoneyness = (batchLogMoneyness.values - self.MeanLogMoneyness) / self.StdLogMoneyness
        logMoneynessTensor = torch.Tensor(np.expand_dims(scaledMoneyness, 1)).float() #Log moneyness
        
        # for j in np.random.choice(len(test[k]), 10):
        # filt  = test[k].nBizDays >= 10
        batchLogMat = self.getLogMaturities(batch)
        scaledMat = (batchLogMat.values - self.MeanLogMaturity) / self.StdLogMaturity
        logMaturity = torch.tensor( np.expand_dims(scaledMat, 1)  , requires_grad=True).float() 
        
        scaledFwd = (batch[2].values - self.MeanFwd) / self.StdFwd
        fwdTensor = torch.tensor( np.expand_dims(scaledFwd, 1) ).float() 
        
        codeTensor = code.repeat(batch[0].shape[0], 1).float()
        refVol = torch.tensor(batch[0].values)
        
        inputTensor = torch.cat((logMoneynessTensor, logMaturity, fwdTensor, codeTensor), dim=1)
        outputTensor = self.fe( inputTensor )[:, 0]
        
        loss = torch.mean( (outputTensor - refVol)[~torch.isnan(outputTensor)] ** 2 )#torch.nanmean( (outputTensor - refVol) ** 2 ) 
        return inputTensor, outputTensor, loss, logMaturity, codeTensor, logMoneynessTensor
        
    
    def commonEvalSingleDayWithoutCalibration(self, 
                                              initialValueForFactors,
                                              dataSetList,
                                              computeSensi = False):
        
        #Rebuild tensor graph
        self.restoringGraph()
        #Build tensor for reconstruction
        nbObs = 1 if initialValueForFactors.ndim == 1 else initialValueForFactors.shape[0]
        nbPoints = dataSetList[1].shape[0] if dataSetList[1].ndim == 1 else dataSetList[1].shape[1]
        nbFactors = self.nbFactors 
        
        reshapedValueForFactors = np.reshape([initialValueForFactors],
                                             (nbObs,nbFactors))
        
        
        self.code = pytorchModel.Code(nbObs, self.nbFactors, initialValue = reshapedValueForFactors) #Latent variables
        codeTensor = self.code.code[k, :].repeat(nbPoints, 1)
        
        batchLogMoneyness = self.getLogMoneyness(dataSetList)
        scaledMoneyness = (batchLogMoneyness.values - self.MeanLogMoneyness) / self.StdLogMoneyness
        logMoneynessTensor = torch.Tensor(np.expand_dims(scaledMoneyness.values, 1)).float() #Log moneyness
        
        scaledFwd = (dataSetList[2].values - self.MeanFwd) / self.StdFwd
        fwdTensor = torch.tensor( np.expand_dims(scaledFwd, 1) ).float() 
            
        # for j in np.random.choice(len(test[k]), 10):
        # filt  = test[k].nBizDays >= 10
        batchLogMat = self.getLogMaturities(dataSetList)
        scaledMat = (batchLogMat.values - self.MeanLogMaturity) / self.StdLogMaturity
        logMaturity = torch.tensor( np.expand_dims(scaledMat, 1) ).float() 
        
        inputTensor = torch.cat((logMoneynessTensor, logMaturity, fwdTensor, codeTensor), dim=1)
        outputTensor = self.fe( inputTensor )[:, 0]
        self.restoreWeights()
        
        #Build tensor for reconstruction
        # print("nbPoints : ", nbPoints)
        # print("initialValueForFactors : ", initialValueForFactors)
        # print("inputFeatures : ", inputFeatures)
        # print("outputFeatures : ", outputFeatures)
        # print("outputTensor : ", self.outputTensor)
        
        reconstructedSurface = outputTensor.detach().numpy().reshape(batch[0].shape)
        
        inputTensor = torch.cat((strikes, logMaturity, codeTensor), dim=1)
        #if computeSensi :
        #    inputTensor.requires_grad = True
        outputTensor = self.fe( inputTensor )[:, 0]
        
        
        reshapedJacobian = None
        if computeSensi :
            reshapedJacobian = np.ones((nbObs, nbPoints, nbFactors)) if initialValueForFactors.ndim != 1 else np.ones((nbPoints, nbFactors))
            #for p in range(nbPoints) :
            #   output.backward()
            #   jacobian = input.grad.data
            #   reshapedJacobian = tf.reshape(jacobian, shape = [nbObs, nbPoints, nbFactors])
            #   if self.verbose :
            #       print(reshapedJacobian)
        
        
        calibratedSurfaces = outputTensor
        factorSensi = None
        
        if initialValueForFactors.ndim == 1 :
            calibratedSurfaces = np.reshape(reconstructedSurface, (nbPoints))
            if reshapedJacobian is not None :
                factorSensi = np.reshape(reshapedJacobian, (nbPoints, nbFactors))
        elif initialValueForFactors.ndim == 2 :
            calibratedSurfaces = np.reshape(reconstructedSurface, (nbObs,nbPoints))
            if reshapedJacobian is not None :
                factorSensi = np.reshape(reshapedJacobian, (nbObs, nbPoints, nbFactors))
        
        
        return calibratedSurfaces, factorSensi 

