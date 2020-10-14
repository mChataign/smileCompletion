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
import pickle

from functools import partial
from abc import ABCMeta, abstractmethod

import plottingTools
import factorialModel
import loadData

tmpFolder = "./tmp/" #"./../tmp/"
#tmpFolder = os.environ['tmp']


class Functional_encoder(nn.Module):
    def __init__(self, n_factor):
        super(Functional_encoder, self).__init__()

        self.linear1 = nn.Linear(n_factor + 2, 20)
        self.linear2 = nn.Linear(20, 1)
        # self.linear2 = nn.Linear(20, 20)
        # self.linear3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        # x = torch.tanh(x)
        # x = self.linear3(x)
        return x


class Code(nn.Module):
    def __init__(self, n_obs, n_factor, initialValue = None):
        super(Code, self).__init__()
        # self.code = nn.Parameter(torch.normal(0, 1, size=(n_obs, n_factor)), requires_grad=True)
        initialValueTorch = torch.zeros((n_obs, n_factor)) if initialValue is None else torch.tensor( np.reshape(initialValue, (n_obs, n_factor) ).astype(np.float32) ).float()
        self.code = nn.Parameter(initialValueTorch, requires_grad=True)







class pytorchModel(factorialModel.FactorialModel) :
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestPyTorchModel"):
        #pytorch objects 
        self.optimizer = None
        self.optimizer_code = None 
        
        self.MeanFwd = 0
        self.StdFwd = 1
        self.MeanLogMaturity = 0
        self.StdLogMaturity = 1
        self.MeanLogMoneyness = 0
        self.StdLogMoneyness = 0
        
        super().__init__(learningRate, hyperParameters, nbUnitsPerLayer, nbFactors, 
                         modelName = modelName)
        
        self.metaModelName = self.metaModelName.replace(".cpkt", "Torch")
        self.metaModelNameInit = self.metaModelNameInit.replace(".cpkt", "Torch") 
        
    #Build the architecture, losses and optimizer.
    def buildModel(self):
        self.fe = Functional_encoder(self.nbFactors) #Neural network architecture
        return
    
        
    #######################################################################################################
    #Training functions
    #######################################################################################################
    #Extract for each day the volatility value as output values the coordinates as input input values
    def getLocationFromDatasetList(self, dataSet):
        if dataSet[1].ndim > 1 :#historical data
            nbObs = dataSet[1].shape[0]
            nbPoints = dataSet[1].shape[1]
            
            vol = dataSet[0].values if dataSet[0] is not None else dataSet[0]
            
            coordinates = dataSet[1]
            yCoor = np.ravel(coordinates.applymap(lambda x : x[1]))
            xCoor = np.ravel(coordinates.applymap(lambda x : x[0]))
            l_Feature = np.reshape(np.vstack([xCoor, yCoor]).T, (nbObs, nbPoints, 2))
        else :#Data for a single day
            nbObs = 1
            nbPoints = dataSet[1].shape[0]
            
            vol = np.expand_dims(dataSet[0].values, 0) if dataSet[0] is not None else dataSet[0]
            
            coordinates = dataSet[1]
            yCoor = np.ravel(coordinates.map(lambda x : x[1]))
            xCoor = np.ravel(coordinates.map(lambda x : x[0]))
            l_Feature = np.reshape(np.vstack([xCoor, yCoor]).T, (nbObs, nbPoints, 2))
            
        return l_Feature, vol
    
    
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
        
        self.MeanFwd = np.nanmean(np.ravel(inputTrain[2]))
        self.StdFwd = np.nanstd(np.ravel(inputTrain[2]))
        self.MeanLogMaturity = np.nanmean(self.getLogMaturities(inputTrain))
        self.StdLogMaturity = np.nanstd(self.getLogMaturities(inputTrain))
        self.MeanLogMoneyness = np.nanmean(self.getLogMoneyness(inputTrain))
        self.StdLogMoneyness = np.nanstd(self.getLogMoneyness(inputTrain))
        
        self.restoringGraph()
        nbDays = inputTrain[0].shape[0]
        self.code = Code(nbDays, self.nbFactors) #Latent variables
        self.optimizer = torch.optim.Adam(self.fe.parameters(), lr=1e-2) #Optimizer for fe
        self.optimizer_code = torch.optim.Adam(self.code.parameters(), lr=1e-2) #Optimizer for code
        
        #x_vec = np.linspace(0, 1, 1000 + 1)[0:-1] 
        #y_vec = np.zeros(len(x_vec))
        
        line1 = []
        trainingLoss = []
        def getLogMaturities(batch):
            return batch[1].applymap(lambda x : np.log(x[0] / 252.0))

        for __ in range(nbEpoch):#range(100000): #Number of epochs
            res_ = []
            
            miniBatchIndex = np.random.choice(nbDays, 200) if (200 <= nbDays) else np.arange(nbDays)
            for k in miniBatchIndex : #random mini-batch
                batch = [(elt.iloc[k] if elt is not None else None)  for elt in inputTrain]
                keptCols = batch[0].dropna().index # batch[0].dropna(axis = 1, how="all").columns
                batch = [(elt[keptCols] if elt is not None else None)  for elt in batch ]
                
                tensorList = self.evalBatch(batch, self.code.code[k, :])
                
                res_.append( tensorList[2] ) #mean of squared errors for each observation
            res = torch.mean(torch.stack(res_)) #Mean along mini-batches : training loss
            
            # res += torch.mean(torch.mean(code.code))**2 + (torch.mean(torch.mean(code.code**2)-1))**2

            if __ < 300:#Pre-train only neural network
                self.optimizer.zero_grad() #set gradient to zero
                res.backward() #compute loss gradient wrt neural weights
                self.optimizer.step() #update neural weights and go to next iteration

            else:#Train neural network and  codes
                self.optimizer.zero_grad() #set gradient wrt neural weights to zero
                self.optimizer_code.zero_grad() #set gradient wrt code
                res.backward() # compute loss gradient by AAD
                self.optimizer.step() #Update neural weights
                self.optimizer_code.step() #Update code
            
            trainingLoss.append(np.sqrt(res.item()))
            print(__, trainingLoss[-1]) #print RMSE
            if trainingLoss[-1] <= min(trainingLoss) :
                if self.verbose :
                    print("New Best error : ", trainingLoss[-1])
                self.saveModel(self.metaModelName + "fe")#save neural weights
                torch.save(self.code.state_dict(), self.metaModelName + "code") #save code
            #y_vec[-1] = trainingLoss[-1]
            #line1 = live_plotter(x_vec, y_vec, line1)
            #y_vec = np.append(y_vec[1:], 0.0)
            
        
        
        return np.array(trainingLoss)
    
    def serializeObject(self, object, fileName):
        #Delete former file version
        #for f in glob.glob(fileName + "*"):
        #    os.remove(f)
        
        with open(fileName, "wb") as f :
            pickle.dump(object, f, protocol=3)
        
        return
    
    def readObject(self, fileName):
        with open(fileName, "rb") as f :
            object = pickle.load(f)
        
        return object
    
    #save metamodel (graph and variable values)
    def saveModel(self, pathFile): 
        #Delete former file version
        for f in glob.glob(pathFile + "*"):
            os.remove(f)
        torch.save(self.fe.state_dict(), pathFile) #save neural weights
        
        scalingValue = [self.MeanFwd, self.StdFwd, self.MeanLogMaturity, self.StdLogMaturity, self.MeanLogMoneyness, self.StdLogMoneyness]
        self.serializeObject(scalingValue, pathFile + "scale")
        return None
    
    
    def restoreWeights(self, fileName = None):
        #Restore graph of operations
        self.fe.load_state_dict(torch.load((self.metaModelName  + "fe") if fileName is None else fileName))
        scalingValue = self.readObject(((self.metaModelName  + "fe") if fileName is None else fileName) + "scale")
        
        self.MeanFwd = scalingValue[0]
        self.StdFwd = scalingValue[1]
        self.MeanLogMaturity = scalingValue[2]
        self.StdLogMaturity = scalingValue[3]
        self.MeanLogMoneyness = scalingValue[4]
        self.StdLogMoneyness = scalingValue[5]
        return
    
    def restoringGraph(self):
        #Restore graph of operations
        self.fe = None
        self.code = None
        
        self.buildModel()
        return
    
    #######################################################################################################
    #Evaluation functions
    #######################################################################################################
    
    #Same but with default session 
    def evalModel(self, inputTest):
        bestLoss, reconstructedSurface, encodings, _ =  self.calibratedFactors(inputTest)
        return bestLoss, reconstructedSurface, encodings
    
    
    def castToDataFrame(self, x):
        if type(x)==type(pd.Series()):
            return pd.DataFrame(x.values, index = x.index, columns = [x.name])
        return x
    
    def getLogMoneyness(self, batch):
        return batch[1].applymap(lambda x : (x[1])) if type(batch[1])==type(pd.DataFrame()) else batch[1].map(lambda x : (x[1]))
        
    def getLogMaturities(self, batch):
        return batch[1].applymap(lambda x : np.log(x[0])) if type(batch[1])==type(pd.DataFrame()) else batch[1].map(lambda x : np.log(x[0]))
    
    def evalBatch(self, batch, code):
        
        batchLogMoneyness = self.getLogMoneyness(batch)
        scaledMoneyness = (batchLogMoneyness.values - self.MeanLogMoneyness) / self.StdLogMoneyness
        logMoneynessTensor = torch.Tensor(np.expand_dims(scaledMoneyness, 1)).float() #Log moneyness
        
        # for j in np.random.choice(len(test[k]), 10):
        # filt  = test[k].nBizDays >= 10
        batchLogMat = self.getLogMaturities(batch)
        scaledMat = (batchLogMat.values - self.MeanLogMaturity) / self.StdLogMaturity
        logMaturity = torch.tensor( np.expand_dims(scaledMat, 1) , requires_grad=True).float() 
        
        codeTensor = code.repeat(batch[0].shape[0], 1).float()
        refVol = torch.tensor(batch[0].values)
        
        inputTensor = torch.cat((logMoneynessTensor, logMaturity, codeTensor), dim=1)
        outputTensor = self.fe( inputTensor )[:, 0]
        
        loss = torch.mean( (outputTensor - refVol)[~torch.isnan(outputTensor)] ** 2 )#torch.nanmean( (outputTensor - refVol) ** 2 ) 
        return inputTensor, outputTensor, loss, logMaturity, codeTensor, logMoneynessTensor
    
    def getArbitrageTheta(self, dataSetList, initialFactorValue):
        
        sparseSurface = dataSetList[0]
        #Build tensor for reconstruction
        def reshapeDataset(df):
            return pd.DataFrame(np.reshape([df.values], (1,df.shape[0])), 
                                           columns = df.index)
        reshapedDatasetList = [reshapeDataset(x) if x is not None else x for x in dataSetList]
        
        reshapedFactorValue = pd.DataFrame(np.reshape([initialFactorValue], (1,initialFactorValue.shape[0])))
        
        self.restoringGraph()
        
        nbDays = reshapedDatasetList[0].shape[0]
        nbFactors = reshapedDatasetList[0].shape[1]
        self.code = Code(nbDays, self.nbFactors, initialValue = reshapedFactorValue.values) #Latent variables
        self.optimizer_code = torch.optim.Adam(self.code.parameters(), lr=1e-2) #Optimizer for code
        
        self.restoreWeights()
        
        volPred = []
        for d in np.arange(nbDays) : 
            batchOriginal = [(elt.iloc[d] if elt is not None else None)  for elt in reshapedDatasetList]
            #get valid coordinates
            filteredBatch, filteredBatchCoordinates = loadData.removePointsWithInvalidCoordinates(batchOriginal[0],
                                                                                                  batchOriginal[1])
            keptCols = filteredBatch.index #batch[0].dropna(axis = 1, how="all").columns
            batch = [(elt[keptCols] if elt is not None else None)  for elt in batchOriginal ]
            
            #gradSeries = pd.Series(np.zeros_like(batch.values), index = batch.index)
            #for idx in gradSeries.index : 
            tensorList = self.evalBatch(batch, self.code.code[d, :])
            outputTensor = tensorList[1]
            logMaturity = tensorList[3]
            logMaturity.retain_grad()
                
            maturityTensor = torch.exp(logMaturity).view(-1)
            impliedTotalVariance = torch.square(outputTensor) * maturityTensor
            self.fe.zero_grad()
            impliedTotalVariance.backward( torch.ones_like(outputTensor).float() )
            gradSeries = pd.Series((logMaturity.grad.view(-1) / maturityTensor.view(-1)).detach().numpy().reshape(batch[0].shape), index = batch[0].index)
            print(gradSeries)
            
            #indexZero = torch.Tensor(0).to(dtype=torch.long)
            #for i, (logMat, output) in enumerate(zip(logMaturity[:,0] ,outputTensor)) : #range(impliedTotalVariance.size(0)):
            #    print((logMat, output))
            #    maturityTensor = torch.exp(logMat)
            #    impliedTotalVariance = torch.square(output) * maturityTensor
            #    
            #    gradTheta = torch.autograd.grad(impliedTotalVariance, logMat,
            #                                    retain_graph=True, allow_unused=True, create_graph=True) / maturityTensor)
            #    gradList.append(gradTheta.item())
            
            #for i in range(batch[0].shape[0]):
            #    gradTheta[i, 0] =  torch.autograd.grad(impliedTotalVariance[i], 
            #                                           [logMaturity[i,0]], 
            #                                           retain_graph=True, allow_unused=True, create_graph=True) / torch.exp(logMaturity[i,0])
            
            gradSeries = gradSeries.append(batchOriginal[0].drop(keptCols))[batchOriginal[0].index]
            volPred.append(gradSeries)
        
        volDf = pd.concat(volPred, axis = 1).transpose()
        reshapedReconstruction = pd.DataFrame(volDf.values, 
                                              index = reshapedDatasetList[0].index, 
                                              columns = reshapedDatasetList[0].columns)
        
        return reshapedReconstruction#.rename(sparseSurface.name)
    
    def calibratedFactors(self, dataSetList, initialFactorValue = None):
        #input, output = self.getLocationFromDatasetList(dataSetList)
        
        self.restoringGraph()
        
        nbDays = dataSetList[0].shape[0]
        self.code = Code(nbDays, self.nbFactors, initialValue = None if initialFactorValue is None else initialFactorValue.values) #Latent variables
        self.optimizer_code = torch.optim.Adam(self.code.parameters(), lr=1e-2) #Optimizer for code
        
        
        self.restoreWeights()
        
        def getLogMaturities(batch):
            return batch[1].applymap(lambda x : np.log(x[0]))
        
        nbCalibrationStep = 1000 if ("nbCalibrationStep" not in self.hyperParameters) else self.hyperParameters["nbCalibrationStep"]
        calibrationLosses = []
        for __ in range(nbCalibrationStep):#range(100000): #Number of epochs
            res_ = []
            
            miniBatchIndex = np.random.choice(nbDays, 200) if (200 <= nbDays) else np.arange(nbDays)
            for k in miniBatchIndex : #random mini-batch
                batch = [(elt.iloc[k] if elt is not None else None)  for elt in dataSetList]
                keptCols = batch[0].dropna().index #batch[0].dropna(axis = 1, how="all").columns
                batch = [(elt[keptCols] if elt is not None else None)  for elt in batch ]
                
                tensorList = self.evalBatch(batch, self.code.code[k, :])
                
                res_.append( tensorList[2] ) #mean of squared errors for each observation
            res = torch.mean(torch.stack(res_)) #Mean along mini-batches : training loss
            
            # res += torch.mean(torch.mean(code.code))**2 + (torch.mean(torch.mean(code.code**2)-1))**2
            self.optimizer_code.zero_grad() #set gradient to zero
            res.backward() #compute loss gradient wrt neural weights
            self.optimizer_code.step() #update neural weights and go to next iteration
            
            calibrationLosses.append(np.sqrt(res.item()))
            if self.verbose :
                print(__, calibrationLosses[-1]) #print RMSE
            
            if calibrationLosses[-1]<=min(calibrationLosses) :
                if self.verbose : 
                    print("New minimal error", calibrationLosses[-1]) #print RMSE
                torch.save(self.code.state_dict(), self.metaModelName + "code" + "Tmp") #save code
        
        volPred = []
        self.code.load_state_dict(torch.load(self.metaModelName + "code" + "Tmp"))
        for d in np.arange(nbDays) : 
            batchOriginal = [(elt.iloc[d] if elt is not None else None)  for elt in dataSetList]
            #get valid coordinates
            filteredBatch, filteredBatchCoordinates = loadData.removePointsWithInvalidCoordinates(batchOriginal[0],
                                                                                                  batchOriginal[1])
            keptCols = filteredBatch.index #batch[0].dropna(axis = 1, how="all").columns
            batch = [(elt[keptCols] if elt is not None else None)  for elt in batchOriginal ]
            
            tensorList = self.evalBatch(batch, self.code.code[d, :])
            predSeries = pd.Series(tensorList[1].detach().numpy().reshape(batch[0].shape), index = keptCols)
            predSeries = predSeries.append(batchOriginal[0].drop(keptCols))[batchOriginal[0].index]
            volPred.append(predSeries)
        
        volDf = pd.concat(volPred, axis = 1).transpose()
        reshapedReconstruction = pd.DataFrame(volDf.values, 
                                              index = dataSetList[0].index, 
                                              columns = dataSetList[0].columns)
        calibratedFactors = np.reshape(self.code.code.data.cpu().numpy(), (nbDays, self.nbFactors))
        bestCalibration = -1
        
        if self.verbose :
            print("Average Loss : ", calibrationLosses[bestCalibration])
        
        return calibrationLosses[bestCalibration], reshapedReconstruction, pd.DataFrame(calibratedFactors, index = dataSetList[0].index), calibrationLosses
    
    def completeDataTensor(self, 
                           sparseSurfaceList, 
                           initialValueForFactors, 
                           nbCalibrationStep, 
                           *args):
        
        
        #Rebuild tensor graph
        self.restoringGraph()
        
        sparseSurface = sparseSurfaceList[0]
        #Build tensor for reconstruction
        def reshapeDataset(df):
            return pd.DataFrame(np.reshape([df.values], (1,df.shape[0])), 
                                           columns = df.index)
        reshapedDatasetList = [reshapeDataset(x) if x is not None else x for x in sparseSurfaceList]
        
        reshapedValueForFactors = pd.DataFrame(np.reshape([initialValueForFactors],
                                                          (1,initialValueForFactors.shape[0])))
        
        tmp = self.calibratedFactors(reshapedDatasetList, initialFactorValue = reshapedValueForFactors)
        return tmp[0], np.ravel(tmp[2].values), tmp[1].iloc[0].rename(sparseSurface.name), pd.Series(tmp[3])
        
    
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
        
        
        self.code = Code(nbObs, self.nbFactors, initialValue = reshapedValueForFactors) #Latent variables
        codeTensor = self.code.code[k, :].repeat(nbPoints, 1)
        
        batchLogMat = self.getLogMaturities(dataSetList)
        scaledMat = (batchLogMat.values - self.MeanLogMaturity) / self.StdLogMaturity
        logMaturity = torch.tensor( np.expand_dims(scaledMat, 1) ).float() 
        
        batchLogMoneyness = self.getLogMoneyness(dataSetList)
        scaledMoneyness = (batchLogMoneyness.values - self.MeanLogMoneyness) / self.StdLogMoneyness
        logMoneynessTensor = torch.Tensor(np.expand_dims(scaledMoneyness, 1)).float() #Log moneyness
        
            
        inputTensor = torch.cat((logMoneynessTensor, logMaturity, codeTensor), dim=1)
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
        raise NotImplementedError("Implicit encoder model")
        return
    
        
    
    #Return None if not supported
    def getDecoderCoefficients(self):
        raise NotImplementedError("Abstract Class !")
        return None

