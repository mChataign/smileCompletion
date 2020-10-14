import os
import pandas as pd
import numpy as np
from pandas import DatetimeIndex
import dask
import scipy
import time

import seaborn as sns
from IPython.display import HTML, Image, display
from functools import partial
from abc import ABCMeta, abstractmethod

import loadData 
import plottingTools 

import factorialModel 
import dynamicModel
import teacher

def lossReconstruction(s1,s2):#One day
    return np.mean((s1-s2)**2)**(0.5)
    #return np.mean(np.mean(((s1-s2))**2, axis=1)**(0.5))




class TeacherDynamic(teacher.Teacher) :
    def __init__(self,
                 modelCompression, 
                 dataSet,
                 nbEpochs,
                 nbStepCalibrations,
                 dynamicModel):
        self.dynamicModel = dynamicModel
        #Dynamic model needs modelCompression for the relationship between surface and latentVariables
        self.dynamicModel.setEncodingModel(modelCompression) 
        
        super().__init__(modelCompression, 
                         dataSet, 
                         nbEpochs, 
                         nbStepCalibrations)
        
    #Fit hold model no training data
    def fit(self, restoreResults = False):
        if restoreResults : 
            self.testingLoss = self.readObject("trainingLoss")
        else :
            super().fit()
            
            _, factors = self.evalModel(self.dataSet.getTrainingDataForModel()[0].index)
            
            self.dynamicModel.train(self.dataSet.getTrainingDataForModel(), 
                                    factors)
        if self.saveResults :
            self.serializeObject(self.testingLoss, "trainingLoss")
        return
    
    #Plot some results for compression
    def diagnoseCompression(self, restoreResults = False):
        super().diagnoseCompression()
        
        
        if restoreResults :
            resCompression = self.readObject("compressionResult")
            filteredValue = resCompression["filteredValue"]
            filteredValueTest = resCompression["filteredValueTest"]
        else : 
            filteredValue = self.dynamicModel.iterateOnSeries(self.dataSet.getTrainingDataForModel(),
                                                              self.codings_Train)
            
            filteredValueTest = self.dynamicModel.iterateOnSeries(self.dataSet.getTestingDataForModel(),
                                                                  self.codings_val)
        plottingTools.printDelimiter(3)
        plottingTools.printIsolated("Filtered values for factors on the training data")
                                               
        plottingTools.plotFactor(filteredValue["muSerie"]-self.codings_Train)
        
        plottingTools.printIsolated("Filtered values for factors on the testing data")
        
        plottingTools.plotFactor(filteredValueTest["muSerie"]-self.codings_val)
        
        if self.saveResults :
            #Read results from parent class 
            resCompression = self.readObject("compressionResult")
            
            resCompression["filteredValue"] = filteredValue
            resCompression["filteredValueTest"] = filteredValueTest
            self.serializeObject(resCompression, "compressionResult")
        
        return 
        #Select in the past factor values yielding the closest surface
    

    
    #Assess completion along testing data history and recalibrate from latest availble factor values
    def backTestCompletion(self, restoreResults = False) :
        if self.outputs_val is None :
            raise ValueError("Diagnose compression before completing one day")
        
        if restoreResults : 
            result = self.readObject("completion")
        else :
            testingSet = self.dataSet.getTestingDataForModel()
            latestFactorValuesForTrainingSet = self.codings_Train.iloc[-1]
            deletedIndex = self.dataSet.maskedPoints
            
            #Evaluate factors on the testingSet to get real factors
            _ , trueSurface, trueFactors = self.model.evalModel(testingSet)
            trueSurface = self.dataSet.formatModelDataAsDataSet(testingSet)
                
            data = np.reshape(latestFactorValuesForTrainingSet.to_frame().transpose().append(trueFactors.head(-1)).values, 
                              trueFactors.shape)
            
            #Gather data for dynamic model
            wholeHistoryOfFactors = self.codings_Train.append(self.codings_val)
            wholeDataHistory = self.dataSet.getDataForModel()
            trainingDataSetSize = self.dataSet.getTrainingDataForModel()[0].shape
            
            #Make a forward pass on testing data
            factoryPenalizationSerie = None
            if "regularizeCompletion" in self.model.hyperParameters and self.model.hyperParameters["regularizeCompletion"]:
                factoryPenalizationSerie = self.dynamicModel.buildPenalizationFactoriesAlongHistory(wholeDataHistory, 
                                                                                                    wholeHistoryOfFactors)
            
            
            initialValuesForFactors = None
            if "inferCompletionFactor" in self.model.hyperParameters and self.model.hyperParameters["inferCompletionFactor"]:
                forwardPassResult = self.dynamicModel.iterateOnSeries(wholeDataHistory,
                                                                      wholeHistoryOfFactors)
                predictions = forwardPassResult["muConditionalSerie"]
                initialValuesForFactors = pd.DataFrame(predictions.loc[trueFactors.index].values, 
                                                       index = trueFactors.index, 
                                                       columns = latestFactorValuesForTrainingSet.index).astype(np.float32)
            else: 
                initialValuesForFactors = pd.DataFrame(data, 
                                                       index = trueFactors.index, 
                                                       columns = latestFactorValuesForTrainingSet.index)
            
            #Iterate on testing Set observations to obtain
            calibratedFactorValues = []
            calibrationLosses = []
            completedSurfaces = []
            
            deletedValues = None 
            
            counter = 0
            for counter in range(testingSet.shape[0]) : 
                observation=testingSet.iloc[counter]
                
                if deletedValues is None : 
                    #Choose which points will be deleted
                    deletedValues = deletedIndex
                
                #Make a deep copy to prevent observation instance from being polluted with np.NaN 
                observationToComplete = observation.copy()
                observationToComplete[deletedValues] = np.NaN
                
                #Create the penalization from the dynamic model
                dayNumber = trainingDataSetSize[0] + counter
                
                if factoryPenalizationSerie is not None :
                    penalization = factoryPenalizationSerie.iloc[dayNumber]
                    self.model.setDynamicPenalizationFactory(penalization)
                
                initialFactorValues = self.selectClosestObservationsInThePast(testingSet.index[counter],
                                                                              wholeHistoryOfFactors, 
                                                                              observationToComplete)
                
                
                l, f, s, _ = self.executeCompletion(observationToComplete, 
                                                    initialFactorValues,#initialValuesForFactors.iloc[counter], 
                                                    self.nbStepCalibrations)
                #if self.model.hyperParameters["filterCompletionFactor"]:
                #filter
                
                #completedSurfaces.append(self.dataSet.formatModelDataAsDataSet(s))
                completedSurfaces.append(s)
                calibratedFactorValues.append(f)
                calibrationLosses.append(l)
            
            result = {'calibratedFactorValues' : pd.DataFrame(calibratedFactorValues, index = testingSet.index), 
                      'calibrationLosses' : pd.Series(calibrationLosses, index = testingSet.index),
                      'completedSurfacesUntransformed' : pd.DataFrame(np.reshape(completedSurfaces,testingSet.shape), 
                                                                      index = testingSet.index, 
                                                                      columns = testingSet.columns),
                      'trueFactors' : pd.DataFrame(trueFactors, index = testingSet.index),
                      'trueSurface' : trueSurface[0],
                      'deletedValues' : deletedValues}
            result['completedSurfaces'] = self.dataSet.formatModelDataAsDataSet(result['completedSurfacesUntransformed'])
            if self.saveResults : 
                self.serializeObject(result, "completion")
        self.plotBackTestCompletion(result)
        
        return result
    
    #
    def diagnoseCompletion(self):
        return self.backTestCompletion()
    
    
    #Complete surface for a given date
    def completionTest(self, date):
        if self.outputs_val is None :
            raise ValueError("Diagnose compression before complteing one day")
        
        deletedIndex = self.dataSet.maskedPoints
        
        allDays = self.dataSet.getDataForModel()[0].index
        dayObserved = allDays[(allDays <= date)]
        wholeDataHistory = self.dataSet.getDataForModel(dayObserved)
        wholeHistoryOfFactors = self.codings_Train.append(self.codings_val).loc[dayObserved]
        
        #Delete points inside the surface
        surfaceToComplete = wholeDataHistory.loc[date]
        surfaceSparse = surfaceToComplete.copy()
        surfaceSparse[deletedIndex] = np.NaN 
        
        #Get latest available values for latent variables, -1 today, -2 yesterday
        if wholeHistoryOfFactors.shape[0] > 1: 
            lastFactorsValues = wholeHistoryOfFactors.iloc[-2]
        else :
            lastFactorsValues = pd.Series(np.zeros(wholeHistoryOfFactors.shape[1]), 
                                          index = wholeHistoryOfFactors.columns)
        
        lastFactorsValues = self.selectClosestObservationsInThePast(date,
                                                                    wholeHistoryOfFactors,
                                                                    surfaceSparse)
        
        factoryPenalization = None
        if "regularizeCompletion" in self.model.hyperParameters and self.model.hyperParameters["regularizeCompletion"]:
            factoryPenalization = self.dynamicModel.buildPenalizationFactory(wholeDataHistory, 
                                                                             wholeHistoryOfFactors)
            self.model.setDynamicPenalizationFactory(factoryPenalization)
        
        #Complete the surface
        l, f, s, lSerie = self.executeCompletion(surfaceSparse, 
                                                 lastFactorsValues, 
                                                 self.nbStepCalibrations)
        
        plottingTools.plotLossThroughEpochs(lSerie, 
                                            title = "Calibration loss on non-missing points through epochs")
        
        originalSurface = self.dataSet.formatModelDataAsDataSet(surfaceToComplete)
        outputSurface = pd.Series(self.dataSet.formatModelDataAsDataSet(s), index = surfaceToComplete.index)
        
        plottingTools.printIsolated("L2 Reconstruction loss : ", 
                                    lossReconstruction(originalSurface,outputSurface))
        return l, f, outputSurface, originalSurface
    


