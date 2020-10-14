import os
import pandas as pd
import numpy as np
from pandas import DatetimeIndex
import dask
import scipy
import time
import glob

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from matplotlib import animation, rc
from IPython.display import HTML, Image, display
from functools import partial
from abc import ABCMeta, abstractmethod
import scipy
from scipy.optimize import minimize, LinearConstraint, linprog, NonlinearConstraint

import loadData 
import plottingTools 
import pickle

import factorialModel 
import gaussianProcess

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def lossReconstruction(s1,s2):#One day
    return np.nanmean((s1-s2)**2)**(0.5)
    #return np.mean(np.mean(((s1-s2))**2, axis=1)**(0.5))




class Teacher :
    def __init__(self,
                 model, 
                 dataSet,
                 nbEpochs,
                 nbStepCalibrations):
        self.model = model
        self.dataSet = dataSet
        self.nbEpochs = nbEpochs
        self.nbStepCalibrations = nbStepCalibrations
        self.saveResults = True
        
        #Hard-coded parameters for expert user
        self.diagnoseOriginalData = False
        self.colorMapSystem = None
        self.plotType = None
        
        #Temporary results in cache
        self.testingLoss = None
        self.outputs_val = None
        self.codings_val = None
        self.training_val = None
        self.codings_Train = None
        
    #Fit hold model no training data
    def fit(self, restoreResults = False):
        if restoreResults : 
            self.testingLoss = self.readObject("trainingLoss")
        else :
            self.testingLoss = self.model.train(self.dataSet.getTrainingDataForModel(), 
                                                self.nbEpochs, 
                                                self.dataSet.getTestingDataForModel())
        if self.saveResults :
            self.serializeObject(self.testingLoss, "trainingLoss")
        return
    
    def serializeObject(self, object, objectName):
        fileName = self.model.metaModelName.replace(".cpkt","") + objectName
        #Delete former file version
        #for f in glob.glob(fileName + "*"):
        #    os.remove(f)
        
        with open(fileName, "wb") as f :
            pickle.dump(object, f, protocol=3)
        
        return
    
    def readObject(self, objectName):
        fileName = self.model.metaModelName.replace(".cpkt","") + objectName
        with open(fileName, "rb") as f :
            object = pickle.load(f)
        
        return object
    
    def saveModel(self, session, pathFile): 
        cpkt = self.saver.save(session, pathFile, 
                               latest_filename="LatestSave")
        return cpkt
    #Evaluate the model for given list of dates
    def evalModel(self, dates):
        _, outputs, factors = self.model.evalModel(self.dataSet.getDataForModel(dates))
        return self.dataSet.formatModelDataAsDataSet(outputs), factors
    
    
    
    #Two behaviours : comparison of allocation, comparison of vega strategy
    #Comparison of allocation : Compute the cumulative sum of discounted completion loss weighted by vega exposure (VegaAllocation)
    #Comparison of vega strategy : Compute the cumulative sum of discounted implied volatility moves weighted by one the two vega strategy
    #Here vega exposure means of much move our portfolio when implied volatility surface moves
    #Another metric normalized P&L explains how much percentage of the portfolio variation can be explained with the completed surface
    #Compute the sum of discounted completion loss weighted by vega exposure (VegaAllocation)
    #Here vega exposure means of much move our portfolio when implied volatility surface moves
    #Another metric normalized P&L explains how much percentage of the portfolio variation can be explained with the completed surface
    def ProfitAndLossVegaStrategies(self,
                                    trueDataSerie, 
                                    testedVegaAllocation,
                                    originalVegaAllocation, 
                                    riskFreeRate = 0):
        #Cumulative sum of daily P&L 
        #where daily P&L is the inner product of volatility variation and benchmark vega allocation
        PAndLOriginalVega = 0 
        #Cumulative sum of daily P&L 
        #where daily P&L is the inner product of volatility variation and projected vega allocation
        PAndLTestedVega = 0 
        #Gap between PAndLOriginalVega and PAndLTestedVega
        PAndLTrackingCost = 0 
        PAndLTrackingCostL2 = 0 
        
        PAndLOriginalVegaHistory = []
        PAndLTestedVegaHistory = []
        PAndLTrackingCostHistory = []
        PAndLTrackingCostL2History = []
        dailyTrackingCostHistory = []
        investablePoints = self.dataSet.decideInvestableInstruments()
        
        #If no benchmark vega allocation is provided then we assume a uniform exposure normalized 
        #such that sum of vega is equal to one
        
        #trueVariation  = (trueDataSerie[0].diff()).dropna(how="all", axis=0)
        
        #Determine previous day as the previous business day (ignoring holiday for now)
        previousDay = trueDataSerie[0].index[0]
        #previousDay += pd.tseries.offsets.BusinessDay(n = -1)
        
        for day, gapSurface in trueDataSerie[0].tail(-1).iterrows(): #for each day
            #Cumulative sum of relative signed gap weighted by vega exposure
            deltaDay = day - previousDay
            discountStep = (riskFreeRate * float(deltaDay.days) / 365.0)
            
            dailyTrueVariation = (trueDataSerie[0].loc[day][investablePoints[previousDay][1]].values - 
                                  trueDataSerie[0].loc[previousDay][investablePoints[previousDay][0]].values)
            dailyTrueVariationSerie = pd.Series(dailyTrueVariation, 
                                                index = investablePoints[previousDay][0])
            #dailyPAndLOriginalVega = np.nansum(gapSurface * originalVegaAllocation.loc[day])
            dailyPAndLOriginalVega = np.nanmean(dailyTrueVariation)
            PAndLOriginalVega += PAndLOriginalVega * discountStep - dailyPAndLOriginalVega
            PAndLOriginalVegaHistory.append(PAndLOriginalVega)
            
            dailyAllocation = testedVegaAllocation.loc[day]
            dailyPAndLTestedVega = np.nansum(dailyTrueVariationSerie[dailyAllocation.index] * dailyAllocation) 
            PAndLTestedVega += PAndLTestedVega * discountStep - dailyPAndLTestedVega
            PAndLTestedVegaHistory.append(PAndLTestedVega)
            
            dailyTrackingCost = dailyPAndLTestedVega - dailyPAndLOriginalVega 
            PAndLTrackingCost += PAndLTrackingCost * discountStep - dailyTrackingCost
            PAndLTrackingCostHistory.append(PAndLTrackingCost)
            dailyTrackingCostHistory.append(dailyTrackingCost)
            
            dailyTrackingCostL2 = np.square(dailyPAndLTestedVega - dailyPAndLOriginalVega)
            PAndLTrackingCostL2 = np.sqrt( np.square(PAndLTrackingCostL2) + np.square(PAndLTrackingCostL2 * discountStep) + dailyTrackingCostL2)
            PAndLTrackingCostL2History.append(PAndLTrackingCostL2)
            
            previousDay = day
            
        
        originalVegaPAndLSerie = pd.Series(PAndLOriginalVegaHistory, index=trueDataSerie[0].tail(-1).index).rename("P&L Vanilla Vega")
        testedVegaPAndLSerie = pd.Series(PAndLTestedVegaHistory, index=trueDataSerie[0].tail(-1).index).rename("P&L Projected Vega")
        trackingCostPAndLSerie = pd.Series(PAndLTrackingCostHistory, index=trueDataSerie[0].tail(-1).index).rename("Tracking cost")
        trackingCostL2PAndLSerie = pd.Series(PAndLTrackingCostL2History, index=trueDataSerie[0].tail(-1).index).rename("Tracking L2 cost")
        dailyTrackingCostSerie = pd.Series(dailyTrackingCostHistory, index=trueDataSerie[0].tail(-1).index).rename("Daily error")
        
        nbHedgingPeriods = trueDataSerie[0].tail(-1).shape[0] 
        index=["P&L variation divided by True P&L variation",
               "Percentage of cumulative P&L variation explained", 
               "Tracking cost of completion portfolio", 
               "Tracking cost L2 of completion portfolio",
               "Tracking cost of completion portfolio divided by total variation of true portfolio"]
        columns = ["Daily", "Annualized (255 days a year)", "Total"]
        
        PAndLQuotient = PAndLTestedVega / PAndLOriginalVega
        RelativeTrackingCost = PAndLTrackingCost/PAndLOriginalVega
        cumulativePAndExplained = abs((originalVegaPAndLSerie.cumprod() / testedVegaPAndLSerie.cumprod()).iloc[-1])
        cells = [[PAndLQuotient,PAndLQuotient,PAndLQuotient],
                 [cumulativePAndExplained**(1.0 / nbHedgingPeriods),
                  cumulativePAndExplained**(255.0 / nbHedgingPeriods),
                  cumulativePAndExplained],
                 [PAndLTrackingCost*(1.0 / nbHedgingPeriods),
                  PAndLTrackingCost*(255.0 / nbHedgingPeriods),
                  PAndLTrackingCost],
                 [PAndLTrackingCostL2*np.sqrt(1.0 / nbHedgingPeriods), #standard deviation
                  PAndLTrackingCostL2*np.sqrt(255.0 / nbHedgingPeriods), 
                  PAndLTrackingCostL2],
                 [RelativeTrackingCost,RelativeTrackingCost,RelativeTrackingCost]]
        
        summary = pd.DataFrame(cells, index=index, columns=columns)
        print()
        display(HTML(summary.to_html()))
        print()
        
        plottingTools.plotSeries([originalVegaPAndLSerie, testedVegaPAndLSerie, trackingCostPAndLSerie],
                                 title="P&L performance")
        plottingTools.plotSeries([originalVegaPAndLSerie, testedVegaPAndLSerie, trackingCostL2PAndLSerie],
                                 title="P&L performance")
        if dailyTrackingCostSerie.std() > 1e-6 :
            dailyTrackingCostSerie.plot.kde(bw_method=0.5)
        refSize=5
        plt.ylabel("Density", fontsize=2*refSize, labelpad=3*refSize)
        plt.xlabel("Tracking error", fontsize=2*refSize, labelpad=3*refSize)
        plt.show()
        return summary
    
        
    #Compute the sum of discounted completion loss weighted by vega exposure (VegaAllocation)
    #Here vega exposure means of much move our portfolio when implied volatility surface moves
    #Another metric normalized P&L explains how much percentage of the portfolio variation can be explained with the completed surface
    def ProfitAndLoss(self,
                      trueDataSerie, 
                      ApproximatedDataSerie, 
                      VegaAllocation = None, 
                      riskFreeRate = 0):
        totalVariationAccount = 0 #Cost of tracking real portfolio with a cash position
        completionVariationAccount = 0 #Cost of tracking completion portfolio with a cash position
        PAndLCompletion = 0 #Tracking cost between completion portfolio and real portfolio
        trackingErrorL2 = 0
        PAndL = 0 #
        cumulativeAccountExplained = 1
        
        totalVariationAccountHistory = []
        PAndLCompletionHistory = []
        TrackingErrorL2History = []
        dailyLossSerie = []
        
        
        
        #If no vega allocation is provided then we assume a uniform exposure normalized 
        #such that sum of vega is equal to one
        nbPoints = trueDataSerie.shape[1]
        usedVegaAllocation = VegaAllocation if VegaAllocation else (np.ones(nbPoints)/float(nbPoints))
        
        approximationCompletionError = ApproximatedDataSerie - trueDataSerie
        #approximatedVariation = (ApproximatedDataSerie - trueDataSerie.shift()).dropna(how="all")
        #completionVariation  = (ApproximatedDataSerie.diff()).dropna(how="all")
        #trueVariation  = (trueDataSerie.diff()).dropna(how="all")
        
        investablePoints = self.dataSet.decideInvestableInstruments()
        
        #Determine previous day as the previous business day (ignoring holiday for now)
        previousDay = trueDataSerie.index[0]
        #previousDay += pd.tseries.offsets.BusinessDay(n = -1)
        
        for day, gapSurface in approximationCompletionError.iterrows(): #for each day
            #Cumulative sum of relative signed gap weighted by vega exposure
            deltaDay = day - previousDay
            
            if day != approximationCompletionError.index[0] :
                #dailyLoss = np.sum(gapSurface * usedVegaAllocation) 
                dailyCompletionError = gapSurface[investablePoints[previousDay][1]]
                dailyLoss =  np.nanmean(dailyCompletionError)
                PAndL += PAndL * (riskFreeRate * float(deltaDay.days) / 365.0) - dailyLoss
                
                #dailyTrueVariation = np.nansum(trueVariation.loc[day] * usedVegaAllocation)
                dailyDiff = (trueDataSerie.loc[day][investablePoints[previousDay][1]].values - 
                             trueDataSerie.loc[previousDay][investablePoints[previousDay][0]].values)
                dailyTrueVariation = np.nanmean(dailyDiff)
                
                #dailyLoss = np.nansum(approximationCompletionError.loc[day] * usedVegaAllocation)
                
                dailyApproxDiff = (ApproximatedDataSerie.loc[day][investablePoints[previousDay][1]].values - 
                                   trueDataSerie.loc[previousDay][investablePoints[previousDay][0]].values)
                dailyApproxVariation = np.nanmean(dailyApproxDiff)
                
                dailyCompletionDiff = (ApproximatedDataSerie.loc[day][investablePoints[previousDay][1]].values - 
                                       ApproximatedDataSerie.loc[previousDay][investablePoints[previousDay][0]].values)
                dailyCompletionVariation = np.nanmean(dailyCompletionDiff)
                
                discountStep = (riskFreeRate * float(deltaDay.days) / 365.0)
                
                totalVariationAccount += totalVariationAccount * discountStep - dailyTrueVariation
                completionVariationAccount += completionVariationAccount * discountStep - dailyCompletionVariation
                PAndLCompletion += PAndLCompletion * discountStep - dailyLoss
                trackingErrorL2 = np.sqrt(np.square(trackingErrorL2) + np.square(trackingErrorL2 * discountStep) + np.square(dailyLoss))
                
                PAndLCompletionHistory.append(PAndLCompletion)
                dailyLossSerie.append(dailyLoss)
                totalVariationAccountHistory.append(totalVariationAccount)
                TrackingErrorL2History.append(trackingErrorL2)
                
                cumulativeAccountExplained *= abs(dailyApproxVariation/dailyTrueVariation)
                
            else :
                PAndL = 0
            
            previousDay = day
            
        
        
        
        nbHedgingPeriods = approximationCompletionError.shape[0] - 1
        index=["Completion P&L variation divided by True P&L variation",
               "Percentage of cumulative P&L variation explained", 
               "Tracking cost of completion portfolio",
               "Tracking L2 cost of completion portfolio",
               "Tracking cost of completion portfolio divided by total variation of true portfolio"]
        columns = ["Daily", "Annualized (255 days a year)", "Total"]
        
        PAndLQuotient = completionVariationAccount / totalVariationAccount
        RelativeTrackingCost = PAndLCompletion/totalVariationAccount
        cells = [[PAndLQuotient,PAndLQuotient,PAndLQuotient],
                 [cumulativeAccountExplained**(1.0 / nbHedgingPeriods),
                  cumulativeAccountExplained**(255.0 / nbHedgingPeriods),
                  cumulativeAccountExplained],
                 [PAndLCompletion*(1.0 / nbHedgingPeriods),
                  PAndLCompletion*(255.0 / nbHedgingPeriods),
                  PAndLCompletion],
                 [trackingErrorL2 * np.sqrt(1.0 / nbHedgingPeriods), #standard deviation
                  trackingErrorL2 * np.sqrt(255.0 / nbHedgingPeriods), 
                  trackingErrorL2],
                 [RelativeTrackingCost,RelativeTrackingCost,RelativeTrackingCost]]
        
        summary = pd.DataFrame(cells, index=index, columns=columns)
        print()
        display(HTML(summary.to_html()))
        print()
        
        plottingTools.plotSeries([pd.Series(PAndLCompletionHistory, index=trueDataSerie.tail(-1).index).rename("Tracking cost"),
                                  pd.Series(totalVariationAccountHistory, index=trueDataSerie.tail(-1).index).rename("Total Variation")],
                                 title="P&L performance")
        plottingTools.plotSeries([pd.Series(TrackingErrorL2History, index=trueDataSerie.tail(-1).index).rename("Tracking L2 cost"),
                                  pd.Series(totalVariationAccountHistory, index=trueDataSerie.tail(-1).index).rename("Total Variation")],
                                 title="P&L performance")
        #Plot loss density
        
        if np.std(dailyLossSerie) > 1e-6 :
            pd.Series(dailyLossSerie, index=trueDataSerie.tail(-1).index).plot.kde(bw_method=0.5)
        refSize=5
        plt.ylabel("Density", fontsize=2*refSize, labelpad=3*refSize)
        plt.xlabel("Tracking error", fontsize=2*refSize, labelpad=3*refSize)
        plt.show()
        return summary
    
    def latentfactorSanityCheck(self) : 
        encodingCorrelation = self.codings_Train.corr()
        print(encodingCorrelation)
        if encodingCorrelation.dropna().size > 0 :
            pca = PCA(n_components=encodingCorrelation.shape[0])
            _ = pca.fit_transform(scale(self.codings_Train.corr()))
            plt.plot(pca.explained_variance_ratio_)
            plt.title("Eigen value for latent space")
            plt.show()
    
    def printThetaArbitrage(self, historyPred, historyRef, codings):
        
        plottingTools.printDelimiter()
        print("Calendar arbitrage")
        errorsAbsRMSE = pd.Series(np.nanmean(np.square(historyRef - historyPred),axis=1)**0.5, 
                              index = historyRef.index)
        worstDayPred, worstDayRef = plottingTools.getWorstGrids(historyPred,
                                                                historyRef, 
                                                                errorsAbsRMSE)
        modelData = self.dataSet.getDataForModel(worstDayRef.name)
        coordinates = self.dataSet.formatModelDataAsDataSet(modelData)[1]
        encodings = codings.loc[worstDayRef.name]
        thetaSurface = self.model.getArbitrageTheta(modelData, encodings)
        plottingTools.plotGrid(thetaSurface.iloc[0],
                               coordinates,    
                               "Calendar condition for worst reconstruction on testing dataset", 
                               colorMapSystem=self.colorMapSystem, 
                               plotType=self.plotType,
                               refPoints = None)
        
        print("Minimal theta : ", thetaSurface.min().min())
        
        plottingTools.printDelimiter()
        return
    
    #Plot some results for compression
    def diagnoseCompression(self, restoreResults = False):
        if self.testingLoss is None :
            plottingTools.printIsolated("Please fit model on data before any diagnosis")
            return
        
        if restoreResults :
            resCompression = self.readObject("compressionResult")
            self.outputs_val = resCompression["outputs_val"]
            self.codings_val = resCompression["codings_val"]
            self.training_val = resCompression["training_val"]
            self.codings_Train = resCompression["codings_Train"]
        else : 
            _, self.outputs_val, self.codings_val = self.model.evalModel(self.dataSet.getTestingDataForModel())
            _, self.training_val, self.codings_Train = self.model.evalModel(self.dataSet.getTrainingDataForModel())
        
        predTestingValues = self.dataSet.formatModelDataAsDataSet(self.outputs_val)
        refTestingValues = self.dataSet.formatModelDataAsDataSet(self.dataSet.getTestingDataForModel())[0]
        predTrainingValues = self.dataSet.formatModelDataAsDataSet(self.training_val)
        refTrainingValues = self.dataSet.formatModelDataAsDataSet(self.dataSet.getTrainingDataForModel())[0]
        
        if self.saveResults :
            resCompression = {}
            resCompression["outputs_val"] = self.outputs_val
            resCompression["codings_val"] = self.codings_val
            resCompression["training_val"] = self.training_val
            resCompression["codings_Train"] = self.codings_Train
            resCompression["predTestingValues"] = predTestingValues
            resCompression["refTestingValues"] = refTestingValues
            resCompression["predTrainingValues"] = predTrainingValues
            resCompression["refTrainingValues"] = refTrainingValues
            self.serializeObject(resCompression, "compressionResult")
        
        plottingTools.diagnoseModels(self.codings_val, 
                                     predTrainingValues, 
                                     refTrainingValues, 
                                     predTestingValues,
                                     refTestingValues, 
                                     self.testingLoss,
                                     self.dataSet,
                                     colorMapSystem=self.colorMapSystem, 
                                     plotType=self.plotType)
        
        plottingTools.printDelimiter()
        
        self.printThetaArbitrage(predTestingValues, refTestingValues, self.codings_val)
        
        plottingTools.printDelimiter()
        
        self.latentfactorSanityCheck()
        
        plottingTools.printDelimiter()
        plottingTools.printIsolated("P&L training set")
        _ = self.ProfitAndLoss(refTrainingValues, predTrainingValues)
        
        plottingTools.printIsolated("P&L testing set")
        _ = self.ProfitAndLoss(refTestingValues, predTestingValues)
        
        if self.diagnoseOriginalData : 
            plottingTools.printDelimiter(3)
            plottingTools.printIsolated("With data used as input for learning model")
            plottingTools.diagnoseModels(self.codings_val, 
                                         self.training_val,
                                         self.dataSet.getTrainingDataForModel()[0], 
                                         self.outputs_val,
                                         self.dataSet.getTestingDataForModel()[0], 
                                         self.testingLoss, 
                                         self.dataSet,
                                         colorMapSystem=self.colorMapSystem, 
                                         plotType=self.plotType)
            
            plottingTools.printDelimiter()
            # _ = self.ProfitAndLoss(self.dataSet.getTestingDataForModel(), 
                                   # self.outputs_val)
        return 
    
    #Complete surface for a given date
    def completionTest(self, date):
        if self.outputs_val is None :
            raise ValueError("Diagnose compression before completing one day")
        
        fullDataSet = self.dataSet.getDataForModel(date)
        factorHistory = self.codings_Train.append(self.codings_val)
        deletedIndex = self.dataSet.maskedPoints
        
        #Delete points inside the surface
        surfaceToComplete = fullDataSet[0]
        surfaceSparse = self.dataSet.maskDataset(surfaceToComplete)
        
        #Get latest available values for latent variables
        #lastFactorsValues = factorHistory[ factorHistory.index < date].iloc[-1]
        lastFactorsValues = self.selectClosestObservationsInThePast(date,
                                                                    factorHistory,
                                                                    surfaceSparse)
        
        #Complete the surface
        l, f, s, lSerie = self.executeCompletion([surfaceSparse] + fullDataSet[1:], 
                                                 lastFactorsValues, 
                                                 self.nbStepCalibrations)
        
        plottingTools.plotLossThroughEpochs(lSerie, 
                                            title = "Calibration loss on non-missing points through epochs")
        
        originalSurface = self.dataSet.formatModelDataAsDataSet(surfaceToComplete)
        outputSurface = pd.Series(self.dataSet.formatModelDataAsDataSet(s), index = surfaceToComplete.index)
        
        plottingTools.printIsolated("L2 Reconstruction loss : ", 
                                    lossReconstruction(originalSurface,outputSurface))
        return l, f, outputSurface, originalSurface
        
    #Show surface sensitivity with respect to each factor
    def printOutputSensiToFactors(self, factorCalibrated, date):
        
        allData = self.dataSet.getDataForModel(date)
        s, JFactors = self.model.evalSingleDayWithoutCalibrationWithSensi(factorCalibrated, allData)
        sIndexed = pd.Series(s,index = allData[0].index)
        JFactorsDf = pd.DataFrame(JFactors, index = allData[0].index)
        
        title = "Original Data"
        originalCoordinates = self.dataSet.formatModelDataAsDataSet(allData)[1]
        plottingTools.plotGrid(sIndexed, originalCoordinates, title, 
                               colorMapSystem=self.colorMapSystem, 
                               plotType=self.plotType)
        
        plottingTools.printDelimiter()
        plottingTools.printIsolated("Sensitivities to each factor")
        for k in JFactorsDf.columns:
            title = "Data Sensitivity to factor number " + str(k)
            plottingTools.plotGrid(JFactorsDf[k], originalCoordinates, title, 
                                   colorMapSystem=self.colorMapSystem, 
                                   plotType=self.plotType)
        return
    
    def correctExtrapolationDomain(self, sparseSurface, completedSurface, coordinates):
        extrapolationMode = (self.model.hyperParameters["extrapolationMode"] 
                             if "extrapolationMode" in self.model.hyperParameters else "NoExtrapolation")
        #remove values from extrapolation domain 
        interpolatedPoint = None
        if extrapolationMode == 'InnerDomain' :
            interpolatedPoint = gaussianProcess.areInInnerPolygon(sparseSurface, coordinates)
        elif extrapolationMode == 'OuterDomain' :
            interpolatedPoint = gaussianProcess.areInOuterPolygon(sparseSurface, coordinates)
        else : #NoExtrapolation, keep all points
            interpolatedPoint = coordinates.loc[sparseSurface.index]
        
        #Hide data not in the domain
        pointTokeep = interpolatedPoint.index
        pointToRemove = sparseSurface.index.difference(pointTokeep)
        completedSurface.loc[pointToRemove] = np.NaN
        
        extrapolatedSurface = gaussianProcess.extrapolationFlat(completedSurface, coordinates)
        
        return extrapolatedSurface.rename(sparseSurface.name)
        
    
    #Intermediate step for possibly using flat extrapolation
    def executeCompletion(self, 
                          sparseSurfaceList, 
                          initialValueForFactors, 
                          nbCalibrationStep):
        
        l, f, S, lSerie = self.model.completeDataTensor(sparseSurfaceList,
                                                        initialValueForFactors, 
                                                        nbCalibrationStep)
        #Retain only completed points
        completedPoints = S[sparseSurfaceList[0].isna()].append(sparseSurfaceList[0].dropna()).rename(S.name)[S.index]
        
        
        extrapolatedS = self.correctExtrapolationDomain(sparseSurfaceList[0], completedPoints, sparseSurfaceList[1])
        
        return l, f, extrapolatedS, lSerie 
    
    def selectClosestObservationsInThePast(self, dayObs, factorHistory, incompleteSurface):
        wholeHistory = self.dataSet.getDataForModel()[0]
        #Get all previous observations
        history = wholeHistory[wholeHistory.index < dayObs]
        error = np.square(history - incompleteSurface).dropna(axis=1)
        argMinError = error.mean(axis=1).idxmin()
        return factorHistory.loc[argMinError]    
    
    #Assess completion along testing data history and recalibrate from latest available factor values
    def backTestCompletion(self, restoreResults = False) :
        if self.outputs_val is None :
            raise ValueError("Diagnose compression before completing one day")
        
        if restoreResults : 
            result = self.readObject("completion")
        else :
            testingSet = self.dataSet.getTestingDataForModel()
            latestFactorValuesForTrainingSet = self.codings_Train.iloc[-1]
            
            #Gather data for dynamic model
            wholeHistoryOfFactors = self.codings_Train.append(self.codings_val)
            
            #Evaluate factors on the testingSet to get real factors
            _ , trueSurface, trueFactors = self.model.evalModel(testingSet)
            trueSurface = self.dataSet.formatModelDataAsDataSet(testingSet[0])
                
            data = np.reshape(latestFactorValuesForTrainingSet.to_frame().transpose().append(trueFactors.head(-1)).values, 
                              trueFactors.shape)
            initialValuesForFactors = pd.DataFrame(data, 
                                                   index = trueFactors.index, 
                                                   columns = latestFactorValuesForTrainingSet.index)
            
            #Iterate on testing Set observations to obtain
            calibratedFactorValues = []
            calibrationLosses = []
            completedSurfaces = []
            
            counter = 0
            for counter in range(testingSet[0].shape[0]) : 
                funcDateSelect = (lambda x : x.iloc[counter] if x is not None else None)
                dayData = list(map(funcDateSelect, testingSet))
                observation=dayData[0]
                #Make a deep copy to prevent observation instance from being polluted with np.NaN 
                observationToComplete = self.dataSet.maskDataset(observation)
                
                observedData = [observationToComplete] + dayData[1:]
                
                initialFactorValues = self.selectClosestObservationsInThePast(observation.name,
                                                                              wholeHistoryOfFactors, 
                                                                              observationToComplete)
                l, f, s, _ = self.executeCompletion(observedData, 
                                                    initialFactorValues,#initialValuesForFactors.iloc[counter], 
                                                    self.nbStepCalibrations)
                
                #completedSurfaces.append(self.dataSet.formatModelDataAsDataSet(s))
                completedSurfaces.append(s)
                calibratedFactorValues.append(f)
                calibrationLosses.append(l)
            
            result = {'calibratedFactorValues' : pd.DataFrame(calibratedFactorValues, index = testingSet[0].index), 
                      'calibrationLosses' : pd.Series(calibrationLosses, index = testingSet[0].index),
                      'completedSurfacesUntransformed' : pd.DataFrame(np.reshape(completedSurfaces,testingSet[0].shape), 
                                                                      index = testingSet[0].index, 
                                                                      columns = testingSet[0].columns),
                      'trueFactors' : pd.DataFrame(trueFactors, index = testingSet[0].index),
                      'trueSurface' : trueSurface}
            result['completedSurfaces'] = self.dataSet.formatModelDataAsDataSet(result['completedSurfacesUntransformed'])
            if self.saveResults : 
                self.serializeObject(result, "completion")
        self.plotBackTestCompletion(result)
        return result
    
    def plotBackTestCompletion(self, result):
        testingSet = self.dataSet.getTestingDataForModel()
        
        #plot calibration loss along history
        plottingTools.printDelimiter()
        worstCalibration = np.argmax(result['calibrationLosses'].values)
        plottingTools.printIsolated("Worst day in term of calibration : ", worstCalibration)
        
        plottingTools.historyPlot(result['calibrationLosses'], 
                                  title = "Calibration loss on non-missing Points")
        
        #plot gap between factor values returned by encoder and calibrated ones 
        plottingTools.printDelimiter()
        plottingTools.plotFactor(result['trueFactors'], 
                                 title = "Factors obtained from encoder with complete surface")
        plottingTools.plotFactor(result['trueFactors'] - result['calibratedFactorValues'],
                                 title = "Gap between factors obtained from encoder and factors calibrated from a non-complete surface")
        
        #plot results for worst day in term of calibration Loss
        plottingTools.printDelimiter()
        errorPlotTitle = "Daily reconstruction loss between completed surface and complete surface"
        errorsAbsRMSE, _, _, _ = plottingTools.errorPlot(result['completedSurfaces'], 
                                                         result['trueSurface'],
                                                         title = errorPlotTitle)
        
        plottingTools.printDelimiter()
        worstDayPred, worstDayGridRef = plottingTools.getWorstGrids(result['completedSurfaces'], 
                                                                    result['trueSurface'],
                                                                    errorsAbsRMSE)
        
        originalCoordinates = self.dataSet.formatModelDataAsDataSet(testingSet)[1]
        coordinates = originalCoordinates.loc[worstDayGridRef.name]
        plottingTools.plotResults(worstDayPred, 
                                  worstDayGridRef,
                                  coordinates,
                                  title = "Worst completion on testing data", 
                                  colorMapSystem=self.colorMapSystem, 
                                  plotType=self.plotType,
                                  refPoints = self.dataSet.maskDataset(worstDayGridRef).dropna())
        
        
        plottingTools.plotResults(plottingTools.getTotalImpliedVariance(worstDayPred, coordinates), 
                                  plottingTools.getTotalImpliedVariance(worstDayGridRef, coordinates),
                                  coordinates,
                                  title = "Worst Implied Total Variance completion on testing data", 
                                  colorMapSystem=self.colorMapSystem, 
                                  plotType=self.plotType,
                                  refPoints = self.dataSet.maskDataset(plottingTools.getTotalImpliedVariance(worstDayGridRef, coordinates)).dropna())
        
        
        
        self.printThetaArbitrage(result['completedSurfaces'], result['trueSurface'], result['calibratedFactorValues'])
        
        
        plottingTools.plotGrid(self.dataSet.getDataForModel(worstDayGridRef.name)[0],
                               coordinates,
                               "Model data for worst reconstruction on testing dataset", 
                               colorMapSystem=self.colorMapSystem, 
                               plotType=self.plotType,
                               refPoints = self.dataSet.maskDataset(worstDayGridRef).dropna())
                                  
        
        plottingTools.printDelimiter()
        _ = self.ProfitAndLoss(result['trueSurface'], result['completedSurfaces'])
        
        if self.diagnoseOriginalData : 
            plottingTools.printDelimiter(3)
            plottingTools.printIsolated("With data used as input for learning model")
            errorPlotTitle = "Daily reconstruction loss between completed surface and complete surface"
            errorsAbsRMSEUntransformed, _, _, _ = plottingTools.errorPlot(result['completedSurfacesUntransformed'], 
                                                                          testingSet[0],
                                                                          title = errorPlotTitle)
            worstDayPred, worstDayGridRef = plottingTools.getWorstGrids(result['completedSurfacesUntransformed'], 
                                                                        testingSet[0], 
                                                                        errorsAbsRMSEUntransformed)
            plottingTools.printDelimiter()
            coordinates = testingSet[1].loc[worstDayGridRef.name]
            plottingTools.plotResults(worstDayPred, 
                                      worstDayGridRef,
                                      coordinates,
                                      title = "Worst completion on testing data", 
                                      colorMapSystem=self.colorMapSystem, 
                                      plotType=self.plotType,
                                      refPoints = self.dataSet.maskDataset(worstDayGridRef).dropna())
        return 
    #
    def diagnoseCompletion(self):
        return self.backTestCompletion()
        
    ##Evaluate likelyhood for independant [-1,1] uniform marginal distribution
    #def uniformLikelyhood(self, factor):
    #    return tf.cumprod(tf.clip_by_value(factor, 
    #                                       clip_value_min=-1, 
    #                                       clip_value_max=1) / 2, 
    #                      axis = 1)
    
    #def gaussianLogLikelyhood(self, meanTensor, covTensor, factor, independantMarginals = False):
    #                            
    #    #Clean version is not possible because tensorflow_probability module is not available
    #    #dist = tfd.MultivariateNormalFullCovariance(loc=meanTensor, scale=covTensor)
    #    #return  dist.log_prob(currentLatentVariables,name="entropyOfInferredValue")
    #    #pi = tf.constant(m.pi)
    #    
    #    centeredFactor = tf.transpose(factor) - meanTensor
    #    if independantMarginals : 
    #        invCovMatrix = tf.linalg.inv(covTensor)
    #    else : 
    #        invCovMatrix = covTensor
    #    exponent = tf.matmul(centeredFactor,tf.matmul(invCovMatrix,centeredFactor), 
    #                         transpose_a = True, 
    #                         transpose_b = False)
    #    logpdf =  - 0.5 * exponent #- tf.pow(2 * pi, nbMarginals / 2.0) - tf.sqrt(tf.linalg.det(covTensor))
    #    return logpdf
    
    #
    def completeObservationForSingleDay(self, date):
        tmp = self.completionTest(date)
        calibLoss = tmp[0]
        factorCalibrated = tmp[1]
        completedSurface = tmp[2]
        trueSurface = tmp[3]
        
        coordinates = self.dataSet.formatModelDataAsDataSet(self.dataSet.getDataForModel(date))[1]
        plottingTools.plotCompletion(trueSurface, completedSurface[trueSurface.index], 
                                     coordinates,
                                     colorMapSystem=self.colorMapSystem, 
                                     plotType=self.plotType,
                                     refPoints = self.dataSet.maskDataset(trueSurface).dropna())
        
        plottingTools.printDelimiter()
        self.printOutputSensiToFactors(factorCalibrated, date)
        return tmp
    
    def evalDecoder(self, date):
        listOfFactor = pd.DataFrame(self.getFactor(date), 
                                    index = date, 
                                    columns = self.codings_Train.columns)
        listOfDecodedSurface = pd.DataFrame(self.model.evalSingleDayWithoutCalibration(listOfFactor.values, 
                                                                                       self.dataSet.getDataForModel(date)), 
                                            index = date, 
                                            columns = self.outputs_val.columns)
        return listOfDecodedSurface
    
    def getFactor(self,date):
        #Behaviour for serie of dates
        if isinstance(date, pd.Index): 
            arrayRes = np.reshape(list(map((lambda x : self.getFactor(x)),date)),
                                  (date.size,self.codings_Train.shape[1]))
                                 
            return pd.DataFrame(arrayRes, index = date, columns = self.codings_Train.columns)
        
        #Behaviour for a single date
        if date in self.codings_Train.index: 
            return self.codings_Train.loc[date]
        elif date in self.codings_val.index : 
            return self.codings_val.loc[date]
        return None
    #
    def evaluateBumpedInterdependanciesForSingleDay(self, date):
        if self.outputs_val is None :
            raise ValueError("Diagnose compression before completing one day")
        
        #Delete points inside the surface
        surfaceToEvaluate = self.dataSet.formatModelDataAsDataSet(self.dataSet.getDataForModel(date)) 
        nbPoints = surfaceToEvaluate[0].size
        
        #Generate for each hedging point a bumped surface
        bumpedSurfaces = []
        bumpSize = 0.01#0.01 * surfaceToEvaluate.max()
        for i in surfaceToEvaluate[0].index:
            bumpedSurface = surfaceToEvaluate[0].copy()
            bumpedSurface.loc[i] += bumpSize
            bumpedSurfaces.append(self.dataSet.convertRealDataToModelFormat(bumpedSurface))
            
            bumpedSurface = surfaceToEvaluate[0].copy()
            bumpedSurface.loc[i] -= bumpSize
            bumpedSurfaces.append(self.dataSet.convertRealDataToModelFormat(bumpedSurface))
            
        #bumpedSurfaces.append(self.dataSet.convertRealDataToModelFormat(surfaceToEvaluate))
        
        #Reconstruct the surfaces and compute the approximated jacobian
        #_,s,_ = self.model.evalModel(pd.DataFrame(np.reshape(bumpedSurfaces,(nbPoints + 1,nbPoints)), 
        #                                          columns = surfaceToEvaluate.index))
        
        bumpedSurfacesDf = pd.DataFrame(np.reshape(bumpedSurfaces,(2 * nbPoints,nbPoints)), 
                                        columns = surfaceToEvaluate[0].index)
        def repeatRow(rowSerie):
            if rowSerie is None :
                return rowSerie 
            return pd.DataFrame(np.repeat(np.expand_dims(rowSerie.values,0),
                                          bumpedSurfacesDf.shape[0],
                                          axis=0), 
                                index = bumpedSurfacesDf.index,
                                columns = rowSerie.index)
        #Repeat data for other datasets (forward, coordinates ...)
        bumpedData = [bumpedSurfacesDf] + list(map(repeatRow, surfaceToEvaluate[1:])) 
        
        _,s,_ = self.model.evalModel(bumpedData)
        applyInverseDataTransform = lambda x : self.dataSet.formatModelDataAsDataSet(x.rename(date))
        surfaceInOriginalFormat = s.apply(applyInverseDataTransform, axis = 1) #(160,80)
        
        #reconstructionJacobian = pd.DataFrame((surfaceInOriginalFormat - surfaceInOriginalFormat.iloc[-1]).head(-1).values / bumpSize, 
        #                                      index = surfaceToEvaluate.index, 
        #                                      columns = surfaceToEvaluate.index) #(80,80)
        reconstructionJacobianValues = (surfaceInOriginalFormat.iloc[::2].values - 
                                        surfaceInOriginalFormat.drop(surfaceInOriginalFormat.iloc[::2].index).values) / 2 / bumpSize #(80,80)
        reconstructionJacobian = pd.DataFrame(reconstructionJacobianValues , 
                                              index = surfaceToEvaluate[0].index, 
                                              columns = surfaceToEvaluate[0].index)
        reconstrutedSurface = surfaceInOriginalFormat.iloc[-1] #(80,)
        
        return reconstrutedSurface , self.jacobianPostTreatment(reconstructionJacobian)
    
    
    def jacobianPostTreatment(self, estimatedJac):
        ##First set diagonal equal to one
        #normalizedJac = estimatedJac / np.diag(estimatedJac)
        
        ##Jacobian matrix in our case should respect transpose(J)=1/J (element wise inverse)
        ##First we estimate inferior triangle coefficients
        #triLowerJac = 0.5 * (np.tril(normalizedJac.values) + 1 / np.triu(normalizedJac.values).T)
        #correctedJac = triLowerJac + (1 /  triLowerJac.T)  
        correctedJac = pd.DataFrame(np.where(estimatedJac == 0, 
                                             estimatedJac, 
                                             estimatedJac / estimatedJac.sum(axis=0)), 
                                    index = estimatedJac.index, 
                                    columns = estimatedJac.columns) #Taylor order one approximation
        return correctedJac.dropna(how="all", axis=0) #Remove rows for missing values
    #
    def evaluateInterdependanciesForSingleDay(self, date):
        if self.testingLoss is None :
            raise ValueError("Fit the model before evaluating one day")
        
        #Delete points inside the surface
        surfaceToEvaluate = self.dataSet.getDataForModel(date)
        
        #Reconstruct the surface and compute the jacobian
        reconstrutedSurface, reconstructionJacobian = self.model.evalInterdependancy(surfaceToEvaluate)
        
        reshapedInterdependancies = pd.DataFrame(reconstructionJacobian , 
                                                 index = surfaceToEvaluate[0].index, 
                                                 columns = surfaceToEvaluate[0].index) 
        reshapedReconstructedSurface = pd.Series(reconstrutedSurface, 
                                                 index = surfaceToEvaluate[0].index).rename(date)
        outputSurface = self.dataSet.formatModelDataAsDataSet(reshapedReconstructedSurface)
        
        return outputSurface , self.jacobianPostTreatment(reshapedInterdependancies)
    
    #Assess interdependancies between surface points and compare risk projection to an unhedged fictive portfolio
    def backTestRiskProjection(self, restoreResults = False) :
        if restoreResults : 
            result = self.readObject("riskProjection")
        else : 
            fullDataSet = self.dataSet.getDataForModel() #Whole history of volatility surface 
            nbPoints = fullDataSet[0].shape[1] #Number of points per surface
            testingSet = self.dataSet.getDataForModel()[0].iloc[1:]#self.dataSet.getTestingDataForModel() #History of volatility surface in the testing set
            testingSetDays = testingSet.index #Testing set days
            nbTrainingDays = 1#self.dataSet.getTrainingDataForModel().shape[0] #Number of days in the training set
            observedPoints = self.dataSet.maskDataset(testingSet).dropna(how="all", axis=1).columns  #Points which are known (not masked)
            nbObservedPoints = observedPoints.size #Numper of points hence number of projected vega
            
            def budgetPenalty(x):
                return np.sum(np.abs(x))
            def budgetPenaltyGrad(x):
                return np.sign(x)
            def budgetPenaltyHess(x, v):
                return np.zeros((nbHedgingPoints, nbHedgingPoints))
            
            #Iterate on testing Set observations to obtain
            #Fixing day : day on which we compute the projected vega
            #Testing day : Day following the fixing day when we measure the PAndL of our projected Vega
            fixingDay = [] #Days used for computing the vega projection matrix
            jacobians = [] #Jacobian observed on the fixing day
            projectedVegas = [] #Product of the jacobian and volatility variation
            refVegas = []
            equallyProjectedVegas = []
            
            deletedValues = None 
            
            counter = 0
            
            investablePoints = self.dataSet.decideInvestableInstruments()
            
            for counter in range(testingSetDays.size) : 
                dayNumber = counter + nbTrainingDays #Day Number in fullDataSet 
                
                currentDay = fullDataSet[0].iloc[dayNumber].name #Testing day
                previousDay = fullDataSet[0].iloc[dayNumber-1].name #Fixing day
                fixingDay.append(previousDay)
                
                hedgeablePoints = investablePoints[previousDay][0] #Index of points which are still quoted at next day
                nbHedgeablePoints = hedgeablePoints.size #Number of instruments we can use for hedging
                
                hedgingPoints = hedgeablePoints.intersection(observedPoints)
                nbHedgingPoints = hedgingPoints.size
                
                #The jacobian is of size (nbPoints, nbPoints), index stand for input x, columns for output J(D(E(x)))
                surface, Jacobian = self.evaluateBumpedInterdependanciesForSingleDay(previousDay) #self.evaluateInterdependanciesForSingleDay(previousDay)
                
                ##Method 1
                
                ##Jacobian : Matrix of dimensions (nbPoints, hedgingPoints.size), we keep only sensitivities of hedging points w.r.t. inputs
                #jacobians.append(Jacobian.loc[hedgingPoints])#Jacobian[hedgingPoints]) #dataframe Index -> inputs, dataframe columns -> ouputs 
                ##projectionMatrix : Matrix of dimensions (hedgingPoints.size, nbPoints)
                ##projectionMatrix = np.linalg.pinv(jacobians[-1]) #Pseudo inverse of the jabobian
                #projectionMatrix = jacobians[-1].values #Pseudo inverse of the jabobian
                #projectedVega = np.dot( projectionMatrix,  originalVega) #Vega computed on hedging points
                
                ##Method 2
                HedgingPointJacobian = Jacobian[hedgingPoints].transpose()[hedgeablePoints].transpose() #Select sensitivities of hedging points w.r.t. every surface point
                if HedgingPointJacobian.isnull().any().any():
                    #HedgingPointJacobian = pd.DataFrame(np.eye(HedgingPointJacobian.shape[0]), 
                    #                                    index = HedgingPointJacobian.index, 
                    #                                    columns = HedgingPointJacobian.columns)
                    print("Backtest risk projection failed because of invalid Jacobian (nans values)")
                    return {}
                #Shape is (80,8)
                #projectedVega = np.linalg.lstsq(HedgingPointJacobian.values, originalVega, rcond=None)[0] #Solve with least square method the allocation which reproduces the equally weighted vega
                #(80,8),(80,) -> (8,)
                #We normalize projected vega such that their sum is equal to the sum of original vega
                #projectedVegas.append(projectedVega * np.sum(originalVega) / np.sum(projectedVega)) 
                
                equallyProjectedVega = (np.ones(nbHedgeablePoints)/float(nbHedgeablePoints)) #Default vega which is assumed to be uniform
                budgetVega = budgetPenalty(equallyProjectedVega)
                budgetConstraint = NonlinearConstraint(budgetPenalty, budgetVega, budgetVega,
                                                       jac = budgetPenaltyGrad,
                                                       hess = budgetPenaltyHess)
                def objectiveFunction(x) : 
                    return np.sum(np.square(HedgingPointJacobian.values @ x - equallyProjectedVega))
                def gradient(x):
                    return 2 * ( - HedgingPointJacobian.values.T @ equallyProjectedVega + 
                                HedgingPointJacobian.values.T @ HedgingPointJacobian.values @ x)
                def hessian(x):
                    return 2 * HedgingPointJacobian.values.T @ HedgingPointJacobian.values
                x0 = (np.ones(nbHedgingPoints)/float(nbHedgingPoints))
                res = minimize(objectiveFunction, x0, #bounds = bounds, 
                               constraints = budgetConstraint, 
                               method = "trust-constr", 
                               jac = gradient,
                               hess = hessian)
                projectedVega = res.x
                projectedVegas.append(pd.Series(projectedVega, index = hedgingPoints)) 
                refVegas.append(pd.Series(equallyProjectedVega, index = hedgeablePoints))
                equallyProjectedVegas.append(pd.Series(x0, index = hedgingPoints))
                
            
            result = {}
            # result['OriginalVega'] = pd.DataFrame(np.reshape(np.repeat(originalVega, testingSetDays.size, axis=0),
                                                             # (testingSetDays.size, nbPoints)),
                                                  # index = testingSetDays,
                                                  # columns = fullDataSet[0].columns)
            # result['ProjectedVega'] = pd.DataFrame(np.reshape(projectedVegas,(testingSetDays.size, nbHedgingPoints)),
                                                   # index = testingSetDays,
                                                   # columns = hedgingPoints)
            
            # result['EquallyProjectedVega'] = pd.DataFrame(np.reshape(np.repeat(equallyProjectedVega, testingSetDays.size, axis=0),
                                                                     # (testingSetDays.size, nbHedgingPoints)),
                                                          # index = testingSetDays,
                                                          # columns = hedgingPoints)
            result['OriginalVega'] = pd.Series( refVegas , index = testingSetDays)
            result['ProjectedVega'] = pd.Series( projectedVegas , index = testingSetDays)
            result['EquallyProjectedVega'] = pd.Series( equallyProjectedVegas , index = testingSetDays)
            
            if self.saveResults : 
                self.serializeObject(result, "riskProjection")
        
        
        
        self.plotBackTestRiskProjection(result)
        return result
    
    def plotBackTestRiskProjection(self, result):
        #plot projected vega and original vega along history
        testingSet = self.dataSet.getDataForModel(result['ProjectedVega'].index)#self.dataSet.getTestingDataForModel() #History of volatility surface in the testing set
        
        #Plot P&L gap between original and projected vega
        plottingTools.printDelimiter()
        _ = self.ProfitAndLossVegaStrategies(self.dataSet.formatModelDataAsDataSet(testingSet), 
                                             result['ProjectedVega'],
                                             result['OriginalVega'])
        
        #Plot P&L gap between original and equally weighted projected vega
        plottingTools.printDelimiter()
        _ = self.ProfitAndLossVegaStrategies(self.dataSet.formatModelDataAsDataSet(testingSet), 
                                             result['EquallyProjectedVega'],
                                             result['OriginalVega'])
        return 
    
    def assignNewModel(self, newLearningModel):
        self.model = newLearningModel
        
        #Reset temporary results in cache
        self.testingLoss = None
        self.outputs_val = None
        self.codings_val = None
        self.training_val = None
        self.codings_Train = None
        
    def testDaniel(self, date): 
        cLoss, fCalibrated, completedSurface, trueSurface = self.completionTest(date)
        reconstructedHistory = self.training_val.append(self.outputs_val) 
        reconstructedSurface = self.dataSet.formatModelDataAsDataSet(reconstructedHistory[ reconstructedHistory.index < date].iloc[-1])[0]
        observedValues = self.dataSet.maskDataset(reconstructedSurface).dropna()
        plottingTools.printIsolated("Show completed surface against yesterday reconstructed surface")
        coordinates = self.dataSet.formatModelDataAsDataSet(self.dataSet.getDataForModel(date))[1]
        plottingTools.plotCompletion(reconstructedSurface, 
                                     completedSurface, 
                                     coordinates,
                                     colorMapSystem=self.colorMapSystem, 
                                     plotType=self.plotType,
                                     refPoints = observedValues)
        return

