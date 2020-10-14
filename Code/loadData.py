#Import modules
import os
import pandas as pd
import numpy as np
from pandas import DatetimeIndex
import dask
import scipy
from scipy.optimize import minimize, LinearConstraint
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

#Define Column Name
indexName = 'date'
indexExpiry = 'optionExpiry'
indexTenor = 'underlyingTerm'
indexStrike = 'Strike'
indexRelStrike = 'RelativeStrike'

def getTTMFromCoordinates(dfList):
    return dfList[1].applymap(lambda x : x[0])
def getMoneynessFromCoordinates(dfList):
    return dfList[1].applymap(lambda x : x[1])

def readfile(file):
    print("file")
    print(file)
def iterateOnFolderContent(folderName):
    for elt in os.scandir(folderName):
        if os.DirEntry.is_dir(elt):
            print("Folder")
            print(elt)
            iterateOnFolderContent(elt)
        else :
            readfile(elt)

            
def parseTerm(stringTerm):
    if 'M' == stringTerm[-1]: 
        return float(stringTerm[:-1])/12
    elif 'Y' == stringTerm[-1]: 
        return float(stringTerm[:-1])
    else :
        raise Exception("Can not parse term")
def parseTenor(row):
    return [parseTerm(row['underlyingTerm']), parseTerm(row['optionExpiry'])]


def smileFromSkew(skew):
    atmVol = skew['A'] 
    #smile = atmVol + skew[skewShift]
    #return smile#.append(skew.drop(smile.index))
    return atmVol + skew.drop('A')

def parseStrike(relStrike):
    if relStrike.name[3] == 'A':
        return relStrike['forward']
    if "+" in relStrike.name[3]:
        shift = int(relStrike.name[3].split("+")[1])
        return relStrike['forward'] + shift/1000
    if "-" in relStrike.name[3]:
        shift = int(relStrike.name[3].split("-")[1])
        return relStrike['forward'] - shift/1000
    raise Exception(' Can not parse Strike ')

#intersection of all dates across history
def intersectionGrid(grid) :
    nbDates = grid.index.get_level_values(0).unique().shape[0]
    if nbDates <= 1:
        return grid.index.droplevel(0)
    else :
        midDate = grid.index.get_level_values(0).unique()[int(nbDates/2)]
        g1 = grid[grid.index.get_level_values(0) < midDate]
        g2 = grid[grid.index.get_level_values(0) >= midDate]
        return intersectionGrid(g1).intersection(intersectionGrid(g2))

def splitTrainTestDataRandomly(gridHistory, trainingSetPercentage):
    nbDates = gridHistory.index.get_level_values(0).unique().shape[0] 
    trainingDates = np.random.choice(gridHistory.index.get_level_values(0).unique(),
                                     replace=False,
                                     size=int(nbDates * trainingSetPercentage))
    trainingData = gridHistory.loc[pd.IndexSlice[trainingDates,:,:], :]
    testingData = gridHistory.drop(trainingData.index)
    trainingData.index = trainingData.index.droplevel([1,2])
    testingData.index = testingData.index.droplevel([1,2])
    return trainingData, testingData 

def splitTrainTestDataChronologically(gridHistory, trainingSetPercentage):
    firstTestingDate = int(gridHistory.index.get_level_values(0).unique().shape[0] 
                           * trainingSetPercentage)
    trainingDates = gridHistory.index.get_level_values(0).unique()[:firstTestingDate]
    trainingData = gridHistory.loc[pd.IndexSlice[trainingDates,:,:], :]
    testingData = gridHistory.drop(trainingData.index)
    trainingData.index = trainingData.index.droplevel([1,2])
    testingData.index = testingData.index.droplevel([1,2])
    return trainingData, testingData 

def sampleBatchOfDays(dataSet, nbDrawn):
    trainingDates = np.random.choice(dataSet.index.get_level_values(0).unique(),
                                     replace=False,
                                     size=nbDrawn)
    return dataSet.loc[trainingDates, :]


def splitHistory(history, colName):
    return pd.pivot_table(history, 
                          values = colName,
                          index = history.index.names,
                          columns=['Expiry','Tenor'])

def extractDataFromCSV(dataSetPath):
    #Read csv file
    data = pd.read_csv(dataSetPath)
    #Parse tenor and expiry as float years
    data['Tenor'],data['Expiry'] = zip(*data.apply(parseTenor,axis=1))
    
    #Parse date as a datetime
    data[indexName] = pd.to_datetime(data['businessDate'], dayfirst=True)
    
    #Set Index as as a three dimension vector and sort observation
    indexedData = data.set_index([indexExpiry, indexTenor, indexName]).sort_index()
    
    #Keep relevant features
    #Columns used for representing a Strike Value
    skewShift = [shift for shift in indexedData.columns if ('A' in shift )]#and 'A' != shift
    #Other Columns to keep
    otherColumns = ['forward', 'Tenor', 'Expiry']
    #Get columns indexed by a relative strike
    skewHistory = indexedData[skewShift + otherColumns]#.apply(smileFromSkew,axis=1)
    
    #Merge with other useful columns
    
    #Stacking Smile
    
    #Left outer Join on (tenor, expiry, date)
    joinColumns = skewHistory.index.names
    leftTable = skewHistory.drop(otherColumns, axis = 1).stack().rename("Vol")#Features depending on strike value
    leftTable.index.names = [leftTable.index.names[0],
                             leftTable.index.names[1],
                             leftTable.index.names[2],
                             'RelativeStrike']
    formattedHistory = leftTable.reset_index().merge(skewHistory[otherColumns].reset_index(), 
                                                     on=joinColumns,
                                                     validate = "m:1").set_index(leftTable.index.names).sort_index()
    
    #Convert strike shift as a float from a stringTerm
    formattedHistory[indexStrike] = formattedHistory.apply(parseStrike,axis=1)
    return formattedHistory
    


def equalDf(df1, df2):
    if df1.shape == df2.shape :
        if np.sum(np.isnan(df1.values)) != np.sum(np.isnan(df2.values)) :
            print("Not the same number of nan")
            return False 
        tol = 1e-6
        gap = np.nansum(np.abs(df1.values - df2.values))
        if gap < tol :
            return True
        else :
            print("Large df error : ", gap)
            return False
    print("Not the same shape")
    return False


def sampleSwaptionsToDelete(dataSet, completionRate):
    return dataSet.iloc[0].sample(frac = completionRate).index
    
def removeSwaptionsToDelete(dataSet):
    listToDelete = [(0.08333333333333333,0.25),(0.08333333333333333,10.0),
                    (0.08333333333333333,30.0),(0.5,2.0),(0.5,15.0),
                    (5.0,1.0),(5.0,20.0),(10.0,5.0)]
    return dataSet.iloc[0].index.difference(listToDelete)

#Different from minmax scaler of scikit learn
#Min and Max are computed on the dataset, not column wise
class customMinMaxScale:
    def __init__(self, feature_range = (0,1)):
        self.min = feature_range[0]
        self.max = feature_range[1]
    #We can enforce the minimum if we expect smaller data in the testing set
    def fit(self, dataset, 
            enforceDataSetMin = None, 
            enforceDataSetMax = None):
        self.datasetMin = dataset.min().min()
        if enforceDataSetMin is not None :
            self.datasetMin = min(enforceDataSetMin, self.datasetMin)
        
        self.datasetMax = dataset.max().max()
        if enforceDataSetMax is not None :
            self.datasetMax = max(enforceDataSetMax, self.datasetMax)
        
        return
        
    def transform(self, dataset):
        scale = (self.max - self.min) / (self.datasetMax - self.datasetMin)
        return (dataset - self.datasetMin) * scale + self.min
    
    def inverse_transform(self, scaledDataset):
        scale = (self.max - self.min) / (self.datasetMax - self.datasetMin)
        return (scaledDataset - self.min) / scale + self.datasetMin

#Encapsulation class for Sklearn Standard scaling
class customMeanStdScale:
    def __init__(self, feature_range = (0,1)):
        self.scalerList = []
    #We can enforce the minimum if we expect smaller data in the testing set
    def fit(self, dataset, 
            enforceDataSetMin = None, 
            enforceDataSetMax = None):
        hasTupleElt = (type(dataset.iloc[0,0] if dataset.ndim==2 else dataset.iloc[0])==type(tuple()))
        if hasTupleElt :
            tupleSize = len(dataset.iloc[0,0] if dataset.ndim==2 else dataset.iloc[0])
            self.scalerList = [StandardScaler() for i in range(tupleSize)]
            for k in range(tupleSize):
                funcAccess = lambda x : x[k]
                scaler = self.scalerList[k]
                dfElt = dataset.applymap(funcAccess) if (type(dataset) != type(pd.Series())) else dataset.map(funcAccess)
                scaler.fit(dfElt)
            
        else :
            self.scalerList = []
            self.scalerList.append(StandardScaler())
            self.scalerList[0].fit(dataset)
        return
    
    def transformSingleDf(self, scaler, dfElt):
        totalVariance = np.sum(scaler.var_)
        if totalVariance <= 1e-6 : #Avoid mean scaling for constant data
            return dfElt
        if type(dfElt) == type(pd.Series()):
            return pd.Series(np.ravel(scaler.transform(dfElt.values.reshape(1, -1))), 
                             index = dfElt.index).rename(dfElt.name)
        return pd.DataFrame(scaler.transform(dfElt), 
                            index = dfElt.index, 
                            columns = dfElt.columns)
    
    def transform(self, dataset):
        hasTupleElt = (type(dataset.iloc[0,0] if dataset.ndim==2 else dataset.iloc[0])==type(tuple()))
        if hasTupleElt :
            tupleSize = len(dataset.iloc[0,0] if dataset.ndim==2 else dataset.iloc[0])
            scaledDfList = []
            for k in range(tupleSize):
                funcAccess = lambda x : x[k]
                dfElt = dataset.applymap(funcAccess) if (type(dataset) != type(pd.Series())) else dataset.map(funcAccess)
                scaler = self.scalerList[k]
                scaledDfList.append(np.ravel(self.transformSingleDf(scaler, dfElt).values)) 
            #Flattened list of tuples
            tupleList= list(zip(*scaledDfList))
            #Merge all datasets into a single structure
            if dataset.ndim==2 :
                reshapedList = [tupleList[(i*dataset.shape[1]):((i+1)*dataset.shape[1])] for i in range(dataset.shape[0])]
                return pd.DataFrame(reshapedList, 
                                    index = dataset.index, 
                                    columns = dataset.columns)
            else :
                reshapedList = tupleList
                return pd.Series(reshapedList, index = dataset.index)
        else :
            return self.transformSingleDf(self.scalerList[0], dataset)
        
        return None
    
    def inverTransformSingleDf(self, scaler, dfElt):
        totalVariance = np.sum(scaler.var_)
        if totalVariance <= 1e-6 : #Avoid mean scaling for constant data
            return dfElt
        if type(dfElt) == type(pd.Series()):
            return pd.Series(np.ravel(scaler.inverse_transform(dfElt.values.reshape(1, -1))), 
                             index = dfElt.index).rename(dfElt.name)
        return pd.DataFrame(scaler.inverse_transform(dfElt), 
                            index = dfElt.index, 
                            columns = dfElt.columns)
                            
    def inverse_transform(self, scaledDataset):
        hasTupleElt = (type(scaledDataset.iloc[0,0] if scaledDataset.ndim==2 else scaledDataset.iloc[0])==type(tuple()))
        if hasTupleElt :
            tupleSize = len(scaledDataset.iloc[0,0] if scaledDataset.ndim==2 else scaledDataset.iloc[0])
            scaledDfList = []
            for k in range(tupleSize):
                funcAccess = lambda x : x[k]
                dfElt = scaledDataset.applymap(funcAccess) if (type(scaledDataset) != type(pd.Series())) else scaledDataset.map(funcAccess)
                scaler = self.scalerList[k]
                scaledDfList.append(np.ravel(self.inverTransformSingleDf(scaler, dfElt).values)) 
            #Flattened list of tuples
            tupleList= list(zip(*scaledDfList))
            #Merge all datasets into a single structure
            if scaledDataset.ndim==2 :
                reshapedList = [tupleList[(i*scaledDataset.shape[1]):((i+1)*scaledDataset.shape[1])] for i in range(scaledDataset.shape[0])]
                return pd.DataFrame(reshapedList, 
                                    index = scaledDataset.index, 
                                    columns = scaledDataset.columns)
            else :
                reshapedList = tupleList
                return pd.Series(reshapedList, index = scaledDataset.index)
        else :
            return self.inverTransformSingleDf(self.scalerList[0], scaledDataset)
        
        return None

#Encapsulation class for Sklearn min max scaling
class standardMinMaxScale(customMeanStdScale):
    def __init__(self, feature_range = (0,1)):
        super().__init__() 
    #We can enforce the minimum if we expect smaller data in the testing set
    def fit(self, dataset, 
            enforceDataSetMin = None, 
            enforceDataSetMax = None):
        hasTupleElt = (type(dataset.iloc[0,0] if dataset.ndim==2 else dataset.iloc[0])==type(tuple()))
        if hasTupleElt :
            tupleSize = len(dataset.iloc[0,0] if dataset.ndim==2 else dataset.iloc[0])
            self.scalerList = [MinMaxScaler() for i in range(tupleSize)]
            for k in range(tupleSize):
                funcAccess = lambda x : x[k]
                scaler = self.scalerList[k]
                dfElt = dataset.applymap(funcAccess) if (type(dataset) != type(pd.Series())) else dataset.map(funcAccess)
                scaler.fit(dfElt)
            
        else :
            self.scalerList = []
            self.scalerList.append(MinMaxScaler())
            self.scalerList[0].fit(dataset)
        return

def selectLessCorrelatedFeatures(featureCorr, nbPoints):
    objectiveFunction = lambda x : x.T @ featureCorr.values @ x
    gradient = lambda x : (featureCorr.values + featureCorr.values.T) @ x
    hessian = lambda x : featureCorr.values + featureCorr.values.T
    nbRestart = 5
    x0s = np.random.uniform(size=(nbRestart , featureCorr.shape[1]))
    x0s = x0s * nbPoints / np.sum(x0s, axis = 1, keepdims=True)
    bestSol = x0s[0,:]
    bestVar = featureCorr.shape[1]
    
    bounds = [[0,1]] * featureCorr.shape[1]
    budgetAllocation = LinearConstraint(np.ones((1,featureCorr.shape[1])), [nbPoints], [nbPoints], keep_feasible = True)
    for k in range(nbRestart):
        res = minimize(objectiveFunction, x0s[k,:], 
                       bounds = bounds, 
                       constraints = budgetAllocation, 
                       method = "trust-constr", 
                       jac = gradient,
                       hess = hessian)
        if (res.fun < bestVar) or (k==0) :
            bestSol = res.x
            bestVar = res.fun
        print("Attempt no ", k, " ; best solution : ", bestSol, " ; best inertia : ", bestVar)
    
    topnbPointsValue = -(np.sort(-bestSol)[nbPoints - 1])
    optimalAllocation = pd.Series(bestSol, index = featureCorr.index)
    return optimalAllocation[optimalAllocation >= topnbPointsValue].index

def isCSVFile(filename):
    extension = filename[-3:]
    return (extension == "csv")
    
#These class are responsible for :
# - passing the right data to the model for trainingData
# - converting data to the original format for plotting 
class datasetATM:
    def __init__(self, pathToDataset, 
                 trainingSetPercentage, 
                 minExpiry, 
                 completionRate,
                 scaleFeatures = False):
        self.trainingSetPercentage = trainingSetPercentage
        self.pathToDataset = pathToDataset
        self.activateScaling = scaleFeatures
        self.isGridStable = True
        
        self.testVol = None
        self.trainVol = None
        self.VolSerie = None
        self.volScaler = None
        self.scaledTrainVol = None
        self.scaledTestVol = None
        
        self.testCoordinates = None
        self.trainCoordinates = None
        self.CoordinatesSerie = None
        self.coordinatesScaler = None
        self.scaledTrainCoordinates = None
        self.scaledTestCoordinates = None
        
        self.testFwd = None
        self.trainFwd = None
        self.FwdSerie = None
        self.fwdScaler = None
        self.scaledTrainFwd = None
        self.scaledTestFwd = None
        
        self.testStrike = None
        self.trainStrike = None
        self.StrikeSerie = None
        
        self.loadData()
        self.scaleDataSets()
        
        lambdaAppend = (lambda x : x[0].append(x[1]) if x[0] is not None else None)
        self.fullHistory = list(map(lambdaAppend, zip(self.getTrainingDataForModel(),self.getTestingDataForModel())))
        self.fullScaler = [self.volScaler, self.coordinatesScaler, self.fwdScaler, None]
        
        self.gridSize = self.getTestingDataForModel()[0].shape[1]
        
        
        return
    def loadData(self):
        raise NotImplementedError("Abstract class")
        return
        
    
    def sanityCheck(self):
        
        print("Testing formatModelDataAsDataSet")
        assert(equalDf(self.testVol.dropna(how="all").head(), 
               self.formatModelDataAsDataSet(self.getTestingDataForModel())[0].head()))
        
        origData = self.formatModelDataAsDataSet(self.getTrainingDataForModel())
        
        print("Testing coordinates")
        assert(equalDf(self.trainCoordinates.head().applymap(lambda x : x[0]), 
                       origData[1].head().applymap(lambda x : x[0])))
        assert(equalDf(self.trainCoordinates.head().applymap(lambda x : x[1]), 
                       origData[1].head().applymap(lambda x : x[1])))
                       
        print("Testing Forward")
        assert(equalDf(self.getTrainingDataForModel()[2].head(), 
                       self.convertRealDataToModelFormat(self.formatModelDataAsDataSet(self.getTrainingDataForModel()))[2].head()))
        
        print("Testing masking function")
        maskedDf = self.maskDataset(self.getTrainingDataForModel()[1]).dropna(how="all",axis=1).head()
        assert(maskedDf.shape[1] == (self.gridSize - self.maskedPoints.size))
        
        print("Testing convertRealDataToModelFormat")
        assert(equalDf(self.trainVol.loc[origData[0].index].head(), 
                       self.formatModelDataAsDataSet(self.convertRealDataToModelFormat(origData))[0].head()))
        
        print("Success")
        return
        
    #When the grid is not fixed - i.e. volatilities time to maturities are sliding - 
    #we need to decide which instruments can be compared between two dates
    def decideInvestableInstruments(self):
        coordinatesDf = self.formatModelDataAsDataSet(self.getDataForModel())[1]
        
        pairIndexHistory = []#series of pair of index 
        nextTTMDf = coordinatesDf.shift(-1).dropna(how = "all")
        for serie in coordinatesDf.head(-1).iterrows():
            currentDay = serie[1]
            nextDay = nextTTMDf.loc[serie[0]]
            currentRankForHedgeablePoints = currentDay.index
            nextRankForHedgeablePoints = nextDay.index
            pairIndexHistory.append((currentRankForHedgeablePoints, nextRankForHedgeablePoints))
        pairIndexHistory.append((nextRankForHedgeablePoints, nextRankForHedgeablePoints))
        pairIndexHistory = pd.Series(pairIndexHistory, index = coordinatesDf.index)
        return pairIndexHistory
        
    #List Format : First position vol, second position coordinates, third position forward, fourth position strike
    def getTestingDataForModel(self):
        return [self.scaledTestVol, self.scaledTestCoordinates, self.scaledTestFwd, self.testStrike]
        
    def getTrainingDataForModel(self):
        return [self.scaledTrainVol, self.scaledTrainCoordinates, self.scaledTrainFwd, self.trainStrike]
    
    def getDataForModel(self, dates = None):
        if dates is None : 
            return self.fullHistory
        funcExtractDate = lambda x : x.loc[dates] if x is not None else None
        return list(map(funcExtractDate, self.fullHistory))
    
    #Tranform synthetic surfaces as model data
    #Name of surfaces should be the date
    def convertRealDataToModelFormat(self, unformattedSurface):
        if(self.activateScaling): 
            if (type(unformattedSurface)==type(list())) and (len(unformattedSurface)==4):
                lambdaTransform = lambda x : x[0] if x[1] is None else x[1].transform(x[0])
                return list(map(lambdaTransform, zip(unformattedSurface, self.fullScaler)))
            elif (type(unformattedSurface)!=type(list())) :
                return self.volScaler.transform(unformattedSurface)
            else :
                raise("Can not format as model data")
            return 
        return unformattedSurface
    
    #Format data returned by a model to format 
    #For instance variation are transformed as level with yesterday volatilities
    def formatModelDataAsDataSet(self, modelData):
        if(self.activateScaling): 
            if (type(modelData)==type(list())) and (len(modelData)==4):
                lambdaTransform = lambda x : x[0] if x[1] is None else x[1].inverse_transform(x[0])
                return list(map(lambdaTransform, zip(modelData, self.fullScaler)))
            elif (type(modelData)!=type(list())) :
                return self.volScaler.inverse_transform(modelData)
            else :
                raise("Can not format as model data")
            return 
        return modelData
    
    def scaleDataSets(self):
        if(self.activateScaling): 
            #Define MinMax scaling for volatility
            self.volScaler = customMeanStdScale() #customMinMaxScale()
            self.volScaler.fit(self.trainVol, enforceDataSetMin = 0)#Positive volatilities of course
            self.scaledTrainVol = self.volScaler.transform(self.trainVol)
            self.scaledTestVol = self.volScaler.transform(self.testVol)
            
            #Define MinMax scaling for volatility
            self.coordinatesScaler = customMeanStdScale() #customMinMaxScale()
            self.coordinatesScaler.fit(self.trainCoordinates, enforceDataSetMin = 0)#Positive volatilities of course
            self.scaledTrainCoordinates = self.coordinatesScaler.transform(self.trainCoordinates)
            self.scaledTestCoordinates = self.coordinatesScaler.transform(self.testCoordinates)
            
            #Define MinMax scaling for forward swap rates
            self.fwdScaler = customMeanStdScale() # customMinMaxScale()
            self.fwdScaler.fit(self.trainFwd)
            self.scaledTrainFwd = self.fwdScaler.transform(self.trainFwd)
            self.scaledTestFwd = self.fwdScaler.transform(self.testFwd)
        else : 
            self.scaledTrainVol = self.trainVol
            self.scaledTestVol = self.testVol
            
            self.scaledTrainCoordinates = self.trainCoordinates
            self.scaledTestCoordinates = self.testCoordinates
            
            self.scaledTrainFwd = self.trainFwd
            self.scaledTestFwd = self.testFwd
        return 
    
    

def getATMDataFromCSV(dataSetPath, trainingSetPercentage=0.8):
    formattedHistory = extractDataFromCSV(dataSetPath)
    #Filter only ATM volatility
    ATMHistory = (formattedHistory[formattedHistory.index.get_level_values(indexRelStrike)=='A']
                  .reorder_levels([indexName, indexExpiry, indexTenor, indexRelStrike])
                  .sort_index())
    #Remove strike from index as we consider only ATM
    ATMHistory.index = ATMHistory.index.droplevel(3)
    #Get Expiry and tenors shared by all dates
    commonGridPoints = intersectionGrid(ATMHistory)
    #Get indexer for multiindex
    idx = pd.IndexSlice
    #Filter data for Expiry and tenors common to all dates
    commonATMHistory = ATMHistory.loc[idx[:,commonGridPoints.get_level_values(0),
                                          commonGridPoints.get_level_values(1)],:]
    #Feeding Data
    #Take the first 80% dates as training set and the remaining ones as testing set
    trainTmp,testTmp = splitTrainTestDataChronologically(commonATMHistory,trainingSetPercentage)
    #Separate features between volatility, forward rate and Strike
    testVol = splitHistory(testTmp,"Vol")
    trainVol = splitHistory(trainTmp,"Vol")
    
    testFwd = splitHistory(testTmp,"forward")
    trainFwd = splitHistory(trainTmp,"forward")
    
    testStrike = None
    trainStrike = None
    
    indexFunc = lambda x : pd.Series(x.index.values, 
                                     index = x.index)
    trainCoordinates = trainVol.apply(indexFunc, axis=1)
    testCoordinates = testVol.apply(indexFunc, axis=1)
    trainVol = pd.DataFrame(trainVol.values, index=trainVol.index)
    testVol = pd.DataFrame(testVol.values, index=testVol.index)
    trainCoordinates = pd.DataFrame(trainCoordinates.values, index=trainCoordinates.index)
    testCoordinates = pd.DataFrame(testCoordinates.values, index=testCoordinates.index)
    
    return testVol, trainVol, testFwd, trainFwd, testCoordinates, trainCoordinates, testStrike, trainStrike

    
class dataSetATMCSV(datasetATM):
    def __init__(self, pathToDataset, 
                 trainingSetPercentage, 
                 minExpiry, 
                 completionRate,
                 scaleFeatures = False):
        
        self.nbExpiry = 0
        self.nbTenors = 0
        self.minExpiry = minExpiry
        self.expiryTenorToRankSerie = None
        
        super().__init__(pathToDataset, 
                         trainingSetPercentage, 
                         minExpiry, 
                         completionRate,
                         scaleFeatures = scaleFeatures)
        
        listTokeep = [(0.08333333333333333,0.25),(0.08333333333333333,10.0),
                      (0.08333333333333333,30.0),(0.5,2.0),(0.5,15.0),
                      (5.0,1.0),(5.0,20.0),(10.0,5.0)]
        self.setMaskedPoints(listTokeep)
        
    def setMaskedPoints(self, completionPoints):
        # self.maskedPoints = sampleSwaptionsToDelete(self.getTestingDataForModel(), 
                                                    # completionRate)
        fullObs = self.getTestingDataForModel()[1]
        self.maskedPoints = fullObs.columns.difference(completionPoints) 
        
        if self.isGridStable :#Surface coordinates are the same for each day
            
            
            #Matrix where True indicates that this point is completed (i.e. hidden on the market), false otherwise
            maskMatrix = pd.Series(False, index = self.expiryTenorToRankSerie.index)
            maskMatrix.loc[fullObs.iloc[0].loc[self.maskedPoints]] = True
            self.maskSerie = pd.Series(maskMatrix.values, index = self.expiryTenorToRankSerie.values)
            self.maskMatrix = maskMatrix.unstack(level=-1)
    
    #Return a deep copy with masked values
    def maskDataset(self, completeDataset):
        maskedRank = self.maskedPoints
        maskedDataset = completeDataset.copy()
        if completeDataset.ndim == 1 :
            maskedDataset.loc[maskedRank] = np.NaN
        elif completeDataset.ndim == 2 :
            maskedDataset[maskedRank] = np.NaN
        return maskedDataset
    
    def removeShortestExpiry(self, dataset):
        if dataset is None :
            return
        #remove data with expiry inferior than minExpiry
        hasExpiryColumn = ("Expiry" in dataset.columns.names)
        columnsFilter = ((dataset.columns.get_level_values("Expiry")>=self.minExpiry) if hasExpiryColumn else 
                          self.expiryTenorToRankSerie[self.expiryTenorToRankSerie.index.get_level_values("Expiry")>=self.minExpiry].values)
        return dataset.filter(items=dataset.columns[columnsFilter])
    
    def loadData(self):
        tmp = getATMDataFromCSV(self.pathToDataset, self.trainingSetPercentage)
        self.expiryTenorToRankSerie = pd.Series(tmp[4].columns,
                                                index = pd.MultiIndex.from_tuples(tmp[4].iloc[0].values, 
                                                                                  names=('Expiry', 'Tenor')))
        self.expiryTenorToRankSerie = self.expiryTenorToRankSerie[self.expiryTenorToRankSerie.index.get_level_values("Expiry")>=self.minExpiry]                                                                    
        
        self.testVol = self.removeShortestExpiry(tmp[0])
        self.trainVol = self.removeShortestExpiry(tmp[1])
        
        self.testCoordinates = self.removeShortestExpiry(tmp[4])
        self.trainCoordinates = self.removeShortestExpiry(tmp[5])
        
        self.testFwd = self.removeShortestExpiry(tmp[2])
        self.trainFwd = self.removeShortestExpiry(tmp[3])
        
        self.testStrike = self.removeShortestExpiry(tmp[6])
        self.trainStrike = self.removeShortestExpiry(tmp[7])
        
        self.nbExpiry = self.trainFwd.columns.get_level_values("Expiry").unique().size
        self.nbTenors = self.trainFwd.columns.get_level_values("Tenor").unique().size
        self.gridSize = self.trainFwd.columns.size
        
        return
        
    def datasetSummary(self):
        print("Number of days in dataset", 
              self.getDataForModel()[0].shape[0])
        print("Number of days for testing", self.getTestingDataForModel()[0].shape[0])
        print("Number of days for training", self.getTrainingDataForModel()[0].shape[0])
        print("Working on ATM volatility level")
        print("Number of points in the grid : ", self.gridSize)
        print("Number of expiries : ", self.nbExpiry)
        print("List : ", self.getTrainingDataForModel()[2].columns.get_level_values("Expiry").unique())
        print("Number of tenors : ", self.nbTenors)
        print("List : ", self.getTrainingDataForModel()[2].columns.get_level_values("Tenor").unique())
        return

def getATMDataFromPickle(dataSetPath, 
                         trainingSetPercentage=0.8, 
                         minStrikeIndex = 0,
                         maturityStrikeIndex = 0):
    with open(dataSetPath, "rb") as f :
        objectRead = pickle.load(f)
    def rankCalDays(dfDay):
        return dfDay["nBizDays"].rank()
    listRank = list(map(rankCalDays, objectRead))
    dfRank = pd.concat(listRank)
    dfConcat = pd.concat(objectRead)
    dfConcat["Rank"] = dfRank
    volDf = dfConcat.reset_index().set_index(["index", "Rank"]).drop(["Date", "Forwards", "nBizDays", "nCalDays", "diff Days"], axis=1, errors="ignore").unstack()
    volDf.columns = volDf.columns.set_names("Moneyness",level=0)
    volDf = volDf.dropna(how="all",axis=1).astype("float64")
    
    fwdDf = dfConcat.reset_index().set_index(["index", "Rank"])["Forwards"].unstack()
    coordinatesRankDf = dfConcat.reset_index().set_index(["index", "Rank"])["nBizDays"].unstack()
    
    def bindBizDays(rows):
        bizDays = coordinatesRankDf.loc[rows.name].astype("float64")
        return pd.Series(list(zip(bizDays[rows.index.get_level_values("Rank")].values / 252.0, 
                                  np.log(rows.index.get_level_values("Moneyness").astype("float64")) )),
                         index = rows.index)
    coordinatesDf = volDf.apply(bindBizDays, axis=1)
    
    def getFwd(rowVol):
        ttmRank = rowVol.index.get_level_values("Rank")
        return pd.Series(fwdDf.loc[rowVol.name, ttmRank].values, index = rowVol.index)
    #Search for point in the vol dataframe the corresponding forward
    fwdDf = volDf.apply(getFwd, axis=1).dropna(how="all",axis=1).astype("float64")
    
    
    firstTestingDate = int(volDf.index.shape[0] * trainingSetPercentage)
    trainingDates = volDf.index[:firstTestingDate]
    
    trainVol = volDf.loc[trainingDates]
    testVol = volDf.drop(trainVol.index)
    trainVol = pd.DataFrame(trainVol.values, index=trainVol.index)
    testVol = pd.DataFrame(testVol.values, index=testVol.index)
    
    trainFwd = fwdDf.loc[trainVol.index]
    trainFwd = pd.DataFrame(trainFwd.values, index=trainFwd.index)[trainVol.columns]
    testFwd = fwdDf.drop(trainVol.index)
    testFwd = pd.DataFrame(testFwd.values, index=testFwd.index)[testVol.columns]
    
    
    testStrike = None
    trainStrike = None
    
    trainCoordinates = coordinatesDf.loc[trainingDates]
    trainCoordinates = pd.DataFrame(trainCoordinates.values, index=trainCoordinates.index)[trainVol.columns]
    testCoordinates = coordinatesDf.drop(trainVol.index)
    testCoordinates = pd.DataFrame(testCoordinates.values, index=testCoordinates.index)[testVol.columns]
    
    strikeDf = trainCoordinates.applymap(lambda x : x[1]).iloc[0]
    strikeList = np.sort(strikeDf.unique())
    minStrike = strikeList[minStrikeIndex]
    strikesKept = strikeDf[strikeDf >= minStrike].index
    
    maturityDf = trainCoordinates.applymap(lambda x : x[0]).iloc[0][strikesKept]
    maturityList = np.sort(maturityDf.unique())
    minMaturity = maturityList[minStrikeIndex]
    maturityKept = maturityDf[maturityDf >= minMaturity].index
    
    testVol = testVol[maturityKept]
    trainVol = trainVol[maturityKept]
    
    trainCoordinates = trainCoordinates[maturityKept]
    testCoordinates = testCoordinates[maturityKept]
    
    trainFwd = trainFwd[maturityKept]
    testFwd = testFwd[maturityKept]
    
    return testVol, trainVol, testFwd, trainFwd, testCoordinates, trainCoordinates, testStrike, trainStrike

def saveInterpolationResult(pathFile, paramDf, interpDf):
    pathTestFileInterp = pathFile + 'Interp'
    dictPickle = {}
    dictPickle["InterpParam"] = paramDf
    dictPickle["InterpolatedDf"] = interpDf
    with open(pathTestFileInterp, "wb") as f :
        pickle.dump(dictPickle, f, protocol=3)
    return

def removePointsWithInvalidCoordinates(incompleteSurface, coordinates):
    #Filter location with incomplete observations
    def invalidCoordinates(x):
        if isinstance(x, tuple):
            return not any(np.isnan(x))
        return not np.isnan(x)
    filteredCoordinates = np.array(list(map(invalidCoordinates, coordinates)))
    return incompleteSurface[filteredCoordinates], coordinates[filteredCoordinates]
    
def readInterpolationResult(pathFile):
    pathTestFileInterp = pathFile + 'Interp'
    with open(pathTestFileInterp, "rb") as f :
        dictPickle = pickle.load(f)
    return dictPickle["InterpParam"], dictPickle["InterpolatedDf"]

class dataSetATMPickle(datasetATM):
    def __init__(self, pathToDataset, 
                 trainingSetPercentage, 
                 minExpiry, 
                 completionRate,
                 scaleFeatures = False):
        
        self.nbMoneyness = 0
        self.MoneynessList = []
        self.nbTTM = 0
        self.ttmList = []
        self.minTTM = None
        self.isGridStable = False
        
        self.minStrike = 4
        self.minMaturity = 0
        self.logTransform = True
        
        super().__init__(pathToDataset, 
                         trainingSetPercentage, 
                         minExpiry, 
                         completionRate,
                         scaleFeatures = scaleFeatures)
        
        listTokeep = [1.0, 2.0, 3.0, 4.0]
        self.setMaskedPoints(listTokeep)
    
    def datasetSummary(self):
        print("Number of days in dataset", 
              self.getDataForModel()[0].shape[0])
        print("Number of days for testing", self.getTestingDataForModel()[0].shape[0])
        print("Number of days for training", self.getTrainingDataForModel()[0].shape[0])
        print("Working on Equity volatility level")
        print("Number of points in the grid : ", self.gridSize)
        print("Number of Moneyness : ", self.nbMoneyness)
        print("List : ", self.MoneynessList)
        print("Number of Time to maturities : ", self.nbTTM)
        print("List : ", self.ttmList)
        return
    
    def loadData(self):
        tmp = getATMDataFromPickle(self.pathToDataset, self.trainingSetPercentage, self.minStrike, self.minMaturity)
        
        self.testVol = tmp[0]
        self.trainVol = tmp[1]
        
        self.testCoordinates = tmp[4]
        self.trainCoordinates = tmp[5]
        
        self.testFwd = tmp[2]
        self.trainFwd = tmp[3]
        
        self.testStrike = tmp[6]
        self.trainStrike = tmp[7]
        
        def extractSingleton(df, coordIndex):
            valueList = np.unique(list(map(lambda x : x[coordIndex], np.ravel(df.values))))
            return valueList[~np.isnan(valueList)] 
        
        fullCoordinatedDf = self.testCoordinates.append(self.trainCoordinates)
        self.MoneynessList = extractSingleton(fullCoordinatedDf, 1)
        self.ttmList = extractSingleton(fullCoordinatedDf, 0)
        self.nbMoneyness = self.MoneynessList.size
        self.nbTTM = self.ttmList.size
        self.gridSize = self.trainVol.columns.size
        
        return
    
    def setMaskedPoints(self, completionPoints):
        # self.maskedPoints = sampleSwaptionsToDelete(self.getTestingDataForModel(), 
                                                    # completionRate)
        fullObs = self.getTestingDataForModel()[0].iloc[0]
        self.maskedPoints = fullObs.index.difference(completionPoints) 
        
        #Matrix where True indicates that this point is completed (i.e. hidden on the market), false otherwise
        maskMatrix = pd.Series(False, index = fullObs.index)
        maskMatrix.loc[self.maskedPoints] = True
        self.maskSerie = maskMatrix
        #self.maskMatrix = maskMatrix.unstack(level=-1)
    
    #Return a deep copy with masked values
    def maskDataset(self, completeDataset):
        maskedRank = self.maskedPoints
        maskedDataset = completeDataset.copy()
        if completeDataset.ndim == 1 :
            maskedDataset.loc[maskedRank] = np.NaN
        elif completeDataset.ndim == 2 :
            maskedDataset[maskedRank] = np.NaN
        return maskedDataset
    #When the grid is not fixed - i.e. volatilities time to maturities are sliding - 
    #we need to decide which instruments can be compared between two dates
    def decideInvestableInstruments(self):
        ttmDf = getTTMFromCoordinates(self.formatModelDataAsDataSet(self.getDataForModel()))
        
        pairIndexHistory = []#series of pair of index 
        nextTTMDf = ttmDf.shift(-1).dropna(how = "all")
        for serie in ttmDf.head(-1).iterrows():
            currentDay = serie[1]
            nextDay = nextTTMDf.loc[serie[0]]
            currentRankForHedgeablePoints = currentDay[(currentDay - 1).isin(nextDay) & (~currentDay.isna())].index
            nextRankForHedgeablePoints = nextDay[(nextDay).isin(currentDay - 1) & (~nextDay.isna())].index
            if currentRankForHedgeablePoints.empty :#case where current or day is not considered as a business day
                currentRankForHedgeablePoints = currentDay[(currentDay).isin(nextDay) & (~currentDay.isna())].index
                nextRankForHedgeablePoints = nextDay[(nextDay).isin(currentDay) & (~nextDay.isna())].index
                
            pairIndexHistory.append((currentRankForHedgeablePoints, nextRankForHedgeablePoints))
        #Last day 
        pairIndexHistory.append((nextRankForHedgeablePoints, nextRankForHedgeablePoints))
        
        pairIndexHistory = pd.Series(pairIndexHistory, index = ttmDf.index)
        return pairIndexHistory
    

class datasetATMVariation(dataSetATMCSV):
    def __init__(self, pathToDataset, 
                 trainingSetPercentage, 
                 minExpiry, 
                 completionRate,
                 scaleFeatures = False):
        self.trainingVolVariation = None
        self.testingVolVariation = None
        self.yesterdayVolSerie = None
        
        self.trainingCoordinatesVariation = None
        self.testingCoordinatesVariation = None
        
        self.trainingFwdVariation = None
        self.testingFwdVariation = None
        self.yesterdayFwdSerie = None
        
        self.trainingStrikeVariation = None
        self.testingStrikeVariation = None
        self.yesterdayStrikeSerie = None
        
        #No variation 
        
        super().__init__(pathToDataset, 
                         trainingSetPercentage, 
                         minExpiry, 
                         completionRate, 
                         scaleFeatures = scaleFeatures)
    
    def addYesterdayLevel(self, variationDataset, levelDataSet):
        if variationDataset.ndim == 1 :
            return variationDataset + levelDataSet.loc[variationDataset.name]
        elif variationDataset.ndim == 2 :
            return variationDataset + levelDataSet.loc[variationDataset.index]
        raise ValueError("Incorrect tensor order !")
        return None 
        
    def removeYesterdayLevel(self, todayDataset, yesterdayDataSet):
        if todayDataset.ndim == 1 :
            return todayDataset - yesterdayDataSet.loc[todayDataset.name]
        elif todayDataset.ndim == 2 :
            return todayDataset - yesterdayDataSet.loc[todayDataset.index]
        raise ValueError("Incorrect tensor order !")
        return None 
    
    
    #Apply scaling and various transform to fall on model data
    #Name of surface should be the date
    def convertRealDataToModelFormat(self, unformattedSurface):
        if (type(unformattedSurface)==type(list())) and (len(unformattedSurface)==4):
            date = unformattedSurface[0].index
            variation = [unformattedSurface[0] - self.yesterdayVolSerie.loc[date],
                         unformattedSurface[1],
                         unformattedSurface[2] - self.yesterdayFwdSerie.loc[unformattedSurface[2].index],
                         unformattedSurface[3]]
                         
            if(self.activateScaling):
                lambdaTransform = lambda x : x[0] if x[1] is None else x[1].transform(x[0])
                return list(map(lambdaTransform, zip(variation, self.fullScaler)))
            else :
                return variation
        elif (type(unformattedSurface)!=type(list())) :
            date = unformattedSurface.name
            variation = unformattedSurface - self.yesterdayVolSerie.loc[date]
            if(self.activateScaling):
                return self.volScaler.transform(variation)
            else :
                return variation
        else :
            raise("Can not format as model data")
        return None
    
    #Format data returned by a model to format 
    #For instance variation are transformed as level with yesterday
    def formatModelDataAsDataSet(self,modelData):
        unScaledModelData = super().formatModelDataAsDataSet(modelData)
        if (type(modelData)==type(list())) and (len(modelData)==4):
            originalFormat = [self.addYesterdayLevel(unScaledModelData[0], self.yesterdayVolSerie),
                              unScaledModelData[1],
                              self.addYesterdayLevel(unScaledModelData[2], self.yesterdayFwdSerie),
                              unScaledModelData[3]]
        elif (type(modelData)!=type(list())) :
            originalFormat = self.addYesterdayLevel(unScaledModelData, self.yesterdayVolSerie)
        else :
            raise("Can not format as model data")
        return originalFormat
    
    def formatDataAsVariation(self, trainingDataSet, testingDataSet):
        trainingVariation = trainingDataSet.diff().dropna(how='all')
        testingVariation = testingDataSet.diff()
        testingVariation.iloc[0] = testingDataSet.iloc[0] - trainingDataSet.iloc[-1]
        
        #Shift date to have a serie of past values
        yesterdayTraining = trainingDataSet.shift().dropna(how='all')
        yesterdayTesting = testingDataSet.shift()
        yesterdayTesting.iloc[0] = trainingDataSet.iloc[-1]
        return trainingVariation, testingVariation, yesterdayTraining.append(yesterdayTesting)
    
    def loadData(self):
        super().loadData()
        
        tmp1 = self.formatDataAsVariation(self.trainVol, self.testVol) 
        self.trainingVolVariation = tmp1[0]
        self.testingVolVariation = tmp1[1]
        self.yesterdayVolSerie = tmp1[2]
        
        #Coordiantes are not formatted as variation
        self.trainingCoordinatesVariation = self.trainCoordinates.loc[self.trainingVolVariation.index]
        self.testingCoordinatesVariation = self.testCoordinates.loc[self.testingVolVariation.index]
        
        tmp2 = self.formatDataAsVariation(self.trainFwd, self.testFwd) 
        self.trainingFwdVariation = tmp2[0]
        self.testingFwdVariation = tmp2[1]
        self.yesterdayFwdSerie = tmp2[2]
        
        # tmp3 = self.formatDataAsVariation(self.trainStrike, self.testStrike) 
        # self.trainingStrikeVariation = tmp3[0]
        # self.testingStrikeVariation = tmp3[1]
        # self.yesterdayStrikeSerie = tmp3[2]
        
        return
    
    
    def scaleDataSets(self):
        if(self.activateScaling): 
            #Define MinMax scaling for volatility
            self.volScaler =  customMeanStdScale() #customMinMaxScale()
            self.volScaler.fit(self.trainingVolVariation)#Positive volatilities of course
            self.scaledTrainVol = self.volScaler.transform(self.trainingVolVariation)
            self.scaledTestVol = self.volScaler.transform(self.testingVolVariation)
            
            #Define MinMax scaling for volatility
            self.coordinatesScaler = customMeanStdScale() #customMinMaxScale()
            self.coordinatesScaler.fit(self.trainCoordinates, enforceDataSetMin = 0)#Positive volatilities of course
            self.scaledTrainCoordinates = self.coordinatesScaler.transform(self.trainingCoordinatesVariation)
            self.scaledTestCoordinates = self.coordinatesScaler.transform(self.testingCoordinatesVariation)
            
            #Define MinMax scaling for forward swap rates
            self.fwdScaler =  customMeanStdScale() #customMinMaxScale()
            self.fwdScaler.fit(self.trainingFwdVariation)
            self.scaledTrainFwd = self.fwdScaler.transform(self.trainingFwdVariation)
            self.scaledTestFwd = self.fwdScaler.transform(self.testingFwdVariation)
            
        else : 
            self.scaledTrainVol = self.trainingVolVariation
            self.scaledTestVol = self.testingVolVariation
            
            self.scaledTrainCoordinates = self.trainingCoordinatesVariation
            self.scaledTestCoordinates = self.testingCoordinatesVariation
            
            self.scaledTrainFwd = self.trainingFwdVariation
            self.scaledTestFwd = self.testingFwdVariation
        return
    





def getSkewDataFromCSV(dataSetPath, trainingSetPercentage=0.8):
    formattedHistory = (extractDataFromCSV(dataSetPath)
                        .reorder_levels([indexName, indexExpiry, indexTenor, indexRelStrike])
                        .sort_index())
    #Get Expiry and tenors shared by all dates
    commonGridPoints = intersectionGrid(formattedHistory)
    #Get indexer for multiindex
    idx = pd.IndexSlice
    #Filter data for Expiry, Tenors and Strike common to all dates
    commonHistory = formattedHistory.loc[idx[:,commonGridPoints.get_level_values(0),
                                             commonGridPoints.get_level_values(1), 
                                             commonGridPoints.get_level_values(2)],:]
    #Feeding Data
    #Take the first 80% dates as training set and the remaining ones as testing set
    trainTmp,testTmp = splitTrainTestDataChronologically(commonHistory,trainingSetPercentage)
    #Separate features between volatility, forward rate and Strike
    testVol = splitHistory(testTmp,"Vol")
    trainVol = splitHistory(trainTmp,"Vol")
    trainVol = pd.DataFrame(trainVol.values, index=trainVol.index)
    testVol = pd.DataFrame(testVol.values, index=testVol.index)
    
    
    testFwd = splitHistory(testTmp,"forward")
    trainFwd = splitHistory(trainTmp,"forward")
    testStrike = splitHistory(testTmp,indexStrike)
    trainStrike = splitHistory(trainTmp,indexStrike)
    
    indexFunc = lambda x : pd.Series(x.index.values, 
                                     index = x.index)
    trainCoordinates = trainVol.apply(indexFunc, axis=1)
    testCoordinates = testVol.apply(indexFunc, axis=1)
    trainCoordinates = pd.DataFrame(trainCoordinates.values, index=trainCoordinates.index)
    testCoordinates = pd.DataFrame(testCoordinates.values, index=testCoordinates.index)
    
    return testVol, trainVol, testFwd, trainFwd, testCoordinates, trainCoordinates, testStrike, trainStrike

class datasetStrike(dataSetATMCSV):
    def __init__(self, pathToDataset, 
                 trainingSetPercentage, 
                 minExpiry, 
                 completionRate,
                 scaleFeatures = False):
        self.nbStrike = 0
        super().__init__(pathToDataset, 
                         trainingSetPercentage, 
                         minExpiry, 
                         completionRate, 
                         scaleFeatures = scaleFeatures)
        
    def loadData(self):
        tmp = getSkewDataFromCSV(pathToDataset, self.trainingSetPercentage)
        self.expiryTenorToRankSerie = pd.Series(tmp[4].columns,
                                                index = pd.MultiIndex.from_tuples(tmp[4].iloc[0].values, 
                                                                                  names=('Expiry', 'Tenor')))
        
        self.testVol = self.removeShortestExpiry(tmp[0])
        self.trainVol = self.removeShortestExpiry(tmp[1])
        
        self.testCoordinates = self.removeShortestExpiry(tmp[4])
        self.trainCoordinates = self.removeShortestExpiry(tmp[5])
        
        self.testFwd = self.removeShortestExpiry(tmp[2])
        self.trainFwd = self.removeShortestExpiry(tmp[3])
        
        self.testStrike = self.removeShortestExpiry(tmp[6])
        self.trainStrike = self.removeShortestExpiry(tmp[7])
        
        self.nbExpiry = self.testFwd.columns.get_level_values("Expiry").unique().size
        self.nbTenors = self.testFwd.columns.get_level_values("Tenor").unique().size
        self.nbStrike = self.testStrike.columns.get_level_values(3).unique().size
        return
        
    def datasetSummary(self):
        super().datasetSummary()
        print("Number of relative strikes : ", self.nbStrike)
        print("List : ", self.testVol.columns.get_level_values(3).unique())
        return
        
class datasetATMLogVariation(datasetATMVariation):
    def __init__(self, pathToDataset, 
                 trainingSetPercentage, 
                 minExpiry, 
                 completionRate, 
                 scaleFeatures = False):
        super().__init__(pathToDataset, 
                         trainingSetPercentage, 
                         minExpiry, 
                         completionRate, 
                         scaleFeatures = scaleFeatures)
        
    
    def formatDataAsVariation(self, trainingDataSet, testingDataSet):
        trainingVariation = np.log(trainingDataSet).diff().dropna(how='all')
        testingVariation = np.log(testingDataSet).diff()
        testingVariation.iloc[0] = np.log(testingDataSet).iloc[0] - np.log(trainingDataSet).iloc[-1]
        
        #Shift date to have a serie of past values
        yesterdayTraining = trainingDataSet.shift().dropna(how='all')
        yesterdayTesting = testingDataSet.shift()
        yesterdayTesting.iloc[0] = trainingDataSet.iloc[-1]
        return trainingVariation, testingVariation, yesterdayTraining.append(yesterdayTesting)
    
    def addYesterdayLevel(self, variationDataset, levelDataSet):
        if variationDataset.ndim == 1 :
            return np.exp(variationDataset) * levelDataSet.loc[variationDataset.name]
        elif variationDataset.ndim == 2 :
            return np.exp(variationDataset) * levelDataSet.loc[variationDataset.index]
        raise ValueError("Incorrect tensor order !")
        return variationDataset 
    
    
    #Apply scaling and various transform to fall on model data
    #Name of surface should be the date
    def convertRealDataToModelFormat(self, unformattedSurface):
        if (type(unformattedSurface)==type(list())) and (len(unformattedSurface)==4):
            date = unformattedSurface[0].index
            variation = [np.log(unformattedSurface[0]) - np.log(self.yesterdayVolSerie.loc[date]),
                         unformattedSurface[1],
                         np.log(unformattedSurface[2]) - np.log(self.yesterdayFwdSerie.loc[date]),
                         unformattedSurface[3]]
                         
            if(self.activateScaling):
                lambdaTransform = lambda x : x[0] if x[1] is None else x[1].transform(x[0])
                return list(map(lambdaTransform, zip(variation, self.fullScaler)))
            else :
                return variation
        elif (type(unformattedSurface)!=type(list())) :
            date = unformattedSurface.name
            variation = np.log(unformattedSurface) - np.log(self.yesterdayVolSerie.loc[date])
            if(self.activateScaling):
                return self.volScaler.transform(variation)
            else :
                return variation
        else :
            raise("Can not format as model data")
        return None
