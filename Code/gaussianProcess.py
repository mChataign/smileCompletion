
import pandas as pd
import numpy as np
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import point_in_polygon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct, WhiteKernel

import factorialModel
import loadData
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
import SSVI
import bootstrapping

#######################################################################################################
class InterpolationModel(factorialModel.FactorialModel):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestInterpolationModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
        
        
        
    
    #Build the learner
    def buildModel(self):
        #raise NotImplementedError()
        return 
    
    def trainWithSession(self, session, inputTrain, nbEpoch, inputTest = None):
        raise NotImplementedError("Not a tensorflow model")
        return super().trainWithSession(session, 
                                        inputTrain, 
                                        nbEpoch, 
                                        inputTest = inputTest)
    
    def train(self, inputTrain, nbEpoch, inputTest = None):
        #Do nothing
        return np.array([0.0])
    
    
    def evalModelWithSession(self, sess, inputTest):
        raise NotImplementedError("Not a tensorflow model")
        return super().evalModelWithSession(sess, inputTest)
    
    
    def evalModel(self, inputTestList):
        #No loss since we interpolate exactly
        inputTest = inputTestList[0]
        coordinates = inputTestList[1]
        loss = pd.Series(np.zeros(inputTest.shape[0]), index = inputTest.index)
        
        #Return the inputs as compressed values
        inputs = inputTest.apply(lambda x : self.interpolate(x, coordinates.loc[x.name]), axis=1)
        
        #We do not have any factors so we assign a dummy value of 1
        factors = pd.DataFrame(np.ones((inputTest.shape[0],self.nbFactors)), 
                               index=inputTest.index)
        return loss, inputs, factors
        
    
    def getWeightAndBiasFromLayer(self, layer):
        raise NotImplementedError("Not a tensorflow model")
        return super().getWeightAndBiasFromLayer(layer)
    
    #Interpolate or extrapolate certain values given the knowledge of other ones
    def interpolate(self, incompleteSurface, coordinates):
        raise NotImplementedError()
        return pd.Series()
    
    def completeDataTensor(self, 
                           sparseSurfaceList, 
                           initialValueForFactors, 
                           nbCalibrationStep):
        # knownValues = sparseSurface.dropna()
        # locationToInterpolate = sparseSurface[sparseSurface.isna()].index
        sparseSurface = sparseSurfaceList[0]
        coordinates = sparseSurfaceList[1]
        interpolatedValues = self.interpolate(sparseSurface, coordinates)
        
        #Not a factorial model, we assign a dummy value
        bestFactors = np.ones(self.nbFactors)
        #Exact inteprolation
        calibrationLoss = 0.0
        calibrationSerie = pd.Series([calibrationLoss])
        #Complete surface with inteporlated values
        bestSurface = interpolatedValues
        return calibrationLoss, bestFactors, bestSurface, calibrationSerie
    
    
    #Interpolation does not assume any factors but relies on some known values
    def evalSingleDayWithoutCalibrationWithSensi(self, initialValueForFactors, dataSetList):
        raise NotImplementedError("Not a factorial model")
        return super().evalSingleDayWithoutCalibrationWithSensi(initialValueForFactors, dataSetList)
    
    def plotInterpolatedSurface(self,valueToInterpolate, calibratedFactors, 
                                colorMapSystem=None, 
                                plotType=None):
        raise NotImplementedError("Not a factorial model")
        return
        
    def evalInterdependancy(self, fullSurfaceList):
        raise NotImplementedError("Not a Factorial model")
        return
        
    def evalSingleDayWithoutCalibration(self, initialValueForFactors, dataSetList):
        raise NotImplementedError("Not a Factorial model")
        return

#ToolBox

#######################################################################################################
def getMaskedPoints(incompleteSurface, coordinates):
    return coordinates.loc[incompleteSurface.isna()]

def getMaskMatrix(incompleteSurface):
    maskMatrix = incompleteSurface.copy().fillna(True)
    maskMatrix.loc[~incompleteSurface.isna()] = False
    return maskMatrix

#maskedGrid : surface precising missing value with a NaN
#Assuming indexes and columns are sorted
#Select swaption coordinates (expiry, tenor) whose value is known and are on the boundary
#This defined a polygon whose vertices are known values
def selectPolygonOuterPoints(coordinates):
    outerPoints = []
    
    #Group coordinates by first coordinate
    splittedCoordinates = {}
    for tple in coordinates.values :
        if tple[0] not in splittedCoordinates :
            splittedCoordinates[tple[0]] = []
        splittedCoordinates[tple[0]].append(tple[1])
    
    #Get maximum and minimum for the second dimension
    for key in splittedCoordinates.keys():
        yMin = np.nanmin(splittedCoordinates[key])
        yMax = np.nanmax(splittedCoordinates[key])
        outerPoints.append((key,yMin))
        outerPoints.append((key,yMax))
        
    return outerPoints

def removeNaNcooridnates(coordinatesList):
    isNotNaN = [False if (np.isnan(x[0])  or np.isnan(x[1])) else True for x in coordinatesList]
    return coordinatesList[isNotNaN]

#Order a list of vertices to form a polygon 
def orderPolygonVertices(outerPointList):
    sortedPointList = np.sort(outerPointList) #np sort supports array of tuples 
    #Points are built as a pair of two points for value in the first dimension
    #Hence the polygon starts with points having the first value for the second dimension
    #(and order them along the first dimension)
    orderedListOfVertices = sortedPointList[::2]
    #We then browse the remaining points but in the reverse order for the second dimension
    orderedListOfVertices = sortedPointList[1::2][::-1]
    return orderedListOfVertices

#Select swaption coordinates (expiry, tenor) whose value is known and are on the boundary
#This defined a polygon whose vertices are known values
def buildInnerDomainCompletion(incompleteSurface, coordinates):
    coordinatesWithValues = coordinates.loc[~incompleteSurface.isna()]
    outerPointsList = selectPolygonOuterPoints(coordinatesWithValues)
    verticesList = orderPolygonVertices(outerPointsList)
    expiryVertices, tenorVectices = zip(*verticesList)
    return expiryVertices, tenorVectices

#Select swaption coordinates (expiry, tenor) whose value is known 
#and their coordinate corresponds to maximum/minimum value for x axis and y axis
#This defines a quadrilateral
def buildOuterDomainCompletion(incompleteSurface, coordinates):
    coordinatesWithValues = coordinates.loc[~incompleteSurface.isna()].values
    firstDimValues = list(map(lambda x : x[0], coordinatesWithValues))
    secondDimValues = list(map(lambda x : x[1], coordinatesWithValues))
    
    maxExpiry = np.amax(firstDimValues)
    minExpiry = np.nanmin(firstDimValues)
    
    maxTenor = np.amax(secondDimValues)
    minTenor = np.nanmin(secondDimValues)
    
    expiryVertices = [maxExpiry, maxExpiry, minExpiry, minExpiry, maxExpiry]
    tenorVectices = [maxTenor, minTenor, minTenor, maxTenor, maxTenor]
    
    return expiryVertices, tenorVectices

#verticesList : list of vertices defining the polygon
#Points : multiIndex serie for which we want to check the coordinates belongs to the domain defined by the polygon
#Use Winding number algorithm
def areInPolygon(verticesList, points):
    return pd.Series(points.map(lambda p : point_in_polygon.wn_PnPoly(p, verticesList) != 0).values, 
                     index = points.index)

#Return the list (pandas Dataframe) of points which are located in the domain (as a closed set) 
#The closure ( i.e. edge of the domain ) is also returned
#defined by points which are not masked
def areInInnerPolygon(incompleteSurface, coordinates, showDomain = False):
    #Add the frontier
    gridPoints = coordinates.loc[~incompleteSurface.isna()]
    
    #Build polygon from the frontier
    expiriesPolygon, tenorsPolygon = buildInnerDomainCompletion(incompleteSurface, coordinates)
    polygon = list(zip(expiriesPolygon,tenorsPolygon))
    
    #Search among masked points which ones lie inside the polygon
    maskedPoints = getMaskedPoints(incompleteSurface, coordinates)
    interiorPoints = areInPolygon(polygon, maskedPoints)
    if not interiorPoints.empty :
        gridPoints = gridPoints.append(maskedPoints[interiorPoints]).drop_duplicates()
    
    
    if showDomain :
        plt.plot(expiriesPolygon,tenorsPolygon)
        plt.xlabel("First dimension")
        plt.xlabel("Second dimension")
        plt.plot(gridPoints.map(lambda x : x[0]).values,
                 gridPoints.map(lambda x : x[1]).values,
                 'ro')
        plt.show()
    
    return gridPoints
    
#Return the list (pandas Dataframe) of points which are located in the outer domain (as a closed set) 
#Outer domain is delimited by the maximum and minimum coordinates of the known values
#inner domain is delimited by the polygon whose vertices are the known points
#showDomain plots the boundary ( i.e. edge of the domain ) and the points which are inside the quadrilateral
def areInOuterPolygon(incompleteSurface, coordinates, showDomain = False):
    #Add the frontier
    gridPoints = coordinates.loc[~incompleteSurface.isna()]
    
    #Build polygon from the frontier
    expiriesPolygon, tenorsPolygon = buildOuterDomainCompletion(incompleteSurface, coordinates)
    polygon = list(zip(expiriesPolygon,tenorsPolygon))
    
    #Search among masked points which ones lie inside the polygon
    maskedPoints = getMaskedPoints(incompleteSurface, coordinates)
    interiorPoints = areInPolygon(polygon, maskedPoints)
    if not interiorPoints.empty :
        gridPoints = gridPoints.append(maskedPoints[interiorPoints]).drop_duplicates()
    
    
    if showDomain :
        plt.plot(expiriesPolygon,tenorsPolygon)
        plt.xlabel("First dimension")
        plt.xlabel("Second dimension")
        plt.plot(gridPoints.map(lambda x : x[0]).values,
                 gridPoints.map(lambda x : x[1]).values,
                 'ro')
        plt.show()
    
    return gridPoints

#######################################################################################################
#Linear interpolation with flat extrapolation
#Assume row are non empty
def interpolateRow(row, coordinates): 
    definedValues = row.dropna()
    if definedValues.size == 1 :
        return pd.Series(definedValues.iloc[0] * np.ones_like(row), 
                         index = row.index)
    else : 
        #Flat extrapolation and linear interpolation based on index (Tenor) value
        filledRow = row.interpolate(method='index', limit_direction = 'both')
        return filledRow
        
        

def formatCoordinatesAsArray(coordinateList):
    x = np.ravel(list(map(lambda x : x[0], coordinateList)))
    y = np.ravel(list(map(lambda x : x[1], coordinateList)))
    return np.vstack((x, y)).T

#Linear interpolation combined with Nearest neighbor extrapolation
# drawn from https://github.com/mChataign/DupireNN
def customInterpolator(interpolatedData, formerCoordinates, NewCoordinates):
    knownPositions = formatCoordinatesAsArray(formerCoordinates)
    
    xNew = np.ravel(list(map(lambda x : x[0], NewCoordinates)))
    yNew = np.ravel(list(map(lambda x : x[1], NewCoordinates)))
    # print(type(xNew))
    # print(type(yNew))
    # print(np.array((xNew, yNew)).T.shape)
    # print(type(interpolatedData))
    # print(type(knownPositions))
    # print()
    
    fInterpolation = griddata(knownPositions,
                              np.ravel(interpolatedData),
                              np.array((xNew, yNew)).T,
                              method = 'linear',
                              rescale=True)
    
    fExtrapolation =  griddata(knownPositions,
                               np.ravel(interpolatedData),
                               np.array((xNew, yNew)).T,
                               method = 'nearest',
                               rescale=True)
    
    return np.where(np.isnan(fInterpolation), fExtrapolation, fInterpolation)

def interpolate(incompleteSurface, coordinates):
    knownValues = incompleteSurface.dropna()
    knownLocation = coordinates.loc[knownValues.index]
    locationToInterpolate = coordinates.drop(knownValues.index)
    interpolatedValues = customInterpolator(knownValues.values, 
                                            knownLocation.values, 
                                            locationToInterpolate.values)
    completeSurface = pd.Series(interpolatedValues, 
                                index = locationToInterpolate.index).append(knownValues)
    
    return completeSurface.loc[incompleteSurface.index].rename(incompleteSurface.name)


def extrapolationFlat(incompleteSurface, coordinates):
    filteredSurface, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(incompleteSurface, coordinates)
    correctedSurface = interpolate(filteredSurface, filteredCoordinates)
    correctedSurface = correctedSurface.append(pd.Series(incompleteSurface.drop(filteredCoordinates.index),
                                                         index = coordinates.drop(filteredCoordinates.index).index))
    return correctedSurface.sort_index()

#######################################################################################################
class LinearInterpolation(InterpolationModel):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestLinearInterpolationModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    
    #Extrapolation is flat and interpolation is linear
    def interpolate(self, incompleteSurface, coordinates):
        filteredSurface, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(incompleteSurface, coordinates)
        interpolatedSurface = interpolate(filteredSurface, filteredCoordinates)
        nanSurface = incompleteSurface.drop(interpolatedSurface.index)
        return interpolatedSurface.append(nanSurface)[coordinates.index].rename(incompleteSurface.name)
        
    
    # #Build the learner
    # def buildModel(self):
        # raise NotImplementedError()
        # return 


#######################################################################################################
class SplineInterpolation(LinearInterpolation):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestSplineInterpolationModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    def customInterpolator(self, interpolatedData, formerCoordinates, NewCoordinates):
        knownPositions = formatCoordinatesAsArray(formerCoordinates)
        
        xNew = np.ravel(list(map(lambda x : x[0], NewCoordinates)))
        yNew = np.ravel(list(map(lambda x : x[1], NewCoordinates)))
        
        
        fInterpolation = griddata(knownPositions,
                                  np.ravel(interpolatedData),
                                  (xNew, yNew),
                                  method = 'cubic',
                                  rescale=True)
        
        fExtrapolation =  griddata(knownPositions,
                                   np.ravel(interpolatedData),
                                   (xNew, yNew),
                                   method = 'nearest',
                                   rescale=True)
        
        return np.where(np.isnan(fInterpolation), fExtrapolation, fInterpolation)
    
    #Extrapolation is flat and interpolation is linear
    def interpolate(self, incompleteSurface, coordinates):
        filteredSurface, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(incompleteSurface, coordinates)
        
        knownValues = filteredSurface.dropna()
        knownLocation = filteredCoordinates.loc[knownValues.index]
        locationToInterpolate = filteredCoordinates.drop(knownValues.index)
        interpolatedValues = self.customInterpolator(knownValues.values, 
                                                     knownLocation.values,
                                                     locationToInterpolate.values)
        completeSurface = pd.Series(interpolatedValues, 
                                    index = locationToInterpolate.index).append(knownValues)
        
        interpolatedSurface = completeSurface.loc[filteredSurface.index].rename(filteredSurface.name)
        nanSurface = incompleteSurface.drop(interpolatedSurface.index)
        return interpolatedSurface.append(nanSurface)[coordinates.index].rename(incompleteSurface.name)
    


#######################################################################################################
class GaussianProcess(InterpolationModel):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestGaussianModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
                         
        self.TrainGaussianHyperparameters = (self.hyperParameters["Train Interpolation"] 
                                             if ("Train Interpolation" in self.hyperParameters) else False)
        self.sigmaF = self.hyperParameters["sigmaF"] if ("sigmaF" in self.hyperParameters) else 50.0
        self.bandwidth = self.hyperParameters["bandwidth"] if ("bandwidth" in self.hyperParameters) else 0.5
        
        self.sigmaBounds = self.hyperParameters["sigmaBounds"] if ("sigmaBounds" in self.hyperParameters) else (1.0, 200.0)
        self.bandwidthBounds = self.hyperParameters["bandwidthBounds"] if ("bandwidthBounds" in self.hyperParameters) else (0.01, 10.0)
        
        self.kernel = (ConstantKernel(constant_value=self.sigmaF, 
                                      constant_value_bounds=self.sigmaBounds) 
                       * RBF(length_scale=self.bandwidth, 
                             length_scale_bounds=self.bandwidthBounds))
    
    def kernelRBF(self, X1, X2, sigma_f=1.0, l=1.0):
        '''
        Isotropic squared exponential kernel. Computes 
        a covariance matrix from points in X1 and X2.
            
        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).
        
        Returns:
            Covariance matrix (m x n).
        '''
        #print("sigma_f : ",sigma_f," l : ",l)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
    
    def predictGaussianModel(self, X, XStar, Y, sigma_f, l):
        
        KStar = self.kernelRBF(X, XStar, sigma_f, l)
        KStarT = KStar.T
        
        K = self.kernelRBF(X, X, sigma_f, l) 
        #Add noise to avoid singular matrix problem
        noise = (1e-9) * np.eye(K.shape[0])
        KInv = np.linalg.inv(K + noise)
        KStarStar = self.kernelRBF(XStar, XStar, sigma_f, l)
        
        YStar = np.dot(np.dot(KStarT,KInv),Y)
        YStarUncertainty = KStarStar - np.dot(np.dot(KStarT,KInv),KStar)
        
        return YStar, YStarUncertainty
    
    def predictGaussianModelFormatted(self, knownValues, locationToInterpolate, coordinates):
        knownLocation = coordinates.loc[knownValues.index]
        #Optimize on log parameters 
        interpolatedValues, _ = self.predictGaussianModel(formatCoordinatesAsArray(knownLocation.values), 
                                                          formatCoordinatesAsArray(locationToInterpolate.values),
                                                          knownValues.values,
                                                          np.exp(self.kernel.theta[0]),
                                                          np.exp(self.kernel.theta[1]))
        return pd.Series(interpolatedValues, index = locationToInterpolate.index)
    
    #Interpolate or extrapolate certain values given the knowledge of other ones
    def interpolate(self, incompleteSurface, coordinates):
        filteredSurface, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(incompleteSurface, 
                                                                                           coordinates)
        nanSurface = incompleteSurface.drop(filteredSurface.index)
        
        
        extrapolationMode = self.hyperParameters["extrapolationMode"] if "extrapolationMode" in self.hyperParameters else None 
        
        #NoExtrapolation : NoExtrapolation | InnerDomain | OuterDomain
        #LocationToInterpolate : Index of missing values
        #knownValues : Serie of values which are known
        knownValues = filteredSurface.dropna()
        if knownValues.size == filteredSurface.size : #No value to interpolate
            return incompleteSurface
        
        resSurface = filteredSurface.copy()
        
        interpolatedPoint = None
        if extrapolationMode == 'InnerDomain' :
            interpolatedPoint = areInInnerPolygon(filteredSurface, filteredCoordinates)
        elif extrapolationMode == 'OuterDomain' :
            interpolatedPoint = areInOuterPolygon(filteredSurface, filteredCoordinates)
        else : #NoExtrapolation
            interpolatedPoint = filteredCoordinates.drop(knownValues.index)
        
        
        if self.TrainGaussianHyperparameters :
            interpolatedValues = self.predictGaussianModelFormatted(knownValues, 
                                                                    interpolatedPoint,
                                                                    filteredCoordinates)
        else : 
            knownLocation = filteredCoordinates.loc[knownValues.index]
            interpolator = GaussianProcessRegressor(kernel=self.kernel,
                                                    random_state=0, 
                                                    normalize_y=True).fit(formatCoordinatesAsArray(knownLocation.values), 
                                                                          knownValues.values)
            
            interpolatedValues = pd.Series(interpolator.predict(formatCoordinatesAsArray(interpolatedPoint.values), return_std=False), 
                                           index = interpolatedPoint.index)
            
        resSurface.loc[interpolatedValues.index] = interpolatedValues
        
        
        return extrapolationFlat(resSurface.append(nanSurface)[incompleteSurface.index].rename(incompleteSurface.name), 
                                 coordinates)
    
    def nll_fn(self, X_trainSerie, Y_trainSerie, theta, noise=1e-3):
        '''
        Computes the negative log marginal
        likelihood for training data X_train and Y_train and given 
        noise level.
        
        Args:
            X_train: training locations (m x d).
            Y_train: training targets (m x 1).
            noise: known noise level of Y_train. 
            theta: gaussian hyperparameters [sigma_f, l]
        '''
        
        filteredSurface, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(Y_trainSerie, 
                                                                                           X_trainSerie)
        Y_Train = filteredSurface.dropna().values
        X_train = formatCoordinatesAsArray(filteredCoordinates.loc[filteredSurface.dropna().index].values)
        
        
        # Numerically more stable implementation of Eq. (7) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        K = (self.kernelRBF(X_train, X_train, sigma_f=theta[0], l=theta[1]) + 
             noise**2 * np.eye(len(X_train)))
        L = np.linalg.cholesky(K)
        return (np.sum(np.log(np.diagonal(L))) + 
                0.5 * Y_train.T.dot(np.linalg.lstsq(L.T, np.linalg.lstsq(L, Y_train)[0])[0]) + 
                0.5 * len(X_train) * np.log(2*np.pi))
        
    #Apply nll_fn for each day of YSerie and sum results
    def computeTrainHistoryLogLikelyhood(self, kernelParams, dataSetList):
        error = 0
        locations = dataSetList[1]  #YSerie.iloc[0].index.to_frame().values
        func = lambda x : self.nll_fn(locations.loc[x.name], x, np.exp(kernelParams)) 
        marginalLogLikelyhood = dataSetList[0].apply(func, axis = 1)
        return marginalLogLikelyhood.sum()
    
    
    def train(self, inputTrain, nbEpoch, inputTest = None):
        if self.TrainGaussianHyperparameters :
            #Calibrate globally gaussian process hyperparameters l and sigma on the training set
            objectiveFuntion = lambda x : self.computeTrainHistoryLogLikelyhood(x,inputTrain)
            
            nbRestart = 5#15
            bestValue = None
            bestParam = None
            
            #As loglikelyhood function is nonconvex we try l-bfgs algorithms several times 
            def randomStart(bounds, nbStart):
                return np.random.uniform(low=bounds[0], high=bounds[1], size=nbStart)
            
            optimStarts = np.apply_along_axis(lambda x : randomStart(x,nbRestart), 
                                              1, 
                                              self.kernel.bounds).T
            start = time.time()
            for i in range(nbRestart):
                print("bounds", np.exp(self.kernel.bounds))
                print("random Starts", np.exp(optimStarts[i]))
                resOptim = scipy.optimize.fmin_l_bfgs_b(objectiveFuntion, 
                                                        optimStarts[i], 
                                                        approx_grad = True,
                                                        maxiter = 20,
                                                        bounds = self.kernel.bounds)
                if self.verbose :
                    print(resOptim)
                if (bestParam is None) or (resOptim[1] < bestValue) : 
                    bestValue = resOptim[1]
                    bestParam = resOptim[0]
                print("Attempt : ", i, " nnLogLikelyHood : ", bestValue, " bestParam : ", np.exp(bestParam))
            
            optimalValues = {'k1__constant_value' : np.exp(bestParam)[0], 
                             'k2__length_scale' : np.exp(bestParam)[1]}
            self.kernel.set_params(**optimalValues)
            
            print("Time spent during optimization : ", time.time() - start)
            
        #Else
        return super().train(inputTrain, nbEpoch, inputTest = None)

def getTTMFromCoordinates(dfList):
    return dfList[1].applymap(lambda x : x[0])
def getMoneynessFromCoordinates(dfList):
    return dfList[1].applymap(lambda x : x[1])


#######################################################################################################

class NelsonSiegelCalibrator:    
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 order,
                 hyperparameters):
        self.hyperParameters = hyperparameters
        self.order = order
        self.beta = []
        self.alpha = []
        self.verbose = False
    
    def objectiveFunction(self, ttm, beta, alpha):
        slopeTime = (1 - np.exp(-alpha[0] * ttm))/(alpha[0] * ttm)
        nelsonSiegel = beta[0] + slopeTime * beta[1] + (slopeTime - np.exp(-alpha[0] * ttm)) * beta[2]
        if self.order == 4 :
            nelsonSiegelSvensson = nelsonSiegel + ((1 - np.exp(-alpha[1] * ttm))/(alpha[1] * ttm) - np.exp(-alpha[1] * ttm)) * beta[3]
            return nelsonSiegelSvensson
        return nelsonSiegel
    
    def drawnStartingPoints(self, bounds):
        randPos = np.random.rand(len(bounds))
        return [x[0][0] + (x[0][1] - x[0][0]) * x[1] for x in zip(bounds, randPos)]
        
    
    def calibrate(self, curvesVol, ttms):
        
        if self.order == 4 : 
            #Bounds taken from "Calibrating the Nelson–Siegel–Svensson model", M. Gilli, Stefan Große, E. Schumann
            #See https://comisef.eu/files/wps031.pdf
            bounds = [(-10000,10000), (-10000,10000), (-10000,10000), (-10000,10000), (0,100), (0,200)]
            startBounds = [(-1,1), (-1,1), (-1,1), (-1,1), (0,30), (0,30)]
            func = lambda x : np.sqrt(np.nanmean(np.square(self.objectiveFunction(ttms/250, x[:4], x[4:]) - curvesVol)))
        else :
            bounds = [(-10000,10000), (-10000,10000), (-10000,10000), (0,200)]
            startBounds = [(-1,1), (-1,1), (-1,1), (0,30)]
            func = lambda x : np.sqrt(np.nanmean(np.square(self.objectiveFunction(ttms/250, x[:3], x[3:]) - curvesVol)))
        
        bestFit = None
        nbInit = 10
        for k in range(nbInit) : 
            startingPoints = self.drawnStartingPoints(startBounds)
            resOptim = scipy.optimize.minimize(func, startingPoints, bounds=bounds, method='L-BFGS-B')
            
            if bestFit is None or resOptim.fun < bestFit :
                bestFit = resOptim.fun
                self.beta = resOptim.x[:4] if self.order == 4 else resOptim.x[:3]
                self.alpha = resOptim.x[4:] if self.order == 4 else resOptim.x[3:]
            if self.verbose :
                print(resOptim.fun, " ; ", bestFit)
        if self.verbose :
            print("best error : ", bestFit)
        return
        
    def interpolate(self, ttm):
        return self.objectiveFunction(ttm/250, self.beta, self.alpha)

#Post-treatments for calibrateModelMoneynessWiseDaily
def mergeResults(xCoordinates, 
                 xAvgCoordinates, 
                 xVol, 
                 interpList, 
                 refList, 
                 nelsonList,
                 dfList):
    interpVolDf = pd.concat(interpList,axis=1)
    refVolDf = pd.concat(refList,axis=1)
    
    moneynesses = np.unique(getMoneynessFromCoordinates(dfList))
    nelsonIndex = pd.MultiIndex.from_product( [moneynesses, nelsonList[0].columns], 
                                             names=["Moneyness", "Nelson-Siegel Parameters"])
    nelsonDf = pd.DataFrame(pd.concat(nelsonList,axis=1).values, 
                            index = nelsonList[0].index, 
                            columns = nelsonIndex)
    
    coordinatesIndex = pd.MultiIndex.from_product([moneynesses, xCoordinates[0].columns], 
                                                  names=["Moneyness", "Rank"])
    coordinatesDf = pd.DataFrame(pd.concat(xCoordinates,axis=1).values, 
                                 index=nelsonList[0].index, 
                                 columns =  coordinatesIndex)
    volDf = pd.DataFrame(pd.concat(xVol,axis=1).values, 
                         index=nelsonList[0].index, 
                         columns =  coordinatesIndex)
    return interpVolDf, refVolDf, nelsonDf, coordinatesDf, volDf

#Interpolate Volatility along maturity for predefined nelson-seigel parameters
def getVolFromNelsonParameters(nelsonDf, coordinatesDf):
    def getVolFromNelsonParametersApply(nelsonRow, coordinatesRow):
        #iterate on moneyness
        interpolatedValues = []
        for m in coordinatesRow.index.get_level_values("Moneyness").unique():
            coordinatesForM = coordinatesRow[coordinatesRow.index.get_level_values("Moneyness") == m]
            parametersForM = nelsonRow[nelsonRow.index.get_level_values("Moneyness") == m]
            
            interpolatorNelsonSiegel = NelsonSiegelCalibrator(3, {})
            interpolatorNelsonSiegel.beta = parametersForM.head(3).values
            interpolatorNelsonSiegel.alpha = parametersForM.tail(1).values
            interpolatedValues.append(interpolatorNelsonSiegel.interpolate(coordinatesForM.values))
        return pd.Series(np.ravel(interpolatedValues), coordinatesRow.index)
        #Format with same format as ttms
    
    
    
    nelsonList = list(map(lambda x : x[1], nelsonDf.iterrows()))
    coordinatesList = list(map(lambda x : x[1], coordinatesDf.iterrows()))
    interpolatedVol = list(map(lambda x : getVolFromNelsonParametersApply(x[0],x[1]), 
                               zip(nelsonList, coordinatesList)))
    
    
    return pd.DataFrame(np.reshape(interpolatedVol, coordinatesDf.shape), 
                        index = coordinatesDf.index, 
                        columns = coordinatesDf.columns )

#Calibrate nelson siegel interpolation for each day and each moneyness
def calibrateModelMoneynessWiseDaily(dataSet):
    dfList = dataSet.getDataForModel()#(dataSet.trainVol.head(20).index)
    moneynesses = np.unique(getMoneynessFromCoordinates(dfList))
    moneynessDf = getMoneynessFromCoordinates(dfList)
    ttmDf = getTTMFromCoordinates(dataSet.formatModelDataAsDataSet(dfList))
    volDf = dataSet.formatModelDataAsDataSet(dfList)[0]
    rankForMList = []
    TTMForMList = []
    AvgTTMForMList = []
    volForMList = []
    interpolationCurvesList = []
    interpolatedCurveList = []
    refCurveList = []
    nelsonList = []
    
    def treatCurve(curveVol, curveTTM):
        ttmToInterpolate = curveTTM.dropna()
        volToInteporlate = curveVol.dropna()
        
        interpolatorNelsonSiegel = NelsonSiegelCalibrator(3, {})
        interpolatorNelsonSiegel.calibrate(volToInteporlate.values, ttmToInterpolate[volToInteporlate.index].values)
        interpolationCurve = interpolatorNelsonSiegel.interpolate(ttmToInterpolate.values)
        
        calibratedCurve = pd.Series(volToInteporlate.values, 
                                    index = volToInteporlate.index).rename(curveVol.name)
        nonCalibratedTTM = curveVol.index.difference(calibratedCurve.index)
        calibratedCurve = calibratedCurve.append(pd.Series([np.NaN]*nonCalibratedTTM.size, 
                                                           index = nonCalibratedTTM)).sort_index()
        
        interpolatedCurve = pd.Series(interpolationCurve, index = ttmToInterpolate.index).rename(curveVol.name)
        nonInterpolatedTTM = curveVol.index.difference(interpolatedCurve.index)
        interpolatedCurve = interpolatedCurve.append(pd.Series([np.NaN]*nonInterpolatedTTM.size, 
                                                               index = nonInterpolatedTTM)).sort_index()
        
        
        return (calibratedCurve, interpolatedCurve, np.append(interpolatorNelsonSiegel.beta , interpolatorNelsonSiegel.alpha))
    
    for m in moneynesses:#For a fixed moneyness
        #Gather values for corresponding moneyness
        rankForM = moneynessDf[moneynessDf == m].dropna(how="all", axis=1).columns
        rankForMList.append(rankForM)
        
        TTMForM =  ttmDf[rankForM] #.dropna(how="any", axis=0)
        TTMForMList.append(TTMForM)
        AvgTTMForMList.append(TTMForM.mean(axis=0).round())
        
        volForMList.append(volDf[rankForM]) #.dropna(how="any", axis=0))
        
        #Turn dataframe as a list of series for applying operation jointly on two dataframe 
        volSeriesListForM = list(map(lambda x : x[1], volForMList[-1].iterrows()))
        coordinatesSeriesListForM = list(map(lambda x : x[1], TTMForMList[-1].iterrows()))
        #Estimate Nelson siegel paramters for every day 
        interpolationCurvesList.append(list(map(lambda x : treatCurve( x[0], x[1]) , 
                                                zip(volSeriesListForM, coordinatesSeriesListForM))))
        
        #Data used for nelson siegle calibration, should be equal to volForMList
        refCurveList.append(pd.DataFrame(list(map(lambda x : x[0], interpolationCurvesList[-1])), 
                                         index = volForMList[-1].index, 
                                         columns = volForMList[-1].columns))
        #Interpolated volatility
        interpolatedCurveList.append(pd.DataFrame(list(map(lambda x : x[1], interpolationCurvesList[-1])), 
                                                  index = volForMList[-1].index, 
                                                  columns = volForMList[-1].columns))
        
        #Parameters estimated every day
        nelsonList.append(pd.DataFrame(list(map(lambda x : x[2], interpolationCurvesList[-1])), 
                                       index = volForMList[-1].index))
        print(m)
        
        
        
    return mergeResults(TTMForMList, AvgTTMForMList, volForMList, interpolatedCurveList, refCurveList, nelsonList, dfList)

#Calibrate a model that interpolates a whole surface (not a single smile) with single parameters
def calibrateModelDayWise(dfList):
    moneynesses = np.unique(getMoneynessFromCoordinates(dfList))
    moneynessDf = getMoneynessFromCoordinates(dfList)
    ttmDf = getTTMFromCoordinates(dataSet.formatModelDataAsDataSet(dfList))
    volDf = dataSet.formatModelDataAsDataSet(dfList)[0]
    rankForMList = []
    TTMForMList = []
    AvgTTMForMList = []
    volForMList = []
    interpolationCurvesList = []
    interpolatedCurveList = []
    refCurveList = []
    nelsonList = []
    
    def treatSurface(surfaceVol, surfaceTTM, surfaceMoneyness):
        ttmToInterpolate = surfaceTTM.dropna()
        moneynessToInterpolate = surfaceMoneyness.dropna()
        ttmToInterpolate = ttmToInterpolate[ttmToInterpolate.index.intersection(moneynessToInterpolate.index)]
        moneynessToInterpolate = moneynessToInterpolate[ttmToInterpolate.index]
        volToInterpolate = surfaceVol.dropna()
        
        interpolatorSpline = gaussianProcess.Spline(3, {})
        interpolatorSpline.calibrate(volToInterpolate.values, 
                                     ttmToInterpolate[volToInterpolate.index].values, 
                                     moneynessToInterpolate[volToInterpolate.index].values)
        interpolationCurve = interpolatorSpline.interpolate(ttmToInterpolate.values, moneynessToInterpolate.values)
        
        calibratedCurve = pd.Series(volToInterpolate.values, 
                                    index = volToInterpolate.index).rename(surfaceVol.name)
        nonCalibratedTTM = surfaceVol.index.difference(calibratedCurve.index)
        calibratedCurve = calibratedCurve.append(pd.Series([np.NaN]*nonCalibratedTTM.size, 
                                                           index = nonCalibratedTTM)).sort_index()
        
        interpolatedCurve = pd.Series(interpolationCurve, index = ttmToInterpolate.index).rename(surfaceVol.name)
        nonInterpolatedTTM = surfaceVol.index.difference(interpolatedCurve.index)
        interpolatedCurve = interpolatedCurve.append(pd.Series([np.NaN]*nonInterpolatedTTM.size, 
                                                               index = nonInterpolatedTTM)).sort_index()
        
        
        return (calibratedCurve, interpolatedCurve, interpolatorSpline.beta)
    
    volSeriesList = list(map(lambda x : x[1], volDf.iterrows()))
    moneynessSeriesList = list(map(lambda x : x[1], moneynessDf.iterrows()))
    ttmSeriesList = list(map(lambda x : x[1], ttmDf.iterrows()))
    dailyData = list(map(lambda x : treatSurface( x[0], x[1], x[2]) , 
                         zip(volSeriesList, ttmSeriesList, moneynessSeriesList)))
    interpolatedDf = pd.DataFrame(pd.concat(list(map(lambda x : x[1], dailyData))),
                                  index = volDf.index, 
                                  columns = volDf.columns)
    refDf = pd.DataFrame(pd.concat(list(map(lambda x : x[0], dailyData))),
                         index = volDf.index, 
                         columns = volDf.columns)
    paramDf = pd.DataFrame(pd.concat(list(map(lambda x : x[2], dailyData))),
                           index = volDf.index)
    #paramIndex = pd.MultiIndex.from_product([moneynesses, paramDf.columns], 
    #                                        names=["Moneyness", "Spline Parameters"])
    volIndex = pd.MultiIndex.from_product([moneynesses, np.arange(1, int(interpolatedDf.shape[1] / moneynesses.size) + 1, 1)], 
                                          names=["Moneyness", "Rank"])
                                          
    reindexedVolDf = pd.DataFrame(volDf.values, 
                                  index = volDf.index, 
                                  columns = volIndex)
    reindexedCoordinatesDf = pd.DataFrame(coordinatesDf.values, 
                                          index = coordinatesDf.index, 
                                          columns = volIndex)
    return interpolatedDf, refDf, paramDf, reindexedCoordinatesDf, reindexedVolDf

def calibrateDataSetWithNelsonSiegel(pathTestFile, dataSet, restoreResults = True):
    if restoreResults :
        nelsonDf, interpVolDf = loadData.readInterpolationResult(pathTestFile)
    else : 
        interpVolDf, refVolDf, nelsonDf, coordinatesDf, volDf = calibrateModelMoneynessWiseDaily(dataSet)
        loadData.saveInterpolationResult(pathTestFile, nelsonDf, interpVolDf)

    moneynesses = np.unique(getMoneynessFromCoordinates(dataSet.getDataForModel()))
    volDf = dataSet.formatModelDataAsDataSet(dataSet.getDataForModel())[0]
    volIndex = pd.MultiIndex.from_product([moneynesses, np.arange(1, int(volDf.shape[1] / moneynesses.size) + 1, 1)], 
                                          names=["Moneyness", "Rank"])
    volDf = pd.DataFrame(volDf.values, index = volDf.index, columns = volIndex)
    coordinatesDf = getTTMFromCoordinates(dataSet.formatModelDataAsDataSet(dataSet.getDataForModel()))
    coordinatesDf = pd.DataFrame(coordinatesDf.values, index = coordinatesDf.index, columns = volIndex)

    ######################## Plot parameters
    plt.plot(nelsonDf.iloc[:,0], label = "Beta1")
    plt.show()
    plt.plot(nelsonDf.iloc[:,1], label = "Beta2")
    plt.show()
    plt.plot(nelsonDf.iloc[:,2], label = "Beta3")
    plt.show()
    plt.plot(nelsonDf.iloc[:,3], label = "alpha1")
    plt.show()

    print(nelsonDf.head())

    ######################## Plot error
    maeInterp = np.abs(np.nanmean(np.abs(interpVolDf.values - volDf.values)/volDf.values, axis=1))
    plt.plot(interpVolDf.index, maeInterp)
    plt.show()
    rmseInterp = np.sqrt(np.nanmean(np.square(interpVolDf.values - volDf.values), axis=1))
    plt.plot(interpVolDf.index, rmseInterp)
    plt.show()

    ############################## Analyse worst estimation

    moneynessPlot = 1.0
    rowVol = volDf.transpose()[volDf.columns.get_level_values("Moneyness") == moneynessPlot].transpose()
    rowInterpVol = interpVolDf.transpose()[volDf.columns.get_level_values("Moneyness") == moneynessPlot].transpose()
    rowTTM = coordinatesDf[rowVol.columns]

    rowImpliedTotalVariance = np.square(rowVol * rowTTM / 250)
    rowInterpImpliedTotalVariance = np.square(pd.DataFrame(rowInterpVol.values, 
                                                           index = rowVol.index,
                                                           columns = rowVol.columns) * rowTTM / 250)
    dayPlot = np.argmax(rmseInterp)
    plt.plot(rowTTM.dropna(how="all",axis=1).iloc[dayPlot].values,
             rowInterpVol.dropna(how="all",axis=1).iloc[dayPlot].values, 
             "-", 
             label = "Nelson-Siegel")
    plt.plot(rowTTM.dropna(how="all",axis=1).iloc[dayPlot].values,
             rowVol.dropna(how="all",axis=1).iloc[dayPlot].values, 
             "+", 
             label = "Ref")
    plt.legend()
    plt.show()
    plt.plot(rowTTM.dropna(how="all",axis=1).iloc[dayPlot].values,
             rowInterpImpliedTotalVariance.dropna(how="all",axis=1).iloc[dayPlot].values, 
             "-", 
             label = "Nelson-Siegel")
    plt.title("Implied vol")
    plt.plot(rowTTM.dropna(how="all",axis=1).iloc[dayPlot].values,
             rowImpliedTotalVariance.dropna(how="all",axis=1).iloc[dayPlot].values, 
             "+", 
             label = "Ref")
    plt.title("Implied total variance")
    plt.legend()
    plt.show()
    plt.plot(rowTTM.dropna(how="all",axis=1).iloc[-2].values,
             (rowVol.dropna(how="all",axis=1).iloc[-2].values - rowInterpVol.dropna(how="all",axis=1).iloc[-1].values)/rowVol.dropna(how="all",axis=1).iloc[-1].values, 
             "+", 
             label = "Ref")
    plt.title("Implied vol relative mae")
    plt.show()

    #absolute error
    #interp2Df = getVolFromNelsonParameters(nelsonDf, coordinatesDf)
    #interp2Df.head()
    #plt.plot(interpVolDf.index, np.sqrt(np.nanmean(np.square(interpVolDf.values - volDf.values), axis=1)))
    #relative error
    #plt.plot(interpVolDf.index, np.abs(np.nanmean(np.abs(interpVolDf.values - interp2Df.values)/interp2Df.values, axis=1)))
    #plt.show()
    return 


class NelsonSiegel(LinearInterpolation):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestNelsonSiegelInterpolationModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    def interpolate(self, incompleteSurface, coordinates):
        filteredSurface, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(incompleteSurface, 
                                                                                           coordinates)
        nanSurface = incompleteSurface.drop(filteredSurface.index)
        
        knownValues = filteredSurface.dropna() #No value is interpolated   
        if knownValues.size == filteredSurface.size : #No value to interpolate
            return incompleteSurface
        knownLocation = filteredCoordinates.loc[knownValues.index]
        locationToInterpolate = filteredCoordinates.drop(knownValues.index)
        
        interpolatedValues = self.customInterpolator(knownValues, 
                                                     knownLocation,
                                                     locationToInterpolate)
        
        completeSurface = pd.Series(interpolatedValues, 
                                    index = locationToInterpolate.index).append(knownValues)
        
        
        interpolatedSurface = completeSurface.loc[filteredSurface.index].rename(filteredSurface.name)
        return interpolatedSurface.append(nanSurface)[incompleteSurface.index].rename(incompleteSurface.name)
        
    def customInterpolator(self, interpolatedData, formerCoordinates, NewCoordinates):
        knownPositions = formatCoordinatesAsArray(formerCoordinates)
        
        xNew = np.ravel(list(map(lambda x : x[0], NewCoordinates))) #Maturity
        yNew = np.ravel(list(map(lambda x : x[1], NewCoordinates))) #Moneyness
        
        
        #Group coordinates by moneyness
        curveNewDict = {}
        for idx in NewCoordinates.index : 
            m = NewCoordinates[idx][1]
            ttm = NewCoordinates[idx][0]
            if m not in curveNewDict :
                curveNewDict[m] = [[ttm], [idx]]
            else : 
                curveNewDict[m][0].append(ttm)
                curveNewDict[m][1].append(idx)
                
        #Group coordinates by moneyness
        curveOldDict = {}
        
        for idx in formerCoordinates.index : 
            m = formerCoordinates[idx][1]
            ttm = formerCoordinates[idx][0]
            v = interpolatedData[idx]
            if m not in curveOldDict :
                curveOldDict[m] = [[ttm], [v]]
            else : 
                curveOldDict[m][0].append(ttm)
                curveOldDict[m][1].append(v)
        fInterpolation = pd.Series()
        #Iterate on moneyness and interpolate the associated curve
        for m in curveNewDict : 
            if m in curveOldDict : 
                interpolatorNelsonSiegel = NelsonSiegelCalibrator(3, {})
                interpolatorNelsonSiegel.calibrate(np.array(curveOldDict[m][1]), np.array(curveOldDict[m][0]))
                interpolationCurve = interpolatorNelsonSiegel.interpolate(np.array(curveNewDict[m][0]))
            else : #Return nan
                interpolationCurve = np.full_like(curveNewDict[m][0], np.nan, dtype=np.float32)
            fInterpolation = fInterpolation.append(pd.Series(interpolationCurve, index = curveNewDict[m][1]))
        
        fInterpolation = fInterpolation[NewCoordinates.index] #Get the same order as NewCoordinates
        
        fExtrapolation =  griddata(knownPositions,
                                   np.ravel(interpolatedData.values),
                                   np.array((xNew, yNew)).T,
                                   method = 'nearest',
                                   rescale=True)
        
        return np.where(np.isnan(fInterpolation), fExtrapolation, fInterpolation)
    #nelsonDf, interpVolDf = loadData.readInterpolationResult(pathTestFile)

class SSVIModel(NelsonSiegel):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestNelsonSiegelInterpolationModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    def interpolate(self, incompleteSurface, coordinates, forwards):
        #print("incompleteSurface",incompleteSurface)
        #print("coordinates",coordinates)
        #print("forwards",forwards)
        filteredSurface, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(incompleteSurface, 
                                                                                           coordinates)
        nanSurface = incompleteSurface.drop(filteredSurface.index)
        
        knownValues = filteredSurface.dropna() #No value is interpolated   
        if knownValues.size == filteredSurface.size : #No value to interpolate
            return incompleteSurface
        knownLocation = filteredCoordinates.loc[knownValues.index]
        locationToInterpolate = filteredCoordinates.drop(knownValues.index)
        
        interpolatedValues = self.customInterpolator(knownValues, 
                                                     knownLocation,
                                                     forwards.loc[knownValues.index],
                                                     locationToInterpolate)
        
        completeSurface = pd.Series(interpolatedValues, 
                                    index = locationToInterpolate.index).append(knownValues)
        
        
        interpolatedSurface = completeSurface.loc[filteredSurface.index].rename(filteredSurface.name)
        return interpolatedSurface.append(nanSurface)[incompleteSurface.index].rename(incompleteSurface.name)
    
    
    def evalModel(self, inputTestList):
        #No loss since we interpolate exactly
        inputTest = inputTestList[0]
        coordinates = inputTestList[1]
        forwards = inputTestList[2]
        loss = pd.Series(np.zeros(inputTest.shape[0]), index = inputTest.index)
        
        #Return the inputs as compressed values
        inputs = inputTest.apply(lambda x : self.interpolate(x, coordinates.loc[x.name], forwards.loc[x.name]), axis=1)
        
        #We do not have any factors so we assign a dummy value of 1
        factors = pd.DataFrame(np.ones((inputTest.shape[0],self.nbFactors)), 
                               index=inputTest.index)
        return loss, inputs, factors
    
    
    def completeDataTensor(self, 
                           sparseSurfaceList, 
                           initialValueForFactors, 
                           nbCalibrationStep):
        # knownValues = sparseSurface.dropna()
        # locationToInterpolate = sparseSurface[sparseSurface.isna()].index
        sparseSurface = sparseSurfaceList[0]
        coordinates = sparseSurfaceList[1]
        forward = sparseSurfaceList[2]
        interpolatedValues = self.interpolate(sparseSurface, 
                                              coordinates, 
                                              forward)
        
        #Not a factorial model, we assign a dummy value
        bestFactors = np.ones(self.nbFactors)
        #Exact inteprolation
        calibrationLoss = 0.0
        calibrationSerie = pd.Series([calibrationLoss])
        #Complete surface with inteporlated values
        bestSurface = interpolatedValues
        return calibrationLoss, bestFactors, bestSurface, calibrationSerie
    
    def customInterpolator(self, interpolatedData, formerCoordinates, forwards, NewCoordinates):
        knownPositions = formatCoordinatesAsArray(formerCoordinates)
        
        xNew = np.ravel(list(map(lambda x : x[0], NewCoordinates))) #Maturity
        yNew = np.ravel(list(map(lambda x : x[1], NewCoordinates))) #Moneyness
        
        dummyBootstrap = bootstrapping.bootstrappingDummy(None, None, None)
        
        impVol = interpolatedData
        maturity = knownPositions[:,0]
        logMoneyness = np.log(knownPositions[:,1])
        optionType = np.ones_like(logMoneyness)
        
        #Estimate the underlying from forwards
        x, idx = np.unique(maturity, return_index=True)
        y = forwards.iloc[idx]
        S0 = scipy.interpolate.interp1d(x,y, fill_value="extrapolate")(0)
        Strike = S0 * knownPositions[:,1]
        
        reformulatedDf = pd.DataFrame(np.vstack([impVol, maturity, logMoneyness, optionType, Strike]).T, 
                                      columns = ["ImpliedVol", "Maturity", "logMoneyness", "OptionType", "Strike"])
        calibrator = SSVI.SSVIModel(S0, dummyBootstrap)
        calibrator.fit(reformulatedDf)
        
        
        maturity = xNew
        logMoneyness = np.log(yNew)
        Strike = S0 * yNew
        optionType = np.ones_like(logMoneyness)
        newDf = pd.DataFrame(np.vstack([maturity, logMoneyness, optionType, Strike]).T, 
                             columns = ["Maturity", "logMoneyness", "OptionType", "Strike"])
        fInterpolation = calibrator.eval(newDf).values
        
        fExtrapolation =  griddata(knownPositions,
                                   np.ravel(interpolatedData.values),
                                   np.array((xNew, yNew)).T,
                                   method = 'nearest',
                                   rescale=True)
        
        
        return np.where(np.isnan(fInterpolation), fExtrapolation, fInterpolation)
