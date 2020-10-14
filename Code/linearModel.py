
import pandas as pd
import numpy as np
import tensorflow as tf
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression 

import tensorflowModel


class LinearProjection(tensorflowModel.tensorflowModel):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestLinearModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
        
    def buildArchitecture(self):
        
        self.inputTensor = tf.placeholder(tf.float32,
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#batch size along
        if self.verbose :
            print(self.inputTensor)
                                            
        self.factorTensor = self.buildDenseLayer(self.nbFactors, 
                                                 self.inputTensor,
                                                 activation = None)
        if self.verbose :
            print(self.factorTensor)
        self.nbEncoderLayer = len(self.layers)
        # DECODE --------------------------------------------------------------------
        
        lastTensor = self.factorTensor
        for k in range(self.nbEncoderLayer):
            lastTensor = self.buildInverseLayer(lastTensor)
        
        
        self.outputTensor = lastTensor
        
        if self.verbose :
            print(self.outputTensor)
        return
        
    
    def getDecoderCoefficients(self):
        #Rebuild tensor graph
        self.restoringGraph()
        
        #Evaluating surface for these factor Values
        projectionMatrix = None
        with tf.Session() as sess :
            #Restoring Model
            sess.run(self.init)
            self.restoreWeights(sess)
            
            #We know that latest layer is fylly connected one without activation function
            projectionMatrix = self.layers[-1].weights.eval()
        
        return projectionMatrix
    
    

class PCAScaled(LinearProjection):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestPCAScaledModel"):
        self.components = None
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    #Build a tensor that construct 
    def buildReconstructionTensor(self, factorTensor):
        l = tf.matmul(factorTensor, self.components, adjoint_b=True)
        return l * self.std + self.mean
    
    def buildArchitecture(self):
        self.inputTensor = tf.placeholder(tf.float32,
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#batch size along
        self.components = tf.Variable(np.ones(shape=(self.nbUnitsPerLayer['Input Layer'],
                                                     self.nbFactors)).astype(np.float32))
        self.mean = tf.Variable(np.zeros(shape=(self.nbUnitsPerLayer['Input Layer'])).astype(np.float32))
        self.std = tf.Variable(np.ones(shape=(self.nbUnitsPerLayer['Input Layer'])).astype(np.float32))
        
        scaledInputTensor = (self.inputTensor - self.mean) / self.std
        
        self.factorTensor = tf.matmul(scaledInputTensor, self.components)
        
        scaledOutputTensor = tf.matmul(self.factorTensor, self.components, adjoint_b=True)
        
        self.outputTensor = scaledOutputTensor * self.std + self.mean
        
        return
    #Build the architecture, losses and optimizer.
    def buildModel(self):
        tf.reset_default_graph()
        if self.verbose :
            print("build architecture, loss and penalisations")
        self.buildArchitecture()
        self.reconstructionLoss = self.buildReconstructionLoss(self.outputTensor, 
                                                               self.inputTensor,
                                                               "reconstructionLoss") 
        self.reducedReconstructionLoss = self.normalizeLoss(self.reconstructionLoss, 
                                                            "reducedReconstructionLoss")
        self.penalizationList = self.buildPenalization()
        self.loss = tf.add_n([self.reducedReconstructionLoss] + self.penalizationList, 
                             name="loss")
        self.optimizer = tf.train.AdamOptimizer(self.learningRate, name="optimizer")
        
        self.componentsHolder = tf.placeholder(self.components.dtype,
                                               shape=self.components.get_shape())
        self.meansHolder = tf.placeholder(self.mean.dtype,
                                          shape=self.mean.get_shape())
        self.stdsHolder = tf.placeholder(self.std.dtype,
                                         shape=self.std.get_shape())
        
        self.trainingMean = self.mean.assign(self.meansHolder)
        self.trainingStd = self.std.assign(self.stdsHolder) 
        self.trainingOperator = self.components.assign(self.componentsHolder)
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(name="saver", save_relative_paths=True)
        return
    
    def createFeedDictEncoder(self, *args):
        feedDict = None
        if len(args)==1 : 
            feedDict = {self.inputTensor : args[0][0]}
        else :
            feedDict = {self.inputTensor : args[0][0],
                        self.componentsHolder : args[1],
                        self.meansHolder : args[2], 
                        self.stdsHolder : args[3]}
            
        return feedDict
        
    #######################################################################################################
    #Training functions
    #######################################################################################################
    def gradientDescent(self, 
                        session, 
                        datasetTrain, 
                        nbEpoch, 
                        dataSetTest, 
                        trainingLoss, 
                        gradientStep, 
                        validationLoss):
        epochLosses = []
        useValidationDataSet = (('validationPercentage' in self.hyperParameters) &
                                (self.hyperParameters['validationPercentage'] > 0.001))
        validationSet, trainingSet = self.splitValidationAndTrainingSet(datasetTrain)
        validationLosses = []
        epsilon = 0.00001
        
        testingSet = dataSetTest if (dataSetTest is not None) else trainingSet
        
        pca = PCA(n_components=self.nbFactors)
        _ = pca.fit_transform(scale(trainingSet[0]))
        muTrain = trainingSet[0].mean(axis=0)
        stdTrain = trainingSet[0].std(axis=0)
        feedDict = self.createFeedDictEncoder(trainingSet, pca.components_.T, muTrain, stdTrain)
        session.run([self.trainingOperator, self.trainingMean, self.trainingStd], 
                    feed_dict=feedDict)
        epochLosses.append(trainingLoss.eval(feed_dict=self.createFeedDictEncoder(testingSet)))
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        
        validationLosses.append(
            validationLoss.eval(feed_dict=self.createFeedDictEncoder(validationSet)))
        
        if self.verbose :
            print("Epoch : ", 0, " , Validation Loss : ", validationLosses)      
        save_path = self.saveModel(session, self.metaModelName)
        
        if self.verbose :
            print(validationLoss.eval(feed_dict=self.createFeedDictEncoder(validationSet)))
        return np.array(epochLosses), np.array(validationLosses)
    
    def getDecoderCoefficients(self):
        #raise NotImplementedError("Not allowed for PCA because of mean rescaling !")
        return None
    
        #Build a tensor that construct a surface from factors values
    def buildAutoEncoderTensor(self, surfaceTensor):
        lastTensor = (surfaceTensor - self.mean) / self.std
        lastTensor = tf.matmul(lastTensor, self.components)
        lastTensor = tf.matmul(lastTensor, self.components, adjoint_b=True)
        
        return lastTensor * self.std + self.mean
    
    # def buildCompletionLoss(self, factorTensor, calibrationLoss, completedSurfaceTensor):
        # previousPenalization = super().buildCompletionLoss(factorTensor, calibrationLoss, completedSurfaceTensor)
        
        # finalCalibrationLoss = previousPenalization
        # if "lambdaCompletionEncodings" in self.hyperParameters :
            #completedEncodings = self.buildEncoderTensor(completedSurfaceTensor)
            # reconstructedSurface = self.buildAutoEncoderTensor(completedSurfaceTensor)
            # calibrationLoss = self.buildLoss(reconstructedSurface, 
                                             # self.sparseSurfaceTensor, 
                                             # name="calibrationLossOutlier" )
            #outlierRegularization = tf.reduce_mean(self.buildReconstructionLoss(completedSurfaceTensor, 
            #                                                                    reconstructedSurface, 
            #                                                                    "EncodingRegularization"))
            
            #finalCalibrationLoss += self.hyperParameters["lambdaCompletionEncodings"] * outlierRegularization
            # finalCalibrationLoss = calibrationLoss
        # return finalCalibrationLoss
    
    def completeDataTensor(self, 
                           sparseSurfaceList, 
                           initialValueForFactors, 
                           nbCalibrationStep, 
                           *args):
        #Rebuild tensor graph
        self.restoringGraph()
        
        #Build tensor for reconstruction
        reshapedSparseSurface = np.reshape([sparseSurfaceList[0]],
                                           (1,sparseSurfaceList[0].shape[0]))
        reshapedValueForFactors = np.reshape([initialValueForFactors],
                                             (1,initialValueForFactors.shape[0]))
        dataSetList = [reshapedSparseSurface]
        for k in args :
            dataSetList.append(np.reshape([k], (1,k.shape[0])))
        
        #Opening session for calibration
        with tf.Session() as sess :
            #Restoring Model
            #self.saver = tf.train.import_meta_graph(self.metaModelName + '.meta')
            sess.run(self.init)
            self.restoreWeights(sess)
            
            reconstructionMatrix = self.components.eval() #(56, 5)
            mu = self.mean.eval() #(56,)
            sigma = self.std.eval() #(56,)
        
        nonMissingColumns = np.argwhere(~np.isnan(reshapedSparseSurface))[:,1] #(8,)
        nonMissingPoints = (reshapedSparseSurface - mu) / sigma #(1,56)
        #factorsCalibrated, residuals, _, _ = np.linalg.lstsq(reconstructionMatrix[nonMissingColumns,:], 
        #                                                     nonMissingPoints[:, nonMissingColumns].T) #((8,5), (8,)) -> (5,)
        factorsCalibrated = np.linalg.pinv(reconstructionMatrix[nonMissingColumns,:]) @ nonMissingPoints[:, nonMissingColumns].T
        
        reconstructedSurface = (reconstructionMatrix @ factorsCalibrated).T * sigma + mu #(56,1)
        
        completedSurface = np.where(np.isnan(reshapedSparseSurface), 
                                    reconstructedSurface,
                                    reshapedSparseSurface)
        
        calibrationLosses = [np.sqrt(np.nanmean(np.square(completedSurface.T - sparseSurfaceList[0].values)[:, nonMissingColumns]))]
        
        #Get results for best calibration
        bestCalibration = np.argmin(calibrationLosses)
        bestFactors = [np.ravel(factorsCalibrated.astype(np.float32))]
        bestSurface = pd.Series(np.reshape(completedSurface, sparseSurfaceList[0].shape), 
                                index=sparseSurfaceList[0].index, 
                                name=sparseSurfaceList[0].name)
        return calibrationLosses[bestCalibration] , bestFactors[0], bestSurface, pd.Series(calibrationLosses) 






