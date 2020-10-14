import numpy as np
import pandas as pd
import tensorflow as tf
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression 

import linearModel




#Regress linearly missing points against non-missing points 
#Adapted for missing not at random data
class OLS(linearModel.LinearProjection):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestLinearRegressionModel"):
        self.components = None
        self.mask = hyperParameters["mask"]
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         int((~self.mask).sum()),
                         modelName)
    #Build a tensor that construct 
    def buildReconstructionTensor(self, factorTensor):
        l = tf.matmul(factorTensor, self.components, adjoint_b=True)
        return l * self.std + self.mean
    
    #Build a tensor that construct a surface from factors values
    def buildAutoEncoderTensor(self, surfaceTensor):
        lastTensor = (surfaceTensor - self.mean) / self.std
        lastTensor = tf.boolean_mask(lastTensor, ~self.mask.values)
        lastTensor = tf.matmul(lastTensor, self.components, adjoint_b=True)
        
        return lastTensor * self.std + self.mean
    
    def setDynamicPenalizationFactory(self, factory):
        raise NotImplementedError("Not allowed for a regression model !")
        return
        
    def buildArchitecture(self):
        self.inputTensor = tf.placeholder(tf.float32,
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#batch size along
        self.components = tf.Variable(np.ones(shape=(self.nbUnitsPerLayer['Input Layer'],
                                                     self.nbFactors)).astype(np.float32))
        self.mean = tf.Variable(np.zeros(shape=(self.nbUnitsPerLayer['Input Layer'])).astype(np.float32))
        self.std = tf.Variable(np.ones(shape=(self.nbUnitsPerLayer['Input Layer'])).astype(np.float32))
        
        scaledInputTensor = (self.inputTensor - self.mean) / self.std
        
        self.factorTensor = tf.boolean_mask(scaledInputTensor, ~self.mask.values, axis=1)
        
        scaledOutputTensor = tf.matmul(self.factorTensor, self.components, adjoint_b=True)
        
        self.outputTensor = scaledOutputTensor * self.std + self.mean
        
        return
    #Build the architecture, losses and optimizer.
    def buildModel(self):
        tf.reset_default_graph()
        if self.verbose :
            print("build architecture, loss and penalisations")
        self.nbEncoderLayer = 0
        self.layers = [] 
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
        
        muTrain = trainingSet[0].mean(axis=0)
        stdTrain = trainingSet[0].std(axis=0)
        
        OLSModel = LinearRegression(fit_intercept=False, normalize=False, copy_X=True)
        #print((trainingSet[0].T[~self.mask.values].T - muTrain[~self.mask.values]).dropna())
        OLSModel.fit(((trainingSet[0].T[~self.mask.values].T - muTrain[~self.mask.values]) / stdTrain[~self.mask.values]).dropna(), 
                     ((trainingSet[0] - muTrain) / stdTrain).dropna())
        components = OLSModel.coef_
        
        feedDict = self.createFeedDictEncoder(trainingSet, components, muTrain, stdTrain)
        session.run([self.trainingOperator, self.trainingMean, self.trainingStd], 
                    feed_dict=feedDict)
        epochLosses.append(trainingLoss.eval(feed_dict=self.createFeedDictEncoder(testingSet)))
        
        validationLosses.append(
            validationLoss.eval(feed_dict=self.createFeedDictEncoder(validationSet)))
        
        if self.verbose :
            print("Epoch : ", 0, " , Validation Loss : ", validationLosses)      
        save_path = self.saveModel(session, self.metaModelName)
        
        if self.verbose :
            print(validationLoss.eval(feed_dict=self.createFeedDictEncoder(validationSet)))
        return np.array(epochLosses), np.array(validationLosses)
    
    def getDecoderCoefficients(self):
        raise NotImplementedError("Not allowed for a regression model !")
        return None
    
    def completeDataTensor(self, 
                           sparseSurfaceList, 
                           initialValueForFactors, 
                           nbCalibrationStep, 
                           *args):
        #Rebuild tensor graph
        self.restoringGraph()
        
        #Build tensor for reconstruction
        sparseSurface = sparseSurfaceList[0]
        reshapedSparseSurface = np.reshape([sparseSurface],
                                           (1,sparseSurface.shape[0]))
        dataSetList = [pd.DataFrame(reshapedSparseSurface, 
                                    index = [sparseSurface.name], 
                                    columns = sparseSurface.index)] + sparseSurfaceList[1:]
        # for k in args :
            # dataSetList.append(np.reshape([k], (1,k.shape[0])))
        loss, inputs, factors = self.evalModel(dataSetList)
        calibrationLosses = [loss]
        bestCalibration = np.argmin(calibrationLosses)
        bestFactors = [np.ravel(factors.values.T.astype(np.float32))]
        bestSurface = pd.Series(np.reshape(inputs.values, sparseSurface.shape), 
                                index=sparseSurface.index, 
                                name=sparseSurface.name)
        
        return calibrationLosses[bestCalibration] , bestFactors[0], bestSurface, pd.Series(calibrationLosses) 
    
    def commonEvalInterdependancy(self, fullDataSet):
        raise NotImplementedError("Not allowed for a regression model !")

#Same but with a neural network
class denseNetwork(OLS):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestDenseNetworkModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    #Build a tensor that construct a surface from factors values
    def buildReconstructionTensor(self, factorTensor):
        nbDeconvLayer = len(self.layers)
        lastTensor = factorTensor
        for k in range(nbDeconvLayer):
            factoryTmp = self.layers[k].copy()
            lastTensor = factoryTmp(lastTensor)
        return lastTensor
    
    #Build a tensor that construct a surface from factors values
    def buildAutoEncoderTensor(self, surfaceTensor):
        lastTensor = surfaceTensor
        lastTensor = tf.boolean_mask(lastTensor, ~self.mask.values)
        nbDeconvLayer = len(self.layers)
        for k in range(nbDeconvLayer):
            factoryTmp = self.layers[k]
            lastTensor = factoryTmp(lastTensor)
        return lastTensor
    
    def setDynamicPenalizationFactory(self, factory):
        raise NotImplementedError("Not allowed for a regression model !")
        return
        
    def buildArchitecture(self):
        #Regularizer
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
                                          
        self.inputTensor = tf.placeholder(tf.float32,
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#batch size along
        self.components = tf.Variable(np.ones(shape=(self.nbUnitsPerLayer['Input Layer'],
                                                     self.nbFactors)).astype(np.float32))
        self.mean = tf.Variable(np.zeros(shape=(self.nbUnitsPerLayer['Input Layer'])).astype(np.float32))
        self.std = tf.Variable(np.ones(shape=(self.nbUnitsPerLayer['Input Layer'])).astype(np.float32))
        
        scaledInputTensor = self.inputTensor
        
        self.factorTensor = tf.boolean_mask(scaledInputTensor, ~self.mask.values, axis=1)
        
        hiddenEncoder1 = self.buildDenseLayer(int(self.nbUnitsPerLayer['Output Layer']/2), 
                                              tf.reshape(self.factorTensor, [-1, self.nbFactors]),
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
                                              
        hiddenEncoder2 = self.buildDenseLayer(self.nbUnitsPerLayer['Output Layer'], 
                                              hiddenEncoder1,
                                              activation = None,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        
        scaledOutputTensor = hiddenEncoder2
        
        self.outputTensor =  scaledOutputTensor 
        
        return
    #Build the architecture, losses and optimizer.
    def buildModel(self):
        tf.reset_default_graph()
        if self.verbose :
            print("build architecture, loss and penalisations")
        self.nbEncoderLayer = 0
        self.layers = [] 
        self.buildArchitecture()
        self.reconstructionLoss = self.buildReconstructionLoss(self.outputTensor, 
                                                               self.inputTensor,
                                                               "reconstructionLoss") 
        self.reducedReconstructionLoss = self.normalizeLoss(self.reconstructionLoss, 
                                                            "reducedReconstructionLoss")
        self.penalizationList = self.buildPenalization()
        self.loss = tf.add_n([self.reducedReconstructionLoss] + self.penalizationList, 
                             name="loss")
        self.learningRateVariable = tf.Variable(self.learningRate, 
                                                dtype = tf.float32 , 
                                                name = "OptimizerLearningRate", 
                                                trainable = False)
        self.learningRatePlaceholder = tf.placeholder(tf.float32, shape=(), 
                                                      name="LearningRatePlaceholder")
        self.learningRateAssign = self.learningRateVariable.assign(self.learningRatePlaceholder)
        self.optimizer = tf.train.AdamOptimizer(self.learningRateVariable, 
                                                name="optimizer")
        
        self.trainingOperator = self.optimizer.minimize(self.loss, 
                                                        name="trainingOperator")
        # self.componentsHolder = tf.placeholder(self.components.dtype,
                                               # shape=self.components.get_shape())
        # self.meansHolder = tf.placeholder(self.mean.dtype,
                                          # shape=self.mean.get_shape())
        # self.stdsHolder = tf.placeholder(self.std.dtype,
                                         # shape=self.std.get_shape())
        
        # self.trainingMean = self.mean.assign(self.meansHolder)
        # self.trainingStd = self.std.assign(self.stdsHolder) 
        # self.trainingOperator = self.components.assign(self.componentsHolder)
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(name="saver", save_relative_paths=True)
        return
    
    
    def createFeedDictEncoder(self, *args):
        feedDict = None
        if len(args)==1 : 
            feedDict = {self.inputTensor : args[0][0]}
        else :
            feedDict = {self.inputTensor : args[0][0]}
            
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
        activateEarlyStopping = (('validationPercentage' in self.hyperParameters) &
                                 (self.hyperParameters['validationPercentage'] > 0.001))
        validationSet, trainingSet = self.splitValidationAndTrainingSet(datasetTrain)
        if activateEarlyStopping :
            #Use early stopping
            patience = self.hyperParameters['earlyStoppingWindow']
            epsilon = 0.00001
            bestEpoch = 0
        else :
            trainingSet = datasetTrain
            
        testingSet = dataSetTest if (dataSetTest is not None) else trainingSet
        epochLosses = []
        validationLosses = []
        save_path = None
        lastLearningRateRefinement = 0
        
        for epoch in range(nbEpoch):
            miniBatches = self.generateMiniBatches(trainingSet, nbEpoch)
            for batch in miniBatches:
                session.run(gradientStep, feed_dict=self.createFeedDictEncoder(batch))
            
            epochLosses.append(trainingLoss.eval(feed_dict=self.createFeedDictEncoder(testingSet)))
            if self.verbose :
                print("Epoch : ", epoch, 
                      " , Penalized Loss on testing dataset : ", epochLosses[epoch]) 
            
            validationFeedDict = self.createFeedDictEncoder(validationSet)
            validationLosses.append( validationLoss.eval(feed_dict=validationFeedDict) )
            if self.verbose :
                print("Epoch : ", epoch, " , Validation Loss : ", validationLosses[epoch])
            
            #Monitor Model Performance
            if activateEarlyStopping :
                #Decide which model to keep
                if ((epoch == 0) or (validationLosses[epoch] <= np.nanmin(validationLosses))):
                    #Reset learning rate for completion
                    formerLearningRate = self.learningRateVariable.eval()
                    self.resetLearningRate(session)
                    
                    #Save Model if it improves validation error
                    save_path = self.saveModel(session, self.metaModelName)
                    bestEpoch = epoch
                    
                    #Restore former learning rate
                    session.run(self.learningRateAssign, 
                                feed_dict = {self.learningRatePlaceholder : formerLearningRate})
                    
                #Early stopping is triggered if performance is not improved during a certain window
                if (((epoch - max(bestEpoch, lastLearningRateRefinement)) >= patience) or (epoch >= (nbEpoch - 1))) : 
                    if ((self.learningRateVariable.eval() > 1e-6) and (epoch < (nbEpoch - 1))):
                        #Record current learning rate
                        formerLearningRate = self.learningRateVariable.eval()
                        
                        self.restoreWeights(session)
                        
                        #Restore former learning rate
                        session.run(self.learningRateAssign, 
                                    feed_dict = {self.learningRatePlaceholder : formerLearningRate})
                        
                        self.refineLearningRate(session)
                        lastLearningRateRefinement = epoch
                        
                    else :
                        #Trigger early stopping and restore best performing model 
                        minPatienceWindow = np.nanmin(validationLosses[-patience:]) 
                        if self.verbose :
                            print("Minimum validation loss for the latest ", patience ," observations : ", minPatienceWindow)
                            print("Minimum validation loss : ", np.nanmin(validationLosses))
                        self.restoreWeights(session)
                        if self.verbose :
                            print("Validation loss from restored model : ", validationLoss.eval(feed_dict=validationFeedDict))
                        break
        return np.array(epochLosses), np.array(validationLosses)