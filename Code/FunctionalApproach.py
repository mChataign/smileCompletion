
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import tensorflowModel
import plottingTools

class FunctionalModel(tensorflowModel.tensorflowModel):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestFunctionalModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    
    
    
    ##Loss computed on a per observation basis
    # def buildReconstructionLoss(self, pred, ref, name, matrixNorm = True):
        ##If reference value contains NaN we ignore it for completion purpose
        # diff = pred - tf.where(tf.is_nan(ref), tf.zeros_like(ref), ref)
        # if matrixNorm : 
            # diff_without_nans = tf.reshape(diff, 
                                           # [-1, self.hyperParameters['nbX'], self.hyperParameters['nbY']])
            # return tf.norm(diff_without_nans, ord = self.lossHolderExponent, axis = [-2,-1], name=name) 
        # diff_without_nans = tf.where(tf.is_nan(ref), tf.zeros_like(ref), diff)
        # return tf.norm(diff_without_nans, ord = self.lossHolderExponent, axis =1, name=name)
    
    def buildArchitecture(self):
        
        nbCoordinates = 2
        #Location to interpolate which are common to each day
        
        #Factors values for each day
        self.factorTensor = tf.placeholder(tf.float32, 
                                           shape=[None, self.nbFactors], 
                                           name = "factorTensor")
        #Get the number of days as dynamic shape
        numObs = tf.shape(self.factorTensor)[0]
        
        #Repeat the locations for each day
        self.inputTensor = tf.placeholder(tf.float32, shape=[None, nbCoordinates], name = "inputTensor")
        reshapedInputTensor = tf.reshape(self.inputTensor, 
                                         tf.stack([numObs, -1, nbCoordinates]), 
                                         name = "reshapedInputTensor") #[None, None, 2]
        # reshapedInputTensor = tf.tile(tf.expand_dims(self.inputTensor, 0),
                                      # tf.stack([numObs, 1, 1]))
        
        #Repeat the factors for each location
        surfaceSize = tf.shape(reshapedInputTensor, 
                               name = "surfaceSize")[1]
        reshapedFactorTensor = tf.tile(tf.expand_dims(self.factorTensor, 1),
                                       tf.stack([1, surfaceSize, 1]), 
                                       name = "reshapedFactorTensor")
        
        if self.verbose :
            print(self.inputTensor)
            print(reshapedInputTensor)
        
        #self.factorTensor = tf.Variable(np.ones(shape=(1, self.nbFactors)).astype(np.float32))
        if self.verbose :
            print(self.factorTensor)
            print(reshapedFactorTensor)
        
         #Dynamic Shape of features shoud be [None, surfaceSize, 2 + self.nbFactors] that is [nbDays, surfaceSize, 2 + self.nbFactors]
        features = tf.reshape(tf.concat([reshapedInputTensor, reshapedFactorTensor],2), 
                                        [-1, nbCoordinates + self.nbFactors], 
                                        name = "features")
         # filteredFeatures = tf.boolean_mask(features, 
                                            # tf.logical_not(tf.reduce_any(tf.is_nan(features), axis=1)), 
                                            # name = "filteredFeatures")
         # partitions = tf.cast(tf.logical_not(tf.reduce_any(tf.is_nan(features), axis=1)), tf.int32)
         # filteredFeatures = tf.dynamic_partition(features, 
                                                 # partitions,
                                                 # num_partitions = 2,
                                                 # name = "filteredFeaturesVar")[1]
        filteredFeatures = tf.where(tf.reduce_any(tf.is_nan(features), axis=1), 
                                    tf.zeros_like(features), 
                                    features)
                                    
        #Factors values for each day
        # self.factorTensor = tf.placeholder(tf.float32, 
                                           # shape=[None, self.nbFactors], 
                                           # name = "factorTensor")
        # self.inputTensor = tf.placeholder(tf.float32, shape=[None, nbCoordinates], name = "inputTensor")
        # filteredFeatures = tf.concat([self.factorTensor, self.inputTensor],1 , name = "features")
        # if self.verbose :
            # print(self.inputTensor)
            # print(self.factorTensor)
            # print(filteredFeatures)
        
        
        # if self.verbose :
            # print(features)
            # print(filteredFeatures)
        
        #Build neuronal architecture
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        
        
        
        hidden1 = self.buildDenseLayer(20, 
                                       filteredFeatures,
                                       activation = tf.nn.softplus,
                                       kernelRegularizer = l2_regularizer,
                                       kernelInitializer = he_init)
        if self.verbose :
            print(hidden1)
        hidden2 = self.buildDenseLayer(20, 
                                       hidden1,
                                       activation = tf.nn.softplus,
                                       kernelRegularizer = l2_regularizer,
                                       kernelInitializer = he_init)
        if self.verbose :
            print(hidden2)
        hidden3 = self.buildDenseLayer(1, 
                                       hidden2,
                                       activation = None,
                                       kernelRegularizer = l2_regularizer,
                                       kernelInitializer = he_init)
        if self.verbose :
            print(hidden3)
        
        #Reshape output as (nbDays, surfaceSize)
        self.outputTensor = hidden3#tf.reshape(hidden3, tf.stack([-1, surfaceSize]))
        if self.verbose :
            print(self.outputTensor)
        
        return
    
    #Loss computed on a per observation basis
    def buildReconstructionLoss(self, pred, ref, name, matrixNorm = True):
        #If reference value contains NaN we ignore it for completion purpose
        diff = pred - ref
        diff_without_nans = tf.where(tf.logical_or(tf.expand_dims(tf.reduce_any(tf.is_nan(self.inputTensor), axis=1), 1), tf.is_nan(ref)), 
                                     tf.zeros_like(diff), 
                                     diff)
        #diff_without_nans = tf.where(tf.is_nan(ref), tf.zeros_like(diff), diff)
        return tf.norm(diff_without_nans, ord = self.lossHolderExponent, name=name)#, axis =1
        #return tf.pow(diff_without_nans, self.lossHolderExponent, name=name)
        
    #Build a tensor that construct a surface from factors values
    def buildReconstructionTensor(self, factorTensor):
        nbCoordinates = 2
        
        #Get the number of days as dynamic shape
        numObs = tf.shape(factorTensor)[0]
        
        #Repeat the locations for each day
        reshapedInputTensor = tf.reshape(self.inputTensor, 
                                         tf.stack([numObs, -1, nbCoordinates]),
                                         name = "reshapedInputTensorVar") 
        # reshapedInputTensor = tf.tile(tf.expand_dims(self.inputTensor, 0),
                                      # tf.stack([numObs, 1, 1]))
        
        #Repeat the factors for each location
        surfaceSize = tf.shape(reshapedInputTensor)[1]
        reshapedFactorTensor = tf.tile(tf.expand_dims(factorTensor, 1),
                                       tf.stack([1, surfaceSize, 1]),
                                       name = "reshapedFactorTensorVar")
        
        if self.verbose :
            print(self.inputTensor)
            print(reshapedInputTensor)
        
        #self.factorTensor = tf.Variable(np.ones(shape=(1, self.nbFactors)).astype(np.float32))
        if self.verbose :
            print(factorTensor)
            print(reshapedFactorTensor)
        
        #Dynamic Shape of features shoud be [None, surfaceSize, 3 + self.nbFactors] that is [nbDays, surfaceSize, 3 + self.nbFactors]
        features = tf.reshape(tf.concat([reshapedInputTensor, reshapedFactorTensor],2), 
                              [-1, nbCoordinates + self.nbFactors],
                              name = "featuresVar")
        self.features = features
        # filteredFeatures = tf.boolean_mask(features, 
                                           # tf.logical_not(tf.reduce_any(tf.is_nan(features), axis=1)),
                                           # name = "filteredFeaturesVar")
        # partitions = tf.cast(tf.logical_not(tf.reduce_any(tf.is_nan(features), axis=1)), tf.int32)
        # filteredFeatures = tf.dynamic_partition(features, 
                                                # partitions,
                                                # num_partitions = 2,
                                                # name = "filteredFeaturesVar")[1]
        filteredFeatures = tf.where(tf.reduce_any(tf.is_nan(features), axis=1), 
                                    tf.zeros_like(features), 
                                    features)
        self.filteredFeatures = filteredFeatures
        if self.verbose :
            print(features)
            print(filteredFeatures)
        
        lastTensor = filteredFeatures
        #Iterate on layers to build the decoder with the same weights as those used for training
        for factory in self.layers :
            lastTensor = factory(lastTensor)
        
        
        reshapedOutputTensor = lastTensor#tf.reshape(lastTensor,[-1, surfaceSize])
        
        self.trainingPred = reshapedOutputTensor
        if self.verbose :
            print(reshapedOutputTensor)
            
        return reshapedOutputTensor
    
    #Here we can not use the same architecture because we need to optimize the layer weights AND the factors values 
    #However tensorflow does not allow for optimizing placeholder and we must redeclare factors as variables
    def buildNeuralNetworkForFixedDataSet(self, 
                                          inputDataSet,
                                          initialValueForFactors = None):
        nbDays = inputDataSet.shape[0]
        
        usedInitialFactorsValues = initialValueForFactors if initialValueForFactors is not None else np.ones(shape=(nbDays, self.nbFactors))
        factorCalibratedOnDataSets = tf.Variable(usedInitialFactorsValues.astype(np.float32))
        
        #Build
        finalOutputDataSet = self.buildReconstructionTensor(factorCalibratedOnDataSets)
        
        reconstructionLoss = self.buildReconstructionLoss(finalOutputDataSet, 
                                                          self.outputTensorRef,
                                                          "trainingReconstructionLoss", 
                                                          matrixNorm = False) 
        
        return reconstructionLoss, finalOutputDataSet, factorCalibratedOnDataSets
    
    
    
    
    def buildNeuralNetworkForFixedTrainingSet(self, 
                                              inputDataSet):
        loss, pred, calibratedFactors = self.buildNeuralNetworkForFixedDataSet(inputDataSet)
        trainingReducedReconstructionLoss = self.normalizeLoss(loss, 
                                                               "trainingReducedReconstructionLoss")
        
        trainingPenalizationList = self.buildPenalization()
        
        trainingLoss = tf.add_n([trainingReducedReconstructionLoss] + trainingPenalizationList, 
                                name="trainingLoss")
        
        #Construct list of trainable variables for every neural network
        trainableVars = [calibratedFactors]
        for l in self.layers : 
            trainableVars += l.getTrainableVariables()
        
        
        trainingOperator = self.optimizer.minimize(trainingLoss,
                                                   var_list=trainableVars,
                                                   name="trainingOperator")
        
        return pred, calibratedFactors, trainingLoss, trainingOperator
    
    
    
    
    def buildNeuralNetworkForFixedTestingSet(self, 
                                             inputDataSet,
                                             initialValueForFactors = None):
        lossPerDay, pred, calibratedFactors = self.buildNeuralNetworkForFixedDataSet(inputDataSet, 
                                                                                     initialValueForFactors = initialValueForFactors)
        calibrationLoss = self.normalizeLoss(lossPerDay, "calibrationLoss")
        
        #Calibrates only factors
        trainableVars = [calibratedFactors]
        
        calibrationOperator = self.optimizer.minimize(calibrationLoss,
                                                      var_list=trainableVars,
                                                      name="testingOperator")
        
        return calibrationOperator, pred, calibratedFactors, lossPerDay, calibrationLoss
    
    
    
    
    #Build the architecture, losses and optimizer.
    def buildModel(self):
        self.nbEncoderLayer = 0
        self.layers = []
        
        tf.reset_default_graph()
        
        if self.verbose :
            print("build architecture, loss and penalisations")
        
        self.outputTensorRef = tf.placeholder(tf.float32,
                                              shape=[None,1])#
        self.buildArchitecture()
        
        self.reconstructionLoss = self.buildReconstructionLoss(self.outputTensor, 
                                                               self.outputTensorRef,
                                                               "reconstructionLoss", 
                                                               matrixNorm = False) 
        
        self.reducedReconstructionLoss = self.normalizeLoss(self.reconstructionLoss, 
                                                            "reducedReconstructionLoss")
        
        self.penalizationList = self.buildPenalization()
        
        self.loss = tf.add_n([self.reducedReconstructionLoss] + self.penalizationList, 
                             name="loss")
        
        #Learning rates refining mechanics
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
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(name="saver", save_relative_paths=True)
        return
    
    
    
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
    
    
    def train(self, 
              inputTrain, 
              nbEpoch, 
              inputTest = None):
        self.restoringGraph()
        
        trainingSet = inputTrain
            
        testingSet = inputTest if (inputTest is not None) else trainingSet
        
        testingInput, testingOutput = self.getLocationFromDatasetList(testingSet)
        trainingInput, trainingOutput = self.getLocationFromDatasetList(trainingSet)
        
        trainingPred , trainingFactors, trainingLoss, trainingOperator = self.buildNeuralNetworkForFixedTrainingSet(trainingInput)
        
        
        # calibrationTest, testPred, testFactors, testLoss = self.buildNeuronNetworkForFixedTestingSet(testingInput, 
                                                                                                     # testingOutput)
        start = time.time()
        if self.verbose :
            print("Calibrate model on training data and return testing loss per epoch")
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(name="saver", save_relative_paths=True)
        
        nbInit = self.hyperParameters["nbInit"] if "nbInit" in self.hyperParameters else 100
        nbEpochInit = self.hyperParameters["nbEpochInit"] if "nbEpochInit" in self.hyperParameters else 1
        lossInit = []
        
        with tf.Session() as session :
            session.run(self.init)
            save_path = self.saveModel(session, self.metaModelNameInit)
            for k in range(nbInit):
                session.run(self.init)
                #Global optimization for all layers
                epochLosses, epochValidation = self.gradientDescent(session, 
                                                                    [trainingInput, trainingOutput], 
                                                                    nbEpochInit,
                                                                    [trainingInput, trainingOutput],  
                                                                    trainingLoss, 
                                                                    trainingOperator, 
                                                                    trainingLoss)
                if (k==0) or (np.nanmin(epochValidation) < np.nanmin(lossInit)): 
                    #save this model
                    save_path = self.saveModel(session, self.metaModelNameInit)
                lossInit.append(np.nanmin(epochValidation))
            
            #Get best estimate
            self.restoreWeights(session, self.metaModelNameInit)
            
            if nbInit > 0 :
                print("Min validation error for initialization : ", np.nanmin(lossInit))
                print("Mean validation error for initialization : ", np.nanmean(lossInit))
                print("Std validation error for initialization : ", np.nanstd(lossInit))
            
            #Global optimization for all layers
            epochLosses, epochValidation = self.gradientDescent(session, 
                                                                [trainingInput, trainingOutput], 
                                                                nbEpoch,
                                                                [trainingInput, trainingOutput],  
                                                                trainingLoss,
                                                                trainingOperator,
                                                                trainingLoss)
            #print("Print best solution : ")
            #self.debugWeightsAndBiases()
        
        print("Training time : % 5d" %(time.time() -  start))
        return np.array(epochLosses)
    
    def createFeedDictEncoder(self, dataSetList):
        feedDict = {self.inputTensor : np.reshape(dataSetList[0],(-1,2)),
                    self.outputTensorRef : np.reshape(dataSetList[1],(-1,1))}
        return feedDict
    
    
    
    def gradientDescent(self, 
                        session, 
                        datasetTrain, 
                        nbEpoch, 
                        dataSetTest, 
                        trainingLoss, 
                        gradientStep, 
                        validationLoss):
        epochLosses = []#np.zeros(nbEpoch)
        validationLosses = []
        activateEarlyStopping = (('validationPercentage' in self.hyperParameters) &
                                 (self.hyperParameters['validationPercentage'] > 0.001))
        patience = self.hyperParameters['calibrationWindow'] if activateEarlyStopping else 10
        bestEpoch = 0
        lastLearningRateRefinement = 0
        feedDictTraining = self.createFeedDictEncoder(datasetTrain)
        for epoch in range(nbEpoch):
            session.run(gradientStep, 
                        feed_dict = feedDictTraining)
            
            epochLosses.append(trainingLoss.eval(feed_dict = feedDictTraining))
            if self.verbose :
                print("Epoch : ", epoch, 
                      " , Penalized Loss on testing dataset : ", epochLosses[epoch]) 
            
            validationLosses.append( validationLoss.eval(feed_dict = feedDictTraining) )
            if self.verbose :
                print("Epoch : ", epoch, " , Validation Loss : ", validationLosses[epoch])
            
            #Monitor Model Performance
            if activateEarlyStopping :
                def improveLearningRate():
                    if ((self.learningRateVariable.eval() > 1e-6) and (epoch < (nbEpoch - 1))):
                        #Record current learning rate
                        formerLearningRate = self.learningRateVariable.eval()
                        
                        self.restoreWeights(session)
                        
                        #Restore former learning rate
                        session.run(self.learningRateAssign, 
                                    feed_dict = {self.learningRatePlaceholder : formerLearningRate})
                        
                        self.refineLearningRate(session)
                    else :
                        #Trigger early stopping and restore best performing model 
                        minPatienceWindow = np.nanmin(validationLosses[-patience:]) 
                        if self.verbose :
                            print("Minimum validation loss for the latest ", patience ," observations : ", minPatienceWindow)
                            print("Minimum validation loss : ", np.nanmin(validationLosses))
                        self.restoreWeights(session)
                        if self.verbose :
                            print("Validation loss from restored model : ", validationLoss.eval(feed_dict = feedDictTraining))
                        return True #Stop
                    return False #Continue training
                    
                if np.isnan(validationLosses[epoch]):
                    self.debugWeightsAndBiases()
                    print("features : ", self.features.eval(feed_dict = feedDictTraining))
                    print("filteredFeatures : ", self.filteredFeatures.eval(feed_dict = feedDictTraining))
                    print("outputTensor : ", self.trainingPred.eval(feed_dict = feedDictTraining))
                    if improveLearningRate():
                        break 
                    
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
                    if improveLearningRate():
                        break 
                    lastLearningRateRefinement = epoch
        return np.array(epochLosses), np.array(validationLosses)
                        
    
    def calibratedFactors(self, dataSetList, initialFactorValue = None):
        input, output = self.getLocationFromDatasetList(dataSetList)
        
        self.restoringGraph()
        calibrationTensors = self.buildNeuralNetworkForFixedTestingSet(input, 
                                                                       initialValueForFactors = initialFactorValue.values if initialFactorValue is not None else initialFactorValue)
        calibrationOperator = calibrationTensors[0] 
        pred = calibrationTensors[1] 
        calibratedFactorsT = calibrationTensors[2]
        reconstructionLossT = calibrationTensors[3]
        calibrationLoss = calibrationTensors[4]
        
        self.init = tf.global_variables_initializer()
        #tf.reset_default_graph()
        
        calibrationLosses = []
        calibratedFactors = None
        calibratedSurfaces = None
        bestCalibration = 0
        activateEarlyStopping = ('calibrationWindow' in self.hyperParameters)
        patience = self.hyperParameters['calibrationWindow'] if activateEarlyStopping else 10
        epsilon = 0.0001
        nbCalibrationStep = 1000
        
        feedDictTraining = self.createFeedDictEncoder([input, output])
        #Opening session for calibrating everyday factor values
        with tf.Session() as sess :
            #Restoring Model
            sess.run(self.init)
            self.restoreWeights(sess)
            
            #Perform Calibration
            for epoch in range(nbCalibrationStep):
                sess.run(calibrationOperator, feed_dict = feedDictTraining)
                
                #Save results for calibration results
                calibrationLosses.append(calibrationLoss.eval(feed_dict = feedDictTraining))
                if self.verbose :
                    print("Epoch : ", epoch, " , Calibration Loss : ", calibrationLosses[epoch])
                
                bestCalibration = np.argmin(calibrationLosses) 
                if ((epoch == 0) or (calibrationLosses[epoch] <= np.nanmin(calibrationLosses))) :
                    calibratedSurfaces = pred.eval(feed_dict = feedDictTraining)
                    calibratedFactors = calibratedFactorsT.eval()
                #No calibration improvement during patience window
                if bestCalibration < (epoch - patience) : 
                    break 
                
            
        if self.verbose :
            print("Average Loss : ", calibrationLosses[bestCalibration])
        reshapedReconstruction = pd.DataFrame(np.reshape([calibratedSurfaces],dataSetList[0].shape), 
                                              index = dataSetList[0].index, 
                                              columns = dataSetList[0].columns)
        
        return calibrationLosses[bestCalibration], reshapedReconstruction, pd.DataFrame(calibratedFactors, index = dataSetList[0].index), calibrationLosses
    
    
    
    #Evaluate the model on a testing dataset
    def evalModel(self, inputTest):
        bestLoss, reconstructedSurface, encodings, _ =  self.calibratedFactors(inputTest)
        return bestLoss, reconstructedSurface, encodings
    
    def completeDataTensor(self, 
                           sparseSurfaceList, 
                           initialValueForFactors, 
                           nbCalibrationStep):
        
        
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
    
    def createFeedDictDecoder(self, *args):
        feedDict = {self.inputTensor : np.reshape(args[0][0], (-1,2)),
                    self.factorTensor : args[1]}
        return feedDict
    
    ##Evaluate the decoder given locations and factors values
    def commonEvalSingleDayWithoutCalibration(self, 
                                              initialValueForFactors, 
                                              dataSetList, 
                                              computeSensi = False):
        #Rebuild tensor graph
        self.restoringGraph()
        #Build tensor for reconstruction
        nbObs = 1 if initialValueForFactors.ndim == 1 else initialValueForFactors.shape[0]
        nbFactors = self.nbFactors 
        
        reshapedValueForFactors = np.reshape([initialValueForFactors],
                                             (nbObs,nbFactors))
        
        inputFeatures, outputFeatures = self.getLocationFromDatasetList(dataSetList)
        
        
        #Build tensor for reconstruction
        nbPoints = inputFeatures.shape[1]
        # print("nbPoints : ", nbPoints)
        # print("initialValueForFactors : ", initialValueForFactors)
        # print("inputFeatures : ", inputFeatures)
        # print("outputFeatures : ", outputFeatures)
        # print("outputTensor : ", self.outputTensor)
        
        reshapedJacobian = None
        if computeSensi : 
            #print("gradient coordinate : ", tf.gradients(self.outputTensor[0, :], self.factorTensor))
            jacobian = tf.stack([tf.gradients(self.outputTensor[i, :], self.factorTensor) for i in range(nbPoints)], 
                                 axis=0)
            #print("jacobian : ", jacobian)
            
            reshapedJacobian = tf.reshape(jacobian, shape = [nbObs * nbPoints, nbFactors])
            #print("reshapedJacobian : ", reshapedJacobian)
            if self.verbose :
                print(reshapedJacobian)
            
            
        calibratedSurfaces = None
        factorSensi = None
            
        feedDict = self.createFeedDictDecoder([inputFeatures, outputFeatures], reshapedValueForFactors)
        #Evaluating surface for these factor Values
        with tf.Session() as sess :
            #Restoring Model
            sess.run(self.init)
            self.restoreWeights(sess)
            
            if initialValueForFactors.ndim == 1 :
                calibratedSurfaces = np.reshape(self.outputTensor.eval(feed_dict=feedDict), 
                                                (nbPoints))
                if reshapedJacobian is not None : 
                    factorSensi = np.reshape(reshapedJacobian.eval(feed_dict=feedDict), 
                                             (nbPoints, nbFactors))
            elif initialValueForFactors.ndim == 2 :
                calibratedSurfaces = np.reshape(self.outputTensor.eval(feed_dict=feedDict), 
                                                (nbObs,nbPoints))
                if reshapedJacobian is not None : 
                    factorSensi = np.reshape(reshapedJacobian.eval(feed_dict=feedDict), 
                                             (nbObs, nbPoints, nbFactors))
        return  calibratedSurfaces, factorSensi
    
    #Generalization of evalSingleDayWithoutCalibration but with customized location 
    #Can be used for interpolation purpose
    #volCoordinates should be a multiIndex
    def evalSingleDayWithoutCalibrationOnCustomLocation(self, 
                                                        initialValueForFactors, 
                                                        dataSetList):
        volCoordinates = dataSetList[1]
        
        s = self.evalSingleDayWithoutCalibration(initialValueForFactors, dataSetList)
        return s

    def plotInterpolatedSurface(self,
                                valueToInterpolate,
                                locationToInterpolate, 
                                calibratedFactors, 
                                colorMapSystem=None, 
                                plotType=None):
        y = list(map( lambda x : x[1] if x is not None else x,
                      locationToInterpolate.values))
        x = list(map( lambda x : x[0] if x is not None else x,
                      locationToInterpolate.values))
        xMax = np.nanmax(x)
        yMax = np.nanmax(y)
        xMin = np.nanmin(x)
        yMin = np.nanmin(y)
        
        xNewValues = np.linspace(xMin,xMax,num=100)
        yNewValues = np.linspace(yMin,yMax,num=100)
        grid = np.meshgrid(xNewValues,yNewValues)
        
        xInterpolated = np.reshape(grid[0],(100 * 100, 1))
        yInterpolated = np.reshape(grid[1],(100 * 100, 1))
        coordinates = np.concatenate([xInterpolated, yInterpolated], axis=1)
        ind = pd.Series(list(zip(xInterpolated,yInterpolated))).rename(locationToInterpolate.name)
        
        interpolatedSurface = self.evalSingleDayWithoutCalibrationOnCustomLocation(calibratedFactors, 
                                                                                   [None, ind, None, None])
        
        interpolatedSurfaceDf = pd.Series(interpolatedSurface, index = ind.index)
        
        plottingTools.plotGrid(interpolatedSurfaceDf, 
                               ind,
                               "Interpolated Data with Functional Approach", 
                               colorMapSystem=colorMapSystem, 
                               plotType=plotType)
        
        plottingTools.standardInterp(valueToInterpolate, 
                                     locationToInterpolate,
                                     colorMapSystem=colorMapSystem, 
                                     plotType=plotType)
        return
    
    #Take a full surface in entry, reconstruct it 
    #and return sensitivities between points i.e. the jacobian of D(E(S)) w.r.t S
    def evalInterdependancy(self, fullSurface):
        raise NotImplementedError("Implicit encoder model")
        return
