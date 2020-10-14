#Import modules
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pandas import DatetimeIndex
import dask
import scipy
import time
import glob

from functools import partial
from abc import ABCMeta, abstractmethod

import plottingTools
import factorialModel

tmpFolder = "./tmp/" #"./../tmp/"
#tmpFolder = os.environ['tmp']

tf.logging.set_verbosity(tf.logging.WARN)

def getTensorByName(tensorName):
    return (tf.get_default_graph().get_tensor_by_name(tensorName + ":0"))

def getOperationByName(operationName):
    return (tf.get_default_graph().get_operation_by_name(operationName))

def registerLayer(layer, name):
    return tf.identity(layer, name=name)
    

#inspired from https://github.com/Seratna/TensorFlow-Convolutional-AutoEncoder/tree/master/models

class Layer(object, metaclass=ABCMeta):
    """

    """
    def __init__(self):
        self.layerType = "NonDefined"
        self.inputTensor = None
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.inputTensor = args[0] #Input tensor should be the first arguments
        return self.call(*args, **kwargs)
        
    def isTrainable(self):
        return False

    @abstractmethod
    def getTrainableVariables(self):
        return []
    
    @abstractmethod
    def copy(self):
        raise NotImplementedError
        
    def createDynamicShape(self, staticShape):
        return tf.shape(tf.constant(0, shape=staticShape))




class FullyConnected(Layer):
    """

    """
    def __init__(self,
                 output_dim,
                 weights=None,
                 bias=None,
                 activation=None,
                 scope='',
                 kernelRegularizer = None,
                 kernelInitializer = None):
        Layer.__init__(self)
        
        self.layerType = "FullyConnected"
        self.output_dim = output_dim
        self.input_dim = None
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.scope = scope
        self.kernelRegularizer = kernelRegularizer
        self.kernelInitializer = kernelInitializer if kernelInitializer else tf.keras.initializers.TruncatedNormal(stddev=0.1)

    def build(self, input_tensor):
        
        tensorDimension = input_tensor.get_shape()
        num_batch = tensorDimension[0] 
        self.input_dim = tensorDimension[-1]
        
        # build weights
        if self.weights:            
            assert self.weights.get_shape() == (self.input_dim.value, self.output_dim)
        else:
            self.weights = tf.Variable(self.kernelInitializer((self.input_dim.value, self.output_dim)),
                                       name='weights')
            if self.kernelRegularizer :
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.kernelRegularizer(self.weights))
            
        # build bias
        if self.bias:
            assert self.bias.get_shape() == (self.output_dim, )
        else:
            self.bias = tf.Variable(tf.constant(0.0, shape=[self.output_dim]), name='bias')

        # fully connected layer
        fc = tf.tensordot(input_tensor, self.weights, axes=1) + self.bias

        # activation
        if self.activation:
            return self.activation(fc)
        return fc

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
    
    def isTrainable(self):
        return True
    
    def getTrainableVariables(self):
        return [self.weights, self.bias]
    
    def copy(self):
        return FullyConnected(self.output_dim, 
                              weights = self.weights, 
                              bias = self.bias,
                              activation= self.activation,
                              scope= self.scope,
                              kernelRegularizer = self.kernelRegularizer,
                              kernelInitializer = self.kernelInitializer)
    def getInputDim(self):
        if self.input_dim :
            return self.input_dim.value
        raise Exception("Input dimension not defined")
        return None





class tensorflowModel(factorialModel.FactorialModel) :
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestTensorflowModel"):
        #Tensors common to any architecture
        self.trainingOperator = None
        self.factorTensor = None
        self.outputTensor = None
        self.inputTensor = None
        self.reconstructionLoss = None #Loss for a given day
        self.reducedReconstructionLoss = None #Mean of self.reconstructionLoss along dataSetDay
        self.loss = None #Loss with Penalization
        self.penalizationList = None 
        self.learningRatePlaceholder = None
        self.learningRateAssign = None
        self.learningRateVariable = None
        self.optimizer = None
        self.init = None
        self.saver = None
        self.dynamicPenalization = None
        self.sparseSurfaceTensor = None
        super().__init__(learningRate, hyperParameters, nbUnitsPerLayer, nbFactors, 
                         modelName = modelName)
        
    #Return all hyperameters used by the learning model 
    # def getModelHyperameters(self):
        # self.learningRate = learningRate
        # self.hyperParameters = hyperParameters
        # self.nbUnitsPerLayer = nbUnitsPerLayer
        # self.nbFactors = nbFactors
        # self.layers = []
        # self.nbEncoderLayer = 0
        # self.batchSize = -1
        # self.lossHolderExponent = (hyperParameters["lossHolderExponent"] if "lossHolderExponent" in hyperParameters else 4)
        
        # hyperParametersDict = {}
        # hyperParametersDict[""] = 
        # return
        
    def restoringGraph(self):
        #Restore graph of operations
        tf.reset_default_graph()
        
        self.buildModel()
        return
        
    def restoreWeights(self,session, fileName = None):
        #Restore graph of operations
        self.saver.restore(session, self.metaModelName if fileName is None else fileName )
        return
        
    #Build the layers, input-output-factor tensors
    def buildArchitecture(self):
        raise NotImplementedError()
        return
    
    #Return a list of penalization to add reconstruction loss
    def buildPenalization(self, **kwargs):
        return []
    
    def setDynamicPenalizationFactory(self, factory):
        self.dynamicPenalization = factory
        return
    
    #Return a list of penalization to add to completion loss
    
    def buildCompletionLoss(self, factorTensor, calibrationLoss, completedSurfaceTensor):
        if self.dynamicPenalization is None :
            finalCalibrationLoss = calibrationLoss
        else : 
            finalCalibrationLoss = (calibrationLoss + 
                                    self.hyperParameters["lambdaDynamic"] * self.dynamicPenalization(factorTensor))
        return finalCalibrationLoss
    
    def debugWeightsAndBiases(self):
        for l in self.layers :
            vars = l.getTrainableVariables()
            for v in vars :
                print(v.eval())
    #Loss computed on a per observation basis
    def buildReconstructionLoss(self, pred, ref, name, matrixNorm = True):
        #If reference value contains NaN we ignore it for completion purpose
        diff = pred - ref
        if matrixNorm : 
            diff_without_nans = tf.reshape(tf.where(tf.is_nan(ref), tf.zeros_like(diff), diff), 
                                           [-1, self.hyperParameters['nbX'], self.hyperParameters['nbY']])
            return tf.norm(diff_without_nans, ord = self.lossHolderExponent, axis = [-2,-1], name=name) 
        diff_without_nans = tf.where(tf.is_nan(ref), tf.zeros_like(diff), diff)
        return tf.norm(diff_without_nans, ord = self.lossHolderExponent, axis =1, name=name)
        #return tf.pow(diff_without_nans, self.lossHolderExponent, name=name)
    
    #Aggregate errors on batch dimension
    def normalizeLoss(self, dayLoss, name):
        #return tf.pow(tf.reduce_mean(tf.reduce_mean(dayLoss, axis=-1)), 1.0 / self.lossHolderExponent, name=name)
        #return tf.reduce_mean(dayLoss, name=name)
        return tf.norm(dayLoss, ord = self.lossHolderExponent, name=name)
    
    #Build loss for batch sample
    def buildLoss(self, pred, ref, name, matrixNorm = True):
        return self.normalizeLoss(self.buildReconstructionLoss(pred, ref, name + "R", matrixNorm=matrixNorm), 
                                  name)
    
    def refineLearningRate(self, session):
        formerLearningRate = self.learningRateVariable.eval(session = session)
        session.run(self.learningRateAssign, 
                    feed_dict = {self.learningRatePlaceholder : formerLearningRate * 0.1})
        #print("New Learning Rate : ", self.learningRateVariable.eval(session = session))
        return 
        
    def resetLearningRate(self, session):
        session.run(self.learningRateAssign, 
                    feed_dict = {self.learningRatePlaceholder : self.learningRate})
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
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(name="saver", 
                                    save_relative_paths=True, 
                                    filename = self.metaModelName) 
        return
    
        
    #save metamodel (graph and variable values)
    def saveModel(self, session, pathFile): 
        #Delete former file version
        for f in glob.glob(pathFile + "*"):
            os.remove(f)
        cpkt = self.saver.save(session, pathFile, 
                               latest_filename="LatestSave")
        return cpkt
    
    
    def isFirstEncodingLayer(self, layerIndex):
        if self.layers[layerIndex].isTrainable() : #Is this layer trainable
            for lIndex in range(layerIndex): #For each lower layer
                if self.layers[lIndex].isTrainable() : #A lower layer is trainable 
                    return False
            #All lower layers are not trainable hence we are layerIndex is the first trainable encoding layer
            return True
        return False
    
    def buildDenseLayer(self,
                        nbOutput, 
                        inputTensor, 
                        activation = tf.nn.softplus,
                        kernelRegularizer = None,
                        kernelInitializer = None):
        self.layers.append(FullyConnected(nbOutput, 
                                          activation=activation,
                                          scope="dense_" + str(len(self.layers)),
                                          kernelRegularizer = kernelRegularizer,
                                          kernelInitializer = kernelInitializer))
        return self.layers[-1](inputTensor)
    
    #Build a tensor that construct a surface from factors values
    def buildReconstructionTensor(self, factorTensor):
        nbDeconvLayer = len(self.layers) - self.nbEncoderLayer
        lastTensor = factorTensor
        for k in range(nbDeconvLayer,0,-1):
            currentLayerIndex = -k
            factoryTmp = self.layers[currentLayerIndex].copy()
            lastTensor = factoryTmp(lastTensor)
        
        return lastTensor
    
    def buildInverseLayer(self, 
                          inputTensor):
        
        encoderLayerIndex = 2 * self.nbEncoderLayer - len(self.layers) - 1 
        if self.verbose :
            print("Building  inserve operation for layer ", 
                  encoderLayerIndex, 
                  " of type ", 
                  self.layers[encoderLayerIndex].layerType)
        #If only python could propose a proper switch !!!
        if self.layers[encoderLayerIndex].layerType == "FullyConnected" :
            if self.verbose :
                print("Dense layer so inverse layer is a dense layer")
            #Output decoder layer receives linear activation function
            buildOutputLayer = self.isFirstEncodingLayer(encoderLayerIndex)
            activationFunction = None if buildOutputLayer else self.layers[encoderLayerIndex].activation
            self.layers.append(FullyConnected(self.layers[encoderLayerIndex].getInputDim(), 
                                              activation=activationFunction,
                                              scope="dense_" + str(len(self.layers)),
                                              kernelRegularizer = self.layers[encoderLayerIndex].kernelRegularizer,
                                              kernelInitializer = self.layers[encoderLayerIndex].kernelInitializer))
        else:
            raise Exception("No legal inverse operation for layer type " + self.layers[encoderLayerIndex].layerType)
        return self.layers[-1](inputTensor)
        
    #######################################################################################################
    #Training functions
    #######################################################################################################
    
    #Create a feed_dict (see tensorflow documentation) for evaluating encoder
    #args : List dataset to feed, order meaning is proper to each model 
    def createFeedDictEncoder(self, dataSetList):
        feedDict = {self.inputTensor : dataSetList[0]}
        return feedDict
    
    #Create a feed_dict (see tensorflow documentation) for completing a surface
    def createFeedDictCompletion(self, dataSetList):
        feedDict = {self.sparseSurfaceTensor : np.expand_dims(dataSetList[0], 0) if dataSetList[0].ndim <=1 else dataSetList[0]}
        return feedDict
    
    #Create a feed_dict (see tensorflow documentation) for evaluating decoder
    #from 
    def createFeedDictDecoder(self, *args):
        feedDict = {}
        return feedDict
    
    
    #Sample Mini-batch
    def generateMiniBatches(self, dataSetList, nbEpoch):
        batchSize = 100
        #return self.selectMiniBatchWithoutReplacement(dataSetList, batchSize)
        return [dataSetList]
        
    #Train the factorial model
    def trainWithSession(self, session, inputTrain, nbEpoch, inputTest = None):
        start = time.time()
        if self.verbose :
            print("Calibrate model on training data and return testing loss per epoch")
        
        nbInit = self.hyperParameters["nbInit"] if "nbInit" in self.hyperParameters else 100
        nbEpochInit = self.hyperParameters["nbEpochInit"] if "nbEpochInit" in self.hyperParameters else 1
        session.run(self.init)
        save_path = self.saveModel(session, self.metaModelNameInit)
        lossInit = []
        for k in range(nbInit):
            session.run(self.init)
            #Global optimization for all layers
            epochLosses, epochValidation = self.gradientDescent(session, 
                                                                inputTrain, 
                                                                nbEpochInit,
                                                                inputTest, 
                                                                self.loss, 
                                                                self.trainingOperator, 
                                                                self.reducedReconstructionLoss)
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
                                                            inputTrain, 
                                                            nbEpoch,
                                                            inputTest, 
                                                            self.loss, 
                                                            self.trainingOperator, 
                                                            self.reducedReconstructionLoss)
        
        print("Detailed performances")
        if inputTest is not None : 
            totalLoss = session.run(self.loss ,
                                    feed_dict=self.createFeedDictEncoder(inputTest if inputTest is not None else inputTrain))
        print("Penalized Loss on testing dataset : ", totalLoss)
        print("Training time : % 5d" %(time.time() -  start))
        return epochLosses
    
    
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
        with tf.Session() as sess :
            res = self.trainWithSession(sess, inputTrain, nbEpoch, inputTest)
        return res
    
    
    
    #Stacked network
    def buildAndTrainPartialAutoencoder(self,
                                        stepEncoderLayer,
                                        stepDecoderLayer,
                                        trainableVariableList,
                                        inputToReproduce, 
                                        session,
                                        inputTrain, 
                                        nbEpoch, 
                                        inputTest,
                                        encoderLayerIndex):
        stepFactorLayer = inputToReproduce
        for layerFactory in stepEncoderLayer :
            stepFactorLayer = layerFactory(stepFactorLayer)
        
        stepOutputLayer = stepFactorLayer
        for layerFactory in stepDecoderLayer :
            stepOutputLayer = layerFactory(stepOutputLayer)
        
        #Build Losses
        stepLoss = self.buildLoss( stepOutputLayer, 
                                   inputToReproduce, 
                                   "stepOutputLayer"+str(encoderLayerIndex),
                                   matrixNorm = False )
        stepPenalization = []
        if 'l2_reg' in self.hyperParameters :
            kwargsPenalization = {'layersKernels' : trainableVariableList[0::2], #kernels are located at even indices
                                  'outputLayer' : stepOutputLayer,
                                  'inputLayer' : inputToReproduce,
                                  'factors' : stepFactorLayer}
            stepPenalization = self.buildPenalization(**kwargsPenalization)
        
        stepTotalLosses = tf.add_n([stepLoss] + stepPenalization)
        
        stepVariableToTrain = self.getVariableFromTensor(trainableVariableList)
        stepTrainingOperator = self.optimizer.minimize(stepTotalLosses, 
                                                       var_list=stepVariableToTrain)
        
        self.gradientDescent(session, 
                             inputTrain, 
                             nbEpoch, 
                             inputTest, 
                             stepTotalLosses, 
                             stepTrainingOperator, 
                             stepLoss)
        return stepFactorLayer
    
    #Train autoencoder by pair following unsuperviser layer-wise training paradigm 
    def pretrainNetwork(self, 
                        session, 
                        inputTrain, 
                        nbEpoch, 
                        inputTest = None):
        nbDecoderLayer = len(self.layers) - self.nbEncoderLayer
        nbSteps = min(self.nbEncoderLayer,nbDecoderLayer)
        layerShift = max(self.nbEncoderLayer,nbDecoderLayer) - nbSteps
        
        inputToReproduce = self.inputTensor
        #List of variable to train, kernel/weights are located on even indexes, bias on odd indexes
        trainableVariableList = [] 
        #List of ordered layer factory for encoder
        stepEncoderLayer = [] 
        #List of ordered layer factory for decoder
        stepDecoderLayer = [] 
        
        #Train each autoencoder layer by pair if possible
        for l in range(nbSteps):
            #Decoder and encoder layers pair for this step
            encoderLayerIndex = l + (0 if (self.nbEncoderLayer == nbSteps) else layerShift)
            decoderLayerIndex = -l-1 - (0 if (nbDecoderLayer == nbSteps) else layerShift)
            if self.verbose :
                print("Train layers nb " + str(encoderLayerIndex + 1) + " and " + str(len(self.layers) + decoderLayerIndex + 1))
            
            if ((l==0) & (layerShift != 0)):#first training step with non-symetric autoencoder
                #add external layers to handle asymetry for training first step
                if (nbDecoderLayer == nbSteps) :#additional encoder layer
                    for k in range(layerShift,0,-1):
                        print("Train layers nb " + str(encoderLayerIndex-k+1))
                        stepEncoderLayer.append(self.layers[encoderLayerIndex-k])
                        trainableVariableList+=stepEncoderLayer[-1].getTrainableVariables()
                    
                    stepEncoderLayer.append(self.layers[encoderLayerIndex])
                    trainableVariableList+=stepEncoderLayer[-1].getTrainableVariables()
                    
                    stepDecoderLayer.insert(0,self.layers[decoderLayerIndex])
                    trainableVariableList+=stepDecoderLayer[0].getTrainableVariables()
                else :#additional decoder layer
                    stepEncoderLayer.append(self.layers[encoderLayerIndex])
                    trainableVariableList+=stepEncoderLayer[-1].getTrainableVariables()
                    
                    for k in range(1,layerShift+1,1):
                        print("Train layers nb " + str(len(self.layers) + decoderLayerIndex + 1 + k))
                        stepDecoderLayer.append(self.layers[decoderLayerIndex+k])
                        trainableVariableList+=stepDecoderLayer[-1].getTrainableVariables()
                    
                    stepDecoderLayer.insert(0,self.layers[decoderLayerIndex])
                    trainableVariableList+=stepDecoderLayer[0].getTrainableVariables()
                    
            else :
                stepEncoderLayer.append(self.layers[encoderLayerIndex])
                trainableVariableList+=stepEncoderLayer[-1].getTrainableVariables()
                
                stepDecoderLayer.insert(0,self.layers[decoderLayerIndex])
                trainableVariableList+=stepDecoderLayer[0].getTrainableVariables()
            
            #A pair of layer is trainable
            if self.layers[encoderLayerIndex].isTrainable() & self.layers[decoderLayerIndex].isTrainable():
                inputToReproduce = self.buildAndTrainPartialAutoencoder(stepEncoderLayer,
                                                                        stepDecoderLayer,
                                                                        trainableVariableList,
                                                                        inputToReproduce, 
                                                                        session,
                                                                        inputTrain, 
                                                                        nbEpoch, 
                                                                        inputTest,
                                                                        encoderLayerIndex)
                trainableVariableList = [] #Reset list of trainable variables
                stepEncoderLayer = []
                stepDecoderLayer = []
                
        
        return
    
    #######################################################################################################
    #Evaluation functions
    #######################################################################################################
    #Evaluate the model on a testing dataset
    def evalModelWithSession(self, sess, inputTest):
        feedDict = self.createFeedDictEncoder(inputTest)
        factors = self.factorTensor.eval(feed_dict=feedDict, 
                                         session = sess)
        reconstructedInputs = self.outputTensor.eval(feed_dict=feedDict, 
                                                     session = sess)
        reconstructionLoss = self.reducedReconstructionLoss.eval(feed_dict=feedDict, 
                                                                 session = sess)
        if self.verbose :
            print("Average Loss : ", reconstructionLoss)
        dfReconstructedInputs = pd.DataFrame(reconstructedInputs,
                                             index=inputTest[0].index,
                                             columns=inputTest[0].columns)
        return reconstructionLoss, dfReconstructedInputs, pd.DataFrame(factors, index = inputTest[0].index)
    
    #Same but with default session 
    def evalModel(self, inputTest):
        self.restoringGraph()
        with tf.Session() as sess:
            self.restoreWeights(sess)
            loss, inputs, factors = self.evalModelWithSession(sess, inputTest)
        return loss, inputs, factors
    
        
    def getWeightAndBiasFromLayer(self, layer):
        kernel = getTensorByName(os.path.split(layer.name)[0] + '/kernel')
        bias = getTensorByName(os.path.split(layer.name)[0] + '/bias')
        return kernel, bias
    
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
        reshapedValueForFactors = np.reshape([initialValueForFactors],
                                             (1,initialValueForFactors.shape[0]))
        dataSetList = [reshapedSparseSurface]
        for k in args :
            dataSetList.append(np.reshape([k], (1,k.shape[0])))
        
        self.sparseSurfaceTensor = tf.placeholder(tf.float32,
                                                  shape=reshapedSparseSurface.shape)
        factorsToCalibrate = tf.Variable(reshapedValueForFactors,
                                         name = "factorsToCalibrate", 
                                         trainable = True)
        reconstructedSurface = self.buildReconstructionTensor(factorsToCalibrate)
        completedSurface = tf.where(tf.is_nan(self.sparseSurfaceTensor), 
                                    reconstructedSurface, 
                                    self.sparseSurfaceTensor)
        calibrationLoss = self.buildLoss(reconstructedSurface, 
                                         self.sparseSurfaceTensor, 
                                         name="calibrationLoss" )
        
        finalCalibrationLoss = self.buildCompletionLoss(factorsToCalibrate, calibrationLoss, reconstructedSurface)
        
        #Factors calibrators
        factorCalibrator = self.optimizer.minimize(finalCalibrationLoss, 
                                                   var_list=[factorsToCalibrate])
        self.init = tf.global_variables_initializer()
        #tf.reset_default_graph()
        
        calibrationLosses = []
        penalizedCalibrationLosses = []
        calibratedFactors = []
        calibratedSurfaces = []
        activateEarlyStopping = ('calibrationWindow' in self.hyperParameters)
        patience = self.hyperParameters['calibrationWindow'] if activateEarlyStopping else 10
        epsilon = 0.0001
        
        
        #Opening session for calibration
        with tf.Session() as sess :
            #Restoring Model
            #self.saver = tf.train.import_meta_graph(self.metaModelName + '.meta')
            sess.run(self.init)
            self.restoreWeights(sess)
            
            #Perform Calibration
            feedDict = self.createFeedDictCompletion(*dataSetList)
            for epoch in range(nbCalibrationStep):
                sess.run(factorCalibrator, feed_dict=feedDict)
                
                #Save results for calibration results
                calibrationLosses.append(calibrationLoss.eval(feed_dict=feedDict))
                penalizedCalibrationLosses.append(finalCalibrationLoss.eval(feed_dict=feedDict))
                if self.verbose :
                    print("Epoch : ", epoch, " , Calibration Loss : ", calibrationLosses[epoch])
                    print("Penalized Loss : " , penalizedCalibrationLosses[epoch])
                
                calibratedFactors.append(factorsToCalibrate.eval())
                calibratedSurfaces.append(completedSurface.eval(feed_dict=feedDict))
                
                #No calibration improvement during patience window
                bestCalibration = np.argmin(penalizedCalibrationLosses) 
                if bestCalibration < (epoch - patience) : 
                    break 
            
        #Get results for best calibration
        bestCalibration = np.argmin(calibrationLosses)
        bestFactors = calibratedFactors[bestCalibration]
        bestSurface = pd.Series(np.reshape([calibratedSurfaces[bestCalibration]], sparseSurface.shape), 
                                index=sparseSurface.index, 
                                name=sparseSurface.name)
        return calibrationLosses[bestCalibration] , bestFactors[0], bestSurface, pd.Series(calibrationLosses) 
        
    
    def commonEvalSingleDayWithoutCalibration(self, 
                                              initialValueForFactors,
                                              dataSetList,
                                              computeSensi = False):
        #Rebuild tensor graph
        self.restoringGraph()
        #Build tensor for reconstruction
        nbObs = 1
        nbFactors = 0
        if initialValueForFactors.ndim == 1 :
            nbFactors = initialValueForFactors.shape[0]
        elif initialValueForFactors.ndim == 2 :
            nbObs = initialValueForFactors.shape[0]
            nbFactors = initialValueForFactors.shape[1]
        else :
            raise NotImplementedError("Tensor of rank greater than 2")
        
        reshapedValueForFactors = np.reshape([initialValueForFactors],
                                             (nbObs,nbFactors))
        factorsToCalibrate = tf.Variable(reshapedValueForFactors,
                                         name = "factorsToCalibrate", 
                                         trainable = False)
        reshapedDatasetList = ([np.expand_dims(x,0) if (x is not None) else x for x in dataSetList] 
                               if (initialValueForFactors.ndim <= 1) else dataSetList)
        
        reconstructedSurface = self.buildReconstructionTensor(factorsToCalibrate)
        
        nbPoints = reconstructedSurface.get_shape().as_list()[1]
        reshapedJacobian = None
        if computeSensi :
            jacobian = tf.stack([tf.gradients(reconstructedSurface[:, i], factorsToCalibrate) for i in range(nbPoints)], 
                                axis=2)
            reshapedJacobian = tf.reshape(jacobian, shape = [nbObs, nbPoints, nbFactors])
            if self.verbose :
                print(reshapedJacobian)
            
        self.init = tf.global_variables_initializer()
            
        calibratedSurfaces = None
        factorSensi = None
            
                
        #Evaluating surface for these factor Values
        with tf.Session() as sess :
            #Restoring Model
            #self.saver = tf.train.import_meta_graph(self.metaModelName + '.meta')
            sess.run(self.init)
            self.restoreWeights(sess)
            feedDict = self.createFeedDictDecoder(reshapedDatasetList)
            if initialValueForFactors.ndim == 1 :
                calibratedSurfaces = np.reshape(reconstructedSurface.eval(feed_dict=feedDict), (nbPoints))
                if reshapedJacobian is not None :
                    factorSensi = np.reshape(reshapedJacobian.eval(feed_dict=feedDict), (nbPoints, nbFactors))
            elif initialValueForFactors.ndim == 2 :
                calibratedSurfaces = np.reshape(reconstructedSurface.eval(feed_dict=feedDict), (nbObs,nbPoints))
                if reshapedJacobian is not None :
                    factorSensi = np.reshape(reshapedJacobian.eval(feed_dict=feedDict), (nbObs, nbPoints, nbFactors))
        
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
        #Rebuild tensor graph
        self.restoringGraph()
        
        nbObs = fullDataSet[0].shape[0]
        nbPoints = fullDataSet[0].shape[1]
        jacobian = tf.stack([tf.gradients(self.outputTensor[:, i], self.inputTensor) for i in range(nbPoints)],
                            axis=2)
        batchSize = tf.shape(self.outputTensor)[0]
        reshapedJacobian = tf.reshape(jacobian, shape = [batchSize, nbPoints, nbPoints])
        
        if self.verbose :
            print(reshapedJacobian)
            
        self.init = tf.global_variables_initializer()
            
        reconstructedSurface = None
        interdependancies = None
            
                
        #Evaluating surface for these factor Values
        reshapedInterdependancies = None
        reshapedReconstructedSurface = None
        with tf.Session() as sess :
            #Restoring Model
            sess.run(self.init)
            self.restoreWeights(sess)
            
            reconstructedSurface, interdependancies = sess.run([self.outputTensor , reshapedJacobian],
                                                               feed_dict=self.createFeedDictEncoder(fullDataSet))
            reshapedInterdependancies = np.reshape(interdependancies,(nbObs,nbPoints,nbPoints))
            reshapedReconstructedSurface = np.reshape(reconstructedSurface,(nbObs,nbPoints))
        
        return reshapedReconstructedSurface, reshapedInterdependancies
        
    
    #Return None if not supported
    def getDecoderCoefficients(self):
        #raise NotImplementedError("Only implemnted for linear deconding model !")
        return None

