
import pandas as pd
import numpy as np
import tensorflow as tf
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

import shallowAutoencoder


#Taken from https://stackoverflow.com/questions/39354566/what-is-the-equivalent-of-np-std-in-tensorflow
#Available in next version of tensorflow
def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


class StackedAutoEncoder(shallowAutoencoder.ShallowAutoEncoder):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestStackedAEModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
        
    def buildArchitecture(self):
        
        #Kernel initializer
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        # def positiveKernelInitializer(shape, dtype=None, partition_info=None):
            # return 1 + tf.abs(tf.contrib.layers.variance_scaling_initializer()(shape,dtype))
        
        #Regularizer
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        
        self.inputTensor = tf.placeholder(tf.float32, 
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#batch size along 
        if self.verbose :
            print(self.inputTensor)
        
        #Layers 1
        hiddenEncoder1 = self.buildDenseLayer(self.nbUnitsPerLayer['LayerEncoder1'], 
                                              self.inputTensor,
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        if self.verbose :
            print(hiddenEncoder1)
        
        #Layer 2
        hiddenEncoder2 = self.buildDenseLayer(self.nbUnitsPerLayer['LayerEncoder2'], 
                                              hiddenEncoder1,
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        if self.verbose :
            print(hiddenEncoder2)
        
        #Layer 3
        hiddenEncoder3 = self.buildDenseLayer(self.nbUnitsPerLayer['LayerEncoder3'], 
                                              hiddenEncoder2,
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        if self.verbose :
            print(hiddenEncoder3)
        
        #Layer 4 / Hidden Layer
        self.factorTensor = self.buildDenseLayer(self.nbFactors, 
                                                 hiddenEncoder3,
                                                 activation = None,
                                                 kernelRegularizer = l2_regularizer,
                                                 kernelInitializer = he_init)
        
        self.nbEncoderLayer = len(self.layers)
        # DECODE --------------------------------------------------------------------
        
        lastTensor = self.factorTensor
        for k in range(self.nbEncoderLayer):
            if self.verbose :
                print(lastTensor)
            lastTensor = self.buildInverseLayer(lastTensor)
        
        # if self.verbose :
            # print(lastTensor)
        # lastTensor = self.buildDenseLayer(self.nbUnitsPerLayer['Output Layer'], 
                                          # lastTensor,
                                          # kernelRegularizer = l2_regularizer,
                                          # kernelInitializer = he_init,
                                          # activation = None)
        
        self.outputTensor = lastTensor
        
        if self.verbose :
            print(self.outputTensor)
        return
    
    
    def buildPenalization(self,**kwargs):
        return super().buildPenalization(**kwargs)



class StackedAutoEncoderOptimized(StackedAutoEncoder):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestStackedAEOptimizedModel"):
        
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    #Train the factorial model
    def trainWithSession(self, session, inputTrain, nbEpoch, inputTest = None):
        start = time.time()
        if self.verbose :
            print("Calibrate model on training data and return testing loss per epoch")
        
        nbInit = self.hyperParameters["nbInit"] if "nbInit" in self.hyperParameters else 100
        nbEpochInit = self.hyperParameters["nbEpochInit"] if "nbEpochInit" in self.hyperParameters else 1
        lossInit = []
        session.run(self.init)
        save_path = self.saveModel(session, self.metaModelNameInit)
        for k in range(nbInit):
            session.run(self.init)
            
            ##Layer wise pretraining
            #self.pretrainNetwork(session, inputTrain, nbEpochInit, inputTest = inputTest)
            
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
        
        
        
        #Layer wise pretraining
        self.pretrainNetwork(session, inputTrain, nbEpoch, inputTest = inputTest)
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

class ContractiveAutoEncoder(StackedAutoEncoderOptimized):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestContractiveAEModel"):
        
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)    
    
    def buildPenalization(self, **kwargs):
        firstPenalizations = super().buildPenalization(**kwargs)
        #Aims at reducing factor sensitivity to inputs values
        def contractiveLoss(inputTensor, factorTensor):
            nbFactor = factorTensor.get_shape().as_list()[1]
            jacobian = tf.stack([tf.gradients(factorTensor[:, i], inputTensor) for i in range(nbFactor)], 
                                axis=2)
            cLoss = tf.norm(jacobian)
            return tf.reduce_mean(cLoss)
        contractivePenalization = None
        
        if len(kwargs)==0:#Train all layers
            contractivePenalization = contractiveLoss(self.inputTensor, self.factorTensor) 
        else :#Train a subpart of neural network
            #contractivePenalization = contractiveLoss(kwargs['inputLayer'], kwargs['factors'])
            return firstPenalizations
        return firstPenalizations + [self.hyperParameters['lambdaContractive'] * contractivePenalization]
























#Denoising autoencoder where each intermediate layer received masked input during training as explained in
#Vincent, Pascal, et al. "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." 
#Journal of machine learning research 11.12 (2010).
class StackedAutoEncoderDenoised(StackedAutoEncoderOptimized):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestStackedAEModelDenoised"):
        self.IsTraining = None
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    #Train the factorial model
    def trainWithSession(self, session, inputTrain, nbEpoch, inputTest = None):
        start = time.time()
        if self.verbose :
            print("Calibrate model on training data and return testing loss per epoch")
        
        
        nbInit = self.hyperParameters["nbInit"] if "nbInit" in self.hyperParameters else 100
        nbEpochInit = self.hyperParameters["nbEpochInit"] if "nbEpochInit" in self.hyperParameters else 100
        lossInit = []
        session.run(self.init)
        save_path = self.saveModel(session, self.metaModelNameInit)
        for k in range(nbInit):
            session.run(self.init)
            
            ##Layer wise pretraining
            #self.pretrainNetwork(session, inputTrain, nbEpochInit, inputTest = inputTest)
            
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
            print("Mean validation error for initialization : ", np.mean(lossInit))
            print("Std validation error for initialization : ", np.std(lossInit))
        
        
        #Layer wise pretraining
        self.pretrainNetwork(session, inputTrain, nbEpoch, inputTest = inputTest)
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
        
    def buildArchitecture(self):
        
        #Kernel initializer
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        # def positiveKernelInitializer(shape, dtype=None, partition_info=None):
            # return 1 + tf.abs(tf.contrib.layers.variance_scaling_initializer()(shape,dtype))
        
        #Regularizer
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        #Tensor for complete data 
        self.inputTensor = tf.placeholder(tf.float32, 
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#batch size along
        #Tensor for noisy data                                           
        # self.corruptedTensor = tf.placeholder(tf.float32,
                                              # shape=[None, self.nbUnitsPerLayer['Input Layer']])
        if self.verbose :
            print(self.inputTensor)
        
        #self.IsTraining = tf.placeholder_with_default(False, (),
        #                                              name="ActivateCorruption")
                                                 
        #inputLayer = self.corruptTensor(self.inputTensor) if self.IsTraining else self.inputTensor
        inputLayer = self.inputTensor
        
        #Layers 1
        hiddenEncoder1 = self.buildDenseLayer(self.nbUnitsPerLayer['LayerEncoder1'], 
                                              inputLayer,
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        if self.verbose :
            print(hiddenEncoder1)
        
        #Layer 2
        hiddenEncoder2 = self.buildDenseLayer(self.nbUnitsPerLayer['LayerEncoder2'], 
                                              hiddenEncoder1,
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        if self.verbose :
            print(hiddenEncoder2)
        
        #Layer 3
        hiddenEncoder3 = self.buildDenseLayer(self.nbUnitsPerLayer['LayerEncoder3'], 
                                              hiddenEncoder2,
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        if self.verbose :
            print(hiddenEncoder3)

        #Layer 4 / Hidden Layer
        self.factorTensor = self.buildDenseLayer(self.nbFactors, 
                                                 hiddenEncoder3,
                                                 activation = None,
                                                 kernelRegularizer = l2_regularizer,
                                                 kernelInitializer = he_init)
        
        self.nbEncoderLayer = len(self.layers)
        # DECODE --------------------------------------------------------------------
        
        lastTensor = self.factorTensor
        for k in range(self.nbEncoderLayer):
            if self.verbose :
                print(lastTensor)
            lastTensor = self.buildInverseLayer(lastTensor)
        
        # if self.verbose :
            # print(lastTensor)
        # lastTensor = self.buildDenseLayer(self.nbUnitsPerLayer['Output Layer'], 
                                          # lastTensor,
                                          # kernelRegularizer = l2_regularizer,
                                          # kernelInitializer = he_init,
                                          # activation = None)
        
        self.outputTensor = lastTensor
        
        if self.verbose :
            print(self.outputTensor)
        return
    
    
    def buildPenalization(self,**kwargs):
        return super().buildPenalization(**kwargs)
        

    
    #Taken from https://github.com/timsainb/GAIA/blob/master/network.py
    def shape(self, tensor):
        """ get the shape of a tensor
        """
        s = tensor.get_shape()
        return tuple([s[i].value for i in range(0, len(s))])
        
    def squared_dist(A):
        """
        Computes the pairwise distance between points
        #http://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
        """
        expanded_a = tf.expand_dims(A, 1)
        expanded_b = tf.expand_dims(A, 0)
        distances = tf.reduce_mean(tf.squared_difference(expanded_a, expanded_b), 2)
        return distances
    
    def distance_loss(self, x, z_x):
        """ Loss based on the distance between elements in a batch
        """
        z_x = tf.reshape(z_x, [shape(z_x)[0], np.prod(shape(z_x)[1:])])
        sdx = squared_dist(x)
        sdx = sdx / tf.reduce_mean(sdx)
        sdz = squared_dist(z_x)
        sdz = sdz / tf.reduce_mean(sdz)
        return tf.reduce_mean(tf.square(tf.log(tf.constant(1.) + sdx) - (tf.log(tf.constant(1.) + sdz))))
    
    def distance_loss_true(self, x, z_x):
        """ Loss based on the distance between elements in a batch
        """
        sdx = squared_dist(x)
        sdz = squared_dist(z_x)
        return tf.reduce_mean(tf.abs(sdz - sdx))
    
    

        
    def corruptDf(self, df):
        def corruptSurface(obs, nbCorruptedSurfaces, maxNbCorruption):
            uncorruptedDfList = [obs]*nbCorruptedSurfaces
            nbCorruptions = np.random.randint(maxNbCorruption, 
                                              size=len(uncorruptedDfList))
            
            def corruptionProcess(obs, nbCorruptions):
                orderedIndexes = np.arange(obs.shape[0])
                np.random.shuffle(orderedIndexes)
                corruptedIndexes = obs.index[orderedIndexes[:nbCorruptions]]
                return obs.mask( obs.index.isin(corruptedIndexes) , other=0.0)
            
            corruptedDfs = pd.concat(list(map(lambda x : corruptionProcess(*x), 
                                              zip(uncorruptedDfList, nbCorruptions) )), 
                                     axis=1)
            return corruptedDfs
        funcCorrupt = lambda x : corruptSurface(x,
                                                self.hyperParameters["nbCorruptedSurfaces"],
                                                self.hyperParameters["nbCorruptionMax"])
        
        return pd.concat(df.apply(funcCorrupt, axis=1).values, axis=1).transpose()
    
    
    def repeatDf(self, df):
        def repeatSurface(obs, nbCorruptedSurfaces, maxNbCorruption):
            uncorruptedDfList = [obs]*nbCorruptedSurfaces
            
            corruptedDfs = pd.concat(uncorruptedDfList, 
                                     axis=1)
            return corruptedDfs
        funcRepeat = lambda x : repeatSurface(x, 
                                              self.hyperParameters["nbCorruptedSurfaces"], 
                                              self.hyperParameters["nbCorruptionMax"])
        return pd.concat(df.apply(funcRepeat, axis=1).values, axis=1).transpose()
    
    #Create a feed_dict (see tensorflow documentation) for evaluating encoder
    #args : List dataset to feed, order meaning is proper to each model 
    # def createFeedDictEncoder(self, dataSetList):
        # feedDict = {self.inputTensor : dataSetList[0]}
        # if len(dataSetList)> 1 :#Add corrupted Data
            # feedDict[self.corruptedTensor] = dataSetList[1]
        # else : 
            # feedDict[self.corruptedTensor] = dataSetList[0]
        # return feedDict
    
    #Sample Mini-batch
    def generateMiniBatches(self, dataSet, nbEpoch):
        batchSize = 1000
        return self.selectMiniBatchWithoutReplacement(dataSet, batchSize)
    
    # def gradientDescent(self, 
                        # session, 
                        # datasetTrain, 
                        # nbEpoch, 
                        # datasetTest, 
                        # trainingLoss, 
                        # gradientStep, 
                        # validationLoss):
        # corruptedDatasetTrain = [self.repeatDf(datasetTrain[0])]#, self.corruptDf(datasetTrain[0])]
        # corruptedDatasetTest = None
        # if datasetTest is not None :
            # corruptedDatasetTest = [self.repeatDf(datasetTest[0])]#, self.corruptDf(datasetTest[0])]
        # return super().gradientDescent(session,
                                       # corruptedDatasetTrain,
                                       # nbEpoch,
                                       # corruptedDatasetTest,
                                       # trainingLoss,
                                       # gradientStep,
                                       # validationLoss)
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
        corruptedLayer = self.corruptTensor(inputToReproduce)
        stepFactorLayer = corruptedLayer
        for layerFactory in stepEncoderLayer :
            stepFactorLayer = layerFactory(stepFactorLayer)
        
        stepOutputLayer = stepFactorLayer
        for layerFactory in stepDecoderLayer :
            stepOutputLayer = layerFactory(stepOutputLayer)
        
        #Build Losses
        stepLoss = self.buildLoss( stepOutputLayer, 
                                   inputToReproduce, 
                                   "stepOutputLayer"+str(encoderLayerIndex),
                                   matrixNorm = False)
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
        
        corruptedDatasetTrain = [self.repeatDf(inputTrain[0])]#, self.corruptDf(datasetTrain[0])]
        corruptedDatasetTest = None
        if inputTest is not None :
            corruptedDatasetTest = [self.repeatDf(inputTest[0])]#, self.corruptDf(datasetTest[0])]
        self.gradientDescent(session, 
                             corruptedDatasetTrain, 
                             nbEpoch, 
                             corruptedDatasetTest, 
                             stepTotalLosses, 
                             stepTrainingOperator, 
                             stepLoss)
        layerFactor = session.run(stepFactorLayer, feed_dict=self.createFeedDictEncoder(inputTrain))
        #print("Factor layer", layerFactor[0,:])
        corruptedInput = session.run(corruptedLayer, feed_dict=self.createFeedDictEncoder(inputTrain))
        #print("corrupted layer", corruptedInput[0,:])
        corruptedOutput = session.run(stepOutputLayer, feed_dict=self.createFeedDictEncoder(inputTrain))
        #print("output layer", corruptedOutput[0,:])
        #Build input for next step
        newInputToReproduce = inputToReproduce
        for layerFactory in stepEncoderLayer :
            newInputToReproduce = layerFactory(newInputToReproduce)
        return newInputToReproduce
    
    def corruptTensor(self, tensor):
        shape = tf.shape(tensor)
        #nbFeatures = shape[1]
        #shape_x, shape_y = shape[0], shape[1]
        # nbCorruptedData = tf.random.uniform((),
                                            # maxval=shape_y / 2,
                                            # dtype=tf.dtypes.int32)
        # nbCorruptedData = tf.random.uniform([nbCorruptedData, ],
                                            # maxval=shape_y,
                                            # dtype=tf.dtypes.float32)
        # return res = tf.where(mask, tf.zeros_like(data), data)
        #mask=np.random.binomial(1, 1 - corruption_level,tensor.shape ) #mask with several zeros at certain position
        mask = tf.keras.backend.random_binomial(shape, 
                                                p = self.hyperParameters["CorruptionLevel"])
        return mask * tensor
        
























#Denoising autoencoder where only input layer received masked input during training 
class StackedAutoEncoderDenoised1(StackedAutoEncoderDenoised):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestStackedAEModelDenoised1"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
        
    
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
        corruptedLayer = self.corruptTensor(inputToReproduce) if (inputToReproduce==self.inputTensor) else inputToReproduce
        stepFactorLayer = corruptedLayer
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
        
        corruptedDatasetTrain = [self.repeatDf(inputTrain[0])]#, self.corruptDf(datasetTrain[0])]
        corruptedDatasetTest = None
        if inputTest is not None :
            corruptedDatasetTest = [self.repeatDf(inputTest[0])]#, self.corruptDf(datasetTest[0])]
        self.gradientDescent(session, 
                             corruptedDatasetTrain, 
                             nbEpoch, 
                             corruptedDatasetTest, 
                             stepTotalLosses, 
                             stepTrainingOperator, 
                             stepLoss)
        layerFactor = session.run(stepFactorLayer, feed_dict=self.createFeedDictEncoder(inputTrain))
        #print("Factor layer", layerFactor[0,:])
        corruptedInput = session.run(corruptedLayer, feed_dict=self.createFeedDictEncoder(inputTrain))
        #print("corrupted layer", corruptedInput[0,:])
        corruptedOutput = session.run(stepOutputLayer, feed_dict=self.createFeedDictEncoder(inputTrain))
        #print("output layer", corruptedOutput[0,:])
        return stepFactorLayer
        
    























class StackedAutoEncoderDisentanglement(StackedAutoEncoderOptimized):
    def __init__(self,
                 learningRate,
                 hyperParameters, 
                 nbUnitsPerLayer,
                 nbFactors,
                 modelName = "./bestStackedAEModelDisentanglement"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
        
        
    def buildPenalization(self,**kwargs):
        firstPenalizations = super().buildPenalization(**kwargs)
        if ("lambdaDisentangle" not in self.hyperParameters) :
            return firstPenalizations
        
        #Penalize unlikely factor values with respect to a gaussian distribution
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_mean(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)
        
        def log_gaussian_independant_vector_pdf(factors, raxis=1):
            exampleWiseScalarProduct = tf.reduce_mean(tf.square(factors), axis = raxis)
            return tf.reduce_mean(exampleWiseScalarProduct)
            
        
        penalization = None
        if len(kwargs)==0:
            #Standard case, training all layers
            penalization = log_gaussian_independant_vector_pdf(self.factorTensor)
            if self.verbose :
                print(penalization)
        else:
            #Training only some layers typically for layer wise learning
            penalization = log_gaussian_independant_vector_pdf(kwargs['factors'])
            if self.verbose :
                print(penalization)
        return firstPenalizations + [self.hyperParameters["lambdaDisentangle"] * penalization]






















#Train autoencoders by masking points as in completion problem
#Completion amounts then to evaluate the model
#See https://github.com/RaptorMai/Deep-AutoEncoder-Recommendation
class StackedAutoEncoderRecommender(StackedAutoEncoderOptimized):
    def __init__(self,
                 learningRate,
                 hyperParameters, 
                 nbUnitsPerLayer,
                 nbFactors,
                 modelName = "./bestStackedAEModelRecommender"):
        self.mask = hyperParameters["mask"]
        self.maskTensor = None
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
        
    def maskValues(self, unmaskedData):
        
        maskedValue = tf.zeros_like(unmaskedData)
        #tf.boolean_mask(unmaskData, mask, axis=1)
        
        return tf.transpose(tf.where(tf.transpose(self.maskTensor), 
                                     tf.transpose(maskedValue), 
                                     tf.transpose(unmaskedData)))
        
    def completeDataTensor(self, 
                           sparseSurface, 
                           initialValueForFactors, 
                           nbCalibrationStep, 
                           *args):
        reshapedSparseSurface = pd.DataFrame(np.reshape([sparseSurface.fillna(0.0)], 
                                                        (1,sparseSurface.shape[0])), 
                                             columns = sparseSurface.index)
        if self.verbose :
            print("Completion is assimilated to compression in our case")
        bestLoss, surface, factor = self.evalModel(reshapedSparseSurface)
        
        completedSurface = np.where(np.isnan(np.reshape([sparseSurface], 
                                                        (1,sparseSurface.shape[0]))), 
                                    surface.values,
                                    reshapedSparseSurface.values)
        bestSurface = pd.Series(np.ravel(completedSurface), index = surface.columns)
        bestFactor = np.ravel(factor.values)
        return bestLoss, bestFactor, bestSurface, pd.Series([bestLoss])
        
    def buildArchitecture(self):
        
        #Kernel initializer
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        # def positiveKernelInitializer(shape, dtype=None, partition_info=None):
            # return 1 + tf.abs(tf.contrib.layers.variance_scaling_initializer()(shape,dtype))

        #Regularizer
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        #Tensor for complete data 
        self.inputTensor = tf.placeholder(tf.float32, 
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#batch size along
        #Tensor for noisy data                                           
        # self.corruptedTensor = tf.placeholder(tf.float32,
                                              # shape=[None, self.nbUnitsPerLayer['Input Layer']])
        if self.verbose :
            print(self.inputTensor)
        
        #self.IsTraining = tf.placeholder_with_default(False, (),
        #                                              name="ActivateCorruption")
                                                 
        #inputLayer = self.corruptTensor(self.inputTensor) if self.IsTraining else self.inputTensor
        
        self.maskTensor = tf.Variable(self.mask.values,
                                      dtype=tf.bool)
        inputLayer = self.maskValues(self.inputTensor)
        
        #Layers 1
        hiddenEncoder1 = self.buildDenseLayer(self.nbUnitsPerLayer['LayerEncoder1'], 
                                              inputLayer,
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        if self.verbose :
            print(hiddenEncoder1)
        
        #Layer 2
        hiddenEncoder2 = self.buildDenseLayer(self.nbUnitsPerLayer['LayerEncoder2'], 
                                              hiddenEncoder1,
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        if self.verbose :
            print(hiddenEncoder2)
        
        #Layer 3
        hiddenEncoder3 = self.buildDenseLayer(self.nbUnitsPerLayer['LayerEncoder3'], 
                                              hiddenEncoder2,
                                              activation = tf.nn.softplus,
                                              kernelRegularizer = l2_regularizer,
                                              kernelInitializer = he_init)
        if self.verbose :
            print(hiddenEncoder3)

        #Layer 4 / Hidden Layer
        self.factorTensor = self.buildDenseLayer(self.nbFactors, 
                                                 hiddenEncoder3,
                                                 activation = None,
                                                 kernelRegularizer = l2_regularizer,
                                                 kernelInitializer = he_init)
        
        self.nbEncoderLayer = len(self.layers)
        # DECODE --------------------------------------------------------------------
        
        lastTensor = self.factorTensor
        for k in range(self.nbEncoderLayer):
            if self.verbose :
                print(lastTensor)
            lastTensor = self.buildInverseLayer(lastTensor)
        
        # if self.verbose :
            # print(lastTensor)
        # lastTensor = self.buildDenseLayer(self.nbUnitsPerLayer['Output Layer'], 
                                          # lastTensor,
                                          # kernelRegularizer = l2_regularizer,
                                          # kernelInitializer = he_init,
                                          # activation = None)
        
        self.outputTensor = lastTensor
        
        if self.verbose :
            print(self.outputTensor)
        return
        
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
        corruptedLayer = self.maskValues(inputToReproduce)  if (inputToReproduce==self.inputTensor) else inputToReproduce
        stepFactorLayer = corruptedLayer
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
        layerFactor = session.run(stepFactorLayer, feed_dict=self.createFeedDictEncoder(inputTrain))
        #print("Factor layer", layerFactor[0,:])
        corruptedInput = session.run(corruptedLayer, feed_dict=self.createFeedDictEncoder(inputTrain))
        #print("corrupted layer", corruptedInput[0,:])
        corruptedOutput = session.run(stepOutputLayer, feed_dict=self.createFeedDictEncoder(inputTrain))
        #print("output layer", corruptedOutput[0,:])
        return stepFactorLayer




















#Try to preserve relative distance for encodings w.r.t volatility surfaces relative distances
class StackedAutoEncoderTopology(StackedAutoEncoderOptimized):
    def __init__(self,
                 learningRate,
                 hyperParameters, 
                 nbUnitsPerLayer,
                 nbFactors,
                 modelName = "./bestStackedAEModelTopology"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    #Taken from https://github.com/timsainb/GAIA/blob/master/network.py
    def shape(self, tensor):
        """ get the shape of a tensor
        """
        s = tensor.get_shape()
        return tuple([s[i].value for i in range(0, len(s))])
        
    # def squared_dist(self, A):
        # """
        # Computes the pairwise distance between points
        # http://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
        # """
        # expanded_a = tf.expand_dims(A, 1)
        # expanded_b = tf.expand_dims(A, 0)
        # distances = tf.reduce_mean(tf.squared_difference(expanded_a, expanded_b), 2)
        # return distances
    
    def squared_dist(self, A):
        r = tf.reduce_sum(A*A, 1)
        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
        return D
        
    def distance_loss(self, x, z_x):
        """ Loss based on the distance between elements in a batch
        """
        #print([self.shape(z_x)[0], np.prod(self.shape(z_x)[1:])])
        #z_x = tf.reshape(z_x, [self.shape(z_x)[0], np.prod(self.shape(z_x)[1:])])
        sdx = self.squared_dist(x)
        sdx = sdx / tf.reduce_mean(sdx)
        sdz = self.squared_dist(z_x)
        sdz = sdz / tf.reduce_mean(sdz)
        return tf.reduce_mean(tf.square(tf.log(tf.constant(1.) + sdx) - (tf.log(tf.constant(1.) + sdz))))
    
    def distance_loss_true(self, x, z_x):
        """ Loss based on the distance between elements in a batch
        """
        sdx = self.squared_dist(x)
        sdz = self.squared_dist(z_x)
        return tf.reduce_mean(tf.abs(sdz - sdx))
        
    def buildPenalization(self,**kwargs):
        firstPenalizations = super().buildPenalization(**kwargs)
        if ("lambdaTopology" not in self.hyperParameters) :
            return firstPenalizations
        
        penalization = None
        if (len(kwargs)==0) :
            #Standard case, training all layers
            penalization = self.distance_loss(self.inputTensor, self.factorTensor)
            if self.verbose :
                print(penalization)
        else:
            #Training only some layers typically for layer wise learning
            penalization = self.distance_loss(kwargs['inputLayer'], kwargs['factors'])
            if self.verbose :
                print(penalization)
        return firstPenalizations + [self.hyperParameters["lambdaTopology"] * penalization]





















#Penalize completion with encodings from completed surfaces
class StackedAutoEncoderPenalizedCompletion(StackedAutoEncoderOptimized):
    def __init__(self,
                 learningRate,
                 hyperParameters, 
                 nbUnitsPerLayer,
                 nbFactors,
                 modelName = "./bestStackedAEModelDisentanglement"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    
    #Build a tensor that construct a surface from factors values
    def buildEncoderTensor(self, surfaceTensor):
        lastTensor = surfaceTensor
        for idxFactory in range(self.nbEncoderLayer):
            factoryTmp = self.layers[idxFactory]
            lastTensor = factoryTmp(lastTensor)
        
        return lastTensor
    
    def buildCompletionLoss(self, factorTensor, calibrationLoss, completedSurfaceTensor):
        previousPenalization = super().buildCompletionLoss(factorTensor, calibrationLoss, completedSurfaceTensor)
        completedEncodings = self.buildEncoderTensor(completedSurfaceTensor)
        
        finalCalibrationLoss = previousPenalization
        if "lambdaCompletionEncodings" in self.hyperParameters :
            encodingRegularization = tf.reduce_mean(self.buildReconstructionLoss(completedEncodings, 
                                                                                 factorTensor, 
                                                                                 "EncodingRegularization"))
            
            finalCalibrationLoss += self.hyperParameters["lambdaCompletionEncodings"] * encodingRegularization
        return finalCalibrationLoss