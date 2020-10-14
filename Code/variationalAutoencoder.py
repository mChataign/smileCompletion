
import pandas as pd
import numpy as np
import tensorflow as tf
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

import tensorflowModel
import shallowAutoencoder
import convolutionAutoencoder
#import plottingTools






class VariationalAutoEncoder(convolutionAutoencoder.ConvolutionalAutoEncoderDeep):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestVariationalAEModel"):
        self.nbSimulation = hyperParameters["nbSimulations"] if "nbSimulations" in hyperParameters else 1
        self.sample = None
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)    
        self.batchSize = 100

    def buildPenalization(self,**kwargs):
        firstPenalizations = [] #super().buildPenalization(**kwargs)
        #Penalize unlikely factor values with respect to a gaussian distribution
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_mean(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)
            
        def entropyPenalization(sample, mean, logvar, raxis=1):
            # reshapedSample = tf.reshape(sample,
                                        # shape=[-1, self.nbSimulation, self.nbFactors])
            # expandedMean = tf.expand_dims(mean,1)
            # expandedVar = tf.expand_dims(logvar,1)
            
            # log_pz = log_normal_pdf(reshapedSample,0.0,0.0,raxis)
            # log_qzx = log_normal_pdf(reshapedSample,expandedMean,expandedVar,raxis)
            # batchPenalization = log_qzx-log_pz
            batchPenalization = 0.5 * tf.reduce_mean(tf.exp(logvar) + tf.square(mean) - 1.0 - logvar, axis=1)
            if self.verbose :
                print("batchPenalization ",batchPenalization)
            return tf.reduce_mean(batchPenalization)
            
        penalization = None
        if len(kwargs)==0:
            #Standard case, training all layers
            penalization = entropyPenalization(self.sample, self.mean, self.logVar)
            if self.verbose :
                print(penalization)
        else:
            #Training only some layers typically for layer wise learning
            activateLatentPenalization = (kwargs["activateLatentPenalization"] if ("activateLatentPenalization" in kwargs) else False)
            if activateLatentPenalization : 
                mean = kwargs["mean"]
                logVar = kwargs["logVar"]
                sample = kwargs["sample"]
                penalization = entropyPenalization(sample, mean, logVar)
            else : 
                return firstPenalizations
        return firstPenalizations + [penalization]
            
    def reparametrize(self, codings):
        mean, logVar = tf.split(codings, num_or_size_splits=2, axis=1)
        
        #expandedMean = tf.expand_dims(mean,1)
        #expandedVar = tf.expand_dims(logVar,1)
        
        #simulate sample for each day
        #outputShape = [tf.shape(expandedMean)[0], self.nbSimulation, self.nbFactors]
        noise = tf.random_normal(tf.shape(mean), dtype = tf.float32) # shape (batch size, nbSimul, nbFactor/2)
        # simulatedLatentVariable = tf.reshape(noise * tf.exp(0.5 * expandedVar) + expandedMean, 
                                             # shape=[-1, self.nbFactors])
        simulatedLatentVariable = noise * tf.exp(0.5 * logVar) + mean
        return simulatedLatentVariable, mean, logVar
        
    def buildArchitecture(self):
        self.activatePooling = True
        
        # coding part 
        self.inputTensor = tf.placeholder(tf.float32, 
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#bacth size along 
        #80
        inputReshaped = self.buildFoldLayer(self.inputTensor,
                                            [-1, self.hyperParameters['nbX'], 
                                             self.hyperParameters['nbY'], 
                                             self.hyperParameters['nbChannel']]) 
        if self.verbose :
            print(inputReshaped)
        
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        
        # encode
        
        conv1, pool1 = self.buildEncodingConvolutionLayer(inputReshaped, 
                                                          [3, 3, 1, 9], 
                                                          'SAME',
                                                          activation = tf.nn.relu)
        if self.verbose :
            print(conv1)
            print(pool1)
        
        unfold = self.buildUnfoldLayer(pool1)
        if self.verbose :
            print(unfold)
        
        
        self.codings = self.buildDenseLayer(self.nbFactors*2,
                                            unfold,
                                            activation = tf.nn.relu,
                                            kernelRegularizer = l2_regularizer,
                                            kernelInitializer = he_init)
        
        self.nbEncoderLayer = len(self.layers)
        # DECODE --------------------------------------------------------------------
        self.sample, self.mean, self.logVar = self.reparametrize(self.codings)
        self.factorTensor = self.sample
        
        
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
        
        
        # print(lastTensor)
        # self.outputSimulatedTensor = tf.reshape(lastTensor, 
                                                # shape=[-1, self.nbSimulation, self.nbUnitsPerLayer['Output Layer']])
        
        
        # self.outputTensor = tf.reduce_mean(self.outputSimulatedTensor, axis=1)#Average over simulation dimension
        
        self.outputTensor = lastTensor
        
        if self.verbose :
            print(self.outputTensor)
        return
    
    def createFeedDictSimul(self, randomVariables):
        feedDict = {self.sample : randomVariables}
        return feedDict
    
    def simulateOutputFromLatentVariables(self, randomLatentVariables, gridPoints):
        reshapedRandomVariables = np.reshape(randomLatentVariables,(-1,self.nbFactors))
        self.restoringGraph()
        
        #Opening session for evaluation
        with tf.Session() as sess :
            #Restoring Model
            sess.run(self.init)
            self.restoreWeights(sess)
            
            simulatedOutputs = self.outputTensor.eval(feed_dict=self.createFeedDictSimul(reshapedRandomVariables), 
                                                      session = sess)
        
        reshapedOutputs =  np.reshape(simulatedOutputs,
                                      (reshapedRandomVariables.shape[0],self.nbUnitsPerLayer['Output Layer']))
        return pd.DataFrame(reshapedOutputs,columns=gridPoints)
        

class DeepVariationalAutoEncoder(VariationalAutoEncoder):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestDeepVariationalAEModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    def buildArchitecture(self):
        self.activatePooling = False
        # Three layers a la LeNet
        
        # coding part 
        self.inputTensor = tf.placeholder(tf.float32, 
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#bacth size along
        if self.verbose :
            print(self.inputTensor)
        #Vector 80
        inputReshaped = self.buildFoldLayer(self.inputTensor,
                                            [-1, self.hyperParameters['nbX'], 
                                             self.hyperParameters['nbY'], 
                                             self.hyperParameters['nbChannel']]) 
        if self.verbose :
            print(inputReshaped)
        
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        #Tensor 10,8,1
        
        
        # ENCODE ------------------------------------------------------------------
            
        conv1, pool1 = self.buildEncodingConvolutionLayer(inputReshaped, 
                                                          [4, 4, 1, 3], 
                                                          'VALID')
        if self.verbose :
            print(conv1)
            print(pool1)
        #6,5,9
        
        
        conv2, pool2 = self.buildEncodingConvolutionLayer(pool1, 
                                                          [3, 3, 3, 9], 
                                                          'VALID')
        if self.verbose :
            print(conv2)
            print(pool2)
        #3,3,81
        
        
        conv3, pool3 = self.buildEncodingConvolutionLayer(pool2, 
                                                          [3, 3, 9, 27], 
                                                          'VALID')
        if self.verbose :
            print(conv3)
            print(pool3)
        #1,1,81
        
        unfold = self.buildUnfoldLayer(pool3)
        if self.verbose :
            print(unfold)
        #unfold = self.layers[-1](pool2)
        
        self.codings = self.buildDenseLayer(self.nbFactors*2, 
                                            unfold,
                                            kernelRegularizer = l2_regularizer,
                                            kernelInitializer = he_init,
                                            activation = None)
        
        self.nbEncoderLayer = len(self.layers)
        # DECODE --------------------------------------------------------------------
        self.sample, self.mean, self.logVar = self.reparametrize(self.codings)
        self.factorTensor = self.sample
        
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
        return
    
    
    #######################################################################################################
    #Training functions
    #######################################################################################################
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
    
    #Stacked VAE
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
        
        stepOutputLayer = None
        if encoderLayerIndex==(self.nbEncoderLayer-1) : #only the most inner layer pair is trained with ELBO criteria
            stepSample, stepMean, stepVar = self.reparametrize(stepFactorLayer)
            stepOutputLayer = stepSample
        else : 
            stepOutputLayer = stepFactorLayer
                
        for layerFactory in stepDecoderLayer :
            stepOutputLayer = layerFactory(stepOutputLayer)
        
        #Build Losses
        stepLoss = self.buildLoss( stepOutputLayer, 
                                   inputToReproduce, 
                                   "stepOutputLayer"+str(encoderLayerIndex), 
                                   matrixNorm = False )
        stepPenalization = []
        if encoderLayerIndex==(self.nbEncoderLayer-1) :
            kwargsPenalization = {"layersKernels" : trainableVariableList[0::2], #kernels are located at even indices
                                  "outputLayer" : stepOutputLayer,
                                  "inputLayer" : inputToReproduce,
                                  "factors" : stepFactorLayer,
                                  "mean" : stepMean,
                                  "logVar" : stepVar,
                                  "sample" : stepSample,
                                  "activateLatentPenalization" : True}
            
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


def runSimulation(variationalModel, nbSimulation, fps, gridPoints):
    mu = np.ones(variationalModel.nbFactors)
    cov = np.eye(variationalModel.nbFactors)
    randomLatentVariables = np.random.multivariate_normal(mu, cov, nbSimulation)
    simulatedSurfaces = variationalModel.simulateOutputFromLatentVariables(randomLatentVariables, gridPoints)
    
    #anim = plottingTools.animationSingleVolatility(simulatedSurfacesDf, fps)
    #plt.show()
    
    return simulatedSurfaces

class AlternatingAutoencoder(shallowAutoencoder.ShallowAutoEncoder):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestAlternatingAEModel"):
        
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
        
        
    def buildArchitecture(self):
        #self.IsTraining = tf.placeholder_with_default(False, shape=(), name='IsTraining')
        
        self.inputTensor = tf.placeholder(tf.float32, 
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#bacth size along
        if self.verbose :
            print(self.inputTensor)
        
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        
        self.hiddenEncoderLayer = self.buildDenseLayer(self.nbUnitsPerLayer['Input Layer'], 
                                                       self.inputTensor,
                                                       activation = tf.nn.softplus,
                                                       kernelRegularizer = None,
                                                       kernelInitializer = he_init)
        
        self.factorTensor = self.buildDenseLayer(self.nbFactors, 
                                                 self.hiddenEncoderLayer,
                                                 activation = None,
                                                 kernelRegularizer = None,
                                                 kernelInitializer = he_init)
        if self.verbose :
            print(self.factorTensor)
        self.nbEncoderLayer = len(self.layers)
        # DECODE --------------------------------------------------------------------
        
        # lastTensor = self.factorTensor
        # for k in range(self.nbEncoderLayer):
            # if self.verbose :
                # print(lastTensor)
            # lastTensor = self.buildInverseLayer(lastTensor)
        
        self.hiddenDecoderLayer = self.buildDenseLayer(self.nbUnitsPerLayer['Input Layer'], 
                                                       self.factorTensor,
                                                       activation = tf.nn.softplus,
                                                       kernelRegularizer = None,
                                                       kernelInitializer = he_init)
        
        self.outputTensor = self.buildDenseLayer(self.nbUnitsPerLayer['Input Layer'], 
                                                 self.hiddenDecoderLayer,
                                                 activation = None,
                                                 kernelRegularizer = None,
                                                 kernelInitializer = he_init)
        
        
        # if self.verbose :
            # print(lastTensor)
        # lastTensor = self.buildDenseLayer(self.nbUnitsPerLayer['Output Layer'], 
                                          # lastTensor,
                                          # kernelRegularizer = l2_regularizer,
                                          # kernelInitializer = he_init,
                                          # activation = None)
        
        #self.outputTensor = lastTensor
        
        if self.verbose :
            print(self.outputTensor)
        return
    
    
    # def buildPenalization(self, **kwargs):
        #Build ridge penalization on kernel weights
        # if len(kwargs)==0:
            #Standard case, training all layers
            # return tf.losses.get_regularization_losses()
        # else:
            #Training only some layers
            # l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
            # return [l2_regularizer(k) for k in kwargs['layersKernels']]
    
    
    
    #Train the factorial model
    def trainWithSession(self, session, inputTrain, nbEpoch, inputTest = None):
        start = time.time()
        if self.verbose :
            print("Calibrate model on training data and return testing loss per epoch")
        session.run(self.init)
        
        #First train encoder layers to generate meaningful latent variables
        epochLosses,_ = self.gradientDescent(session, 
                                             inputTrain, 
                                             nbEpoch, 
                                             inputTest, 
                                             self.EncoderLoss, 
                                             self.trainingEncoderOperator, 
                                             self.EncoderLoss)
        
        nbIter = 1
        for k in range(nbIter): 
            #Train decoding layer to reconstruct original layers
            epochLosses,_ = self.gradientDescent(session, 
                                                 inputTrain, 
                                                 int(nbEpoch/nbIter), 
                                                 inputTest, 
                                                 self.DecoderLoss, 
                                                 self.trainingDecoderOperator, 
                                                 self.DecoderLoss)
            
            #Train Encoding layer w.r.t reconstruction loss
            epochLosses,_ = self.gradientDescent(session, 
                                                 inputTrain, 
                                                 int(nbEpoch/nbIter), 
                                                 inputTest, 
                                                 self.loss, 
                                                 self.trainingEncoderOperator2, 
                                                 self.reducedReconstructionLoss)
        
        print("Detailed performances")
        if inputTest is not None : 
            feed_dict = self.createFeedDictEncoder(inputTest if inputTest is not None else inputTrain)
            totalLoss = session.run(self.loss ,
                                    feed_dict=feed_dict)
            encodingLoss  = session.run(self.encodingDistributionLoss ,
                                        feed_dict=feed_dict)
            pairWiseDistance  = session.run(self.pairWiseDistance ,
                                            feed_dict=feed_dict)
            reconstructionError = session.run(self.reducedReconstructionLoss ,
                                              feed_dict=feed_dict)
                                        
            print("Penalized loss on testing dataset : ", totalLoss)
            print("Reconstruction loss on testing dataset : ", reconstructionError)
            print("Distribution loss on testing encodings : ", encodingLoss)
            print("Preserving topology loss on testing encodings : ", pairWiseDistance)
            print("Training time : % 5d" %(time.time() -  start))
        return epochLosses
    
    #KL divergence with gaussian vector centered and scaled
    def gaussianDivergence(self):
        mean = tf.reduce_mean(self.factorTensor, axis=0, keep_dims=True)
        meanSquare = tf.matmul(mean, mean, transpose_a = True)
        latentVariableSquare = tf.divide(tf.matmul(self.factorTensor, self.factorTensor, transpose_a = True), 
                                         tf.cast(tf.shape(self.factorTensor)[0], tf.float32))
        cov = latentVariableSquare - meanSquare
        
        #meanRef = tf.zeros_like(mean)
        #covRef = tf.eye(tf.shape(latentVariable)[0])
        
        divergence = 0.5 * (tf.matmul( mean, mean, transpose_b = True) - 
                            tf.log(tf.linalg.det(cov)) + tf.linalg.trace(cov) )
        return divergence
    
    #Taken from https://github.com/timsainb/GAIA/blob/master/network.py
    def shape(self, tensor):
        """ get the shape of a tensor
        """
        s = tensor.get_shape()
        return tuple([s[i].value for i in range(0, len(s))])
    
    def squared_dist(self, A):
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
        z_x = tf.reshape(z_x, [self.shape(z_x)[0], np.prod(self.shape(z_x)[1:])])
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
    
    #Soft constraint to obtain positive definitive hessian
    def convexityPenalization(self):
        batchHessians =  tf.reshape(tf.Hessians(self.reconstructionLoss, self.factorTensor), 
                                    [-1, self.nbFactors, self.nbFactors])
        smallestEigenValue = tf.reduce_min(tf.linalg.eigh(batchHessians), axis=1)
        return tf.reduce_mean(tf.nn.relu(- smallestEigenValue)) 
    
    def wassersteinGaussianDistance(self):
        def symsqrt(mat, eps=1e-7):
            """Symmetric square root. https://github.com/tensorflow/tensorflow/issues/9202"""
            s, u, v = tf.svd(mat)
            # sqrt is unstable around 0, just use 0 in such case
            si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
            return u @ tf.diag(si) @ tf.transpose(v)
        mean = tf.reduce_mean(self.factorTensor, axis=0, keep_dims=True)
        meanInnerProduct = tf.matmul(mean, mean, transpose_b = True)
        meanOuterProduct = tf.matmul(mean, mean, transpose_a = True)
        latentVariableSquare = tf.divide(tf.matmul(self.factorTensor, self.factorTensor, transpose_a = True), 
                                         tf.cast(tf.shape(self.factorTensor)[0], tf.float32))
        cov = latentVariableSquare - meanOuterProduct
        
        #meanRef = tf.zeros_like(mean)
        covRef = tf.eye(tf.shape(self.factorTensor)[1])
        
        #return (meanInnerProduct + tf.linalg.trace(cov + covRef - 2 * tf.linalg.sqrtm(cov)))
        return (meanInnerProduct + tf.linalg.trace(cov + covRef - 2 * symsqrt(cov)))
    
    #Build the architecture, losses and optimizer.
    def buildModel(self):
        super().buildModel()
        self.pairWiseDistance = self.distance_loss_true(self.inputTensor, self.factorTensor) 
        self.encodingDistributionLoss = self.hyperParameters["GaussianEncodings"] * self.gaussianDivergence()
        #self.encodingDistributionLoss = self.hyperParameters["GaussianEncodings"] * self.wassersteinGaussianDistance()
        self.EncoderLoss = (self.pairWiseDistance + self.encodingDistributionLoss)
        
        self.DecoderLoss = self.loss
        
        var_list_encoder = []
        var_list_decoder = []
        for k in range(len(self.layers)):
            if k < self.nbEncoderLayer : 
                var_list_encoder += self.layers[k].getTrainableVariables()
            else :
                var_list_decoder += self.layers[k].getTrainableVariables()
        self.trainingEncoderOperator = self.optimizer.minimize(self.EncoderLoss, 
                                                               var_list = var_list_encoder,
                                                               name="trainingEncoderOperator")
        self.trainingEncoderOperator2 = self.optimizer.minimize(self.loss, 
                                                                var_list = var_list_encoder,
                                                                name="trainingEncoderOperator2")
        self.trainingDecoderOperator = self.optimizer.minimize(self.DecoderLoss, 
                                                               var_list = var_list_decoder,
                                                               name="trainingDecoderOperator")
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(name="saver", 
                                    save_relative_paths=True, 
                                    filename = self.metaModelName) 
        return
    
    #Build a tensor that construct a surface from factors values
    def buildEncoderTensor(self, surfaceTensor):
        lastTensor = surfaceTensor
        for idxFactory in range(self.nbEncoderLayer):
            factoryTmp = self.layers[idxFactory]
            lastTensor = factoryTmp(lastTensor)
        
        return lastTensor
    
        #Build a tensor that construct a surface from factors values
    def buildAutoEncoderTensor(self, surfaceTensor):
        lastTensor = surfaceTensor
        for idxFactory in range(len(self.layers)):
            factoryTmp = self.layers[idxFactory]
            lastTensor = factoryTmp(lastTensor)
        
        return lastTensor
    
    def buildCompletionLoss(self, factorTensor, calibrationLoss, completedSurfaceTensor):
        previousPenalization = super().buildCompletionLoss(factorTensor, calibrationLoss, completedSurfaceTensor)
        #completedEncodings = self.buildEncoderTensor(completedSurfaceTensor)
        reconstructedSurface = self.buildAutoEncoderTensor(completedSurfaceTensor)
        
        finalCalibrationLoss = previousPenalization
        if "lambdaCompletionEncodings" in self.hyperParameters :
            outlierRegularization = tf.reduce_mean(self.buildReconstructionLoss(completedSurfaceTensor, 
                                                                                reconstructedSurface, 
                                                                                "EncodingRegularization"))
            
            finalCalibrationLoss += self.hyperParameters["lambdaCompletionEncodings"] * outlierRegularization
        return finalCalibrationLoss
