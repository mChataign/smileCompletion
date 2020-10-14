
import pandas as pd
import numpy as np
import tensorflow as tf
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

import tensorflowModel


class ShallowAutoEncoder(tensorflowModel.tensorflowModel):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestShallowAEModel"):
        
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    def getVariableFromTensor(self, tensorList):
        tensorNameList = [k.name for k in tensorList]
        return [k for k in tf.trainable_variables() if (k.name in tensorNameList)]
        
        
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
        
        self.factorTensor = self.buildDenseLayer(self.nbFactors, 
                                                 self.inputTensor,
                                                 activation = tf.nn.softplus,
                                                 kernelRegularizer = l2_regularizer,
                                                 kernelInitializer = he_init)
        if self.verbose :
            print(self.factorTensor)
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
    
    
    def buildPenalization(self, **kwargs):
        #Build ridge penalization on kernel weights
        if len(kwargs)==0:
            #Standard case, training all layers
            return tf.losses.get_regularization_losses()
        else:
            #Training only some layers
            l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
            return [l2_regularizer(k) for k in kwargs['layersKernels']]
    
