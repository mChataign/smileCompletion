
import pandas as pd
import numpy as np
import tensorflow as tf
import dask
import scipy
import time

from functools import partial
from abc import ABCMeta, abstractmethod

import tensorflowModel
import stackedAutoencoder




class Convolution2D(tensorflowModel.Layer):
    """

    """
    def __init__(self,
                 kernel_shape,
                 kernel=None,
                 bias=None,
                 strides=(1, 1, 1, 1),
                 padding='SAME',
                 activation=None,
                 scope=''):
        tensorflowModel.Layer.__init__(self)
        
        self.layerType = "Convolution2D"
        self.kernel_shape = kernel_shape
        self.kernel = kernel
        self.bias = bias
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope

    def build(self, input_tensor):
        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            self.kernel = tf.Variable(tf.truncated_normal(self.kernel_shape, stddev=0.1), name='kernel')

        # build bias
        kernel_height, kernel_width, num_input_channels, num_output_channels = self.kernel.get_shape()
        if self.bias:
            assert self.bias.get_shape() == (num_output_channels, )
        else:
            self.bias = tf.Variable(tf.constant(0.0, shape=[num_output_channels]), name='bias')

        # convolution
        conv = tf.nn.conv2d(input_tensor, self.kernel, strides=self.strides, padding=self.padding)

        # activation
        if self.activation:
            return self.activation(conv + self.bias)
        return conv + self.bias

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
    
    def isTrainable(self):
        return True
    
    def getTrainableVariables(self):
        return [self.kernel, self.bias]
    
    def copy(self):
        return Convolution2D(self.kernel_shape, 
                             kernel = self.kernel, 
                             bias = self.bias,
                             strides = self.strides,
                             padding = self.padding,
                             activation= self.activation,
                             scope= self.scope)


class DeConvolution2D(tensorflowModel.Layer):
    """

    """
    def __init__(self,
                 kernel_shape,
                 outputTensor,
                 kernel=None,
                 bias=None,
                 strides=(1, 1, 1, 1),
                 padding='SAME',
                 activation=None,
                 scope=''):
        tensorflowModel.Layer.__init__(self)
        
        self.layerType = "DeConvolution2D"
        self.kernel_shape = kernel_shape
        self.output_static_shape = outputTensor.get_shape().as_list()
        self.output_shape = tf.shape(outputTensor)
        self.kernel = kernel
        self.bias = bias
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope

    def build(self, input_tensor):
        # build kernel
        if self.kernel:
            assert self.kernel.get_shape() == self.kernel_shape
        else:
            self.kernel = tf.Variable(tf.truncated_normal(self.kernel_shape, stddev=0.1), name='kernel')

        # build bias
        window_height, window_width, num_output_channels, num_input_channels = self.kernel.get_shape()
        if self.bias:
            assert self.bias.get_shape() == (num_output_channels, )
        else:
            self.bias = tf.Variable(tf.constant(0.0, shape=[num_output_channels]), name='bias')
        
        dyn_input_shape = tf.shape(input_tensor)
        static_input_shape = input_tensor.get_shape().as_list()
        batch_size = dyn_input_shape[0]
        
        assert self.padding in {'SAME', 'VALID'}
        if (self.padding is 'SAME'):
            out_h = static_input_shape[1] * self.strides[1]
            out_w = static_input_shape[2] * self.strides[2]
        elif (self.padding is 'VALID'):
            out_h = (static_input_shape[1] - 1) * self.strides[1] + window_height
            out_w = (static_input_shape[2] - 1) * self.strides[2] + window_width
        
        dyn_output_shape = tf.stack([batch_size,
                                     out_h,
                                     out_w, 
                                     num_output_channels])
        
        # convolution
        deconv = tf.nn.conv2d_transpose(input_tensor,
                                        self.kernel,
                                        output_shape=dyn_output_shape,
                                        strides=self.strides,
                                        padding=self.padding)
        
        # activation
        if self.activation:
            return self.activation(deconv + self.bias)
        return deconv + self.bias

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
    
    def isTrainable(self):
        return True
    
    def getTrainableVariables(self):
        return [self.kernel, self.bias]
    
    def copy(self):
        tmp_shape = self.output_static_shape
        tmp_shape[0] = 1
        return DeConvolution2D(self.kernel_shape,
                               tf.constant(0, shape=tmp_shape),
                               kernel = self.kernel, 
                               bias = self.bias,
                               strides = self.strides,
                               padding = self.padding,
                               activation= self.activation,
                               scope= self.scope)

class MaxPooling(tensorflowModel.Layer):
    """

    """
    def __init__(self,
                 kernel_shape,
                 strides,
                 padding,
                 scope=''):
        tensorflowModel.Layer.__init__(self)
        
        self.layerType = "MaxPooling"
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding
        self.scope = scope

    def build(self, input_tensor):
        return tf.nn.max_pool(input_tensor, ksize=self.kernel_shape, strides=self.strides, padding=self.padding)

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
    
    def isTrainable(self):
        return False
    
    def getTrainableVariables(self):
        return []
    
    def copy(self):
        return MaxPooling(self.kernel_shape,
                          self.strides,
                          self.padding, 
                          scope = self.scope)
    def getUnpoolKernelShape(self):
        return (self.kernel_shape[1], self.kernel_shape[2])


class AvgPooling(tensorflowModel.Layer):
    """

    """
    def __init__(self,
                 kernel_shape,
                 strides,
                 padding,
                 scope=''):
        tensorflowModel.Layer.__init__(self)
        
        self.layerType = "AvgPooling"
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding
        self.scope = scope

    def build(self, input_tensor):
        return tf.nn.avg_pool(input_tensor, ksize=self.kernel_shape, strides=self.strides, padding=self.padding)

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
    
    def isTrainable(self):
        return False
    
    def getTrainableVariables(self):
        return []
    
    def copy(self):
        return AvgPooling(self.kernel_shape,
                          self.strides,
                          self.padding, 
                          scope = self.scope)
    def getUnpoolKernelShape(self):
        return (self.kernel_shape[1], self.kernel_shape[2])


class UnPooling(tensorflowModel.Layer):
    """
    Unpool a max-pooled layer.

    Currently this method does not use the argmax information from the previous pooling layer.
    Currently this method assumes that the size of the max-pooling filter is same as the strides.

    Each entry in the pooled map would be replaced with an NxN kernel with the original entry in the upper left.
    For example: a 1x2x2x1 map of

        [[[[1], [2]],
          [[3], [4]]]]

    could be unpooled to a 1x4x4x1 map of

        [[[[ 1.], [ 0.], [ 2.], [ 0.]],
          [[ 0.], [ 0.], [ 0.], [ 0.]],
          [[ 3.], [ 0.], [ 4.], [ 0.]],
          [[ 0.], [ 0.], [ 0.], [ 0.]]]]
    """
    def __init__(self,
                 kernel_shape,
                 outputTensor,
                 scope=''):
        tensorflowModel.Layer.__init__(self)
        
        self.layerType = "UnPooling"
        self.kernel_shape = kernel_shape
        self.output_static_shape = outputTensor.get_shape().as_list()
        self.output_shape = tf.shape(outputTensor)
        self.scope = scope
        self.padding = 'VALID'

    def build(self, input_tensor):
        num_channels = input_tensor.get_shape()[-1]
        input_dtype_as_numpy = input_tensor.dtype.as_numpy_dtype()
        kernel_rows, kernel_cols = self.kernel_shape

        # build kernel
        kernel_value = np.zeros((kernel_rows, kernel_cols, num_channels, num_channels), dtype=input_dtype_as_numpy)
        kernel_value[0, 0, :, :] = np.eye(num_channels, num_channels)
        kernel = tf.constant(kernel_value)
        
        dyn_input_shape = tf.shape(input_tensor)
        static_input_shape = input_tensor.get_shape().as_list()
        batch_size = dyn_input_shape[0]
        
        if (self.padding is 'SAME'):
            out_h = self.output_static_shape[1]#static_input_shape[1] * kernel_rows
            out_w = self.output_static_shape[2]#static_input_shape[2] * kernel_cols
        elif (self.padding is 'VALID'):
            out_h = self.output_static_shape[1]#(static_input_shape[1] - 1) * kernel_rows + window_height
            out_w = self.output_static_shape[2]#(static_input_shape[2] - 1) * kernel_cols + window_width
        dyn_output_shape = tf.stack([batch_size,
                                     out_h,
                                     out_w, 
                                     num_channels])
        
        # do the un-pooling using conv2d_transpose
        unpool = tf.nn.conv2d_transpose(input_tensor,
                                        kernel,
                                        output_shape=dyn_output_shape,
                                        strides=(1, kernel_rows, kernel_cols, 1),
                                        padding='VALID')
        # TODO test!!!
        return unpool

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
    
    def isTrainable(self):
        return False
    
    def getTrainableVariables(self):
        return []
    
    def copy(self):
        tmp_shape = self.output_static_shape
        tmp_shape[0] = 1
        return UnPooling(self.kernel_shape,
                         tf.constant(0, shape=tmp_shape),
                         scope = self.scope)

class Unfold(tensorflowModel.Layer):
    """

    """
    def __init__(self,
                 scope=''):
        tensorflowModel.Layer.__init__(self)
        
        self.layerType = "Unfold"
        self.scope = scope

    def build(self, input_tensor):
        num_batch, height, width, num_channels = input_tensor.get_shape()

        return tf.reshape(input_tensor, [-1, (height * width * num_channels).value])

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
    
    def isTrainable(self):
        return False
    
    def getTrainableVariables(self):
        return []
    
    def copy(self):
        return Unfold(scope = self.scope)

class Fold(tensorflowModel.Layer):
    """

    """
    def __init__(self,
                 fold_shape,
                 scope=''):
        tensorflowModel.Layer.__init__(self)
        
        self.layerType = "Fold"
        self.fold_shape = fold_shape
        self.scope = scope

    def build(self, input_tensor):
        return tf.reshape(input_tensor, self.fold_shape)

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
    
    def isTrainable(self):
        return False
    
    def getTrainableVariables(self):
        return []
    
    def copy(self):
        return Fold(self.fold_shape, 
                    scope = self.scope)


class ConvolutionalAutoEncoder(stackedAutoencoder.StackedAutoEncoder):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestConvolutionalAEModel"):
        
        self.activatePooling = True
        
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)    

    def buildConvolutionLayer(self,
                              inputTensor,
                              convKernel,
                              poolKernel,
                              poolStride,
                              padding,
                              activation = tf.nn.softplus,
                              convStride=[1,1,1,1]):
        self.layers.append(Convolution2D(convKernel, 
                                         activation=activation,
                                         scope='conv_' + str(len(self.layers)),
                                         padding = padding,
                                         strides=convStride))
        conv = self.layers[-1](inputTensor)
        if(self.activatePooling):
            self.layers.append(MaxPooling(kernel_shape=poolKernel, 
                                          strides=poolStride, 
                                          padding=padding,#'SAME', 
                                          scope='pool_' + str(len(self.layers))))
            pool = self.layers[-1](conv)
        else:
            pool = conv
        return conv, pool
    
    
    def buildUnfoldLayer(self, inputTensor):
        self.layers.append(Unfold(scope='unfold_' + str(len(self.layers))))
        return self.layers[-1](inputTensor)
    
    def buildFoldLayer(self, inputTensor, reshaping):
        self.layers.append(Fold(reshaping, 
                                scope='fold_' + str(len(self.layers))))
        return self.layers[-1](inputTensor)
    
    
    def buildInverseLayer(self, 
                          inputTensor):
        
        encoderLayerIndex = 2 * self.nbEncoderLayer - len(self.layers) - 1 
        
        if self.verbose :
            print("Building  inserve operation for layer ", 
                  encoderLayerIndex, 
                  " of type ", 
                  self.layers[encoderLayerIndex].layerType)
        #If only python could propose a proper switch !!!
        if self.layers[encoderLayerIndex].layerType == "Convolution2D" :
            if self.verbose :
                print("Convolution layer so inverse layer is a deconvolution layer")
            #Output decoder layer receives linear activation function
            buildOutputLayer = self.isFirstEncodingLayer(encoderLayerIndex)
            activationFunction = None if buildOutputLayer else self.layers[encoderLayerIndex].activation
            self.layers.append(DeConvolution2D(self.layers[encoderLayerIndex].kernel_shape, 
                                               self.layers[encoderLayerIndex].inputTensor,
                                               activation=activationFunction, 
                                               scope='deconv_' + str(len(self.layers)),
                                               padding = self.layers[encoderLayerIndex].padding))
        elif ((self.layers[encoderLayerIndex].layerType == "MaxPooling") or 
              (self.layers[encoderLayerIndex].layerType == "AvgPooling")) :
            if self.verbose :
                print(self.layers[encoderLayerIndex].layerType + " layer so inverse layer is an unpooling layer")
            self.layers.append(UnPooling(self.layers[encoderLayerIndex].getUnpoolKernelShape(), 
                                         self.layers[encoderLayerIndex].inputTensor, 
                                         scope='unpool_' + str(len(self.layers))))
        elif self.layers[encoderLayerIndex].layerType == "Unfold" :
            if self.verbose :
                print("Unfold layer so inverse layer is a fold layer")
            encodeurInputTensorShape = self.layers[encoderLayerIndex].inputTensor.get_shape().as_list()
            encodeurInputTensorShape[0] = -1
            self.layers.append(Fold(encodeurInputTensorShape, 
                                    scope='fold_' + str(len(self.layers))))
        elif self.layers[encoderLayerIndex].layerType == "Fold" :
            if self.verbose :
                print(self.layers[encoderLayerIndex].layerType + " layer so inverse layer is a unfold layer")
            self.layers.append(Unfold(scope='unfold_' + str(len(self.layers))))
        else:
            super().buildInverseLayer(inputTensor)
        return self.layers[-1](inputTensor)
        
    
    #Build a convolutional layer with suitable kernel shape for pooling layer and convolutional layer
    #Kernel size and strides are adjusted to surfac size
    def buildEncodingConvolutionLayer(self,
                                      inputTensor,
                                      convKernel,
                                      padding,
                                      activation = tf.nn.softplus,
                                      convStride = [1,1,1,1]):
        
        def searchFirstDivider(number):#dummy implementation
            if (number <= 1) or (not isinstance(number, int)) :
                return 1
            for k in range(2,int(number**(0.5))+2):
                if (number % k) == 0 :
                    return k
            return number
            
        staticInputShape = inputTensor.get_shape().as_list()
        adjustedConvStride = convStride
        adjustedConvKernel = convKernel
        poolKernel = [1,1,1,1]
        poolStride = [1,1,1,1]
        staticConvShape = staticInputShape#Shape of convolution layer
        staticConvShape[3] = staticInputShape[3] * adjustedConvKernel[3] / adjustedConvKernel[2] #Channel number        
        
        if padding == 'VALID' :
            #ajust convolution kernel size
            if staticInputShape[1] < staticInputShape[2] :
                adjustedConvKernel[1] += 1
            elif staticInputShape[1] > staticInputShape[2] :
                adjustedConvKernel[0] += 1
            #Kernel size must inferior than tensor size
            adjustedConvKernel[0] = min(staticInputShape[1],adjustedConvKernel[0])
            adjustedConvKernel[1] = min(staticInputShape[2],adjustedConvKernel[1])
            #With valid padding no zero is added on the edge
            staticConvShape[1] = staticInputShape[1] - (adjustedConvKernel[0]-1)
            staticConvShape[2] = staticInputShape[2] - (adjustedConvKernel[1]-1)
        
        staticConvShape[1] = int(staticConvShape[1] / adjustedConvStride[1])
        staticConvShape[2] = int(staticConvShape[2] / adjustedConvStride[2])
        if self.activatePooling :
            #pooling for autoencoder must respect two constraints :
            # - pooling kernel and pooling stride must have the same size
            # - pooling kernel must be divider of convolution tensor to allow for building unpooling layer
            poolKernel = [1, 
                          searchFirstDivider(staticConvShape[1]), 
                          searchFirstDivider(staticConvShape[2]), 
                          1]
            poolStride = poolKernel
        
        return self.buildConvolutionLayer(inputTensor, 
                                          adjustedConvKernel,
                                          poolKernel, 
                                          poolStride, 
                                          padding,
                                          activation = activation,
                                          convStride = adjustedConvStride)
    
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
                                                                 uniform=False)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        
        
        # encode
        conv1, pool1 = self.buildEncodingConvolutionLayer(inputReshaped, 
                                                          [3, 3, 1, 3], 
                                                          'SAME',
                                                          activation = tf.nn.relu)
        if self.verbose :
            print(conv1)
            print(pool1)
        
        unfold = self.buildUnfoldLayer(pool1)
        if self.verbose :
            print(unfold)
        
        
        self.factorTensor = self.buildDenseLayer(self.nbFactors,
                                                 unfold,
                                                 activation = tf.nn.relu,
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
    

class ConvolutionalAutoEncoderSmooth(ConvolutionalAutoEncoder):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestConvolutionalAESmoothModel"):
        
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)    
       

    def buildArchitecture(self):
        self.activatePooling = False
        
        
        # coding part 
        self.inputTensor = tf.placeholder(tf.float32, 
                                          shape=[None, self.nbUnitsPerLayer['Input Layer']])#bacth size along
        if self.verbose :
            print(self.inputTensor)
        #80
        inputReshaped = self.buildFoldLayer(self.inputTensor,
                                            [-1, self.hyperParameters['nbX'], 
                                             self.hyperParameters['nbY'], 
                                             self.hyperParameters['nbChannel']]) 
        if self.verbose :
            print(inputReshaped)
        
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=False)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        
        # encode
        conv1, pool1 = self.buildEncodingConvolutionLayer(inputReshaped, 
                                                          [3, 3, 1, 9], 
                                                          'SAME')
        
        if self.verbose :
            print(conv1)
            print(pool1)
        
        unfold = self.buildUnfoldLayer(pool1)
        if self.verbose :
            print(unfold)
        
        
        self.factorTensor = self.buildDenseLayer(self.nbFactors,
                                                 unfold,
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
        
        
class ContractiveConvolutionalAutoEncoder(ConvolutionalAutoEncoderSmooth):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestContractiveConvolutionalAEModel"):
        
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





class ConvolutionalAutoEncoderDeep(ConvolutionalAutoEncoderSmooth):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestConvolutionalAEDeepModel"):
        
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
        #Tensor 10,8,1
        
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=False)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        
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
        
        
        self.layers.append(Unfold(scope='unfold'))
        unfold = self.layers[-1](pool3)
        if self.verbose :
            print(unfold)
        
        self.factorTensor = self.buildDenseLayer(self.nbFactors, 
                                                 unfold,
                                                 kernelRegularizer = l2_regularizer,
                                                 kernelInitializer = he_init,
                                                 activation = None)
        
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
    
        
class ConvolutionalAutoEncoderDeepLayerWise(ConvolutionalAutoEncoderDeep):
    #######################################################################################################
    #Construction functions
    #######################################################################################################    
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestConvolutionalAEDeepModelLayerWise"):
        
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)

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

class ConvolutionalAutoEncoderDeep2(ConvolutionalAutoEncoderDeepLayerWise):    
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestConvolutionalAEDeepModel2"):
        
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
                                                                 uniform=False)
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
        
        denseE1 = self.buildDenseLayer(20,
                                       unfold,
                                       kernelRegularizer = l2_regularizer,
                                       kernelInitializer = he_init)
        if self.verbose :
            print(denseE1)
        
        self.factorTensor = self.buildDenseLayer(self.nbFactors, 
                                                 denseE1,
                                                 kernelRegularizer = l2_regularizer,
                                                 kernelInitializer = he_init,
                                                 activation = None)
        
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
    

class ConvolutionalAutoEncoderDeep3(ConvolutionalAutoEncoderDeepLayerWise):    
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestConvolutionalAEDeepModel3"):
        
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
                                                                 uniform=False)
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
        
        self.factorTensor = self.buildDenseLayer(self.nbFactors, 
                                                 unfold,
                                                 kernelRegularizer = l2_regularizer,
                                                 kernelInitializer = he_init,
                                                 activation = None)
        
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
    
