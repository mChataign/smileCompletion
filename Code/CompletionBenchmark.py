
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

import tensorflowModel

#class KNNImputer(LinearProjection):
#class IterativeImputer(LinearProjection):
#Matrix factorizarion with SGD algorithm
class FactorizationSGD(tensorflowModel.tensorflowModel):
    #######################################################################################################
    #Construction functions
    #######################################################################################################
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestSVDModel"):
        self.trainingDataSet = None
        self.QTrain = None
        self.PTrain = None
        self.bFeaturesTrain = None
        self.bObsTrain = None
        
        #Tensors saved 
        self.P = None
        self.Q = None
        self.bFeatures = None
        self.bObs = None
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    
    #Build the architecture, losses and optimizer.
    def buildModel(self):
        tf.reset_default_graph()
        if self.verbose :
            print("Do nothing, architecture depends on the dataset shape")
        self.nbEncoderLayer = 0
        self.layers = [] 
        return
    
    #Loss computed on a per observation basis
    def buildReconstructionLoss(self, pred, ref, name, matrixNorm = True):
        #If reference value contains NaN we ignore it for completion purpose
        diff = pred - ref
        diff_without_nans = tf.where(tf.is_nan(ref), tf.zeros_like(ref), diff)
        return tf.norm(diff_without_nans, ord = self.lossHolderExponent, axis =[-2,-1], name=name)
        #return tf.pow(diff_without_nans, self.lossHolderExponent, name=name)
    
    #Aggregate errors on batch dimension, normally should do nothing
    def normalizeLoss(self, dayLoss, name):
        return tf.reduce_mean(dayLoss, name=name)
    
    #Build loss for batch sample
    def buildLoss(self, pred, ref, name, matrixNorm = True):
        return self.normalizeLoss(self.buildReconstructionLoss(pred, ref, name + "R", matrixNorm=matrixNorm), 
                                  name)
    
    def buildPenalization(self,**kwargs):
        firstPenalizations = super().buildPenalization(**kwargs)
        if "l2_reg" not in self.hyperParameters :
            return firstPenalizations
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.hyperParameters['l2_reg'])
        #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.kernelRegularizer(self.weights))
        
        penalization = [self.hyperParameters['l2_reg'] * tf.norm(self.P, ord=2, axis=[-2,-1]), 
                        self.hyperParameters['l2_reg'] * tf.norm(self.Q, ord=2, axis=[-2,-1]), 
                        self.hyperParameters['l2_reg'] * tf.norm(self.bObs, ord=2),
                        self.hyperParameters['l2_reg'] * tf.norm(self.bFeatures, ord=2)]
        #Standard case, training all layers
        if self.verbose :
            print(penalization)
        return firstPenalizations + penalization
    
    #Build the architecture, losses and optimizer.
    def buildModelFromDataSet(self, dataSet):
        tf.reset_default_graph()
        if self.verbose :
            print("build architecture, loss and penalisations")
        self.nbEncoderLayer = 0
        self.layers = []
        self.buildArchitecture(dataSet)
        self.reconstructionLoss = self.buildReconstructionLoss(self.outputTensor, 
                                                               self.inputTensor,
                                                               "reconstructionLoss") 
        self.reducedReconstructionLoss = self.normalizeLoss(self.reconstructionLoss, 
                                                            "reducedReconstructionLoss")
        
        self.penalizationList = self.buildPenalization()
        
        # self.loss = tf.add_n([self.reducedReconstructionLoss] + self.penalizationList, 
                             # name="loss")
        self.loss = self.reducedReconstructionLoss 
        
        self.optimizer = tf.train.AdamOptimizer(self.learningRate, name="optimizer")
        
        self.trainingOperator = self.optimizer.minimize(self.loss,
                                                        var_list = [self.P, self.Q, 
                                                                    self.bObs, self.bFeatures],
                                                        name="trainingOperator")
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(name="saver", 
                                    save_relative_paths=True, 
                                    filename = self.metaModelName) 
        return
        
    def buildArchitecture(self, dataSet):
        
        numObs = dataSet.shape[0]
        self.inputTensor = tf.placeholder(tf.float32,
                                          shape=[numObs, self.nbUnitsPerLayer['Input Layer']])#batch size along
        inputTensorFilled = tf.where(tf.is_nan(self.inputTensor), tf.zeros_like(self.inputTensor), self.inputTensor)
        
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, 
                                                                 mode='FAN_AVG', 
                                                                 uniform=True)
        
        self.P = tf.Variable(he_init((numObs, self.nbFactors)),
                             name='P')
        self.Q = tf.Variable(he_init((self.nbFactors, self.nbUnitsPerLayer['Input Layer'])),
                             name='Q')
        self.bObs = tf.Variable(np.zeros((numObs,1)),
                                dtype = tf.float32,
                                trainable = True,
                                name='bObs')
        self.bFeatures = tf.Variable(np.zeros((1,self.nbUnitsPerLayer['Input Layer'])),
                                     dtype = tf.float32,
                                     trainable = True,
                                     name='bFeatures')
        self.outputTensor = tf.matmul(self.P,self.Q) + self.bObs + self.bFeatures
        #self.outputTensor = tf.matmul(self.P,self.Q)
        
        self.factorTensor = tf.matmul(inputTensorFilled - self.bObs, self.Q, transpose_b = True)
        #self.factorTensor = tf.matmul(inputTensorFilled, self.Q, transpose_b = True)
        return
        
        
    
    def gradientDescent(self, 
                        session, 
                        datasetTrain, 
                        nbEpoch, 
                        trainingLoss, 
                        gradientStep, 
                        earlyStoppingLoss):
        #No validation set since we factorize the whole matrixNorm
        #However crossvalidation can be done through trainging with different datasetTrain
        #Use early stopping
        patience = self.hyperParameters['earlyStoppingWindow']
        epsilon = 0.00001
        bestEpoch = 0
        trainingSet = datasetTrain
        epochLosses = []
        save_path = None
        epochDict = self.createFeedDictEncoder(trainingSet)
        earlyStoppingLosses = []
        
        for epoch in range(nbEpoch):
              
            session.run(gradientStep, feed_dict=epochDict)
            epochLosses.append(trainingLoss.eval(feed_dict=epochDict))
            if self.verbose :
                print("Epoch : ", epoch, 
                      " , Penalized Loss on testing dataset : ", epochLosses[epoch]) 
            earlyStoppingLosses.append(earlyStoppingLoss.eval(feed_dict=epochDict))
            
            #Monitor Model Performance
            #Decide which model to keep
            if ((epoch == 0) or (earlyStoppingLosses[epoch] <= np.nanmin(earlyStoppingLosses))):
                #Save Model if it improves validation error
                save_path = self.saveModel(session, self.metaModelName)
                bestEpoch = epoch
            
            #Early stopping is triggered if performance is not improved during a certain window
            if (((epoch - bestEpoch) >= patience) or (epoch == (nbEpoch - 1))) : 
                #Trigger early stopping and restore best performing model 
                minPatienceWindow = np.nanmin(earlyStoppingLosses[-patience:]) 
                self.restoreWeights(session)
                if self.verbose :
                    print("Minimum loss for the latest ", patience ," observations : ", minPatienceWindow)
                    print("Minimum loss : ", np.nanmin(earlyStoppingLosses))
                    print("Loss from restored model : ", earlyStoppingLoss.eval(feed_dict=epochDict))
                break
        return np.array(epochLosses), np.array(earlyStoppingLosses)
    
        
        
        
    def train(self, inputTrain, nbEpoch, inputTest = None):
        start = time.time()
        self.buildModelFromDataSet(inputTrain[0])
        patience = self.hyperParameters['earlyStoppingWindow']
        with tf.Session() as sess :
            sess.run(self.init)
            epochLosses, earlyStoppingLosses = self.gradientDescent(sess,
                                                                    inputTrain,
                                                                    nbEpoch,
                                                                    self.loss,
                                                                    self.trainingOperator,
                                                                    self.loss)
            epochDict = self.createFeedDictEncoder(inputTrain)
            self.QTrain = self.Q.eval()
            self.PTrain = self.P.eval()
            self.bFeaturesTrain = self.bFeatures.eval()
            self.bObsTrain = self.bObs.eval()
            minPatienceWindow = np.nanmin(earlyStoppingLosses[-patience:]) 
            print("Detailed performances")
            print("Reconstruction loss : ", self.reducedReconstructionLoss.eval(feed_dict=epochDict))
            print("Minimum loss for the latest ", patience ," observations : ", minPatienceWindow)
            print("Minimum loss : ", np.nanmin(earlyStoppingLosses))
            print("Loss from restored model : ", self.loss.eval(feed_dict=epochDict))
        
        self.trainingDataSet = inputTrain
        
        print("Training time : % 5d" %(time.time() -  start))
        return epochLosses
        
    def getDecoderCoefficients(self):
        #raise NotImplementedError("Not allowed for PCA because of mean rescaling !")
        return None
    
    def completeMatrix(self, dataSet):
        self.buildModelFromDataSet(dataSet[0])
        with tf.Session() as sess:
            sess.run(self.init)
            epochLosses,_ = self.gradientDescent(sess,
                                                 dataSet,
                                                 self.hyperParameters["nbEpochs"],
                                                 self.loss,
                                                 self.trainingOperator,
                                                 self.loss)
            epochDict = self.createFeedDictEncoder(dataSet)
            loss = self.reducedReconstructionLoss.eval(feed_dict=epochDict)
            reconstructedInputs = self.outputTensor.eval(feed_dict=epochDict)
            factors = self.factorTensor.eval(feed_dict=epochDict)
        dfReconstructedInputs = pd.DataFrame(reconstructedInputs,
                                             index=dataSet[0].index,
                                             columns=dataSet[0].columns)
        return loss, dfReconstructedInputs, pd.DataFrame(factors, index = dataSet[0].index), epochLosses
    
        #Same but with default session 
    def evalModel(self, inputTest):
        loss, inputs, factors, _ = self.completeMatrix(inputTest)
        if self.verbose :
            print("Reconstruction Loss : ", loss)
        return loss, inputs, factors
    
    
    def completeDataTensor(self, 
                           sparseSurfaceList, 
                           initialValueForFactors, 
                           nbCalibrationStep, 
                           *args):
        
        #Build tensor for reconstruction 
        matrixToComplete = [(x.append(y) if (x[0] is not None and x[1] is not None) else None 
                             for x in zip(self.trainingDataSet, sparseSurfaceList))]
        loss, matrix, factors, calibrationLosses = self.completeMatrix(matrixToComplete)
        
        #Get results for best calibration
        bestCalibration = loss
        bestFactors = factors.iloc[-1].values
        bestSurface = matrix.iloc[-1]
        return bestCalibration , bestFactors, bestSurface, pd.Series(calibrationLosses)
    
    def evalSingleDayWithoutCalibrationWithSensi(self, initialValueForFactors, datasetList):
        return  self.commonEvalSingleDayWithoutCalibration(initialValueForFactors, 
                                                           datasetList, 
                                                           computeSensi = True)
    
    def evalSingleDayWithoutCalibration(self, initialValueForFactors, datasetList):
        s,_ = self.commonEvalSingleDayWithoutCalibration(initialValueForFactors, datasetList)
        return s   
    
        
    def commonEvalSingleDayWithoutCalibration(self, 
                                              initialValueForFactors, 
                                              datasetList,
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
        
        nbPoints = self.QTrain.shape[1]
        reshapedValueForFactors = np.reshape([initialValueForFactors],
                                             (nbObs,nbFactors))
        reconstructedSurface = np.dot(reshapedValueForFactors, self.QTrain) + self.bFeaturesTrain
        reshapedJacobian = None
        if computeSensi :
            reshapedJacobian = np.repeat(np.expand_dims(self.QTrain.T, axis = 0),
                                         nbObs, axis=0)
        if initialValueForFactors.ndim == 1 :
            calibratedSurfaces = np.reshape(reconstructedSurface.T, (nbPoints))
            if reshapedJacobian is not None :
                factorSensi = np.reshape(reshapedJacobian, (nbPoints, nbFactors))
        elif initialValueForFactors.ndim == 2 :
            calibratedSurfaces = np.reshape(reconstructedSurface.T, (nbObs,nbPoints))
            if reshapedJacobian is not None :
                factorSensi = np.reshape(reshapedJacobian, (nbObs, nbPoints, nbFactors))
        
        
        return calibratedSurfaces, factorSensi 
    
    
        
    #Take a full surface in entry, reconstruct it 
    #and return sensitivities between points i.e. the jacobian of D(E(S)) w.r.t S
    def evalInterdependancy(self, fullSurface):
        raise NotImplementedError("Factorization matrix interdependancies are implicit and obtained through optimization") 


class ALS(FactorizationSGD):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestALSModel"):
        
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
                         
    # def gradientDescent(self, 
                        # session, 
                        # datasetTrain, 
                        # nbEpoch, 
                        # trainingLoss, 
                        # gradientStep, 
                        # earlyStoppingLoss):
        
        # nbIterations = 10
        # epochLosses = []
        # earlyStoppingLosses = []
        # for k in range(nbIterations):
            # l, earlyStoppingL = super().gradientDescent(session, 
                                                        # datasetTrain, 
                                                        # int(nbEpoch/2/nbIterations), 
                                                        # trainingLoss, 
                                                        # self.trainingOperatorObs, 
                                                        # earlyStoppingLoss)
            # epochLosses.append(l)
            # earlyStoppingLosses.append(l)
            
            # l, earlyStoppingL = super().gradientDescent(session, 
                                                        # datasetTrain, 
                                                        # int(nbEpoch/2/nbIterations), 
                                                        # trainingLoss, 
                                                        # self.trainingOperatorFeatures, 
                                                        # earlyStoppingLoss)
            # epochLosses.append(l)
            # earlyStoppingLosses.append(l)
            
        # return np.concatenate(epochLosses), np.concatenate(earlyStoppingLosses)
    
    #Build the architecture, losses and optimizer.
    def buildModelFromDataSet(self, dataSet):
        super().buildModelFromDataSet(dataSet)
        numObs = dataSet.shape[0]
        self.trainingOperatorObs = self.optimizer.minimize(self.loss,
                                                           var_list = [self.P, self.bObs],
                                                           name="trainingOperator")
        self.trainingOperatorFeatures = self.optimizer.minimize(self.loss,
                                                                var_list = [self.Q, self.bFeatures],
                                                                name="trainingOperator")
        
        self.PPlaceHolder = tf.placeholder(tf.float32,
                                           shape=[numObs, self.nbFactors])
        self.PAssign = self.P.assign(self.PPlaceHolder)
        
        self.QPlaceHolder = tf.placeholder(tf.float32,
                                           shape=[self.nbFactors, self.nbUnitsPerLayer['Input Layer']])
        self.QAssign = self.Q.assign(self.QPlaceHolder)
        
        self.bObsPlaceHolder = tf.placeholder(tf.float32,
                                              shape=[numObs, 1])
        self.bObsAssign = self.bObs.assign(self.bObsPlaceHolder)
        
        self.bFeaturesPlaceHolder = tf.placeholder(tf.float32,
                                                   shape=[1, self.nbUnitsPerLayer['Input Layer']])
        self.bFeaturesAssign = self.bFeatures.assign(self.bFeaturesPlaceHolder)
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(name="saver", 
                                    save_relative_paths=True, 
                                    filename = self.metaModelName) 
        return
    
    def gradientDescent(self, 
                        session, 
                        datasetTrain, 
                        nbEpoch, 
                        trainingLoss, 
                        gradientStep, 
                        earlyStoppingLoss):
        
        epochLosses = []
        earlyStoppingLosses = []
        epochDict = self.createFeedDictEncoder(datasetTrain)
        patience = self.hyperParameters['earlyStoppingWindow']
        
        #Get initial values
        P = self.P.eval()
        Q = self.Q.eval()
        bFeatures = np.reshape(self.bFeatures.eval(), (1,Q.shape[1]))
        bObs = np.reshape(self.bObs.eval(),(P.shape[0],1))
        
        for epoch in range(int(nbEpoch/2)):
            #Cu = np.where((datasetTrain[0].isna().sum(axis=1) > 0).values, 100, 1)
            Cu = None
            P, bObs, Q, bFeatures = self.ALSStep(P, Q, bObs, bFeatures, datasetTrain[0].values, Cu)
            session.run(self.PAssign, feed_dict={self.PPlaceHolder : P})
            session.run(self.QAssign, feed_dict={self.QPlaceHolder : Q})
            session.run(self.bFeaturesAssign, feed_dict={self.bFeaturesPlaceHolder : bFeatures})
            session.run(self.bObsAssign, feed_dict={self.bObsPlaceHolder : bObs})
            earlyStoppingLosses.append(earlyStoppingLoss.eval(feed_dict=epochDict))
            epochLosses.append(trainingLoss.eval(feed_dict=epochDict))
            
            
            #Monitor Model Performance
            #Decide which model to keep
            if ((epoch == 0) or (earlyStoppingLosses[epoch] <= np.nanmin(earlyStoppingLosses))):
                #Save Model if it improves validation error
                save_path = self.saveModel(session, self.metaModelName)
                bestEpoch = epoch
            
            #Early stopping is triggered if performance is not improved during a certain window
            if (((epoch - bestEpoch) >= patience) or (epoch == (int(nbEpoch/2) - 1))) : 
                #Trigger early stopping and restore best performing model 
                minPatienceWindow = np.nanmin(earlyStoppingLosses[-patience:]) 
                self.restoreWeights(session)
                if self.verbose :
                    print("Minimum loss for the latest ", patience ," observations : ", minPatienceWindow)
                    print("Minimum loss : ", np.nanmin(earlyStoppingLosses))
                    print("Loss from restored model : ", earlyStoppingLoss.eval(feed_dict=epochDict))
                break
        
        
        return np.array(epochLosses), np.array(earlyStoppingLosses)
    
    #Cu matrix weighting loss for different users, practical when few observations have missing values
    # def ALSStep(self,P,Q,bU,bI,R,Cu=None):
        # lambdaReg = self.hyperParameters['l2_reg'] if 'l2_reg' in self.hyperParameters else 0
        # def dropna(x, axis=0): #Axis defines along which dimension na are removed
            # filteredX = x[~np.isnan(x)]
            # newShape = [-1, x.shape[1]] if (axis==0) else [x.shape[0], -1]
            # return np.reshape(filteredX, newShape)
        #for each user
        # def userLS(Ru, Q, bI):
            #nf : nb factors ; nu : nb users (nb observations) ; ni : nb items (nb features)
            #RU : dim (1, ni)
            #Q : dim (nf, ni)
            #bI : dim (1, ni)
            # beta = np.vstack([Q, np.ones_like(bI)]) #dim (nf+1, ni)
            # C = np.eye(beta.shape[0])
            # tmp1 = np.linalg.inv(np.dot(beta,np.dot(beta.T,C)) + lambdaReg * np.eye(self.nbFactors + 1)) #dim (nf+1, nf+1)
            # tmp2 = np.sum(dropna((np.reshape(Ru, bI.shape) - bI).T * beta.T, axis=0).T, axis=1) #dim (nf+1,1)
            # return np.dot(tmp1, tmp2).T       #dim (1,nf+1)  where last row is the user bias
        # newP = np.apply_along_axis(userLS, 1, R, Q, bI) 
        # def itemLS(Ri, P, bU, CU):
            #nf : nb factors ; nu : nb users (nb observations) ; ni : nb items (nb features)
            #Ri : dim (nu, 1)
            #P : dim (nu, nf)
            #bU : dim (nu, 1)
            #CU : dim (nu, 1)
            # beta = np.hstack([P, np.ones_like(bU)]) #dim (nu, nf + 1)
            # C = np.eye(beta.shape[0]) if CU is None else np.diag(CU, k=0)
            # tmp1 = np.linalg.inv(np.dot(beta.T,np.dot(C, beta)) + lambdaReg * np.eye(self.nbFactors + 1)) #dim (nf+1, nf+1)
            # tmp2 = np.sum(dropna(beta.T * (np.reshape(Ri, bU.shape) - bU).T, axis=1).T, axis=0) #dim (nf+1,1)
            # return   np.dot(tmp1, tmp2)       #dim (nf+1,1)  where last row is the user bias
        # newQ = np.apply_along_axis(itemLS, 0, R, newP[:,:-1], np.reshape(newP[:,-1], bU.shape), Cu)
        # return newP[:,:-1], np.reshape(newP[:,-1], bU.shape), newQ[:-1,:], np.reshape(newQ[-1,:], bI.shape) #P,bU,Q,bI

    #Cu matrix weighting loss for different users, practical when few observations have missing values
    def ALSStep(self,P,Q,bU,bI,R,Cu=None):
        lambdaReg = self.hyperParameters['l2_reg'] if 'l2_reg' in self.hyperParameters else 0
        def dropna(x, axis=0): #Axis defines along which dimension na are removed
            filteredX = x[~np.isnan(x)]
            newShape = [-1, x.shape[1]] if (axis==0) else [x.shape[0], -1]
            return np.reshape(filteredX, newShape)
        newR = np.where(np.isnan(R), np.dot(P,Q) + bU + bI, R)
        #for each user
        def userLS(Ru, Q, bI):
            #nf : nb factors ; nu : nb users (nb observations) ; ni : nb items (nb features)
            #RU : dim (1, ni)
            #Q : dim (nf, ni)
            #bI : dim (1, ni)
            beta = np.vstack([Q, np.ones_like(bI)]) #dim (nf+1, ni)
            C = np.eye(beta.shape[0])
            tmp1 = np.linalg.inv(np.dot(beta,np.dot(beta.T,C)) + lambdaReg * np.eye(self.nbFactors + 1)) #dim (nf+1, nf+1)
            tmp2 = np.sum(dropna((np.reshape(Ru, bI.shape) - bI).T * beta.T, axis=0).T, axis=1) #dim (nf+1,1)
            return np.dot(tmp1, tmp2).T       #dim (1,nf+1)  where last row is the user bias
        newP = np.apply_along_axis(userLS, 1, newR, Q, bI) 
        
        newR = np.where(np.isnan(R), np.dot(newP[:,:-1],Q) + np.reshape(newP[:,-1], bU.shape) + bI, R)
        
        #for each item
        def itemLS(Ri, P, bU, CU):
            #nf : nb factors ; nu : nb users (nb observations) ; ni : nb items (nb features)
            #Ri : dim (nu, 1)
            #P : dim (nu, nf)
            #bU : dim (nu, 1)
            #CU : dim (nu, 1)
            beta = np.hstack([P, np.ones_like(bU)]) #dim (nu, nf + 1)
            C = np.eye(beta.shape[0]) if CU is None else np.diag(CU, k=0)
            tmp1 = np.linalg.inv(np.dot(beta.T,np.dot(C, beta)) + lambdaReg * np.eye(self.nbFactors + 1)) #dim (nf+1, nf+1)
            tmp2 = np.sum(dropna(beta.T * (np.reshape(Ri, bU.shape) - bU).T, axis=1).T, axis=0) #dim (nf+1,1)
            return   np.dot(tmp1, tmp2)       #dim (nf+1,1)  where last row is the user bias
        newQ = np.apply_along_axis(itemLS, 0, R, newP[:,:-1], np.reshape(newP[:,-1], bU.shape), Cu)
        return newP[:,:-1], np.reshape(newP[:,-1], bU.shape), newQ[:-1,:], np.reshape(newQ[-1,:], bI.shape) #P,bU,Q,bI
        
class softImpute(ALS):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestSoftImputeModel"):
        self.softEigenValue = nbFactors
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbUnitsPerLayer['Input Layer'],
                         modelName)
    
    def gradientDescent(self, 
                        session, 
                        datasetTrain, 
                        nbEpoch, 
                        trainingLoss, 
                        gradientStep, 
                        earlyStoppingLoss):
        
        epochLosses = []
        earlyStoppingLosses = []
        epochDict = self.createFeedDictEncoder(datasetTrain)
        patience = self.hyperParameters['earlyStoppingWindow']
        
        #Get initial values
        P = self.P.eval()
        Q = self.Q.eval()
        bFeatures = np.reshape(self.bFeatures.eval(), (1,Q.shape[1]))
        bObs = np.reshape(self.bObs.eval(),(P.shape[0],1))
        M = np.zeros_like(P @ Q)
        lambdaEigenValue = np.linalg.svd(np.where(np.isnan(datasetTrain[0].values), M, datasetTrain[0].values), 
                                         full_matrices=False)[1][self.softEigenValue]
        
        for epoch in range(int(nbEpoch/2)):
            #Cu = np.where((datasetTrain[0].isna().sum(axis=1) > 0).values, 100, 1)
            M, U, V, D = self.SVDSoftStep(M, datasetTrain[0].values, lambdaEigenValue)
            session.run(self.PAssign, feed_dict={self.PPlaceHolder : U @ np.sqrt(D)})
            session.run(self.QAssign, feed_dict={self.QPlaceHolder : np.sqrt(D) @ V})
            #session.run(self.bFeaturesAssign, feed_dict={self.bFeaturesPlaceHolder : bFeatures})
            #session.run(self.bObsAssign, feed_dict={self.bObsPlaceHolder : bObs})
            earlyStoppingLosses.append(earlyStoppingLoss.eval(feed_dict=epochDict))
            epochLosses.append(trainingLoss.eval(feed_dict=epochDict))
            
            
            #Monitor Model Performance
            #Decide which model to keep
            if ((epoch == 0) or (earlyStoppingLosses[epoch] <= np.nanmin(earlyStoppingLosses))):
                #Save Model if it improves validation error
                save_path = self.saveModel(session, self.metaModelName)
                bestEpoch = epoch
            
            #Early stopping is triggered if performance is not improved during a certain window
            if (((epoch - bestEpoch) >= patience) or (epoch == (int(nbEpoch/2) - 1))) : 
                #Trigger early stopping and restore best performing model 
                minPatienceWindow = np.nanmin(earlyStoppingLosses[-patience:]) 
                self.restoreWeights(session)
                if self.verbose :
                    print("Minimum loss for the latest ", patience ," observations : ", minPatienceWindow)
                    print("Minimum loss : ", np.nanmin(earlyStoppingLosses))
                    print("Loss from restored model : ", earlyStoppingLoss.eval(feed_dict=epochDict))
                break
        
        
        return np.array(epochLosses), np.array(earlyStoppingLosses)

    #Cu matrix weighting loss for different users, practical when few observations have missing values
    def SVDSoftStep(self, M, R, lambdaEigenValue):
        #lambdaReg = self.hyperParameters['l2_reg'] if 'l2_reg' in self.hyperParameters else 0
        def dropna(x, axis=0): #Axis defines along which dimension na are removed
            filteredX = x[~np.isnan(x)]
            newShape = [-1, x.shape[1]] if (axis==0) else [x.shape[0], -1]
            return np.reshape(filteredX, newShape)
        newR = np.where(np.isnan(R), M, R)
        newU, newD, newV = np.linalg.svd(newR, full_matrices=False) 
        newM = newU @ np.diag(np.reshape(np.maximum(newD - lambdaEigenValue, 0), newD.shape)) @ newV
        return newM, newU, newV, np.diag(np.maximum(newD - lambdaEigenValue, 0)) 
        

class hardImpute(ALS):
    def __init__(self,
                 learningRate, 
                 hyperParameters, 
                 nbUnitsPerLayer, 
                 nbFactors,
                 modelName = "./bestHardImputeModel"):
        super().__init__(learningRate, 
                         hyperParameters,
                         nbUnitsPerLayer,
                         nbFactors,
                         modelName)
    def warmStart(self, 
                  session, 
                  datasetTrain, 
                  nbEpoch, 
                  trainingLoss, 
                  gradientStep,
                  earlyStoppingLoss):
        
        epochLosses = []
        earlyStoppingLosses = []
        epochDict = self.createFeedDictEncoder(datasetTrain)
        patience = self.hyperParameters['earlyStoppingWindow']
        
        #Get initial values
        P = self.P.eval()
        Q = self.Q.eval()
        bFeatures = np.reshape(self.bFeatures.eval(), (1,Q.shape[1]))
        bObs = np.reshape(self.bObs.eval(),(P.shape[0],1))
        M = np.zeros_like(P @ Q)
        lambdaEigenValue = np.linalg.svd(np.where(np.isnan(datasetTrain[0].values), M, datasetTrain[0].values), 
                                         full_matrices=False)[1][self.nbFactors]
        
        for epoch in range(int(nbEpoch/2)):
            #Cu = np.where((datasetTrain[0].isna().sum(axis=1) > 0).values, 100, 1)
            M, U, V, D = self.SVDSoftStep(M, datasetTrain[0].values, lambdaEigenValue)
            session.run(self.PAssign, feed_dict={self.PPlaceHolder : (U @ np.sqrt(D))[:,:self.nbFactors]})
            session.run(self.QAssign, feed_dict={self.QPlaceHolder : (np.sqrt(D) @ V)[:self.nbFactors,:]})
            #session.run(self.bFeaturesAssign, feed_dict={self.bFeaturesPlaceHolder : bFeatures})
            #session.run(self.bObsAssign, feed_dict={self.bObsPlaceHolder : bObs})
            earlyStoppingLosses.append(earlyStoppingLoss.eval(feed_dict=epochDict))
            epochLosses.append(trainingLoss.eval(feed_dict=epochDict))
            
            
            #Monitor Model Performance
            #Decide which model to keep
            if ((epoch == 0) or (earlyStoppingLosses[epoch] <= np.nanmin(earlyStoppingLosses))):
                #Save Model if it improves validation error
                save_path = self.saveModel(session, self.metaModelName)
                bestEpoch = epoch
            
            #Early stopping is triggered if performance is not improved during a certain window
            if (((epoch - bestEpoch) >= patience) or (epoch == (int(nbEpoch/2) - 1))) : 
                #Trigger early stopping and restore best performing model 
                minPatienceWindow = np.nanmin(earlyStoppingLosses[-patience:]) 
                self.restoreWeights(session)
                if self.verbose :
                    print("Minimum loss for the latest ", patience ," observations : ", minPatienceWindow)
                    print("Minimum loss : ", np.nanmin(earlyStoppingLosses))
                    print("Loss from restored model : ", earlyStoppingLoss.eval(feed_dict=epochDict))
                break
        
        
        return np.array(epochLosses), np.array(earlyStoppingLosses)   
        
    def gradientDescent(self, 
                        session, 
                        datasetTrain, 
                        nbEpoch, 
                        trainingLoss, 
                        gradientStep, 
                        earlyStoppingLoss):
        #Warm start with soft impute method
        self.warmStart(session, 
                       datasetTrain, 
                       nbEpoch, 
                       trainingLoss, 
                       gradientStep, 
                       earlyStoppingLoss)
        
        epochLosses = []
        earlyStoppingLosses = []
        epochDict = self.createFeedDictEncoder(datasetTrain)
        patience = self.hyperParameters['earlyStoppingWindow']
        
        #Get initial values
        P = self.P.eval()
        Q = self.Q.eval()
        bFeatures = np.reshape(self.bFeatures.eval(), (1,Q.shape[1]))
        bObs = np.reshape(self.bObs.eval(),(P.shape[0],1))
        M = P @ Q
        
        for epoch in range(int(nbEpoch/2)):
            #Cu = np.where((datasetTrain[0].isna().sum(axis=1) > 0).values, 100, 1)
            M, U, V, D = self.SVDHardStep(M, datasetTrain[0].values)
            session.run(self.PAssign, feed_dict={self.PPlaceHolder : U @ np.sqrt(D)})
            session.run(self.QAssign, feed_dict={self.QPlaceHolder : np.sqrt(D) @ V})
            #session.run(self.bFeaturesAssign, feed_dict={self.bFeaturesPlaceHolder : bFeatures})
            #session.run(self.bObsAssign, feed_dict={self.bObsPlaceHolder : bObs})
            earlyStoppingLosses.append(earlyStoppingLoss.eval(feed_dict=epochDict))
            epochLosses.append(trainingLoss.eval(feed_dict=epochDict))
            
            
            #Monitor Model Performance
            #Decide which model to keep
            if ((epoch == 0) or (earlyStoppingLosses[epoch] <= np.nanmin(earlyStoppingLosses))):
                #Save Model if it improves validation error
                save_path = self.saveModel(session, self.metaModelName)
                bestEpoch = epoch
            
            #Early stopping is triggered if performance is not improved during a certain window
            if (((epoch - bestEpoch) >= patience) or (epoch == (int(nbEpoch/2) - 1))) : 
                #Trigger early stopping and restore best performing model 
                minPatienceWindow = np.nanmin(earlyStoppingLosses[-patience:]) 
                self.restoreWeights(session)
                if self.verbose :
                    print("Minimum loss for the latest ", patience ," observations : ", minPatienceWindow)
                    print("Minimum loss : ", np.nanmin(earlyStoppingLosses))
                    print("Loss from restored model : ", earlyStoppingLoss.eval(feed_dict=epochDict))
                break
        
        return np.array(epochLosses), np.array(earlyStoppingLosses)    
    
    #Cu matrix weighting loss for different users, practical when few observations have missing values
    def SVDSoftStep(self, M, R, lambdaEigenValue):
        #lambdaReg = self.hyperParameters['l2_reg'] if 'l2_reg' in self.hyperParameters else 0
        def dropna(x, axis=0): #Axis defines along which dimension na are removed
            filteredX = x[~np.isnan(x)]
            newShape = [-1, x.shape[1]] if (axis==0) else [x.shape[0], -1]
            return np.reshape(filteredX, newShape)
        newR = np.where(np.isnan(R), M, R)
        newU, newD, newV = np.linalg.svd(newR, full_matrices=False) 
        newM = newU @ np.diag(np.reshape(np.maximum(newD - lambdaEigenValue, 0), newD.shape)) @ newV
        return newM, newU, newV, np.diag(np.maximum(newD - lambdaEigenValue, 0))
    
    #Cu matrix weighting loss for different users, practical when few observations have missing values
    def SVDHardStep(self, M, R):
        def dropna(x, axis=0): #Axis defines along which dimension na are removed
            filteredX = x[~np.isnan(x)]
            newShape = [-1, x.shape[1]] if (axis==0) else [x.shape[0], -1]
            return np.reshape(filteredX, newShape)
        newR = np.where(np.isnan(R), M, R)
        newU, newD, newV = np.linalg.svd(newR, full_matrices=False) 
        newM = newU[:,:self.nbFactors] @ np.diag(newD[:self.nbFactors]) @ newV[:self.nbFactors,:]
        return newM, newU[:,:self.nbFactors], newV[:self.nbFactors,:], np.diag(newD[:self.nbFactors]) 
    
    
        

