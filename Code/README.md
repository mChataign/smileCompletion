Implementation of different expriments executed inside the different notebook :
- Name of notebook files (ipynb) indicates if data are takes as variation or level and filename also precises if short maturities.


Python scripts implement models requiring tensorflow 1.x, anaconda and stratpy environment :
- bootstraping.py implements some bootstrapping methods for discounting and dividend.
- BS.py implements some features related to Black-Scholes model.
- SSVI.py implements SSVI arbitrage free calibration from Gatheral (2013) paper.
- regression.py implements regression  with OLS and neural network.
- CompletionBenchmark.py implements some reference model for matrix completion such that Alternative Least Square, SVD ...
- extendedKalmanFilter.py models latent variables dynamic when output (observed) variables are a non-linear function of the latent variable.
- teacherDynamic.py manages execution of dimension reduction model with a dynamic model.
- teacher.py manages execution (learning, compression, completion) of a dimension reduction model alone.
- point_in_polygon.py implements winding number algorithm for deciding which point lies in a polygon.
- gaussianProcess.py implmeents some interpolation models such linear interpolation, krieging, nelson-siegel and SSVI.
- dynamicModel.py is the interface for model of latent variables dynamic.
- kalmanFilter.py models latent variables dynamic when output (observed) variables are assumed as a linear function of the latent variable.
- FunctionalApproach.py implements functional approach where neural network takes a input latent variables plus option coordinates (tenor, expiry, moneyness ...).
- FunctionalFwdApproach.py implements functional approach with a forward variable as additionnal input in constrast with FunctionalApproach.py
- convolutionAutoeoncoder.py provides different neural autoencoder architectures with convolutional layers.
- factorialModel.py defines the interface of dimension reduction (latent variable) models. 
- linearModel.py implements linear autoencoder such that PCA.
- loadData.py provides features for loading, structuring and standard features engineering in an object called dataSet.
- plottingTools.py implements some graphical representation and some diagnosis tools.
- shallowAutoencoder.py implements single layer autoencoder.
- stackedAutoencoder.py provides different multilayer autoencoder architectures even with corrupted data (stacked denoising autoencoder).
- variationalAutoencoder.py implements autoencoder where the latent distribution is assumed gaussian allowing simulation of target distribution.

These files can be gathered into different categories : 
- Latent variable models :  linearModel.py, variationalAutoencoder.py, stackedAutoencoder.py, shallowAutoencoder.py, factorialModel.py, convolutionAutoeoncoder.py, FunctionalApproach.py, FunctionalFwdApproach.py.
- Dynamic models : dynamicModel.py, extendedKalmanFilter.py, kalmanFilter.py.
- Display features : plottingTools.py.
- User interface and learning orchestrator : teacherDynamic.py, teacher.py.
- Data processing : loadData.py.
- Interpolation : gaussianProcess.py.
- Regression : regression.py.
- Low-Rank completion : CompletionBenchmark.py.
- Dependancies : bootstraping.py, BS.py, SSVI.py, point_in_polygon.py.

