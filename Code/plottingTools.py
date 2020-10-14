import os
import pandas as pd
import numpy as np
from pandas import DatetimeIndex
import dask
import scipy
import time
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from matplotlib import animation, rc
from IPython.display import HTML, Image, display
import statsmodels.api as sm
import loadData


xLabel = "Maturity"
yLabel = "Log-Moneyness"
zLabel = "Volatility"

azimut = 220

def displayEncodingDistrib(encoding):
    for i in encoding.columns :
        fig = sm.qqplot(encoding[i], fit = True)
        plt.title("Factor no " + str(i))
        plt.show()
    return

def printIsolated(*args):   
    print("")
    print(*args)
    print("")
    return
    
    
def printDelimiter(numberOfDelimitation=1):
    print("")
    for k in range(numberOfDelimitation):
        print("----------------------------------------------------------------------------------------------------")
    print("")
    return

def plotSeries(series, title=None, yName = "Cost"):
    for s in series :
        plt.plot(s.dropna(), linewidth=2.0, label=s.name)
    refSize=5
    plt.tick_params(axis='x', labelsize=2*refSize, pad=int(refSize/2))
    plt.tick_params(axis='y', labelsize=2*refSize, pad=int(refSize/2))
    _ = plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Date", fontsize=2*refSize, labelpad=3*refSize)
    plt.ylabel(yName, fontsize=2*refSize, labelpad=3*refSize)
    if (title is not None):
        plt.title(title, pad = 3*refSize, fontsize = 2*refSize)
    plt.show()
    return

def plotTails(obs, obsCompleted):
    logMoneyess = obs[1].map(lambda x : x[1])
    maturity =  obs[1].map(lambda x : x[0])
    minLogMoneyness = logMoneyess.min()
    maxLogMoneyness = logMoneyess.max()
    
    upperTail = obs[0][logMoneyess[logMoneyess == maxLogMoneyness].index].dropna()
    lowerTail = obs[0][logMoneyess[logMoneyess == minLogMoneyness].index].dropna()
    upperTailCompleted = obsCompleted[0][logMoneyess[logMoneyess == maxLogMoneyness].index].dropna()
    lowerTailCompleted = obsCompleted[0][logMoneyess[logMoneyess == minLogMoneyness].index].dropna()
    
    plt.plot(maturity[upperTail.index], upperTail.values, ".", linewidth=2.0, label="Original")
    plt.plot(maturity[upperTailCompleted.index], upperTailCompleted.values, linewidth=2.0, label="Completed")
    
    refSize=5
    plt.tick_params(axis='x', labelsize=2*refSize, pad=int(refSize/2))
    plt.tick_params(axis='y', labelsize=2*refSize, pad=int(refSize/2))
    _ = plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Maturity", fontsize=2*refSize, labelpad=3*refSize)
    plt.ylabel("Implied volatility", fontsize=2*refSize, labelpad=3*refSize)
    plt.title("Implied volatility for highest strike", pad = 3*refSize, fontsize = 2*refSize)
    plt.show()
    
    
    plt.plot(maturity[lowerTail.index], lowerTail.values, ".", linewidth=2.0, label="Original")
    plt.plot(maturity[lowerTailCompleted.index], lowerTailCompleted.values, linewidth=2.0, label="Completed")
    
    refSize=5
    plt.tick_params(axis='x', labelsize=2*refSize, pad=int(refSize/2))
    plt.tick_params(axis='y', labelsize=2*refSize, pad=int(refSize/2))
    _ = plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Maturity", fontsize=2*refSize, labelpad=3*refSize)
    plt.ylabel("Implied volatility", fontsize=2*refSize, labelpad=3*refSize)
    plt.title("Implied volatility for lowest strike", pad = 3*refSize, fontsize = 2*refSize)
    plt.show()
    
    return

def historyPlot(serie, title=None):
    plt.plot(serie.dropna(), linewidth=2.0)
    refSize=5
    plt.tick_params(axis='x', labelsize=2*refSize, pad=int(refSize/2))
    plt.tick_params(axis='y', labelsize=2*refSize, pad=int(refSize/2))
    _ = plt.xticks(rotation=90)
    plt.xlabel("Date", fontsize=2*refSize, labelpad=3*refSize)
    plt.ylabel("RMSE", fontsize=2*refSize, labelpad=3*refSize)
    if (title is not None):
        plt.title(title, pad = 3*refSize, fontsize = 2*refSize)
    plt.show()
    
    return 

#error plot per observation
def errorPlot(predValues, refValue, title=None):
    #Absolute L1 error
    errorsAbsRMSE = pd.Series(np.nanmean(np.square(predValues - refValue),axis=1)**0.5, 
                              index = refValue.index)
    
    #Absolute RMSE
    errorsAbsL1 = pd.Series(np.nanmean(np.absolute(predValues - refValue),axis=1),
                            index = refValue.index)
    
    #Relative L1 error 
    errorsRelativeRMSE = pd.Series(np.nanmean(np.square((predValues/refValue - 1)),axis=1)**0.5, 
                                   index = refValue.index)
    
    #Relative RMSE 
    errorsRelativeL1 = pd.Series(np.nanmean(np.absolute((predValues/refValue - 1)),axis=1), 
                                 index = refValue.index)
    
    cells = [[errorsAbsRMSE.mean(),errorsAbsL1.mean(),errorsRelativeRMSE.mean(),errorsRelativeL1.mean()],
             [errorsAbsRMSE.max(),errorsAbsL1.max(),errorsRelativeRMSE.max(),errorsRelativeL1.max()],
             [errorsAbsRMSE.idxmax().strftime('%Y-%m-%d'),errorsAbsL1.idxmax().strftime('%Y-%m-%d'),
              errorsRelativeRMSE.idxmax().strftime('%Y-%m-%d'),errorsRelativeL1.idxmax().strftime('%Y-%m-%d')]]
    columns = ["Absolute RMSE error","Absolute L1 error","Relative RMSE error","Relative L1 error"]
    index = ["Daily average error", "Worst dataset error", "Worst dataset day"]
    print("")
    display(HTML(pd.DataFrame(cells, index = index, columns = columns).to_html()))
    print("")
    
    
    historyPlot(errorsAbsRMSE, title=(title + " Absolute RMSE ") if (title is not None) else " Absolute RMSE ")
    historyPlot(errorsRelativeRMSE, title= (title + " Relative RMSE ") if (title is not None) else " Relative RMSE " )
          
    return errorsAbsRMSE, errorsAbsL1, errorsRelativeRMSE, errorsRelativeL1

#worst day
def getWorstGrids(predValues, refValue, errors):
    worstDay = errors.idxmax()
    print("Worst Day : ", worstDay)
    
    generatedGrids = pd.DataFrame(predValues , 
                                  index = refValue.index, 
                                  columns = refValue.columns)
    worstDayGeneratedGrid = generatedGrids.loc[worstDay]
    worstDayGrid = refValue.loc[worstDay]
    
    return worstDayGeneratedGrid, worstDayGrid



def tensorMax(tensor):
    maxT = np.amax(tensor)
    return maxT if maxT.ndim==0 else tensorMax(maxT)

def tensorMin(tensor):
    minT = np.nanmin(tensor)
    return min(minT,0) if minT.ndim==0 else tensorMin(minT)

def getCoordinates(coordinatesList):
    xFunc = lambda x : x[0]
    xList = list(map(xFunc, coordinatesList))
    yFunc = lambda x : x[1]
    yList = list(map(yFunc, coordinatesList))
    return xList, yList

def buildGrid(ax, x, y, z, 
              zMin, zMax, 
              coordinates,
              colorMapSystem = None, 
              plotType= None, 
              normOverride = None,
              refPoints = None):
    xNbPoints = len(np.unique(x))
    yNbPoints = len(np.unique(y))
    
    norm = normOverride if normOverride else plt.Normalize(zMin, zMax)
    colorMap = None
    if colorMapSystem=="CMRmap":
        colorMap = plt.cm.CMRmap
    elif colorMapSystem=="hot":
        colorMap = plt.cm.hot
    elif colorMapSystem=="viridis":
        colorMap = plt.cm.viridis
    elif colorMapSystem=="Spectral":
        colorMap = plt.cm.Spectral
    elif colorMapSystem=="hsv":
        colorMap = plt.cm.hsv
    else :
        colorMap = None
    
    ax.tick_params(axis='x', labelsize=25, pad=5)
    ax.tick_params(axis='y', labelsize=25, pad=5)
    ax.tick_params(axis='z', labelsize=25, pad=10)
    
        
    #Smooth Surface
    if plotType=="coloredWire" :#Only lines between points but with colors
        if colorMap is not None :
            colors = None if colorMap is None else colorMap(norm(z.values))
            pltSurface = ax.plot_surface(np.reshape(x,(xNbPoints,yNbPoints)),
                                         np.reshape(y,(xNbPoints,yNbPoints)), 
                                         np.reshape(z.values,(xNbPoints,yNbPoints)), 
                                         facecolors=np.reshape(colors,(xNbPoints,yNbPoints,4)), 
                                         linewidth=2.0, 
                                         shade=False)
            pltSurface.set_facecolor((0,0,0,0))
        else :
            pltSurface = ax.plot_surface(np.reshape(x,(xNbPoints,yNbPoints)),
                                         np.reshape(y,(xNbPoints,yNbPoints)), 
                                         np.reshape(z.values,(xNbPoints,yNbPoints)), 
                                         linewidth=2.0, 
                                         shade=False)
            pltSurface.set_facecolor((0,0,0,0))
    elif plotType=="wireFrame" :#Only lines between points
        pltSurface = ax.plot_wireframe(np.reshape(x,(xNbPoints,yNbPoints)), 
                                       np.reshape(y,(xNbPoints,yNbPoints)), 
                                       np.reshape(z.values,(xNbPoints,yNbPoints)), 
                                       linewidth=2.0, 
                                       antialiased=True)
    elif plotType == "flexibleWire":
        colors = None if colorMap is None else colorMap(norm(z.values))
        pltSurface = ax.plot_trisurf(x, y, z.values, 
                                     linewidth=1.0, 
                                     antialiased=True, 
                                     cmap=colorMap, 
                                     norm=norm, 
                                     color=(0,0,0,0))
        scaleEdgeValue = pltSurface.to_rgba(pltSurface.get_array())
        pltSurface.set_edgecolors(scaleEdgeValue) 
        pltSurface.set_alpha(0)
        ax.set_facecolor('white')
    elif plotType=="transparent":
        pltSurface = ax.plot_trisurf(x, y, z.values, 
                                     cmap='viridis', 
                                     edgecolor='none', 
                                     alpha = 0.5, 
                                     shade=False)
    elif plotType=="scatter" :
        #Points only
        pltSurface = ax.scatter(x, y, z.values, 
                                linewidth=2.0, 
                                antialiased=True)
    else :
        pltSurface = ax.plot_trisurf(x, y, z.values, 
                                     linewidth=1.0, 
                                     antialiased=True, 
                                     cmap=colorMap, 
                                     norm=norm)
    if refPoints is not None :
        xRef, yRef = getCoordinates(coordinates.loc[refPoints.index])
        pltSurface1 = ax.scatter(xRef, yRef, refPoints.values, 
                                 c = "k",
                                 marker = "+",
                                 linewidth=3, 
                                 s = 200,
                                 antialiased=True,
                                 zorder=1)
    else :
        pltSurface1 = None
    
    ax.view_init(elev=10., azim=azimut)
    ax.set_xlabel(xLabel, fontsize=30, labelpad=30)
    ax.set_ylabel(yLabel, fontsize=30, labelpad=30)
    ax.set_zlabel(zLabel, fontsize=30, labelpad=30)
    ax.set_zlim(zMin, zMax)
    
    return pltSurface, pltSurface1

def plotGrid(surface, 
             coordinates, 
             title, 
             colorMapSystem=None, 
             plotType=None,
             refPoints = None,
             minValUser = None):
    filteredSurface, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(surface, coordinates)
    surfaceComplete = filteredSurface.dropna()
    if surfaceComplete.empty :
        print("All points are invalid, nothing can be plotted.")
        return
    x, y = getCoordinates(coordinates.loc[surfaceComplete.index])
    fig = plt.figure(figsize=(40,20))
    ax = fig.gca( projection='3d' )
    
    maxVal = tensorMax(surface)+0.1
    minVal = tensorMin(surface)-0.1 if minValUser is None else minValUser
    norm = plt.Normalize(minVal, maxVal)
    
    
    pltSurface, pltScatter = buildGrid(ax, x, y, surfaceComplete, 
                                       minVal, maxVal,
                                       coordinates, 
                                       colorMapSystem = colorMapSystem,
                                       plotType = plotType, 
                                       normOverride = norm,
                                       refPoints = refPoints)
    ax.set_title(title, pad = 30, fontsize = 20)#ax.set_title(surface.name)
    
    plt.show()
    return 

    
def updatePlotData(worstPred, worstRef, x, y, 
                   coordinates,
                   fig = None, 
                   maxZ = None, 
                   maxZerror = None,
                   colorMapSystem=None, 
                   plotType=None, 
                   minZ =None,
                   refPoints = None):

    #Build figure and subplots
    if fig is None :
        fig = plt.figure(figsize=(40,40), frameon = False)
        grid = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(grid[0,0], projection='3d', frame_on = False)
        ax2 = fig.add_subplot(grid[0,1], projection='3d', frame_on = False)
        ax3 = fig.add_subplot(grid[1,:], projection='3d', frame_on = False)
        
    
    maxVal = (maxZ if (maxZ is not None) else (max(tensorMax(worstPred), tensorMax(worstRef)) + 0.1))
    minVal = (minZ if (minZ is not None) else 0)
    #True Values
    z = worstRef
    ax1 = fig.axes[0]
    ax1.clear()
    artist1,scatter1 = buildGrid(ax1, x, y, z, 
                                 minVal, maxVal, 
                                 coordinates,
                                 colorMapSystem = colorMapSystem,
                                 plotType = plotType, 
                                 normOverride = None,
                                 refPoints = refPoints)
    ax1.set_title('True Vol', pad = 30, fontsize = 20)

    #Generated Values
    z = worstPred
    ax2 = fig.axes[1]
    ax2.clear()
    artist2,scatter2 = buildGrid(ax2, x, y, z, 
                                 minVal, maxVal, 
                                 coordinates,
                                 colorMapSystem = colorMapSystem,
                                 plotType = plotType, 
                                 normOverride = None,
                                 refPoints = refPoints)
    ax2.set_title('Generated Vol', pad = 30, fontsize = 20)
    
    #Errors
    errorGrid = np.abs(worstPred - worstRef)
    maxValErrorPlot = (maxZerror if (maxZerror is not None) else (tensorMax(errorGrid) + 1))
    z = errorGrid
    ax3 = fig.axes[2]
    ax3.clear()
    artist3,scatter3 = buildGrid(ax3, x, y, z, 
                                 0, maxValErrorPlot, 
                                 coordinates,
                                 colorMapSystem = colorMapSystem,
                                 plotType = plotType, 
                                 normOverride = None,
                                 refPoints = z.loc[refPoints.index] if refPoints is not None else None)
    ax3.set_title('Error Vol', pad = 30, fontsize = 20)
    
    return fig, [artist1, artist2, artist3, scatter1, scatter2, scatter3]
    
    
def getTotalImpliedVariance(surface, coordinates):
    maturities = coordinates.map(lambda x : x[0])[surface.index]
    impliedTotalVariance = np.square(surface) * maturities
    return impliedTotalVariance.rename(surface.name)
    

def plotResults(worstPred, worstRef, 
                coordinates,
                title = None, 
                colorMapSystem=None, 
                plotType=None,
                refPoints = None):
    filteredWorstRef, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(worstRef, coordinates)
    completedWorstRef = filteredWorstRef.dropna() 
    #Predicted values
    x, y = getCoordinates(coordinates.loc[completedWorstRef.index])
    minZ = (min(tensorMin(worstPred), tensorMin(worstRef)) - 0.1)
    fig, _ = updatePlotData(worstPred.loc[completedWorstRef.index], 
                            completedWorstRef, 
                            x, y,
                            coordinates,
                            fig = None, 
                            maxZ = None, 
                            maxZerror = None,
                            colorMapSystem=colorMapSystem, 
                            plotType=plotType, 
                            minZ =minZ,
                            refPoints = refPoints)
    if (title is not None):
        fig.suptitle(title)
    plt.show()
    plotTails([worstRef, coordinates, None, None], [worstPred, coordinates, None, None])
    
    return
    
def plotLossThroughEpochs(testingLoss, title=None):
    refSize=5
    plt.plot(testingLoss)
    plt.tick_params(axis='x', labelsize=2*refSize, pad=int(refSize/2))
    plt.tick_params(axis='y', labelsize=2*refSize, pad=int(refSize/2))
    plt.ylabel("Loss", fontsize=2*refSize, labelpad=3*refSize)
    plt.xlabel("Epochs", fontsize=2*refSize, labelpad=3*refSize)
    if (title is not None):
        plt.title(title, pad = 3*refSize, fontsize = 2*refSize)
    plt.show()
    return


def plotFactor(factors, title=None):#Until 84 factors
    refSize=5
    shapeLine = ['-','--','-.',':','o','.','1','s','*','p','D','^']
    colorLine = ['b','g','r','c','m','y','k']
    for i in range(factors.shape[1]):
        formatLine = shapeLine[int(i/len(colorLine))] + colorLine[int(i%len(colorLine))] 
        plt.plot(factors.iloc[:,i], formatLine, label=("factor " + str(i+1)))
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("Factors", fontsize=2*refSize, labelpad=3*refSize)
    plt.xlabel("Date", fontsize=2*refSize, labelpad=3*refSize)
    _ = plt.xticks(rotation=90)
    plt.tick_params(axis='x', labelsize=2*refSize, pad=int(refSize/2))
    plt.tick_params(axis='y', labelsize=2*refSize, pad=int(refSize/2))
    if (title is not None):
        plt.title(title, pad = 3*refSize, fontsize = 2*refSize)
    plt.show()
    return

def diagnoseModels(h1Eval, 
                   trainingApproximation, 
                   vol_Train, 
                   finalApproximation, 
                   vol_Test, 
                   testingLoss, 
                   dataSet,
                   colorMapSystem=None, 
                   plotType=None):
    
    printDelimiter(2)
    printIsolated("Showing results for Training Dataset : ")
    
    printIsolated("Factor values on first testing observation : ",h1Eval.iloc[0])
    
    errors,_,_,_ = errorPlot(trainingApproximation, 
                             vol_Train, 
                             title = "Daily reconstruction loss on training dataset")
    
    printDelimiter(2)
    printIsolated("Showing results for Testing Dataset : ")
    errorsAutoencoders,_,_,_ = errorPlot(finalApproximation, 
                                         vol_Test,
                                         title = "Daily reconstruction loss on testing dataset" )
    
    printDelimiter()
    worstDayPred, worstDayRef = getWorstGrids(finalApproximation, 
                                              vol_Test, 
                                              errorsAutoencoders)
    modelData = dataSet.getDataForModel(worstDayRef.name)
    coordinates = dataSet.formatModelDataAsDataSet(modelData)[1]
    plotResults(worstDayPred, 
                worstDayRef, 
                coordinates,
                title = "Worst reconstruction on testing dataset", 
                colorMapSystem=colorMapSystem, 
                plotType=plotType)
        
        
    plotResults(getTotalImpliedVariance(worstDayPred, coordinates),
                getTotalImpliedVariance(worstDayRef, coordinates),
                coordinates,
                title = "Worst Implied Total Variance reconstruction on testing data", 
                colorMapSystem=colorMapSystem, 
                plotType=plotType)
    
    plotGrid(modelData[0],
             coordinates,    
             "Model data for worst reconstruction on testing dataset", 
             colorMapSystem=colorMapSystem, 
             plotType=plotType,
             refPoints = None)
    
    printDelimiter()
    plotLossThroughEpochs(testingLoss, 
                          title = "Reconstruction loss on testing dataset through epochs")
    
    printDelimiter()
    plotFactor(h1Eval, title = "Factors returned by encoder on training dataset")
    return

def plotCompletion(trueSurface, 
                   completedSurface,
                   coordinates,
                   colorMapSystem=None, 
                   plotType=None,
                   refPoints = None):
    #plotResults(completedSurface, 
    #            trueSurface, 
    #            coordinates,
    #            title = "Completion", 
    #            colorMapSystem=colorMapSystem, 
    #            plotType=plotType)
    plotGrid(trueSurface, 
             coordinates,
             "Reference Surface", 
             colorMapSystem=colorMapSystem, 
             plotType=plotType,
             refPoints = refPoints)
    plotGrid(completedSurface,  
             coordinates,
             "Completed Surface", 
             colorMapSystem=colorMapSystem, 
             plotType=plotType,
             refPoints = refPoints)
    completionDiff = np.abs(trueSurface - completedSurface)
    plotGrid(completionDiff,  
             coordinates,
             "Absolute Error", 
             colorMapSystem=colorMapSystem, 
             plotType=plotType,
             refPoints = completionDiff.loc[refPoints.index] if refPoints is not None else None)
    return

plt.rcParams['animation.embed_limit'] = 2**28
plt.rcParams['animation.html'] = 'html5'


def animationDiagnosis(volTestSeries, volPredSeries, fps, 
                       isVol = True,
                       colorMapSystem = None,
                       plotType = None):
    N = volTestSeries.shape[1] # Meshsize
    
    frn = volTestSeries.shape[0] # frame number of the animation
    maxVal = max(tensorMax(volTestSeries),tensorMax(volPredSeries)) + 0.1
    minVal = min(tensorMin(volTestSeries),tensorMin(volPredSeries)) - 0.1 if not isVol else 0
    maxValError = tensorMax(np.abs(volPredSeries - volTestSeries))
    
    #Reshape volTestSeries as dataframe
    framePred = pd.DataFrame(volPredSeries,
                             index= volTestSeries.index, 
                             columns = volTestSeries.columns)
    
    #Build figure and subplot grid
    filteredVolTest, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(volTestSeries.iloc[0], 
                                                                                       coordinates.iloc[0])
    completeVolTest = filteredVolTest.dropna()
    x, y = getCoordinates(coordinates.iloc[0].loc[completeVolTest.index])
    fig1, artists = updatePlotData(framePred.iloc[0].loc[completeVolTest.index], 
                                   completeVolTest,
                                   x, y, 
                                   coordinates.iloc[0],
                                   fig = None, 
                                   maxZ = maxVal, 
                                   maxZerror = maxValError,
                                   colorMapSystem = colorMapSystem, 
                                   plotType = plotType, 
                                   minZ = minVal)
    
    def update_multiplePlot(frame_number, volTest, volPred, f, coordinates, plots):
        #update data
        filteredVolTest, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(volTest.iloc[frame_number], 
                                                                                           coordinates.iloc[frame_number])
        completeVolTest = filteredVolTest.dropna()
        xValues, yValues = getCoordinates(coordinates.iloc[frame_number].loc[completeVolTest.index])
        f , plots = updatePlotData(volPred.iloc[frame_number].loc[completeVolTest.index],
                                   completeVolTest,
                                   xValues, yValues,  
                                   coordinates.iloc[frame_number],
                                   fig = f, 
                                   maxZ = maxVal, 
                                   maxZerror = maxValError,
                                   colorMapSystem = colorMapSystem, 
                                   plotType = plotType, 
                                   minZ = minVal)
        f.suptitle(volPred.iloc[frame_number].name)
        return plots
    
    ani = animation.FuncAnimation(fig1, 
                                  update_multiplePlot, 
                                  frames = frn, 
                                  fargs=(volTestSeries, framePred, fig1, coordinates, artists), 
                                  interval=1000/fps, 
                                  blit=False)
    
    
    return ani

def animationSingleVolatility(volSeries, 
                              coordinates, 
                              fps, 
                              isVol = True,
                              colorMapSystem = None,
                              plotType = None):
    N = volSeries.shape[1] # Meshsize
    
    frn = volSeries.shape[0] # frame number of the animation
    maxVal = tensorMax(volSeries) + 0.1
    minVal = tensorMin(volSeries) -0.1 if not isVol else 0
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.gca( projection='3d' )
    
    filteredVolTest, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(volSeries.iloc[0], 
                                                                                       coordinates.iloc[0])
    completeVolObs = filteredVolTest.dropna()
    x, y = getCoordinates(coordinates.iloc[0].loc[completeVolObs.index])
    surface, scatter = buildGrid(ax, x, y, completeVolObs, 
                                 minVal, maxVal, 
                                 coordinates.iloc[0],
                                 colorMapSystem = colorMapSystem,
                                 plotType = plotType)
    ax.set_title(volSeries.iloc[0].name)
    
    
    def update_plot(frame_number, soln, figure, coordinates, artist):
        axe = figure.axes[0]
        #artist.remove()
        axe.clear()
        filteredSoln, filteredCoordinates = loadData.removePointsWithInvalidCoordinates(soln.iloc[frame_number], 
                                                                                        coordinates.iloc[frame_number])
        completeVolObs = filteredSoln.dropna()
        xValues, yValues = getCoordinates(coordinates.iloc[frame_number].loc[completeVolObs.index])
        artist,artist1 = buildGrid(axe, xValues, yValues, completeVolObs, 
                                   minVal, maxVal, 
                                   coordinates.iloc[frame_number],
                                   colorMapSystem = colorMapSystem,
                                   plotType = plotType)
        axe.set_title(soln.iloc[frame_number].name)
        return artist,artist1
    
    
    
    ani = animation.FuncAnimation(fig, 
                                  update_plot, 
                                  frames = frn, 
                                  fargs=(volSeries, fig, coordinates, [surface,scatter]), 
                                  interval=1000/fps, 
                                  blit=False)
    
    return ani

def createTupleListFromArray(arrayCoordinates):
    return list(map(lambda x : (x[0],x[1]), arrayCoordinates[:,:2]))

def standardInterp(surface, 
                   coordinates, 
                   kind = 'cubic', 
                   colorMapSystem=None, 
                   plotType=None):
    xPoints, yPoints = getCoordinates(coordinates)
    formerCoordinates = np.vstack((np.ravel(xPoints), np.ravel(yPoints))).T
    
    xMax = np.nanmax(xPoints)
    yMax = np.nanmax(yPoints)
    xMin = np.nanmin(xPoints)
    yMin = np.nanmin(yPoints)
    
    xValues = np.linspace(xMin, xMax, num=100)
    yValues = np.linspace(yMin, yMax, num=100)
    grid = np.meshgrid(xValues,yValues)
    
    
    xInterpolated = np.reshape(grid[0],(100 * 100,1))
    yInterpolated = np.reshape(grid[1],(100 * 100,1))
    newCoordinates = np.concatenate([xInterpolated, yInterpolated], axis=1)
    newCoordinatesSeries = pd.Series(createTupleListFromArray(newCoordinates))
    
    
    f = scipy.interpolate.griddata(formerCoordinates, 
                                   surface, 
                                   newCoordinates, 
                                   method=kind, 
                                   fill_value=np.nan, 
                                   rescale=False)
    
    interpolatedValues = pd.Series(np.reshape(f,(100 * 100,)), 
                                   index = newCoordinatesSeries.index)
    
    plotGrid(interpolatedValues, 
             newCoordinatesSeries,
             "Interpolated Data : " + kind, 
             colorMapSystem=colorMapSystem, 
             plotType=plotType)
    
    return interpolatedValues, newCoordinatesSeries

#Save as a html tag
def saveAnimation(anim, fileName):
    s = anim.to_jshtml()
    f = open(fileName + ".html","w")
    f.write(s)
    f.close()
    return

# anim_file = 'cvae.gif'

# with imageio.get_writer(anim_file, mode='I') as writer:
  # filenames = glob.glob('image*.png')
  # filenames = sorted(filenames)
  # last = -1
  # for i,filename in enumerate(filenames):
    # frame = 2*(i**0.5)
    # if round(frame) > round(last):
      # last = frame
    # else:
      # continue
    # image = imageio.imread(filename)
    # writer.append_data(image)
  # image = imageio.imread(filename)
  # writer.append_data(image)

# import IPython
# if IPython.version_info >= (6,2,0,''):
  # display.Image(filename=anim_file)
