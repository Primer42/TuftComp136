'''
Created on Oct 6, 2012

@author: will
'''
import os
import csv
import numpy
from numpy.lib.shape_base import row_stack
from numpy import dot, array
from numpy.dual import inv
from numpy.core.numeric import arange, identity, ones
from numpy.core.fromnumeric import squeeze, mean, argmin
import matplotlib.pyplot as plt
import sys
from random import shuffle
from numpy.ma.core import abs
from numpy.linalg.linalg import eig
from timeit import timeit
from functools import wraps
from numpy.numarray.numerictypes import Float64

TRAIN = "train"
TRAIN_LABELS = "trainR"
TEST = "test"
TEST_LABELS = "testR"

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwds):
        def timeitFuncWrap():
            func(*args)
        numTests = 5
        elapsed = float(timeit(timeitFuncWrap, number = numTests)) / numTests
        print "%s & %s & %f \\\\    " % (func.__name__, args[2], elapsed)
        return
    return wrapper

def readData(dataDir, modelName):
    filePrefixes = [TRAIN, TRAIN_LABELS, TEST, TEST_LABELS]
    
    dataList = dict()
    
    #maybe use numpy fromfile here instead of rolling my own?
    
    for filePrefix in filePrefixes:
        fileLoc = os.path.join(dataDir, '-'.join([filePrefix, modelName]) + ".csv")
        #handle the test files for the 3 subsets of 1000-100
        if '(' in fileLoc and "test" in filePrefix:
            splitLoc = [x for x in fileLoc.split('-')]
            offendingSegment = splitLoc[-2]
            offendingSegment = offendingSegment[offendingSegment.find("(")+1: offendingSegment.find(')')]
            splitLoc[-2] = offendingSegment
            fileLoc = '-'.join(splitLoc)
            
        with open(fileLoc, 'r') as modelFile:
            data = []
            modelReader = csv.reader(modelFile)
            for row in modelReader:
                rowVector = numpy.array(row, dtype=numpy.float64)
                data.append(rowVector)
            dataList[filePrefix] = data     
    return dataList
    
def phi(dataVectorList):
    return row_stack(dataVectorList)

def tListToTVector(tList):
    return row_stack(tList)

def doubleU(phi, l, tVector):
    #can't call lamba by it's name, because that's a reserved word in python
    #so I'm calling it l
    lIdentity = l*identity(phi.shape[1])
    phiDotPhi = dot(phi.transpose(), phi)
    firstTerm = inv(lIdentity + phiDotPhi)
    phiDotT = dot(phi.transpose(), tVector)
    return squeeze(dot(firstTerm, phiDotT))

def MSE(data, weights, labels):
    v = 0.0;
    for i in range(len(data)):
        v += squeeze(dot(data[i], weights) - labels[i]) ** 2
    v /= len(data)
    return v

def problem1(data, figureDir, figureName, targetValue):
    figureOutLoc = os.path.join(figureDir, '1', figureName + ".eps")
    if os.path.exists(figureOutLoc):
        return
    if not os.path.exists(os.path.dirname(figureOutLoc)):
        os.makedirs(os.path.dirname(figureOutLoc))
    trainList = []
    testList = []
    for l in range(151):
        w = doubleU(phi(data[TRAIN]), l, tListToTVector(data[TRAIN_LABELS]))
        trainMSE = MSE(data[TRAIN], w, data[TRAIN_LABELS])
        testMSE = MSE(data[TEST], w, data[TEST_LABELS])
        trainList.append(trainMSE)
        testList.append(testMSE)
    trainArray = squeeze(row_stack(trainList))
    testArray = squeeze(row_stack(testList))
    
    #find the best l value on the test set
    targetArray = targetValue * ones(151, dtype=numpy.float64)
    targetDiffArray = testArray - targetArray
    bestL = argmin(targetDiffArray)
    
    lArray = arange(151).reshape(-1)

    plt.plot(lArray, trainArray, '-', label="Train")
    plt.plot(lArray, testArray, '--', label="Test")
    plt.plot(lArray, targetArray, ':', label="Target")
    plt.title(figureName)
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.ylim(ymax = min((plt.ylim()[1], 7.0)))
    #add a label showing the min value, and annotate it's lvalue
    if series == 'wine':
        annotateOffset = .02
    else:
        annotateOffset = .2
    plt.annotate("Best lambda value = " + str(bestL) + " MSE = %.3f" %testList[bestL], 
                 xy=(bestL, testArray[bestL]), 
                 xytext=(bestL + 10, testArray[bestL] - annotateOffset),
                 bbox=dict(boxstyle="round", fc="0.8"), 
                 arrowprops=dict(arrowstyle="->"))
     
    plt.legend(loc=0)
    
    plt.savefig(figureOutLoc)
    plt.clf()
    
def selectSample(dataList, indexes):
    sample = []
    for i in indexes:
        sample.append(dataList[i])
    return sample
    
def getProblem2FigureLoc(figureDir, dataName, l, repeat):
    problem2FigureDir = os.path.join(figureDir, '2')
    if not os.path.exists(problem2FigureDir):
        os.makedirs(problem2FigureDir)
    return os.path.join(problem2FigureDir, '-'.join([dataName, str(l), str(repeat), "problem2"]) + '.eps')

def problem2(data, figureDir, dataName, l, maxNumRepetitions, minSampleSize, targetValue):
    if os.path.exists(getProblem2FigureLoc(figureDir, dataName, l, maxNumRepetitions)):
        #this implies that all figures before it have been created already, so we don't need to repeat them
        return
    MSEValues = [[] for x in xrange(minSampleSize, len(data[TRAIN]) + 1)]
    sampleSizeValueList = range(minSampleSize, len(data[TRAIN]) + 1, 1)
    sampleSizeValueArray = array(sampleSizeValueList)
    
    targetArray = targetValue * ones(len(sampleSizeValueList), dtype=numpy.float64)
    
    for repeatNum in range(1, maxNumRepetitions+1):
        #randomly choose ordering of the samples for this run
        #make the range of indexes, then shuffle them into a random order
        randomlySortedIndexes = range(len(data[TRAIN]))
        shuffle(randomlySortedIndexes)
        
        #start with a sample size of one, go to the total training set
        for sampleSizeIndex, sampleSize in enumerate(sampleSizeValueList):
            curSampleIndexesList = randomlySortedIndexes[:sampleSize]
            curTrainSample = selectSample(data[TRAIN], curSampleIndexesList)
            curTrainLabelSample = selectSample(data[TRAIN_LABELS], curSampleIndexesList)
            w = doubleU(phi(curTrainSample), l, tListToTVector(curTrainLabelSample))
            curSampleMSE = MSE(data[TEST], w, data[TEST_LABELS])
            MSEValues[sampleSizeIndex].append(squeeze(curSampleMSE))
    #have a sample size of 0 is meaningless
    curRepeatMeanMSEValues = array([mean(array(x, dtype=numpy.float64)) for x in MSEValues])
        
    plt.plot(sampleSizeValueArray, curRepeatMeanMSEValues, '-', label="Learning curve")
    plt.plot(sampleSizeValueArray, targetArray, '--', label="Target MSE")
    plt.title("lamba = " + str(l) + " - " + str(repeatNum) + " repetitions")
    plt.xlabel("Sample Size - minimum " + str(minSampleSize))
    plt.ylabel("MSE on Full Test Set")
    plt.xlim(xmin=targetValue - .5)
    plt.legend(loc=0)
    plt.savefig(getProblem2FigureLoc(figureDir, dataName, l, repeatNum))
    plt.clf()
     
def problem3(data, outDir, outName, targetValue):
    resultOutLoc = os.path.join(outDir, '3', outName + '.txt')
    if os.path.exists(resultOutLoc):
        return
    if not os.path.exists(os.path.dirname(resultOutLoc)):
        os.makedirs(os.path.dirname(resultOutLoc))
    avgMSEList = []
    numFolds = 10
    trainDataSize = len(data[TRAIN])
    foldSize = len(data[TRAIN]) / numFolds
    for l in range(151):
        lMSEList = []
        for foldNum in range(numFolds):
            trainFoldIndexes = range(foldSize * foldNum) + range(foldSize * (foldNum + 1), trainDataSize)
            testFoldIndexes = range(foldSize * foldNum, foldSize * (foldNum + 1))
            trainFoldData = selectSample(data[TRAIN], trainFoldIndexes)
            trainFoldLabels = selectSample(data[TRAIN_LABELS], trainFoldIndexes)
            testFoldData = selectSample(data[TRAIN], testFoldIndexes)
            testFoldLabels = selectSample(data[TRAIN_LABELS], testFoldIndexes)
            w = doubleU(phi(trainFoldData), l, tListToTVector(trainFoldLabels))
            mse = MSE(testFoldData, w, testFoldLabels)
            lMSEList.append(mse)
            
        #average the results, and add it to the list
        avgMSEList.append(float(sum(lMSEList)) / numFolds)
    #find the best avg MSE
    bestAvgMSE = avgMSEList[0]
    bestL = 0
    for lVal, avgMSE in enumerate(avgMSEList):
        if abs(avgMSE-targetValue) < abs(bestAvgMSE - targetValue):
            bestL = lVal
            bestAvgMSE = avgMSE
            
    outFile = open(resultOutLoc, 'w')
    outFile.write(str(bestL) + '\t' + str(bestAvgMSE))
    outFile.close()
   
   
def gamma(alpha, beta, eignValues):
    gamma = 0.0 
    betaScaledEignValues = beta * eignValues
    for ev in betaScaledEignValues:
        gamma += ev / (alpha + ev)
    return gamma
   
def alphaF(g, mN):
    a = g / squeeze(dot(mN.transpose(), mN))
    return a

def betaF(g, mN, data):
    summation = 0.0
    N = len(data[TRAIN_LABELS])
    for n in range(N):
        summation += (squeeze(data[TRAIN_LABELS][n] - dot(mN.transpose(), data[TRAIN][n])) ** 2)
    if abs(N-g) < .001:
        g += .001
    oneOverBeta = summation / (N-g)
    b = 1.0 / oneOverBeta
    return b
        
def SN(alpha, beta, phi):
    betaPhiTphi = beta * dot(phi.transpose(), phi)
    alphaI = alpha * identity(betaPhiTphi.shape[0])
    SNinverse = alphaI + betaPhiTphi
    return inv(SNinverse)

def mN(alpha, beta, phi, tVect):
    s = SN(alpha, beta, phi)
    sPhiT = dot(s, phi.transpose())
    return squeeze(beta * dot(sPhiT, tVect))
            
def problem4(data, outDir, outName, targetValue):
    resultOutLoc = os.path.join(outDir, '4', outName + '.txt')
    if os.path.exists(resultOutLoc):
        return
    if not os.path.exists(os.path.dirname(resultOutLoc)):
        os.makedirs(os.path.dirname(resultOutLoc))
    
    p = phi(data[TRAIN])
    tVect = tListToTVector(data[TRAIN_LABELS])
    eignValues = eig(dot(p.transpose(), p))[0]
    
    convergenceDelta = .1
    lastAlpha = 0.0
    lastBeta = 0.0
    alpha = 1.0
    beta = 1.0
    while abs(lastAlpha - alpha) > convergenceDelta or abs(lastBeta - beta) > convergenceDelta:
        lastAlpha = alpha
        lastBeta = beta
        m = mN(lastAlpha, lastBeta, p, tVect)
        g = gamma(lastAlpha, lastBeta, eignValues)
        alpha = Float64(alphaF(g, m))
        beta = Float64(betaF(g, m, data))
    testMSE = MSE(data[TEST], mN(alpha, beta, p, tVect), data[TEST_LABELS])
    
    outFile = open(resultOutLoc, 'w')
    outFile.write('\t'.join([str(alpha), str(beta), str(testMSE)]))
    outFile.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: %s <data dir> <figure dir>" % sys.argv[0]
        exit()
        
    (dataDir, figDir) = sys.argv[1:]
    
    allDataFiles = os.listdir(dataDir)
    seriesSet = set(['-'.join(x.replace('.csv','').split('-')[1:]) for x in allDataFiles])

    targetDict = {'1000-100':4.015,'50(1000)-100':4.015, '150(1000)-100':4.015, '100-10':3.78, '100-100':3.78, '100(1000)-100':4.015, 'wine':0.62}
    
    #run problem 1 on all series
    for series in seriesSet:
        data = readData(dataDir, series)
        problem1(data, figDir, series, targetDict[series])
        #problem3(data, figDir, series, targetDict[series])
        #problem4(data, figDir, series, targetDict[series])
    exit()
    #run problem 2 on the 1000-100 series
    prob2series = '1000-100'
    prob2data = readData(dataDir, prob2series)
    problem2(prob2data, figDir, prob2series, 20, 20, 10, targetDict[prob2series])
    problem2(prob2data, figDir, prob2series, 80, 20, 10, targetDict[prob2series])
    problem2(prob2data, figDir, prob2series, 140, 20, 10, targetDict[prob2series])