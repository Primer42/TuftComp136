'''
Created on Oct 27, 2012

@author: will
'''
import os
from numpy.lib.npyio import genfromtxt
import sys
from numpy.numarray.numerictypes import Float64, Int8
from numpy.lib.shape_base import array_split
from numpy.core.shape_base import hstack, vstack
from itertools import izip
from math import exp,log
from numpy import dot, random
from numpy.dual import inv
from numpy.core.fromnumeric import transpose, mean
from numpy.core.numeric import zeros, outer, array, ones, newaxis, array_equal
from numpy.linalg.linalg import det, LinAlgError
from numpy.ma.core import identity, std
from os import path
import matplotlib.pyplot as plt
import numpy
from numpy.lib.scimath import sqrt
from numpy.ma.extras import diagflat
import math
import itertools

def readData(dataDir, datasetName):
    dataFileLoc = os.path.join(dataDir, datasetName + '.csv')
    labelsFileCol = os.path.join(dataDir, 'labels-' + datasetName + '.csv')

    data = genfromtxt(dataFileLoc, dtype=Float64, delimiter=',')
    #this will create a data vector where each ROW is an example
    #this makes indexing easier for the generative models and Fisher's discriminent, as they all work over individual vectors
    #Bayesian regriession uses a Phi matrix where each row is an example, so that is exactly this matrix
    labels = genfromtxt(labelsFileCol, dtype=Int8, delimiter=',')
    
    return (data, labels)

def crossValidation(numFolds, data, labels, algorithm, accuracyList, learningCurveList, numLearningCurveIterations, learningCurveIndexMod):
    dataFolds = array_split(data, numFolds)
    labelFolds = array_split(labels, numFolds)
    for testIndex in range(numFolds):
        print testIndex,
        testData = dataFolds.pop(testIndex)
        testLabels = labelFolds.pop(testIndex)
        trainData = vstack(dataFolds)
        trainLabels = hstack(labelFolds)
        accuracyList.append(algorithm(trainData, trainLabels, testData, testLabels))
        learningCurve(algorithm, learningCurveList, trainData, trainLabels, testData, testLabels, numLearningCurveIterations, learningCurveIndexMod)
        dataFolds.insert(testIndex, testData)
        labelFolds.insert(testIndex, testLabels)
    print ''
    
def learningCurve(algorithm, learningCurveList, trainData, trainLabels, testData, testLabels, numIteration, indexMod):
    #take bigger and bigger slices of the training data
    #train the model, and record the results in the learningCurveList
    #To reduce variance, do multiple tests with each training subset size
    #selecting randomly from the training data
    totalNumTrainExamples = len(trainLabels)
    for trainSize in range(totalNumTrainExamples / 10, totalNumTrainExamples, indexMod):
        learningCurveList[trainSize].append(algorithm(trainData[:trainSize], trainLabels[:trainSize], testData, testLabels))
            
def sigmoid(a):
    return 1.0 / (1.0 + math.e**(-1.0 * a))

def estimateGaussian(trainData, trainLabels):
    numClasses = max(trainLabels) + 1
    N = [0]* numClasses
    mu = [0.0] * numClasses
    Slist = [zeros((trainData.shape[1], trainData.shape[1]))] * numClasses
    pList = [0.0] * numClasses
    
    #calculate N, and sum x's for mu
    for x,t in izip(trainData, trainLabels):
        N[t] += 1
        mu[t] += x
    #normalize mu
    for i in range(numClasses):
        mu[i] = mu[i] / float(N[i])
    #calculate the class probabilities
    for i in range(numClasses):
        pList[i] = float(N[i]) / sum(N)
    #calculate S0 and S1
    for x,t in izip(trainData, trainLabels):
        Slist[t] += outer(x - mu[t], x - mu[t])
        try:
            inv(Slist[t])
        except LinAlgError:
            Slist[t] += 0.1 * identity(Slist[t].shape[0], Float64)
        
    return (numClasses, N, mu, Slist, pList)


def generativeSharedCov(trainData, trainLabels, testData, testLabels):
    (numClasses, N, mu, Slist, pList) = estimateGaussian(trainData, trainLabels)
    #i.e. calculate everything we need for the model
    #normalize the S's, and calculate the final S
    S = zeros(Slist[0].shape)
    for i in range(numClasses):
        Slist[i] = Slist[i] / float(N[i])
        S += pList[i] * Slist[i]
    
    w = dot(inv(S), (mu[1] - mu[0]))
    w0 = -0.5* dot(dot(mu[1], inv(S)), mu[1]) + 0.5*dot(dot(mu[0], inv(S)), mu[0]) + log(pList[1]/pList[0])
    
    numCorrect = 0
    for x,t in izip(testData, testLabels):
        probClass1 = sigmoid(dot(w, x) + w0)
        if probClass1 >= 0.5:
            if t == 1:
                numCorrect += 1
        else:
            if t == 0:
                numCorrect += 1
    return float(numCorrect) / float(len(testLabels))

def generativeSeperateCov(trainData, trainLabels, testData, testLabels):
    (numClasses, N, mu, Slist, pList) = estimateGaussian(trainData, trainLabels)
    
    numCorrect = 0
    for x,t in izip(testData, testLabels):
        pXgivenClassList = []
        for i in range(numClasses):
            pXgivenClassList.append(1/sqrt(det(Slist[i])) + exp(-0.5 * dot(dot((x - mu[i]), inv(Slist[i])), (x-mu[i]))))
        a = log((pXgivenClassList[1]*pList[1]) / (pXgivenClassList[0]*pList[0]))
        probClass1 = sigmoid(a)
        if probClass1 >= 0.5:
            if t == 1:
                numCorrect += 1
        else:
            if t == 0:
                numCorrect += 1
    return float(numCorrect) / float(len(testLabels))

def fishersLinearDiscriminent(trainData, trainLabels, testData, testLabels):
    numClasses = max(trainLabels) + 1
    N = [0] * numClasses
    m = [0] * numClasses
    for x,t in izip(trainData,trainLabels):
        m[t] += x
        N[t] += 1
    for i in range(numClasses):
        m[i] /= N[i]
    Sw = zeros((trainData.shape[1], trainData.shape[1]))
    for x,t in izip(trainData, trainLabels):
        Sw += outer(x-m[t], x-m[t])
    try:
        inv(Sw)
    except LinAlgError:
        Sw += 0.1 * identity(Sw.shape[0], Float64)    
    
    w = dot(inv(Sw),(m[0] - m[1]))
    meanVect = (N[0]*m[0] + N[1]*m[1]) / sum(N)
    
    numCorrect = 0
    for x,t in izip(testData, testLabels):
        if dot(w, (x-meanVect)) > 0:
            if t == 1:
                numCorrect += 1
        else:
            if t == 0:
                numCorrect += 1
    return float(numCorrect) / float(len(testLabels))

def yScalar(w, data):
    return sigmoid(dot(w,data))

def yVector(w, data):
    return hstack([yScalar(w,x) for x in data])

def R(yVect):
    #yVect = yVector(w, data)
    rVals = zeros(yVect.shape)
    for rowNum, val in enumerate(yVect):
        rVals[rowNum] += val * (1-val)
    return diagflat(rVals)
    

def logisticRegression(trainData, trainLabels, testData, testLabels):
    #adjust the data, adding the 'free parameter' to the train data
    trainDataWithFreeParam = hstack((trainData.copy(), ones(trainData.shape[0])[:,newaxis]))
    testDataWithFreeParam = hstack((testData.copy(), ones(testData.shape[0])[:,newaxis]))
    
    alpha = 10
    oldW = zeros(trainDataWithFreeParam.shape[1])
    newW = ones(trainDataWithFreeParam.shape[1])
    iteration = 0
    
    trainDataWithFreeParamTranspose = transpose(trainDataWithFreeParam)
    alphaI = alpha * identity(oldW.shape[0])
    
    while not array_equal(oldW, newW):
        if iteration == 100:
            break
        oldW = newW.copy()
        
        yVect = yVector(oldW, trainDataWithFreeParam)
        r = R(yVect)

        firstTerm = inv(alphaI + dot(dot(trainDataWithFreeParamTranspose, r), trainDataWithFreeParam))
        secondTerm = dot(trainDataWithFreeParamTranspose, (yVect-trainLabels)) + alpha * oldW
        newW = oldW - dot(firstTerm, secondTerm)
        iteration += 1
                              
        
    #see how well we did
    numCorrect  = 0
    for x,t in izip(testDataWithFreeParam, testLabels):
        
        if yScalar(newW, x) >= 0.5:
            if t == 1:
                numCorrect += 1
        else:
            if t == 0:
                numCorrect += 1
    return float(numCorrect) / float(len(testLabels))
        
        
    
    
        
def getAlgorithmLearningCurveFigureLoc(figureDir, algorithmName):
    ret = os.path.join(figureDir, '-'.join((algorithmName, 'learning_curve.pdf')))
    if not os.path.exists(path.dirname(ret)):
        os.makedirs(path.dirname(ret))
    return ret

def getDatasetLearningCurveFigureLoc(figureDir, dataset):
    ret =  os.path.join(figureDir, '-'.join((dataset, 'learning_curve.pdf')))
    if not os.path.exists(path.dirname(ret)):
        os.makedirs(path.dirname(ret))
    return ret

def getLearningCurveFigureLoc(figureDir, algorithmName, datasetName):
    ret = os.path.join(figureDir, '-'.join((algorithmName, datasetName)) + '.pdf')
    if not os.path.exists(path.dirname(ret)):
        os.makedirs(path.dirname(ret))
    return ret
    
    
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: %s <data dir> <figure dir>" % sys.argv[0]
        exit(1)
    (dataDir, figureDir) = sys.argv[1:]

    numCrossValidationFolds = 10
    numLearningCurveIterations = 1
    learningCurveIndexMod = 10
    
    allDataDirFiles = os.listdir(dataDir)
    datasetNames = [x.split('.')[0] for x in allDataDirFiles if not (x.startswith('labels-') or x.startswith('.'))]

    algorithmList = (generativeSharedCov,generativeSeperateCov, fishersLinearDiscriminent, logisticRegression)
    #algorithmList = (generativeSharedCov,generativeSeperateCov, fishersLinearDiscriminent)
    #algorithmList = (logisticRegression,)
    
    #datasetNames = 'B'
    #datasetNames.remove('A')
    #algorithmList = (generativeSharedCov,)

    statList = []
    learningCurveStatList = []
    for dataset in datasetNames:
        print dataset
        (data, labels) = readData(dataDir, dataset)
        for algorithm in algorithmList:
            print algorithm.__name__, '\t',
            accuracyList = []
            #the learning curve list has one list of values for each training set size
            #so if we're trying a training set of size 10, learningCurveList[10] will be a
            #list of the accuracies from the test with training size 10
            learningCurveList = []
            for x in range(len(labels)):
                learningCurveList.append([])
            crossValidation(numCrossValidationFolds, data, labels, algorithm, accuracyList, learningCurveList, numLearningCurveIterations, learningCurveIndexMod)
            statList.append((dataset, algorithm.__name__, mean(accuracyList), std(accuracyList)))
            learningCurveStatList.append((dataset, algorithm.__name__, 
                                          [mean(x) for x in learningCurveList],
                                          [std(x) for x in learningCurveList]))
            
    outFile = open(path.join(figureDir, "table.txt"), 'a')
    outFile.write('algorithm')
    for ds in datasetNames:
        outFile.write('&& ' + ds)
    outFile.write('\\\\ \n')
    
    for alg in [x.__name__ for x in algorithmList]:
        outFile.write(alg)
        for ds in datasetNames:
            relevantTupList = [x for x in statList if x[0] == ds and x[1] == alg]
            if len(relevantTupList) != 1:
                print "Got %d relevant tups for %s, %s" % (len(relevantTupList), alg, ds)
                print relevantTupList
                exit(1)
            rt = relevantTupList[0]
            outFile.write("& %.3f +/- %.3f" % (rt[2], rt[3]))
        outFile.write('\\\\ \n')
    
    lines = ['-','--','-.',':']
    lineCycler = itertools.cycle(lines)
    
    #now graph the learning curves
    for ds in datasetNames:
        #figure out the training set size for the learning curve
        trainingIndexList = []
        relevantTups = [x for x in learningCurveStatList if x[0] == ds]
        for (i, val) in enumerate(relevantTups[0][2]):
            #if not numpy.isnan(val):
            #    if i < minTrainingSize:
            #        minTrainingSize = i
            #    if i > maxTrainingSize:
            #        maxTrainingSize = i
            if not numpy.isnan(val):
                trainingIndexList.append(i)
                    
#        maxTrainingSize += 1
                            
#        trainingSizeValueArrays = array(range(minTrainingSize, maxTrainingSize))
        trainingIndexArray = array(trainingIndexList)
        #trainingSizeValueArrays  = array(range(len(relevantTups[0][2])))
        for rt in relevantTups:
            #trim the data to remove non-nana values
            plt.errorbar(trainingIndexArray, array(rt[2])[trainingIndexArray], fmt=next(lineCycler), yerr=array(rt[3])[trainingIndexArray])
            plt.title('-'.join((rt[0],rt[1])))
            #plt.legend(loc=0)
            plt.savefig(getLearningCurveFigureLoc(figureDir, rt[1], ds))
            plt.clf()
        for rt in relevantTups:
            plt.errorbar(trainingIndexArray, array(rt[2])[trainingIndexArray], yerr=array(rt[3])[trainingIndexArray], label=rt[1], fmt=next(lineCycler))
        plt.title(ds)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=2)
        plt.savefig(getDatasetLearningCurveFigureLoc(figureDir, ds), bbox_inches='tight', pad_inches=.85)
        plt.clf()
            
        