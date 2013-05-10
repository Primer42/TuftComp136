'''
Created on Nov 17, 2012

@author: will
'''
from multiprocessing.pool import ThreadPool
import os
from numpy.lib.npyio import genfromtxt
from numpy.numarray.numerictypes import Float64
import sys
from svm import svm_problem, svm_parameter
from svmutil import svm_train, svm_save_model, svm_load_model, svm_predict,\
    evaluations
from numpy.lib.shape_base import array_split, column_stack
from numpy.core.shape_base import vstack, hstack
from numpy import dot, rint
from numpy.core.fromnumeric import transpose
from numpy.linalg.linalg import norm, LinAlgError, svd
import pickle
from numpy.core.numeric import zeros, where, identity, ones, array, outer
from glob import glob
from numpy.ma.core import sum, exp, arange, mean, absolute, diag
from numpy.dual import inv, eig
import csv
import itertools
from numpy.ma.extras import unique
from scipy.optimize.optimize import brute
from scipy.optimize._minimize import minimize
import shutil
from scipy.optimize.tnc import fmin_tnc, LSFAIL, INFEASIBLE, CONSTANT,\
    NOPROGRESS, USERABORT, MAXFUN
from numpy.lib.type_check import nan_to_num
from matplotlib.pyplot import ylim, plot, savefig, clf

class crossValidationGen:
    def __init__(self, numFolds, data, labels):
        self._numFolds = numFolds
        self._curFold = 0
        self._trainData = array_split(data, self._numFolds)
        self._testData = None
        self._trainLabels = array_split(labels, self._numFolds)
        self._testLabels = None
        
    def __iter__(self):
        return self
    
    def next(self):
        if self._curFold < self._numFolds-1:
            if self._testData is not None:
                self._trainData.insert(self._curFold, self._testData)
                self._trainLabels.insert(self._curFold, self._testLabels)
                self._curFold += 1
            self._testData = self._trainData.pop(self._curFold)
            self._testLabels = self._trainLabels.pop(self._curFold)      
            ret = (vstack(self._trainData), hstack(self._trainLabels), self._testData, self._testLabels)
            #print self._curFold, self._numFolds, [x.shape for x in ret]
            return ret
        else:
            raise StopIteration()

    def get_num_folds(self):
        return self._numFolds


    def get_cur_fold(self):
        return self._curFold


    def get_train_data(self):
        return self._trainData


    def get_test_data(self):
        return self._testData


    def get_train_labels(self):
        return self._trainLabels


    def get_test_labels(self):
        return self._testLabels


    def set_num_folds(self, value):
        self._numFolds = value


    def set_cur_fold(self, value):
        self._curFold = value


    def set_train_data(self, value):
        self._trainData = value


    def set_test_data(self, value):
        self._testData = value


    def set_train_labels(self, value):
        self._trainLabels = value


    def set_test_labels(self, value):
        self._testLabels = value


    def del_num_folds(self):
        del self._numFolds


    def del_cur_fold(self):
        del self._curFold


    def del_train_data(self):
        del self._trainData


    def del_test_data(self):
        del self._testData


    def del_train_labels(self):
        del self._trainLabels


    def del_test_labels(self):
        del self._testLabels

    _numFolds = property(get_num_folds, set_num_folds, del_num_folds, "_numFolds's docstring")
    _curFold = property(get_cur_fold, set_cur_fold, del_cur_fold, "_curFold's docstring")
    _trainData = property(get_train_data, set_train_data, del_train_data, "_trainData's docstring")
    _testData = property(get_test_data, set_test_data, del_test_data, "_testData's docstring")
    _trainLabels = property(get_train_labels, set_train_labels, del_train_labels, "_trainLabels's docstring")
    _testLabels = property(get_test_labels, set_test_labels, del_test_labels, "_testLabels's docstring")
    
def getDataSets(dataDir, datasetNames=None):
    if datasetNames is None:
        datasetNames = [x.split('.')[0] for x in os.listdir(dataDir) if not (x.startswith('.') or x.startswith('labels-'))]
    
    ret = dict()
    
    for dataset in datasetNames:
        print "Reading", dataset
        dataFileLoc = os.path.join(dataDir, dataset + ".csv")
        labelFileLoc = os.path.join(dataDir, 'labels-' + dataset + '.csv')
        
        data = genfromtxt(dataFileLoc, dtype=Float64, delimiter=',')
        labels = genfromtxt(labelFileLoc, Float64, delimiter=',')
        
        ret[dataset] = (data, labels)
        
    return ret

def getModelDir(modelFileLoc):
    return os.path.dirname(modelFileLoc)


def getModelFileLoc(kernelDir, algorithmName, fileIdentifier):
    base = os.path.join(kernelDir, algorithmName)
    ret = os.path.join(base, fileIdentifier + ".model")
    makeParentDir(ret)
    return ret

def trainSVM(kernel, labels):
    #need to add an id number as the first column of the list
    svmKernel = column_stack((arange(1, len(kernel.tolist()) + 1), kernel))
    prob = svm_problem(labels.tolist(), svmKernel.tolist(), isKernel=True)
    param = svm_parameter('-t 4')   

    model = svm_train(prob, param)
    return model

    
def trainSVMAndSave(modelLoc, kernel, labels):
    if os.path.exists(modelLoc):
        return svm_load_model(modelLoc)
    else:
        model = trainSVM(kernel, labels)
        svm_save_model(modelLoc, model)
        return model
    

def trainSVMFromFiles(tp, trainLabels, datasetOutDir, fileIdentifier):
    
    trainJobs = []
    
    kernelFileLocs = getAllKernelFileLocs(datasetOutDir, fileIdentifier, True)
    for kfl in kernelFileLocs:
        curKernelDir = getKernelDir(kfl)
        modelFileLoc = getModelFileLoc(curKernelDir, 'svm', fileIdentifier)
        if not os.path.exists(modelFileLoc):
            kernel = loadPickle(kfl)
            trainJobs.append(tp.apply_async(trainSVMAndSave(modelFileLoc, kernel, trainLabels)))
    return trainJobs

def getClassKernels(fullKernelMatrix, trainLabels):
    #create a matrix where rows correspond to all examples
    #and columns correspond to examples of a specific class
    #so if l is the total number of examples, and lj is the number of examples in class j
    #then we're creating an l x lj matrix
    uniqueLabels = unique(trainLabels)
    ret = []
    for l in uniqueLabels:
        labelIndexes = where(trainLabels == l)[0]
        k = zeros((len(fullKernelMatrix), len(labelIndexes)))
        for r in range(len(k)):
            for c in range(len(k[r])):
                k[r][c] = fullKernelMatrix[r][labelIndexes[c]]
        ret.append(k)
    return ret        
                
def calcClassM(classKernel, trainLabels, relevantLabel):
    (numTotExamples, numLabelExamples) = classKernel.shape
    M = zeros((numTotExamples))
    for j in range(numTotExamples):
        M[j] = sum(classKernel[j]) / float(numLabelExamples)
    return M

def calcM(classKernelList, trainLabels):
    Mlist = []
    for (classKernel, label) in zip(classKernelList, unique(trainLabels)):
        Mlist.append(calcClassM(classKernel, trainLabels, label))
    Mdiff = Mlist[0] - Mlist[1]
    return outer(Mdiff, Mdiff)

def calcN(classKernels, trainLabels):
    N = zeros((len(trainLabels), len(trainLabels)))
    for i, l in enumerate(unique(trainLabels)):
        numExamplesWithLabel = len(where(trainLabels == l)[0])
        Idiff = identity(numExamplesWithLabel, Float64) - (1.0 / numExamplesWithLabel) * ones(numExamplesWithLabel, Float64)
        firstDot = dot(classKernels[i], Idiff)
        labelTerm = dot(firstDot, transpose(classKernels[i]))
        N += labelTerm
    N = nan_to_num(N)
    #make N more numerically stable
    #if I had more time, I would train this parameter, but I don't
    additionToN = ((mean(diag(N)) + 1) / 100.0) * identity(N.shape[0], Float64) 
    N += additionToN
            
    #make sure N is invertable
    for i in range(1000):
        try:
            inv(N)
        except LinAlgError:
            #doing this to make sure the maxtrix is invertable
            #large value supported by section titled
            #"numerical issues and regularization" in the paper
            N += additionToN

    return N

#def classifyKFDValuesRaw(values, x0, k, a, b, c):
def classifyKFDValuesRaw(values, a,b):
    return 1.0 / (1.0 + exp(-1 * a * values + b))
#    return a / (1 + exp(-k*(values-x0) + b)) + c
    
#def classifyKFDValues(values, x0, k, a, b, c):
def classifyKFDValues(values, a,b):
    #rawValues = classifyKFDValuesRaw(values, x0, k, a, b, c)
    rawValues = classifyKFDValuesRaw(values, a,b)
    return rint(rawValues)
    
def trainKFD(trainKernel, trainLabels):
    classKernels = getClassKernels(trainKernel, trainLabels)
    M = calcM(classKernels, trainLabels)
    N = calcN(classKernels, trainLabels)
    '''
    print "train kernel:",trainKernel
    print "Class kernels:", classKernels
    print "M",M
    print "N",N
    '''
    try:
        solutionMatrix = dot(inv(N), M)
    except LinAlgError:
        #if we get a singular matrix here, there isn't much we can do about it
        #just skip this configuration
        solutionMatrix = identity(N.shape[0], Float64)
        
    solutionMatrix = nan_to_num(solutionMatrix)
    
    eVals, eVects = eig(solutionMatrix)
    #find the 'leading' term i.e. find the eigenvector with the highest eigenvalue
    alphaVect = eVects[:, absolute(eVals).argmax()].real.astype(Float64)
    trainProjections = dot(trainKernel, alphaVect)
    '''
    print 'alpha = ', alphaVect
    print 'train kernel = ', trainKernel
    print 'train projction = ', trainProjections
    '''     
    #train sigmoid based on evaluation accuracy
    #accuracyError = lambda x: 100.0 - evaluations(trainLabels, classifyKFDValues(trainProjections, *list(x)))[0]
    accuracyError = lambda x: 100.0 - evaluations(trainLabels, classifyKFDValues(trainProjections, *x))[0]
    #get an initial guess by brute force
    #ranges = ((-100, 100, 1), (-100, 100, 1))
    #x0 = brute(accuracyError, ranges)
    
    #popt = minimize(accuracyError, x0.tolist(), method="Powell").x

    rc = LSFAIL
    niter = 0
    i = 0
    while rc in (LSFAIL, INFEASIBLE, CONSTANT, NOPROGRESS, USERABORT, MAXFUN) or niter <= 1:
        if i == 10:
            break
        #get a 'smarter' x0
        #ranges = ((-1000, 1000, 100), (-1000, 1000, 100))
        ranges = ((-10**(i + 1), 10**(i + 1), 10**i),) * 2
        x0 = brute(accuracyError, ranges)
        (popt, niter, rc) = fmin_tnc(accuracyError, x0, approx_grad=True)
        #popt = fmin_tnc(accuracyError, x0.tolist(), approx_grad=True)[0]
        i += 1
    
    return (alphaVect, popt)
    
def trainKFDAndSave(modelLoc, trainKernel, trainLabels):
    if os.path.exists(modelLoc):
        return loadPickle(modelLoc)
    else:
        modelTup = trainKFD(trainKernel, trainLabels)
        savePickle(modelTup, modelLoc)
        return modelTup

def trainKFDFromFiles(tp, trainLabels, datasetOutDir, fileIdentifier):
    
    trainJobs = []
    
    kernelFileLocs = getAllKernelFileLocs(datasetOutDir, fileIdentifier, True)
    for kfl in kernelFileLocs:
        curKernelDir = getKernelDir(kfl)
        modelFileLoc = getModelFileLoc(curKernelDir, 'kfd', fileIdentifier)
        if not os.path.exists(modelFileLoc):
            kernel = loadPickle(kfl)
            
            #different non-kernel parameters go here
            #do the actual KFD training here
            trainJobs.append(tp.apply_async(trainKFDAndSave(modelFileLoc, kernel, trainLabels)))
    return trainJobs

def getAllKernelFileLocs(datasetOutDir, fileIdentifier, train):
    #do globs for kernels with and without parameters
    return glob(getKernelFileLoc(datasetOutDir, '*', fileIdentifier, train, ''))

def getAllKernelDirLocs(datasetOutDir, fileIdentifier, train):
    kernelFileLocs = getAllKernelFileLocs(datasetOutDir, fileIdentifier, train)
    return [getKernelDir(a) for a in kernelFileLocs]

def makeParentDir(path):
    parentDir = os.path.dirname(path)
    if not os.path.exists(parentDir) and '*' not in parentDir:
        os.makedirs(parentDir)

def getKernelDir(kernelFilePath):
    return os.path.dirname(os.path.dirname(kernelFilePath))

def getKernelFileLocFromDir(kernelDir, fileIdentifier, train):
    testOrTrain = 'trainKernels' if train else 'testKernels'
    return os.path.join(kernelDir, testOrTrain, fileIdentifier + '.kernel')

def getKernelFileLoc(datasetOutDir, kernelFunctionName, fileIdentifier, train, kernelFunctArgs):
    kernelDirName = '-'.join([kernelFunctionName] + [str(a) for a in kernelFunctArgs])
    testOrTrain = 'train' if train else 'test'
    testOrTrain += 'Kernels'
    fileName = fileIdentifier + ".kernel"
    ret = os.path.join(datasetOutDir, kernelDirName, testOrTrain, fileName)
    makeParentDir(ret)
    return ret

def getDatasetOutDir(baseOutDir, datasetName):
    ret = os.path.join(baseOutDir, datasetName)
    makeParentDir(ret)
    return ret

def savePickle(obj, fileLoc):
    objFile = open(fileLoc, 'w')
    pickle.dump(obj, objFile)
    objFile.close()
        
def loadPickle(fileLoc):
    objFile = open(fileLoc, 'r')
    return pickle.load(objFile)
    objFile.close()

def linearKernel(x, y):
    return dot(x, transpose(y))

def polyKernel(x,y, gamma, coef, degree):
    if gamma == 0:
        gamma = .01
    ret = (gamma * dot(x, transpose(y)) + coef) ** degree
    ret = nan_to_num(ret)
    return ret

def rbfKernel(x, y, gamma):
    if gamma <= 0:
        gamma = 1
    return exp(-1 * gamma * (norm(x-y) ** 2))

def createKernel(xList, yList, kernelFunct, *kernelFunctArgs):
    #remember, examples are rows, not columns
    k = zeros((len(xList), len(yList)))
    for i in range(len(xList)):
        for j in range(len(yList)):
            k[i,j] = kernelFunct(xList[i], yList[j], *kernelFunctArgs)
    return k


def createAndSaveKernel(xList, yList, kernelFileLoc, kernelFunct, *kernelFunctArgs):
    k = createKernel(xList, yList, kernelFunct, *kernelFunctArgs)
    savePickle(k, kernelFileLoc)
    return k
        
def createAndSaveTestAndTrainKernels(trainData, testData, datasetOutDir, kernelFunct, fileIdentifier, *kernelFunctArgs):
    #do the train kernel
    trainKernelLoc = getKernelFileLoc(datasetOutDir, kernelFunct.__name__, fileIdentifier, True, kernelFunctArgs)
    if not os.path.exists(trainKernelLoc):
        trainKernel = createAndSaveKernel(trainData, trainData, trainKernelLoc, kernelFunct, *kernelFunctArgs)
    else:
        trainKernel = loadPickle(trainKernelLoc)        
    #do the test kernels
    testKernelLoc = getKernelFileLoc(datasetOutDir, kernelFunct.__name__, fileIdentifier, False, kernelFunctArgs)
    if not os.path.exists(testKernelLoc):
        testKernel = createAndSaveKernel(testData, trainData, testKernelLoc, kernelFunct, *kernelFunctArgs)
    else:
        testKernel = loadPickle(testKernelLoc)
    return (trainKernel, testKernel)

def createAndSaveAllKernels(tp, datasetOutDir, trainData, testData, fileIdentifier):
    kernelJobs = []
    
    #do linear kernel
    kernelJobs.append(tp.apply_async(createAndSaveTestAndTrainKernels(trainData, testData, datasetOutDir, linearKernel, fileIdentifier)))
    
    #do the rbf kernel, with various gammas
    rbfGammas = [10**x for x in range(-5,1)]
    for g in rbfGammas:
        kernelJobs.append(tp.apply_async(createAndSaveTestAndTrainKernels(trainData, testData, datasetOutDir, rbfKernel, fileIdentifier, g)))
    #do the poly kernels
    polyGammas = [10**x for x in range(-3, 4)]
    polyCoef = range(0,10,1)
    polyDegree = range(2,4)
    for polyArgs in itertools.product(polyGammas, polyCoef, polyDegree):
        kernelJobs.append(tp.apply_async(createAndSaveTestAndTrainKernels(trainData, testData, datasetOutDir, polyKernel, fileIdentifier, *polyArgs)))
    return kernelJobs

def getPredictFileLoc(modelDir, fileIdentifier):
    #want to have the same structure as the model file directory, but starting at the predict directory
    ret = os.path.join(modelDir, fileIdentifier + '.predict')
    makeParentDir(ret)
    return ret 

def predictSVM(model, testKernel):
    svmTestKernel = column_stack(([0] * len(testKernel), testKernel))
    #svm_predict returns 3 things - we want the first one, hense the [0] at the end of the following line
    predictedLabels  = svm_predict([0] * len(svmTestKernel), svmTestKernel.tolist(), model)[0]
    return predictedLabels


def predictSVMAndSave(model, testKernel, predictFileLoc):
    if os.path.exists(predictFileLoc):
        return loadPickle(predictFileLoc)
    else:    
        predictions = predictSVM(model, testKernel)
        savePickle(predictions, predictFileLoc)
        return predictions


def predictSVMFromFiles(tp, datasetOutDir, fileIdentifier):
    
    jobs = []
    
    for testKernelFileLoc in getAllKernelFileLocs(datasetOutDir, fileIdentifier, False):
        testKernel = loadPickle(testKernelFileLoc)
        mfl = getModelFileLoc(getKernelDir(testKernelFileLoc), 'svm', fileIdentifier)
        model = svm_load_model(mfl)
        predictFileLoc = getPredictFileLoc(getModelDir(mfl), fileIdentifier)
        if not os.path.exists(predictFileLoc):
            jobs.append(tp.apply_async(predictSVMAndSave(model, testKernel, predictFileLoc)))
    return jobs

def predictKFD(model, pOpt, testKernel):
    projectionsIntoFeatureSpace = dot(testKernel, model)
    
    #print "test proj:", projectionsIntoFeatureSpace
    
    predictedLabels = classifyKFDValues(projectionsIntoFeatureSpace, *pOpt)
    #print predictedLabels
    return predictedLabels

def predictKFDAndSave(model, pOpt, testKernel, predictFileLoc):
    if os.path.exists(predictFileLoc):
        return loadPickle(predictFileLoc)
    else:
        predictions = predictKFD(model, pOpt, testKernel) 
        savePickle(predictions, predictFileLoc)
        return predictions

def predictKFDFromFiles(tp, datasetOutDir, fileIdentifier):
        
    jobs = []
    
    for kernelDir in getAllKernelDirLocs(datasetOutDir, fileIdentifier, True):
        mfl = getModelFileLoc(kernelDir, 'kfd', fileIdentifier)
        predictFileLoc = getPredictFileLoc(getModelDir(mfl), fileIdentifier)
        if not os.path.exists(predictFileLoc):
            testKernelFileLoc = getKernelFileLocFromDir(kernelDir, fileIdentifier, False)
            (model, pOpt) = loadPickle(mfl)
            testKernel = loadPickle(testKernelFileLoc)
            
            jobs.append(tp.apply_async(predictKFDAndSave(model, pOpt, testKernel, predictFileLoc)))
    return jobs

def getAnalysisDir(kernelDir):
    ret = os.path.join(kernelDir, 'analysis')
    if not os.path.exists(ret):
        os.makedirs(ret)
    return ret

def getAccuracyCsvFileLoc(datasetOutDir, fileIdentifier):
    ret = os.path.join(datasetOutDir, fileIdentifier + '-accuracy.csv')
    return ret

def getAccuracyTxtFileLoc(datasetOutDir, fileIdentifier):
    ret = os.path.join(datasetOutDir, fileIdentifier + '-accuracy.txt')
    return ret


def getAllAccuracyCSVFileLocs(baseOutDir, fileIdentifier):
    return glob(getAccuracyCsvFileLoc(getDatasetOutDir(baseOutDir, '*'), fileIdentifier))

def doSameKernelAnalysis(tp, datesetOutDir, datasetName, trainLabels, testLabels, fileIdentifier):

    accuracyCsvFileLoc = getAccuracyCsvFileLoc(datasetOutDir, fileIdentifier)
    accuracyCsvFile = open(accuracyCsvFileLoc, 'w')
    accuracyCsvWriter = csv.writer(accuracyCsvFile, lineterminator='\n')
    
    accuracyTxtFileLoc =  getAccuracyTxtFileLoc(datasetOutDir, fileIdentifier)
    accuracyTxtFile = open(accuracyTxtFileLoc, 'w')


    #get kfd and svm stats on same data

    #find matching predict files
    #both matching kernels and matching this fold num
    
    
    #for now, graph sigmoid results
    for kernelDir in getAllKernelDirLocs(datasetOutDir, fileIdentifier, True):
            kfdModelFileLoc = getModelFileLoc(kernelDir, 'kfd', fileIdentifier)        
        
            '''
            trainKernelFileLoc = getKernelFileLocFromDir(kernelDir, fileIdentifier, True)
            testKernelFileLoc = getKernelFileLocFromDir(kernelDir, fileIdentifier, False)            
        
            (model, pOpt) = loadPickle(kfdModelFileLoc)
            trainKernel = loadPickle(trainKernelFileLoc)
            testKernel = loadPickle(testKernelFileLoc)
            
            trainProjections = dot(trainKernel, model)
            testProjections = dot(testKernel, model)

            #sigmoidResults = classifyKFDValuesRaw(testProjections, *pOpt)
            trainProjectionsClass0 = array([trainProjections[a] for a in range(len(trainLabels)) if trainLabels[a] == 0])
            trainProjectionsClass1 = array([trainProjections[a] for a in range(len(trainLabels)) if trainLabels[a] == 1])
            
            testProjectionsClass0 = array([testProjections[a] for a in range(len(testLabels)) if testLabels[a] == 0])
            testProjectionsClass1 = array([testProjections[a] for a in range(len(testLabels)) if testLabels[a] == 1])
            
            minProjection = min((trainProjections.min(), testProjections.min()))
            maxProjection = max((trainProjections.max(), testProjections.max()))
            step = (maxProjection - minProjection) / 1000
            minProjection -= step
            maxProjection += step
            curveXvals = arange(minProjection, maxProjection, step)
            
            analysisDir = getAnalysisDir(kernelDir)
            graphFileLoc = os.path.join(analysisDir, fileIdentifier + '-sigmoidGraph.pdf')            

            #graph the results
            ylim((-.1, 1.1))
            plot(curveXvals, classifyKFDValuesRaw(curveXvals, *pOpt), 'g-', 
                 testProjectionsClass1, classifyKFDValuesRaw(testProjectionsClass1, *pOpt), 'bs',
                 trainProjectionsClass1, classifyKFDValuesRaw(trainProjectionsClass1, *pOpt), 'bo',
                 trainProjectionsClass0, classifyKFDValuesRaw(trainProjectionsClass0, *pOpt), 'r+',
                 testProjectionsClass0, classifyKFDValuesRaw(testProjectionsClass0, *pOpt), 'r.')
            savefig(graphFileLoc)
            clf()
            '''
            
            #print test accuracy
            kfdPredictFileLoc = getPredictFileLoc(getModelDir(kfdModelFileLoc), fileIdentifier)
            kfdPredictions = loadPickle(kfdPredictFileLoc)
            kfdEvals = evaluations(testLabels, kfdPredictions)
            #print 'kfd', os.path.basename(datasetOutDir), os.path.basename(kernelDir), kfdEvals[0]
            
            svmModelFileLoc = getModelFileLoc(kernelDir, 'svm', fileIdentifier)
            svmPredictFileLoc = getPredictFileLoc(getModelDir(svmModelFileLoc), fileIdentifier)
            svmPredictions = loadPickle(svmPredictFileLoc)
            svmEvals = evaluations(testLabels, svmPredictions)
            #print 'svm', os.path.basename(datasetOutDir), os.path.basename(kernelDir), svmEvals[0]

            if '-' in os.path.basename(kernelDir):
                kernelName, kernelParamStr = os.path.basename(kernelDir).split('-',1)
            else :
                kernelName = os.path.basename(kernelDir)
                kernelParamStr = 'none'
                
            if kfdEvals[0] > svmEvals[0]:
                kfdAccStr = "\\textbf{%.3f}" % kfdEvals[0]
                svmAccStr = "%.3f" % svmEvals[0]
            else:
                kfdAccStr = "%.3f" % kfdEvals[0]
                svmAccStr = "\\textbf{%.3f}" % svmEvals[0]

            accuracyCsvWriter.writerow((datasetName, kernelName, '"' + kernelParamStr + '"', "%.3f" % kfdEvals[0], "%.3f" % svmEvals[0]))
            accuracyTxtFile.write(' & '.join((datasetName, kernelName, kernelParamStr, kfdAccStr, svmAccStr)) + ' \\\\ \n')
                
    accuracyCsvFile.close()
    accuracyTxtFile.close()

def compareAlgorithmsOnSameKernels(tp, trainData, trainLabels, testData, testLabels, datasetName, fileIdentifier):
        print "Creating overall kernels"
        kernelJobs = createAndSaveAllKernels(tp, datasetOutDir, overallTrainData, overallTestData, fileIdentifier)
        for kj in kernelJobs:
            kj.ready()
        
        #do the model training
        print "Training overall models"
        trainJobs = []
        trainJobs.extend(trainSVMFromFiles(tp, overallTrainLabels, datasetOutDir, fileIdentifier))
        trainJobs.extend(trainKFDFromFiles(tp, overallTrainLabels, datasetOutDir, fileIdentifier))
        for tj in trainJobs:
            tj.ready()
            
        #get the predictions out
        print "Predicting on all data"
        predictJobs = []
        predictJobs.extend(predictSVMFromFiles(tp, datasetOutDir, fileIdentifier))
        predictJobs.extend(predictKFDFromFiles(tp, datasetOutDir, fileIdentifier))
        for pj in predictJobs:
            pj.ready()
    
        
        #this can be a lot of hard coded stuff, to get the graphs we want
        print "Doing overall analysis"
        doSameKernelAnalysis(tp, datasetOutDir, datasetName, overallTrainLabels, overallTestLabels, fileIdentifier);
    
        print "******************************************************"
        
class kfdKernelOptimizer:
    def __init__(self, data, labels, kernelFunct, numFolds, datasetDir, fileIdentifier):
        self._data = data
        self._labels = labels
        self._kernelFunct = kernelFunct
        self._totNumFolds = numFolds
        self._datasetDir = datasetDir
        self._fileIdentifier = fileIdentifier
    def evaluateKernel(self, kernelArgs):
        cvg = crossValidationGen(self._totNumFolds, self._data, self._labels)
        
        errorList = []
        
        try:
            iter(kernelArgs)
        except TypeError:
            kernelArgs = kernelArgs.tolist()
        try:
            iter(kernelArgs)
        except TypeError:
            kernelArgs = [kernelArgs]        
        
        for foldTrainData, foldTrainLabels, foldTestData, foldTestLabels in cvg:
            #get the test and train kernels
            #trainKernel = createKernel(foldTrainData, foldTrainData, self._kernelFunct, *kernelArgs)
            #testKernel = createKernel(foldTestData, foldTrainData, self._kernelFunct, *kernelArgs)
            foldFileIdentifier = '-'.join((fileIdentifier, str(cvg.get_cur_fold()), str(cvg.get_num_folds())))
            (trainKernel, testKernel) = createAndSaveTestAndTrainKernels(foldTrainData, foldTestData, self._datasetDir, self._kernelFunct, foldFileIdentifier, *kernelArgs)
            #train the model
            foldModelFileLoc = getModelFileLoc(getKernelDir(getKernelFileLoc(self._datasetDir, self._kernelFunct.__name__, foldFileIdentifier, True, kernelArgs)), 'kfd', foldFileIdentifier)
            (model, pOpt) = trainKFDAndSave(foldModelFileLoc, trainKernel, foldTrainLabels)
            #predict using the model
            predictedLabels = predictKFDAndSave(model, pOpt, testKernel, getPredictFileLoc(getModelDir(foldModelFileLoc), foldFileIdentifier))
            #evaluate the results
            foldAccuracy = evaluations(foldTestLabels, predictedLabels)[0]
            foldError = 100 - foldAccuracy
            errorList.append(foldError)
        meanError = mean(errorList)
        print kernelArgs, meanError
        return meanError

class svmKernelOptimizer:
    def __init__(self, data, labels, kernelFunct, numFolds, datasetDir, fileIdentifier):
        self._data = data
        self._labels = labels
        self._kernelFunct = kernelFunct
        self._totNumFolds = numFolds
        self._datasetDir = datasetDir
        self._fileIdentifier = fileIdentifier
    def evaluateKernel(self, kernelArgs):
        cvg = crossValidationGen(self._totNumFolds, self._data, self._labels)
                
        errorList = []
        
        try:
            iter(kernelArgs)
        except TypeError:
            kernelArgs = kernelArgs.tolist()
        try:
            iter(kernelArgs)
        except TypeError:
            kernelArgs = [kernelArgs]
        
        for foldTrainData, foldTrainLabels, foldTestData, foldTestLabels in cvg:
            foldFileIdentifier = '-'.join((self._fileIdentifier, str(cvg.get_cur_fold()), str(cvg.get_num_folds())))
            #get the test and train kernels
            (trainKernel, testKernel) = createAndSaveTestAndTrainKernels(foldTrainData, foldTestData, self._datasetDir, self._kernelFunct, foldFileIdentifier, *kernelArgs)
            #trainKernel = createKernel(foldTrainData, foldTrainData, self._kernelFunct, *kernelArgs)
            #testKernel = createKernel(foldTestData, foldTrainData, self._kernelFunct, *kernelArgs)
            #train the model
            #model = trainSVM(trainKernel, foldTrainLabels)
            foldModelFileLoc = getModelFileLoc(getKernelDir(getKernelFileLoc(self._datasetDir, self._kernelFunct.__name__, foldFileIdentifier, True, kernelArgs)), 'svm', foldFileIdentifier)
            model = trainSVMAndSave(foldModelFileLoc, trainKernel, foldTrainLabels)
            #predict using the model
            predictedLabels = predictSVMAndSave(model, testKernel, getPredictFileLoc(getModelDir(foldModelFileLoc), foldFileIdentifier))
            #evaluate the results
            foldAccuracy = evaluations(foldTestLabels, predictedLabels)[0]
            foldError = 100 - foldAccuracy
            errorList.append(foldError)
        meanError = mean(errorList)
        print kernelArgs, meanError
        return meanError
    
def getOptArgsDir(datasetOutDir):
    ret = os.path.join(datasetOutDir, 'optArgs')
    if not os.path.exists(ret):
        os.makedirs(ret)
    return ret

def getOptimizedKernelArgFileLoc(datasetOutDir, kernelAlgName, classifierAlgName, fileIdentifier):
    ret = os.path.join(getOptArgsDir(datasetOutDir), kernelAlgName, classifierAlgName, fileIdentifier + ".optArgs")
    makeParentDir(ret)
    return ret

def getOptimizedAccuracyFileLoc(datasetOutDir, fileIdentifier):
    return os.path.join(getOptArgsDir(datasetOutDir), fileIdentifier + "-accuracy.txt")
    
def compareAlgorithmsOnOptimizedKernel(tp, trainData, trainLabels, testData, testLabels, numFolds, datasetOutDir, datasetName, fileIdentifier):
    #set up brute force ranges based on the kernel algorithm
    #have different ranges for the different parameters of algorithms
    #for kernelAlg, bruteForceRanges, bounds in zip( (rbfKernel, polyKernel), (((.1, 1, .1),) , ((.1, 1.7, .2),(-4, 5, 2), (-2, 3,1))), ( ((0, None),), ((None, None), (None, None), (None, None))) ):
    for kernelAlg, bruteForceRanges, bounds in (((rbfKernel), (((.1, 1, .1),)), ( ((0, None),) )), (None, None, None)):
    #for kernelAlg, initArgs in ((rbfKernel, (.5,)), (polyKernel, (25, 5, 3))):
        if kernelAlg is None:
            continue
        print "Optimizing ", kernelAlg.__name__

        #optimize for kfd
        kfdOptimizedArgFileLoc = getOptimizedKernelArgFileLoc(datasetOutDir, kernelAlg.__name__, 'kfd', fileIdentifier)
        if not os.path.exists(kfdOptimizedArgFileLoc):
            kfdOptimizer = kfdKernelOptimizer(trainData, trainLabels, kernelAlg, numFolds, datasetOutDir, fileIdentifier)
            
            #get an initial guess by brute force
            initArgs = brute(kfdOptimizer.evaluateKernel, bruteForceRanges)

            #pOpt = minimize(kfdOptimizer.evaluateKernel, initArgs, method="Powell").x
            pOpt = fmin_tnc(kfdOptimizer.evaluateKernel, initArgs, approx_grad=True, epsilon = .1, bounds=bounds)[0]
            
            print 'kfd', pOpt
            
            savePickle(pOpt, kfdOptimizedArgFileLoc)
            
        #create optimal kfd kernel, model and predictions
        optimalKFDFileIdentifier = 'kfd-' + fileIdentifier
        kfdOptArgsList = loadPickle(kfdOptimizedArgFileLoc).tolist()
        if not type(kfdOptArgsList) is list:
            kfdOptArgsList = [kfdOptArgsList]
        createAndSaveTestAndTrainKernels(trainData, testData, getOptArgsDir(datasetOutDir), kernelAlg, optimalKFDFileIdentifier, *kfdOptArgsList)
        trainJobs = trainKFDFromFiles(tp, trainLabels, getOptArgsDir(datasetOutDir), optimalKFDFileIdentifier)
        for tj in trainJobs:
            tj.ready()
        predictJobs = predictKFDFromFiles(tp, getOptArgsDir(datasetOutDir), optimalKFDFileIdentifier)
        for pj in predictJobs:
            pj.ready()
                            
        #optimize for svm
        svmOptimizedArgFileLoc = getOptimizedKernelArgFileLoc(datasetOutDir, kernelAlg.__name__, 'svm', fileIdentifier)
        if not os.path.exists(svmOptimizedArgFileLoc):
            svmOptimizer = svmKernelOptimizer(trainData, trainLabels, kernelAlg, numFolds, datasetOutDir, fileIdentifier)
            
            #get an initial guess by brute force
            initArgs = brute(svmOptimizer.evaluateKernel, bruteForceRanges)
            
            #pOpt = minimize(svmOptimizer.evaluateKernel, initArgs, method="Powell").x
            pOpt = fmin_tnc(svmOptimizer.evaluateKernel, initArgs, approx_grad=True, epsilon=.1, bounds=bounds)[0]
            
            print 'svm', pOpt
            savePickle(pOpt, svmOptimizedArgFileLoc)
            
        #do the same kernel, model and prediction for svm
        optimalSVMFileIdentifier = 'svm-' + fileIdentifier
        svmOptArgsList = list(loadPickle(svmOptimizedArgFileLoc).tolist())
        createAndSaveTestAndTrainKernels(trainData, testData, getOptArgsDir(datasetOutDir), kernelAlg, optimalSVMFileIdentifier, *svmOptArgsList)
        trainJobs = trainSVMFromFiles(tp, trainLabels, getOptArgsDir(datasetOutDir), optimalSVMFileIdentifier)
        for tj in trainJobs:
            tj.ready()
        predictJobs = predictSVMFromFiles(tp, getOptArgsDir(datasetOutDir), optimalSVMFileIdentifier)
        for pj in predictJobs:
            pj.ready()
        
        #save the accuracy of both optimized algorithms, with their optimized parameters
        optimalKfdTrainKernelFileLoc = getKernelFileLoc(getOptArgsDir(datasetOutDir), kernelAlg.__name__, optimalKFDFileIdentifier, True, kfdOptArgsList)
        optimalKfdModelFileLoc = getModelFileLoc(getKernelDir(optimalKfdTrainKernelFileLoc), 'kfd', optimalKFDFileIdentifier)
        optimalKfdPredictFileLoc = getPredictFileLoc(getModelDir(optimalKfdModelFileLoc), optimalKFDFileIdentifier)
        optimalKfdPredictions = loadPickle(optimalKfdPredictFileLoc)
        optimalKfdEval = evaluations(testLabels, optimalKfdPredictions)
        
        optimalSvmTrainKernelFileLoc = getKernelFileLoc(getOptArgsDir(datasetOutDir), kernelAlg.__name__, optimalSVMFileIdentifier, True, svmOptArgsList)
        optimalSvmModelFileLoc = getModelFileLoc(getKernelDir(optimalSvmTrainKernelFileLoc), 'svm', optimalSVMFileIdentifier)
        optimalSvmPredictFileLoc = getPredictFileLoc(getModelDir(optimalSvmModelFileLoc), optimalSVMFileIdentifier)
        optimalSvmPredictions = loadPickle(optimalSvmPredictFileLoc)
        optimalSvmEval = evaluations(testLabels, optimalSvmPredictions)
        
        optimizedAccuracyFileLoc = getOptimizedAccuracyFileLoc(datasetOutDir, fileIdentifier)
        optimizedAccuracyFile = open(optimizedAccuracyFileLoc, 'a')
        optimizedAccuracyFile.write(' & '.join([kernelAlg.__name__, datasetName, 'kfd', ' & '.join(["%.3f" % a for a in kfdOptArgsList]), "%.3f" % optimalKfdEval[0]]) + ' \\\\ \n')
        optimizedAccuracyFile.write(' & '.join([kernelAlg.__name__, datasetName, 'svm', ' & '.join(["%.3f" % a for a in svmOptArgsList]), "%.3f" % optimalSvmEval[0]]) + ' \\\\ \n')
        optimizedAccuracyFile.close()
        

if __name__ == '__main__':
    numExpectedArgs = 3
    if len(sys.argv) != numExpectedArgs:
        #print "Usage: %s <data dir> <kernel dir> <model dir> <prediciton dir> <analysis dir>" % sys.argv[0]
        print "Usage: %s <data dir> <out dir>"
        print "Got %d arguments, expecting %d" % (len(sys.argv), numExpectedArgs)
        exit(1)
        
    #(dataDir, datasetOutDir, modelDir, predictDir, analysisDir) = sys.argv[1:]
    (dataDir, outDir) = sys.argv[1:]
    
    #datasets = getDataSets(dataDir)
    datasets = getDataSets(dataDir, ['ionosphere', 'iris', 'wine'])
    #datasets = getDataSets(dataDir, ['by_hand'])
    #datasets = getDataSets(dataDir, ['ionosphere'])
    
    
    
    tp = ThreadPool(4)
        
    for name, (data, labels) in datasets.iteritems():
        datasetOutDir = getDatasetOutDir(outDir, name)
        print "Computing on", name
    
        #do a split into overall test and overall train
        overallTestTrainRatio = 1.0 / 3.0
        overallTestTrainSplitIndex = array([int(overallTestTrainRatio * len(data))])
        overallTestData, overallTrainData = array_split(data, array([overallTestTrainSplitIndex]))
        overallTestLabels, overallTrainLabels = array_split(labels, overallTestTrainSplitIndex)        
                
        #test a whole bunch of generic kernels on the overall split data
        fileIdentifier = 'overall'
        #print "train:", overallTrainData
        #print "test:", overallTestData
               
        compareAlgorithmsOnSameKernels(tp, overallTrainData, overallTrainLabels, overallTestData, overallTestLabels, name, fileIdentifier)
        
        #now, try to find an optimal kernel for either svm or kfd
        #do it for each kernel type
        numOptimizationFolds = 3
        fileIdentifier = 'optimized'
        compareAlgorithmsOnOptimizedKernel(tp, overallTrainData, overallTrainLabels, overallTestData, overallTestLabels, numOptimizationFolds, datasetOutDir, name, fileIdentifier)
        
    #finally, cat together all of the accuracy csv files
    allOverallAccuracyCsvFileLoc = os.path.join(outDir, 'all-overall-accuracy.csv')
    allOverallAccuracyCsvFile = open(allOverallAccuracyCsvFileLoc, 'wb')
    for datasetAccuracyCsvFile in getAllAccuracyCSVFileLocs(outDir, 'overall'):
        shutil.copyfileobj(open(datasetAccuracyCsvFile, 'rb'), allOverallAccuracyCsvFile)
    allOverallAccuracyCsvFile.close()
         
    tp.close()