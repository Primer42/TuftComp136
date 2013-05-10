'''
Created on Nov 26, 2012

@author: Will
'''
import sys
from collections import defaultdict
import csv
import os
from numpy.lib.shape_base import array_split
from glob import glob
from random import shuffle
from numpy.ma.core import min

#clean up one example
def cleanFeatures(features):
    #remove any empty strings
    for i in range(features.count('')):
        features.remove('')
        
        

if __name__ == '__main__':
    numArgsExpected = 3
    if len(sys.argv) != numArgsExpected:
        print "Usage: %s <raw data dir> <cleaned data dir>"
        exit(1)
    (rawDataDir, cleanDataDir) = sys.argv[1:]
    
    if not os.path.exists(cleanDataDir):
        os.makedirs(cleanDataDir)
    
    #look for any dataset with a .classcol file, so we know which column to use as the class label
    for infoFileLoc in glob(os.path.join(rawDataDir, '*.info')):
        datasetName = os.path.basename(infoFileLoc).split('.info')[0]
        
        #see if cleaned data files already exist - if they do, don't make them again
        newDataFileLoc = os.path.join(cleanDataDir, datasetName + '.csv')
        newLabelFileLoc = os.path.join(cleanDataDir, 'labels-' + datasetName + '.csv')
        
        if os.path.exists(newDataFileLoc) and os.path.exists(newLabelFileLoc):
            continue
        
        print "Cleaning", datasetName
        
        infoFile = open(infoFileLoc, 'r')
        classCol = int(infoFile.readline())
        delim = infoFile.readline()[0]
        ignoreColList = [int(x) for x in infoFile.readline().split()]
        infoFile.close()    
    
        datasetDict = defaultdict(list)
                
        #read the data
        origDataFile = open(os.path.join(rawDataDir, datasetName + '.data'), 'r')
        dataReader = csv.reader(origDataFile, delimiter=delim)
        for features in dataReader:
            if not len(features):
                continue
            cleanFeatures(features)
            #remove ignored columns and class column, in reverse sorted order
            #do it in reverse sorted order so the indexes stay correct
            for removeCol in sorted(ignoreColList + [classCol], reverse=True):
                if removeCol == classCol:
                    label = features.pop(removeCol)
                else:
                    features.pop(removeCol)
            datasetDict[label].append(features)
        origDataFile.close()
        
    
        #make it into a 2 class problem by lumping classes together
        #don't have rhyme or reason - don't want to favor one class or another, or make our data artificially clean
        numOrigClasses = len(datasetDict.keys())
        #split into 2, possibly unequal, groups of class labels
        newClassMap = array_split(datasetDict.keys(), 2)
        
        #reorganize the data
        dataWithNewLabelMap = defaultdict(list)
        
        for newClassLabel, oldClassLabelList in enumerate(newClassMap):
            for oldClassLabel in oldClassLabelList:
                for featureRow in datasetDict[oldClassLabel]:
                    dataWithNewLabelMap[newClassLabel].append(featureRow)
        
        #make the two datasets the same size
        dataWithNewLabelTupleList = []
        minClassSize = min([len(x) for x in dataWithNewLabelMap.values()])
        for newClassLabel, featureRowList in dataWithNewLabelMap.iteritems():
            for featureRow in featureRowList[:minClassSize]:
                dataWithNewLabelTupleList.append((featureRow, newClassLabel))
        
        
        #shuffle the data
        shuffle(dataWithNewLabelTupleList)
        
        
        newDataFile = open(newDataFileLoc, 'w')
        newLabelsFile = open(newLabelFileLoc, 'w')
        
        newDataWriter = csv.writer(newDataFile, lineterminator='\n')
        
        for (example, label) in dataWithNewLabelTupleList:
            newDataWriter.writerow(example)
            newLabelsFile.write(str(label) + '\n')
            
        newDataFile.close()
        newLabelsFile.close()