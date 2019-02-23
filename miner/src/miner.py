#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:20:40 2019

@author: MattWall
"""
import numpy as np
import pandas as pd
from scipy import stats
from numpy import random as rd
import os
from sklearn.decomposition import PCA
#import multiprocessing, multiprocessing.pool
import matplotlib.pyplot as plt
import time
from collections import Counter

# =============================================================================
# Functions used for reading and writing files
# =============================================================================

def read_pkl(input_file):
    import pickle
    with open(input_file, 'rb') as f:
        dict_ = pickle.load(f)        
    return dict_

def read_json(filename):
    import json    
    with open(filename) as data:
        dict_ = json.load(data) 
    return dict_

def write_json(dict_, output_file):
    import json
    output_file = output_file
    with open(output_file, 'w') as fp:
        json.dump(dict_, fp)
    return

# =============================================================================
# Functions used for pre-processing data
# =============================================================================

def identifierConversion(expressionData,conversionTable=os.path.join("..","data","identifier_mappings.txt")):    
    idMap = pd.read_table(conversionTable)
    genetypes = list(set(idMap.iloc[:,2]))
    previousIndex = np.array(expressionData.index).astype(str)    
    previousColumns = np.array(expressionData.columns).astype(str)  
    bestMatch = []
    for geneType in genetypes:
        subset = idMap[idMap.iloc[:,2]==geneType]
        subset.index = subset.iloc[:,1]
        mappedGenes = list(set(previousIndex)&set(subset.index))
        mappedSamples = list(set(previousColumns)&set(subset.index))
        if len(mappedGenes)>=max(10,0.01*expressionData.shape[0]):
            if len(mappedGenes)>len(bestMatch):
                bestMatch = mappedGenes
                state = "original"
                gtype = geneType
                continue
        if len(mappedSamples)>=max(10,0.01*expressionData.shape[1]):
            if len(mappedSamples)>len(bestMatch):
                bestMatch = mappedSamples
                state = "transpose"
                gtype = geneType
                continue
    
    mappedGenes = bestMatch
    subset = idMap[idMap.iloc[:,2]==gtype] 
    subset.index = subset.iloc[:,1]

    if len(bestMatch) == 0:
        print("Error: Gene identifiers not recognized")
    
    if state == "transpose":
        expressionData = expressionData.T
        
    try:
        convertedData = expressionData.loc[mappedGenes,:]
    except:
        convertedData = expressionData.loc[np.array(mappedGenes).astype(int),:]
    
    conversionTable = subset.loc[mappedGenes,:]
    conversionTable.index = conversionTable.iloc[:,0]
    conversionTable = conversionTable.iloc[:,1]
    conversionTable.columns = ["Name"]
    
    newIndex = list(subset.loc[mappedGenes,"Preferred_Name"])
    convertedData.index = newIndex
    
    duplicates = [item for item, count in Counter(newIndex).items() if count > 1]
    singles = list(set(convertedData.index)-set(duplicates))
    
    corrections = []
    for duplicate in duplicates:
        dupData = convertedData.loc[duplicate,:]
        firstChoice = pd.DataFrame(dupData.iloc[0,:]).T
        corrections.append(firstChoice)
    
    if len(corrections)  == 0:
        print("completed identifier conversion.\n"+str(convertedData.shape[0])+" genes were converted." )
        return convertedData, conversionTable
        
    correctionsDf = pd.concat(corrections,axis=0)
    uncorrectedData = convertedData.loc[singles,:]
    convertedData = pd.concat([uncorrectedData,correctionsDf],axis=0)    
    
    print("completed identifier conversion.\n"+str(convertedData.shape[0])+" genes were converted." )
    
    return convertedData, conversionTable

def zscore(expressionData):
    zero = np.percentile(expressionData,0)
    meanCheck = np.mean(expressionData[expressionData>zero].mean(axis=1,skipna=True))
    if meanCheck<0.1:
        return expressionData
    means = expressionData.mean(axis=1,skipna=True)
    stds = expressionData.std(axis=1,skipna=True)
    try:
        transform = ((expressionData.T - means)/stds).T
    except:
        passIndex = np.where(stds>0)[0]
        transform = ((expressionData.iloc[passIndex,:].T - means[passIndex])/stds[passIndex]).T
    print("completed z-transformation.")
    return transform

# =============================================================================
# Functions used for clustering 
# =============================================================================

def pearson_array(array,vector):    
    #r = (1/n-1)sum(((x-xbar)/sx)((y-ybar)/sy))
    
    ybar = np.mean(vector)
    sy = np.std(vector,ddof=1)    
    yterms = (vector-ybar)/float(sy)
    
    array_sx = np.std(array,axis=1,ddof=1)
    
    if 0 in array_sx:
        passIndex = np.where(array_sx>0)[0]
        array = array[passIndex,:]
        array_sx = array_sx[passIndex]
        
    array_xbar = np.mean(array,axis=1)            
    product_array = np.zeros(array.shape)
    
    for i in range(0,product_array.shape[1]):
        product_array[:,i] = yterms[i]*(array[:,i] - array_xbar)/array_sx
        
    return np.sum(product_array,axis=1)/float(product_array.shape[1]-1)

def getAxes(clusters,expressionData):
    axes = {}
    for key in clusters.keys():
        genes = clusters[key]
        fpc = PCA(1)
        principalComponents = fpc.fit_transform(expressionData.loc[genes,:].T)
        axes[key] = principalComponents.ravel()
    return axes

def FrequencyMatrix(matrix,overExpThreshold = 1):
    
    numRows = matrix.shape[0]
    
    if type(matrix) == pd.core.frame.DataFrame:
        index = matrix.index
        matrix = np.array(matrix)
    else:
        index = np.arange(numRows)
    
    matrix[matrix<overExpThreshold] = 0
    matrix[matrix>0] = 1
            
    hitsMatrix = pd.DataFrame(np.zeros((numRows,numRows)))
    for column in range(matrix.shape[1]):
        geneset = matrix[:,column]
        hits = np.where(geneset>0)[0]
        hitsMatrix.iloc[hits,hits] += 1
        
    frequencyMatrix = np.array(hitsMatrix)
    traceFM = np.array([frequencyMatrix[i,i] for i in range(frequencyMatrix.shape[0])]).astype(float)
    if np.count_nonzero(traceFM)<len(traceFM):
        #subset nonzero. computefm. normFM zeros with input shape[0]. overwrite by slice np.where trace>0
        nonzeroGenes = np.where(traceFM>0)[0]
        normFMnonzero = np.transpose(np.transpose(frequencyMatrix[nonzeroGenes,:][:,nonzeroGenes])/traceFM[nonzeroGenes])
        normDf = pd.DataFrame(normFMnonzero)
        normDf.index = index[nonzeroGenes]
        normDf.columns = index[nonzeroGenes]          
    else:            
        normFM = np.transpose(np.transpose(frequencyMatrix)/traceFM)
        normDf = pd.DataFrame(normFM)
        normDf.index = index
        normDf.columns = index   
    
    return normDf

def f1Binary(similarityMatrix):

    remainingMembers = set(similarityMatrix.index)
    # probeSample is the sample that serves as a seed to identify a cluster in a given iteration
    probeSample = np.argmax(similarityMatrix.sum(axis=1))
    # members are the samples that satisfy the similarity condition with the previous cluster or probeSample
    members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
    # nonMembers are the remaining members not in the current cluster
    nonMembers = remainingMembers-members
    # instantiate list to collect clusters of similar members
    similarityClusters = []
    # instantiate f1 score for optimization
    f1 = 0

    for iteration in range(1500):

        predictedMembers = members
        predictedNonMembers = remainingMembers-predictedMembers

        sumSlice = np.sum(similarityMatrix.loc[:,list(predictedMembers)],axis=1)/float(len(predictedMembers))
        members = set(similarityMatrix.index[np.where(sumSlice>0.8)[0]])

        if members==predictedMembers:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers)==0:
                break    
            similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            probeSample = np.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue

        nonMembers = remainingMembers-members
        TP = len(members&predictedMembers)
        FN = len(predictedNonMembers&members)
        FP = len(predictedMembers&nonMembers)
        tmpf1 = TP/float(TP+FN+FP)

        if tmpf1 <= f1:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers)==0:
                break
            similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            probeSample = np.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue   

        elif tmpf1 > f1:
            f1 = tmpf1
            continue
    
    similarityClusters.sort(key = lambda s: -len(s))

    return similarityClusters

def unmix(df,iterations=25,returnAll=False):    
    frequencyClusters = []
    
    for iteration in range(iterations):
        sumDf1 = df.sum(axis=1)
        maxSum = np.argmax(sumDf1)
        hits = np.where(df.loc[maxSum]>0)[0]
        hitIndex = list(df.index[hits])
        block = df.loc[hitIndex,hitIndex]
        blockSum = block.sum(axis=1)
        coreBlock = list(blockSum.index[np.where(blockSum>=np.median(blockSum))[0]])
        remainder = list(set(df.index)-set(coreBlock))
        frequencyClusters.append(coreBlock)
        if len(remainder)==0:
            return frequencyClusters
        if len(coreBlock)==1:
            return frequencyClusters
        df = df.loc[remainder,remainder]
    if returnAll is True:
        frequencyClusters.append(remainder)
    return frequencyClusters

def remix(df,frequencyClusters):    
    finalClusters = []
    for cluster in frequencyClusters:
        sliceDf = df.loc[cluster,:]
        sumSlice = sliceDf.sum(axis=0)
        cut = min(0.8,np.percentile(sumSlice.loc[cluster]/float(len(cluster)),90))
        minGenes = max(4,cut*len(cluster))
        keepers = list(sliceDf.columns[np.where(sumSlice>=minGenes)[0]])
        keepers = list(set(keepers)|set(cluster))
        finalClusters.append(keepers)
        finalClusters.sort(key = lambda s: -len(s))
    return finalClusters

def decompose(geneset,expressionData,minNumberGenes=6): 
    fm = FrequencyMatrix(expressionData.loc[geneset,:])
    tst = np.multiply(fm,fm.T)
    tst[tst<np.percentile(tst,80)]=0
    tst[tst>0]=1
    unmix_tst = unmix(tst)
    unmixedFiltered = [i for i in unmix_tst if len(i)>=minNumberGenes]
    return unmixedFiltered

def recursiveDecomposition(geneset,expressionData,minNumberGenes=6):
    unmixedFiltered = decompose(geneset,expressionData,minNumberGenes=minNumberGenes)   
    if len(unmixedFiltered) == 0:
        return []
    shortSets = [i for i in unmixedFiltered if len(i)<50]
    longSets = [i for i in unmixedFiltered if len(i)>=50]    
    if len(longSets)==0:
        return unmixedFiltered
    for ls in longSets:
        unmixedFiltered = decompose(ls,expressionData,minNumberGenes=minNumberGenes)
        if len(unmixedFiltered)==0:
            continue
        shortSets.extend(unmixedFiltered)
    return shortSets

def iterativeCombination(dict_,key,iterations=25):    
    initial = dict_[key]
    initialLength = len(initial)
    for iteration in range(iterations):
        revised = [i for i in initial]
        for element in initial:
            revised = list(set(revised)|set(dict_[element]))
        revisedLength = len(revised)
        if revisedLength == initialLength:
            return revised
        elif revisedLength > initialLength:
            initial = [i for i in revised]
            initialLength = len(initial)
    return revised

def decomposeDictionaryToLists(dict_):
    decomposedSets = []
    for key in dict_.keys():
        newSet = iterativeCombination(dict_,key,iterations=25)
        if newSet not in decomposedSets:
            decomposedSets.append(newSet)
    return decomposedSets

def combineClusters(axes,clusters,threshold=0.925):
    combineAxes = {}
    filterKeys = np.array(axes.keys())
    axesMatrix = np.vstack([axes[i] for i in filterKeys])
    for key in filterKeys:
        axis = axes[key]
        pearson = pearson_array(axesMatrix,axis)
        combine = np.where(pearson>threshold)[0]
        combineAxes[key] = filterKeys[combine]
     
    revisedClusters = {}
    combinedKeys = decomposeDictionaryToLists(combineAxes)
    for keyList in combinedKeys:
        genes = list(set(np.hstack([clusters[i] for i in keyList])))
        revisedClusters[len(revisedClusters)] = genes

    return revisedClusters

def reconstruction(decomposedList,expressionData,threshold=0.925):
    clusters = {i:decomposedList[i] for i in range(len(decomposedList))}
    axes = getAxes(clusters,expressionData)
    recombine = combineClusters(axes,clusters,threshold)
    return recombine

def recursiveAlignment(geneset,expressionData,minNumberGenes=6):
    recDecomp = recursiveDecomposition(geneset,expressionData,minNumberGenes)
    if len(recDecomp) == 0:
        return []
    reconstructed = reconstruction(recDecomp,expressionData)
    reconstructedList = [reconstructed[i] for i in reconstructed.keys() if reconstructed[i]>minNumberGenes]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList

def cluster(expressionData,minNumberGenes = 6,minNumberOverExpSamples=4,maxSamplesExcluded=0.50,random_state=12,overExpressionThreshold=80):

    try:
        df = expressionData.copy()
        maxStep = int(np.round(10*maxSamplesExcluded))
        allGenesMapped = []
        bestHits = []

        zero = np.percentile(expressionData,0)
        expressionThreshold = np.mean([np.percentile(expressionData.iloc[:,i][expressionData.iloc[:,i]>zero],overExpressionThreshold) for i in range(expressionData.shape[1])])

        startTimer = time.time()
        trial = -1
        for step in range(maxStep):
            trial+=1
            progress = (100./maxStep)*trial
            print('{:.2f} percent complete'.format(progress))
            genesMapped = []
            bestMapped = []

            pca = PCA(10,random_state=random_state)
            principalComponents = pca.fit_transform(df.T)
            principalDf = pd.DataFrame(principalComponents)
            principalDf.index = df.columns

            for i in range(10):
                pearson = pearson_array(np.array(df),np.array(principalDf[i]))
                if len(pearson) == 0:
                    continue
                highpass = max(np.percentile(pearson,95),0.1)
                lowpass = min(np.percentile(pearson,5),-0.1)
                cluster1 = np.array(df.index[np.where(pearson>highpass)[0]])
                cluster2 = np.array(df.index[np.where(pearson<lowpass)[0]])

                for clst in [cluster1,cluster2]:
                    pdc = recursiveAlignment(clst,expressionData=df,minNumberGenes=minNumberGenes)
                    if len(pdc)==0:
                        continue
                    elif len(pdc) == 1:
                        genesMapped.append(pdc[0])
                    elif len(pdc) > 1:
                        for j in range(len(pdc)-1):
                            if len(pdc[j]) > minNumberGenes:
                                genesMapped.append(pdc[j])

            allGenesMapped.extend(genesMapped)
            try:
                stackGenes = np.hstack(genesMapped)
            except:
                stackGenes = []
            residualGenes = list(set(df.index)-set(stackGenes))
            df = df.loc[residualGenes,:]

            # computationally fast surrogate for passing the overexpressed significance test:
            for ix in range(len(genesMapped)):
                tmpCluster = expressionData.loc[genesMapped[ix],:]
                tmpCluster[tmpCluster<expressionThreshold] = 0
                tmpCluster[tmpCluster>0] = 1
                sumCluster = np.array(np.sum(tmpCluster,axis=0))
                numHits = np.where(sumCluster>0.333*len(genesMapped[ix]))[0]
                bestMapped.append(numHits)
                if len(numHits)>minNumberOverExpSamples:
                    bestHits.append(genesMapped[ix])

            if len(bestMapped)>0:            
                countHits = Counter(np.hstack(bestMapped))
                ranked = countHits.most_common()
                dominant = [i[0] for i in ranked[0:int(np.ceil(0.1*len(ranked)))]]
                remainder = [i for i in np.arange(df.shape[1]) if i not in dominant]
                df = df.iloc[:,remainder]

        bestHits.sort(key=lambda s: -len(s))

        stopTimer = time.time()
        print('\ncoexpression clustering completed in {:.2f} minutes'.format((stopTimer-startTimer)/60.))

    except:
        print('\nClustering failed. Ensure that expression data is formatted with genes as rows and samples as columns.')
        print('Consider transposing data (expressionData = expressionData.T) and retrying')

    return bestHits

def backgroundDf(expressionData):

    low = np.percentile(expressionData,100./3,axis=0)
    high = np.percentile(expressionData,200./3,axis=0)
    evenCuts = zip(low,high)
    
    bkgd = expressionData.copy()
    for i in range(bkgd.shape[1]):
        lowCut = evenCuts[i][0]
        highCut = evenCuts[i][1]    
        bkgd.iloc[:,i][bkgd.iloc[:,i]>=highCut]=1
        bkgd.iloc[:,i][bkgd.iloc[:,i]<=lowCut]=-1
        bkgd.iloc[:,i][np.abs(bkgd.iloc[:,i])!=1]=0    

    return bkgd
            
def assignMembership(geneset,background,p=0.05):

    cluster = background.loc[geneset,:]
    classNeg1 = len(geneset)-np.count_nonzero(cluster+1,axis=0)
    class0 = len(geneset)-np.count_nonzero(cluster,axis=0)
    class1 = len(geneset)-np.count_nonzero(cluster-1,axis=0)
    observations = zip(classNeg1,class0,class1)
    
    highpass = stats.binom.ppf(1-p/3,len(geneset),1./3)
    classes = []
    for i in range(len(observations)):
        check = np.where(np.array(observations[i])>=highpass)[0]
        if len(check)>1:
            check = np.array([np.argmax(np.array(observations[i]))])
        classes.append(check)
    
    return classes

def clusterScore(membership,pMembership=0.05):
    hits = len([i for i in membership if len(i)>0])
    N = len(membership)
    return 1-stats.binom.cdf(hits,N,pMembership)

def getClusterScores(coexpressionLists,background,p=0.05):
    clusterScores = []
    for i in range(len(coexpressionLists)):    
        members = assignMembership(coexpressionLists[i],background,p)    
        score = clusterScore(members)
        clusterScores.append(score)
    return np.array(clusterScores)

def filterCoexpressionDict(coexpressionDict,clusterScores,threshold=0.01):   
    filterPoorClusters = np.where(clusterScores>threshold)[0]
    for x in filterPoorClusters:
        del coexpressionDict[x]
    keys = coexpressionDict.keys()
    filteredDict = {i:coexpressionDict[keys[i]] for i in range(len(coexpressionDict))}
    return filteredDict

def biclusterMembershipDictionary(revisedClusters,background,label=2):

    if label == "excluded":
        members = {}
        for key in revisedClusters.keys():
            assignments = assignMembership(revisedClusters[key],background)
            nonMembers = np.array([i for i in range(len(assignments)) if len(assignments[i])==0])
            if len(nonMembers) == 0:
                members[key] = []
                continue
            members[key] = list(background.columns[nonMembers])
        print("done!")
        return members
    
    if label == "included":
        members = {}
        for key in revisedClusters.keys():
            assignments = assignMembership(revisedClusters[key],background)
            included = np.array([i for i in range(len(assignments)) if len(assignments[i])!=0])
            if len(included) == 0:
                members[key] = []
                continue
            members[key] = list(background.columns[included])
        print("done!")
        return members
    
    members = {}
    for key in revisedClusters.keys():
        assignments = assignMembership(revisedClusters[key],background)
        overExpMembers = np.array([i for i in range(len(assignments)) if label in assignments[i]])
        if len(overExpMembers) ==0:
            members[key] = []
            continue
        members[key] = list(background.columns[overExpMembers])
    print("done!")
    return members

def membershipToIncidence(membershipDictionary,expressionData):
    
    incidence = np.zeros((len(membershipDictionary),expressionData.shape[1]))
    incidence = pd.DataFrame(incidence)
    incidence.index = membershipDictionary.keys()
    incidence.columns = expressionData.columns
    for key in membershipDictionary.keys():
        samples = membershipDictionary[key]
        incidence.loc[key,samples] = 1
    
    try:
        orderIndex = np.array(incidence.index).astype(int)
        orderIndex = np.sort(orderIndex)
    except:
        orderIndex = incidence.index
    try:
        incidence = incidence.loc[orderIndex,:]
    except:
        incidence = incidence.loc[orderIndex.astype(str),:]
        
    return incidence

def processCoexpressionLists(lists,expressionData,threshold=0.925):
    reconstructed = reconstruction(lists,expressionData,threshold)
    reconstructedList = [reconstructed[i] for i in reconstructed.keys()]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList

def reviseInitialClusters(clusterList,expressionData,threshold=0.925):
    coexpressionLists = processCoexpressionLists(clusterList,expressionData,threshold)
    coexpressionLists.sort(key= lambda s: -len(s))
    
    for iteration in range(5):
        previousLength = len(coexpressionLists)
        coexpressionLists = processCoexpressionLists(coexpressionLists,expressionData,threshold)
        newLength = len(coexpressionLists)
        if newLength == previousLength:
            break
    
    coexpressionLists.sort(key= lambda s: -len(s))
    coexpressionDict = {i:list(coexpressionLists[i]) for i in range(len(coexpressionLists))}

    # generate background matrix for cluster significance scoring
    bkgd = backgroundDf(expressionData)
    # score significance of coexpression in each cluster; 
    clusterScores = getClusterScores(coexpressionLists,bkgd,p=0.05)
    # revise clusters to only include those with sufficient coexpression
    revisedClusters = filterCoexpressionDict(coexpressionDict,clusterScores,threshold=0.01)

    return revisedClusters

# =============================================================================
# Functions used for mechanistic inference
# =============================================================================

def regulonDictionary(regulons):
    regulonModules = {}
    #str(i):[regulons[key][j]]}
    df_list = []
    for tf in regulons.keys():
        for key in regulons[tf].keys():
            genes = regulons[tf][key]
            id_ = str(len(regulonModules))
            regulonModules[id_] = regulons[tf][key]
            for gene in genes:
                df_list.append([id_,tf,gene])
    
    array = np.vstack(df_list)
    df = pd.DataFrame(array)
    df.columns = ["Regulon_ID","Regulator","Gene"]
    
    return regulonModules, df

def principalDf(dict_,expressionData,regulons=None,subkey='genes',minNumberGenes=8,random_state=12):

    pcDfs = []
    setIndex = set(expressionData.index)
    
    if regulons is not None:
        dict_ = regulonDictionary(regulons)
    for i in dict_.keys():
        if subkey is not None:
            genes = list(set(dict_[i][subkey])&setIndex)
            if len(genes) < minNumberGenes:
                continue
        elif subkey is None:
            genes = list(set(dict_[i])&setIndex)
            if len(genes) < minNumberGenes:
                continue
            
        pca = PCA(1,random_state=random_state)
        principalComponents = pca.fit_transform(expressionData.loc[genes,:].T)
        principalDf = pd.DataFrame(principalComponents)
        principalDf.index = expressionData.columns
        principalDf.columns = [str(i)]
        
        normPC = np.linalg.norm(np.array(principalDf.iloc[:,0]))
        pearson = stats.pearsonr(principalDf.iloc[:,0],np.median(expressionData.loc[genes,:],axis=0))
        signCorrection = pearson[0]/np.abs(pearson[0])
        
        principalDf = signCorrection*principalDf/normPC
        
        pcDfs.append(principalDf)
    
    principalMatrix = pd.concat(pcDfs,axis=1)
        
    return principalMatrix

def axisTfs(axesDf,tfList,expressionData,correlationThreshold=0.3):
    
    axesArray = np.array(axesDf.T)
    tfArray = np.array(expressionData.loc[tfList,:])
    axes = np.array(axesDf.columns)
    tfDict = {}
    
    if type(tfList) is list:
        tfs = np.array(tfList)
    elif type(tfList) is not list:
        tfs = tfList
    for axis in range(axesArray.shape[0]):
        tfCorrelation = pearson_array(tfArray,axesArray[axis,:])
        tfDict[axes[axis]] = tfs[np.where(np.abs(tfCorrelation)>=correlationThreshold)[0]]
    
    return tfDict

def splitForMultiprocessing(vector,cores):
    
    partition = int(len(vector)/cores)
    remainder = len(vector) - cores*partition
    starts = np.arange(0,len(vector),partition)[0:cores]
    for i in range(remainder):
        starts[cores-remainder+i] = starts[cores-remainder+i] + i    

    stops = starts+partition
    for i in range(remainder):
        stops[cores-remainder+i] = stops[cores-remainder+i] + 1
        
    return zip(starts,stops)

def multiprocess(function,tasks):
    import multiprocessing, multiprocessing.pool
    hydra=multiprocessing.pool.Pool(len(tasks))  
    output=hydra.map(function,tasks)   
    hydra.close()
    hydra.join()
    return output

def hyper(population,set1,set2,overlap):
    
    b = max(set1,set2)
    c = min(set1,set2)
    hyp = stats.hypergeom(population,b,c)
    prb = sum([hyp.pmf(l) for l in range(overlap,c+1)])
    
    return prb 

def tfbsdbEnrichment(task,p=0.05):
    
    start, stop = task[0]
    expressionData,revisedClusters,tfMap,tfToGenes = task[1]
    allGenes = list(expressionData.index)
    keys = revisedClusters.keys()[start:stop]
    
    clusterTfs = {}
    for key in keys:
        print(key)
        for tf in tfMap[str(key)]:    
            hits0TfTargets = list(set(tfToGenes[tf])&set(allGenes))   
            hits0clusterGenes = revisedClusters[key]
            overlapCluster = list(set(hits0TfTargets)&set(hits0clusterGenes))
            pHyper = hyper(len(allGenes),len(hits0TfTargets),len(hits0clusterGenes),len(overlapCluster))
            if pHyper < p:
                if key not in clusterTfs.keys():
                    clusterTfs[key] = {}
                clusterTfs[key][tf] = [pHyper,overlapCluster]
    
    return clusterTfs

def condenseOutput(output):
    
    results = {}
    for i in range(len(output)):
        resultsDict = output[i]
        keys = resultsDict.keys()
        for j in range(len(resultsDict)):
            key = keys[j]
            results[key] = resultsDict[key]
    return results

def mechanisticInference(axes,revisedClusters,expressionData,correlationThreshold=0.3,numCores=5,dataFolder=os.path.join(os.path.expanduser("~"),"Desktop","miner","data")):
    import os
    tfToGenesPath = os.path.join(dataFolder,"network_dictionaries","tfbsdb_tf_to_genes.pkl")
    tfToGenes = read_pkl(tfToGenesPath)
     
    tfs = tfToGenes.keys()     
    tfMap = axisTfs(axes,tfs,expressionData,correlationThreshold=correlationThreshold) 
    taskSplit = splitForMultiprocessing(revisedClusters.keys(),numCores)
    tasks = [[taskSplit[i],(expressionData,revisedClusters,tfMap,tfToGenes)] for i in range(len(taskSplit))]
    tfbsdbOutput = multiprocess(tfbsdbEnrichment,tasks)
    mechanisticOutput = condenseOutput(tfbsdbOutput)
    
    return mechanisticOutput

def coincidenceMatrix(coregulationModules,key,freqThreshold = 0.333):

    tf = coregulationModules.keys()[key]
    subRegulons = coregulationModules[tf]
    srGenes = list(set(np.hstack([subRegulons[i] for i in subRegulons.keys()])))

    template = pd.DataFrame(np.zeros((len(srGenes),len(srGenes))))
    template.index = srGenes
    template.columns = srGenes
    for key in subRegulons.keys():
        genes = subRegulons[key]
        template.loc[genes,genes]+=1
    trace = np.array([template.iloc[i,i] for i in range(template.shape[0])]).astype(float)
    normDf = ((template.T)/trace).T
    normDf[normDf<freqThreshold]=0
    normDf[normDf>0]=1
    
    return normDf

def getCoregulationModules(mechanisticOutput):
    
    coregulationModules = {}
    for i in mechanisticOutput.keys():
        for key in mechanisticOutput[i].keys():
            if key not in coregulationModules.keys():
                coregulationModules[key] = {}
            genes = mechanisticOutput[i][key][1]
            coregulationModules[key][i] = genes
    return coregulationModules

def getRegulons(coregulationModules,minNumberGenes=5,freqThreshold = 0.333):

    regulons = {}
    keys = coregulationModules.keys()
    for i in range(len(keys)):
        tf = keys[i]
        normDf = coincidenceMatrix(coregulationModules,key=i,freqThreshold = 0.333)
        unmixed = unmix(normDf)   
        remixed = remix(normDf,unmixed)
        if len(remixed)>0:
            for cluster in remixed:
                if len(cluster)>minNumberGenes:
                    if tf not in regulons.keys():
                        regulons[tf] = {}
                    regulons[tf][len(regulons[tf])] = cluster                    
    return regulons

def getCoexpressionModules(mechanisticOutput):
    coexpressionModules = {}
    for i in mechanisticOutput.keys():
        genes = list(set(np.hstack([mechanisticOutput[i][key][1] for key in mechanisticOutput[i].keys()])))
        coexpressionModules[i] = genes
    return coexpressionModules

def f1Regulons(coregulationModules,minNumberGenes=5,freqThreshold = 0.1):

    regulons = {}
    keys = coregulationModules.keys()
    for i in range(len(keys)):
        tf = keys[i]
        normDf = coincidenceMatrix(coregulationModules,key=i,freqThreshold = freqThreshold)
        remixed = f1Binary(normDf)
        if len(remixed)>0:
            for cluster in remixed:
                if len(cluster)>minNumberGenes:
                    if tf not in regulons.keys():
                        regulons[tf] = {}
                    regulons[tf][len(regulons[tf])] = cluster                    
    return regulons

# =============================================================================
# Functions used for post-processing mechanistic inference
# =============================================================================

def convertDictionary(dict_,conversionTable):
    converted = {}
    for i in dict_.keys():
        genes = dict_[i]
        conv_genes = conversionTable[genes]
        for j in range(len(conv_genes)):
            if type(conv_genes[j]) is pd.core.series.Series:
                conv_genes[j] = conv_genes[j][0]        
        converted[i] = list(conv_genes)
    return converted

def convertRegulons(df,conversionTable):
    regIds = []
    regs = []
    genes = []
    for i in range(df.shape[0]):
        regIds.append(df.iloc[i,0])
        tmpReg = conversionTable[df.iloc[i,1]]
        if type(tmpReg) is pd.core.series.Series:
            tmpReg=tmpReg[0]
        regs.append(tmpReg)
        tmpGene = conversionTable[df.iloc[i,2]]
        if type(tmpGene) is pd.core.series.Series:
            tmpGene = tmpGene[0]
        genes.append(tmpGene)
    regulonDfConverted = pd.DataFrame(np.vstack([regIds,regs,genes]).T)
    regulonDfConverted.columns = ["Regulon_ID","Regulator","Gene"]
    return regulonDfConverted

# =============================================================================
# Functions used for inferring sample subtypes
# =============================================================================

def sampleCoincidenceMatrix(dict_,freqThreshold = 0.333,frequencies=False):

    keys = dict_.keys()
    lists = [dict_[key] for key in keys]
    samples = list(set(np.hstack(lists)))
    
    template = pd.DataFrame(np.zeros((len(samples),len(samples))))
    template.index = samples
    template.columns = samples
    for key in keys:
        hits = dict_[key]
        template.loc[hits,hits]+=1
    trace = np.array([template.iloc[i,i] for i in range(template.shape[0])])
    normDf = ((template.T)/trace).T
    if frequencies is not False:
        return normDf
    normDf[normDf<freqThreshold]=0
    normDf[normDf>0]=1
    
    return normDf

def f1Decomposition(sampleMembers=None,thresholdSFM=0.333,sampleFrequencyMatrix=None):
    # thresholdSFM is the probability cutoff that makes the density of the binary similarityMatrix = 0.15
    # sampleMembers is a dictionary with features as keys and members as elements

    # sampleFrequencyMatrix[i,j] gives the probability that sample j appears in a cluster given that sample i appears
    if sampleFrequencyMatrix is None:
        sampleFrequencyMatrix = sampleCoincidenceMatrix(sampleMembers,freqThreshold = thresholdSFM,frequencies=True)
    # similarityMatrix is defined such that similarityMatrix[i,j] = 1 iff sampleFrequencyMatrix[i,j] >= thresholdSFM
    similarityMatrix = sampleFrequencyMatrix*sampleFrequencyMatrix.T
    similarityMatrix[similarityMatrix<thresholdSFM] = 0
    similarityMatrix[similarityMatrix>0] = 1
    # remainingMembers is the set of set of unclustered members
    remainingMembers = set(similarityMatrix.index)
    # probeSample is the sample that serves as a seed to identify a cluster in a given iteration
    probeSample = np.argmax(similarityMatrix.sum(axis=1))
    # members are the samples that satisfy the similarity condition with the previous cluster or probeSample
    members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
    # nonMembers are the remaining members not in the current cluster
    nonMembers = remainingMembers-members
    # instantiate list to collect clusters of similar members
    similarityClusters = []
    # instantiate f1 score for optimization
    f1 = 0

    for iteration in range(1500):

        predictedMembers = members
        predictedNonMembers = remainingMembers-predictedMembers

        sumSlice = np.sum(similarityMatrix.loc[:,list(predictedMembers)],axis=1)/float(len(predictedMembers))
        members = set(similarityMatrix.index[np.where(sumSlice>0.8)[0]])

        if members==predictedMembers:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers)==0:
                break    
            similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            probeSample = np.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue

        nonMembers = remainingMembers-members
        TP = len(members&predictedMembers)
        FN = len(predictedNonMembers&members)
        FP = len(predictedMembers&nonMembers)
        tmpf1 = TP/float(TP+FN+FP)

        if tmpf1 <= f1:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers)==0:
                break
            similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            probeSample = np.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue   

        elif tmpf1 > f1:
            f1 = tmpf1
            continue
    
    similarityClusters.sort(key = lambda s: -len(s))
    print('done!')
    return similarityClusters

def plotSimilarity(similarityMatrix,orderedSamples,vmin=0,vmax=0.5,title="Similarity matrix",xlabel="Samples",ylabel="Samples",fontsize=14,figsize=(7,7),savefig=None):
    fig = plt.figure(figsize=figsize)  
    ax = fig.add_subplot(111)
    try:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    except:
        pass    
    ax.imshow(similarityMatrix.loc[orderedSamples,orderedSamples],cmap='viridis',vmin=vmin,vmax=vmax)
    ax.grid(False)
    plt.title(title,FontSize=fontsize+2)
    plt.xlabel(xlabel,FontSize=fontsize)
    plt.ylabel(ylabel,FontSize=fontsize)
    if savefig is not None:
        plt.savefig(savefig)
    return

def f1(vector1,vector2):
    members = set(np.where(vector1==1)[0])
    nonMembers = set(np.where(vector1==0)[0])
    predictedMembers = set(np.where(vector2==1)[0])
    predictedNonMembers = set(np.where(vector2==0)[0])
    
    TP = len(members&predictedMembers)
    FN = len(predictedNonMembers&members)
    FP = len(predictedMembers&nonMembers)
    if TP == 0:
        return 0.0
    F1 = TP/float(TP+FN+FP)
    
    return F1

def centroids(classes,sampleMatrix,f1Threshold = 0.3,returnCentroids=None):
    centroids = []
    for i in range(len(classes)):
        clusterComponents = sampleMatrix.loc[:,classes[i]]
        class1 = np.mean(clusterComponents,axis=1)
        hits = np.where(class1>0.6)[0]
        centroid = pd.DataFrame(sampleMatrix.iloc[:,0])
        centroid.columns = [i]
        centroid[i] = 0
        centroid.iloc[hits,0] = 1
        centroids.append(centroid)

    miss = []
    centroidClusters = [[] for i in range(len(centroids))]
    for smpl in sampleMatrix.columns:
        probeVector = np.array(sampleMatrix[smpl])
        scores = []
        for ix in range(len(centroids)):
            tmp = f1(np.array(probeVector),centroids[ix])
            scores.append(tmp)
        scores = np.array(scores)
        match = np.argmax(scores)
        if scores[match] < f1Threshold:
            miss.append(smpl)
        elif scores[match] >= f1Threshold:
            centroidClusters[match].append(smpl)

    centroidClusters.append(miss)  

    if returnCentroids is not None:
        centroidMatrix = pd.DataFrame(pd.concat(centroids,axis=1))
        return centroidClusters, centroidMatrix
        
    return centroidClusters

def mapExpressionToNetwork(centroidMatrix,membershipMatrix,threshold = 0.05):
    
    miss = []
    centroidClusters = [[] for i in range(centroidMatrix.shape[1])]
    for smpl in membershipMatrix.columns:
        probeVector = np.array(membershipMatrix[smpl])
        scores = []
        for ix in range(centroidMatrix.shape[1]):
            tmp = f1(np.array(probeVector),np.array(centroidMatrix.iloc[:,ix]))
            scores.append(tmp)
        scores = np.array(scores)
        match = np.argmax(scores)
        if scores[match] < threshold:
            miss.append(smpl)
        elif scores[match] >= threshold:
            centroidClusters[match].append(smpl)
    centroidClusters.append(miss)
    
    return centroidClusters

def orderMembership(centroidMatrix,membershipMatrix,mappedClusters,ylabel="",resultsDirectory=None):

    centroidRank = []
    alreadyMapped = []
    for ix in range(centroidMatrix.shape[1]):
        tmp = np.where(centroidMatrix.iloc[:,ix]==1)[0]
        signature = list(set(tmp)-set(alreadyMapped))
        centroidRank.extend(signature)
        alreadyMapped.extend(signature)
    orderedClusters = centroidMatrix.index[np.array(centroidRank)]
    try:
        ordered_matrix = membershipMatrix.loc[orderedClusters,np.hstack(mappedClusters)]
    except:
        ordered_matrix = membershipMatrix.loc[np.array(orderedClusters).astype(int),np.hstack(mappedClusters)]
      
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    except:
        pass    
    ax.imshow(ordered_matrix,cmap='viridis',aspect="auto")
    ax.grid(False)
        
    plt.title(ylabel.split("s")[0]+"Activation",FontSize=16)
    plt.xlabel("Samples",FontSize=14)
    plt.ylabel(ylabel,FontSize=14)
    if resultsDirectory is not None:
        plt.savefig(os.path.join(resultsDirectory,"binaryActivityMap.pdf"))
    return ordered_matrix

def plotDifferentialMatrix(overExpressedMembersMatrix,underExpressedMembersMatrix,orderedOverExpressedMembers,cmap="viridis",aspect="auto",saveFile=None):
    differentialActivationMatrix = overExpressedMembersMatrix-underExpressedMembersMatrix
    fig = plt.figure(figsize=(7,7))  
    ax = fig.add_subplot(111)
    try:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    except:
        pass
    orderedDM = differentialActivationMatrix.loc[orderedOverExpressedMembers.index,orderedOverExpressedMembers.columns]
    ax.imshow(orderedDM,cmap=cmap,vmin=-1,vmax=1,aspect=aspect)
    ax.grid(False)
    if saveFile is not None:
        plt.ylabel("Modules",FontSize=14)
        plt.xlabel("Samples",FontSize=14)
        ax.grid(False)        
        plt.savefig(saveFile,bbox_inches="tight")
    return orderedDM

def kmeans(df,numClusters,random_state=None):
    from sklearn.cluster import KMeans

    if random_state is not None:
        # Number of clusters
        kmeans = KMeans(n_clusters=numClusters,random_state=random_state)

    elif random_state is None:    
        # Number of clusters
        kmeans = KMeans(n_clusters=numClusters)
    
    # Fitting the input data
    kmeans = kmeans.fit(df)
    # Getting the cluster labels
    labels = kmeans.predict(df)
    # Centroid values
    centroids = kmeans.cluster_centers_

    clusters = []
    for i in range(numClusters):
        clstr = df.index[np.where(labels==i)[0]]
        clusters.append(clstr)

    return clusters, labels, centroids

def mosaic(dfr,clusterList,minClusterSize_x=4,minClusterSize_y=5,allow_singletons=True,max_groups=50,saveFile=None,random_state=12):    
    
    import sklearn

    lowResolutionPrograms = [[] for i in range(len(clusterList))]
    sorting_hat = []
    for i in range(len(clusterList)):
        patients = clusterList[i]
        if len(patients) < minClusterSize_x:
            continue
        subset = dfr.loc[:,patients]
        density = subset.sum(axis=1)/float(subset.shape[1])
        sorting_hat.append(np.array(density))
    
    enrichment_matrix = np.vstack(sorting_hat).T    
    choice = np.argmax(enrichment_matrix,axis=1)
    for i in range(dfr.shape[0]):
        lowResolutionPrograms[choice[i]].append(dfr.index[i])
    
    print(len(lowResolutionPrograms))
    
    #Cluster modules into transcriptional programs
    
    y_clusters = []
    for program in range(len(lowResolutionPrograms)):
        regs = lowResolutionPrograms[program]
        if len(regs) == 0:
            continue
        df = dfr.loc[regs,:]
        sil_scores = []
        max_clusters_y = min(max_groups,int(len(regs)/3.))
        for numClusters_y in range(2,max_clusters_y):
            clusters_y, labels_y, centroids_y = kmeans(df,numClusters=numClusters_y,random_state=random_state)
            lens_y = [len(c) for c in clusters_y]
            if min(lens_y) < minClusterSize_y:
                if allow_singletons is True:
                    if min(lens_y) != 1:
                        kmSS = 0
                        sil_scores.append(kmSS)
                        continue
                    elif min(lens_y) == 1:
                        pass
                elif allow_singletons is not True:
                    kmSS = 0
                    sil_scores.append(kmSS)
                    continue
                
            clusters_y.sort(key=lambda s: -len(s))
    
            kmSS=sklearn.metrics.silhouette_score(df,labels_y,metric='euclidean')
            sil_scores.append(kmSS)
    
        if len(sil_scores) > 0:
            top_hit = min(np.where(np.array(sil_scores)>=0.95*max(sil_scores))[0]+2)
            print(top_hit)
            clusters_y, labels_y, centroids_y = kmeans(df,numClusters=top_hit,random_state=random_state)
            clusters_y.sort(key=lambda s: -len(s))
            y_clusters.append(list(clusters_y))
    
        elif len(sil_scores) == 0:
            y_clusters.append(regs)
        
    order_y = np.hstack([np.hstack(y_clusters[i]) for i in range(len(y_clusters))])
    
    #Cluster patients into subtype states
    
    x_clusters = []
    for c in range(len(clusterList)):
        patients = clusterList[c]
        if len(patients) == 0:
            continue
        df = dfr.loc[order_y,patients].T
        sil_scores = []
        
        max_clusters_x = min(max_groups,int(len(patients)/3.))
        for numClusters_x in range(2,max_clusters_x):
            clusters_x, labels_x, centroids_x = kmeans(df,numClusters=numClusters_x,random_state=random_state)
            lens_x = [len(c) for c in clusters_x]
            if min(lens_x) < minClusterSize_x:
                if allow_singletons is True:
                    if min(lens_x) != 1:
                        kmSS = 0
                        sil_scores.append(kmSS)
                        continue
                    elif min(lens_x) == 1:
                        pass
                elif allow_singletons is not True:
                    kmSS = 0
                    sil_scores.append(kmSS)
                    continue
                
            clusters_x.sort(key=lambda s: -len(s))
    
            kmSS=sklearn.metrics.silhouette_score(df,labels_x,metric='euclidean')
            sil_scores.append(kmSS)
    
        if len(sil_scores) > 0:
            top_hit = min(np.where(np.array(sil_scores)>=0.999*max(sil_scores))[0]+2)
            print(top_hit)
            clusters_x, labels_x, centroids_x = kmeans(df,numClusters=top_hit,random_state=random_state)
            clusters_x.sort(key=lambda s: -len(s))
            x_clusters.append(list(clusters_x))
        elif len(sil_scores) == 0:
            x_clusters.append(patients)
    try:
        combine_x = []
        for x in range(len(x_clusters)):
            if type(x_clusters[x][0]) is not str:
                for k in range(len(x_clusters[x])):
                    combine_x.append(x_clusters[x][k])
            else:
                combine_x.append(x_clusters[x])
                               
        order_x = np.hstack(combine_x)
        fig = plt.figure(figsize=(7,7))
        ax = fig.gca()
        ax.imshow(dfr.loc[order_y,order_x],cmap="bwr",vmin=-1,vmax=1)
        ax.set_aspect(dfr.shape[1]/float(dfr.shape[0]))
        ax.grid(False)
        ax.set_ylabel("Regulons",FontSize=14)
        ax.set_xlabel("Samples",FontSize=14)
        if saveFile is not None:
            plt.savefig(saveFile)
            
        return y_clusters, combine_x
    
    except:
        pass
        
    return y_clusters, x_clusters

def transcriptionalPrograms(programs,reference_dictionary):
    transcriptionalPrograms = {}
    programRegulons = {}
    p_stack = []
    programs_flattened = np.array(programs).flatten()
    for i in range(len(programs_flattened)):
        if type(programs_flattened[i][0])==pd.core.indexes.base.Index:
            for j in range(len(programs_flattened[i])):
                p_stack.append(list(programs[i][j]))        
        else:
            p_stack.append(list(programs[i]))

    for j in range(len(p_stack)):
        key = ("").join(["TP",str(j)])
        regulonList = [i for i in p_stack[j]]
        programRegulons[key] = regulonList
        tmp = [reference_dictionary[i] for i in p_stack[j]]
        transcriptionalPrograms[key] = list(set(np.hstack(tmp)))
    return transcriptionalPrograms, programRegulons

def stateProjection(df,programs,states,stateThreshold=0.75,saveFile=None):
    
    df = df.loc[:,np.hstack(states)]
    statesDf = pd.DataFrame(np.zeros((len(programs),df.shape[1])))
    statesDf.index = range(len(programs))
    statesDf.columns = df.columns

    for i in range(len(programs)):
        state = programs[i]
        subset = df.loc[state,:]

        state_scores = subset.sum(axis=0)/float(subset.shape[0])

        keep_high = np.where(state_scores>=stateThreshold)[0]
        keep_low = np.where(state_scores<=-1*stateThreshold)[0]
        hits_high = np.array(df.columns)[keep_high]
        hits_low = np.array(df.columns)[keep_low]

        statesDf.loc[i,hits_high] = 1
        statesDf.loc[i,hits_low] = -1
        
    if saveFile is not None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.gca()
        ax.imshow(statesDf,cmap="bwr",vmin=-1,vmax=1,aspect='auto')
        ax.grid(False)
        ax.set_ylabel("Transcriptional programs",FontSize=14)
        ax.set_xlabel("Samples",FontSize=14)
        plt.savefig(saveFile)
        
    return statesDf

def programsVsStates(statesDf,states,filename=None):
    pixel = np.zeros((statesDf.shape[0],len(states)))
    for i in range(statesDf.shape[0]):
        for j in range(len(states)):
            pixel[i,j] = np.mean(statesDf.loc[statesDf.index[i],states[j]])

    pixel = pd.DataFrame(pixel)
    pixel.index = statesDf.index

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pixel,cmap="bwr",vmin=-1,vmax=1,aspect="auto")
    ax.grid(False)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Transcriptional programs",FontSize=14)
    plt.xlabel("Transcriptional states",FontSize=14)
    if filename is not None:
        plt.savefig(filename)

    return pixel

# =============================================================================
# Functions used for cluster analysis
# =============================================================================

def parallelEnrichment(task):
    partition = task[0]
    test_keys, dict_, reference_dict, reciprocal_dict, population_len, threshold = task[1]
    test_keys = test_keys[partition[0]:partition[1]]
           
    results_dict = {}
    for ix in test_keys:
        basline_ps = {k:1 for k in reference_dict.keys()}
        genes_interrogated = dict_[ix]
        genes_overlapping = list(set(genes_interrogated)&set(reciprocal_dict))
        count_overlapping = Counter(np.hstack([reciprocal_dict[i] for i in genes_overlapping]))
        rank_overlapping = count_overlapping.most_common()

        for h in range(len(rank_overlapping)):
            ct = rank_overlapping[h][1]
            if ct==1:
                break
            key = rank_overlapping[h][0]
            basline_ps[key] = hyper(population_len,len(reference_dict[key]),len(genes_interrogated),rank_overlapping[h][1]) 

        above_basline_ps = {key:basline_ps[key] for key in basline_ps.keys() if basline_ps[key]<threshold}
        results_dict[ix] = above_basline_ps

    return results_dict

def enrichmentAnalysis(dict_,reference_dict,reciprocal_dict,genes_with_expression,resultsDirectory,numCores=5,min_overlap = 3,threshold = 0.05):
    t1 = time.time()
    print('initializing enrichment analysis')
    
    os.chdir(os.path.join(resultsDirectory,"..","data","network_dictionaries"))
    reference_dict = read_pkl(reference_dict)
    reciprocal_dict = read_pkl(reciprocal_dict)
    os.chdir(os.path.join(resultsDirectory,"..","src"))

    genes_in_reference_dict = reciprocal_dict.keys()
    population_len = len(set(genes_with_expression)&set(genes_in_reference_dict))

    reciprocal_keys = set(reciprocal_dict.keys())
    test_keys = []
    for key in dict_.keys():
        genes_interrogated = dict_[key]
        genes_overlapping = list(set(genes_interrogated)&reciprocal_keys)
        if len(genes_overlapping) < min_overlap:
            continue
        count_overlapping = Counter(np.hstack([reciprocal_dict[i] for i in genes_overlapping]))
        rank_overlapping = count_overlapping.most_common()
        if rank_overlapping[0][1] < min_overlap:
            continue

        test_keys.append(key)

    try:
        taskSplit = splitForMultiprocessing(test_keys,numCores)
        taskData = (test_keys, dict_, reference_dict, reciprocal_dict, population_len,threshold)
        tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
        enrichmentOutput = multiprocess(parallelEnrichment,tasks)
        combinedResults = condenseOutput(enrichmentOutput)
    except:
        combinedResults = {}
    
    t2 = time.time()
    print('completed enrichment analysis in {:.2f} seconds'.format(t2-t1))

    return combinedResults

def convertGO(goBio_enriched,resultsDirectory):
    goConversionPath = os.path.join(resultsDirectory,"..","data","network_dictionaries","GO_terms_conversion.csv")
    goBioConversion = pd.read_csv(goConversionPath,index_col=0,header=0)
    go_terms_enriched = {}
    for module in goBio_enriched.keys():
        conv = {}
        for key in goBio_enriched[module].keys():
            tmp = goBioConversion.loc[key,"GO_function"]
            conv[tmp] = goBio_enriched[module][key]
        go_terms_enriched[module] = conv
    return go_terms_enriched

def tsne(matrix,perplexity=100,n_components=2,n_iter=1000,plotOnly=True,plotColor="red",alpha=0.4,dataOnly=False):
    from sklearn.manifold import TSNE
    X = np.array(matrix.T)
    X_embedded = TSNE(n_components=n_components, n_iter=n_iter, n_iter_without_progress=300,init='random',
                             random_state=0, perplexity=perplexity).fit_transform(X)
    if plotOnly is True:
        plt.scatter(X_embedded[:,0],X_embedded[:,1],color=plotColor,alpha=alpha)
        return
    if dataOnly is True:
        return X_embedded
    
    plt.scatter(X_embedded[:,0],X_embedded[:,1],color=plotColor,alpha=alpha)
        
    return X_embedded

def plotStates(statesDf,tsneDf,numCols=3,numRows=None,saveFile=None,size=10,aspect=1,scale=2):

    if numRows is None:
        numRows = int(np.ceil(float(statesDf.shape[0])/numCols))
        
    fig = plt.figure(figsize=(scale*numRows,scale*numCols))
    for ix in range(statesDf.shape[0]):
        ax = fig.add_subplot(numRows,numCols,ix+1)
        # overlay single state onto tSNE plot
        stateIndex = ix

        group = pd.DataFrame(np.zeros(statesDf.shape[1]))
        group.index = statesDf.columns
        group.columns = ["status"]
        group.loc[statesDf.columns,"status"] = list(statesDf.iloc[stateIndex,:])
        group = np.array(group.iloc[:,0])
        ax.set_aspect(aspect)
        ax.scatter(tsneDf.iloc[:,0],tsneDf.iloc[:,1],cmap="bwr",c=group,vmin=-1,vmax=1,s=size)

    if saveFile is not None:
        plt.savefig(saveFile,bbox_inches="tight")

    return

# =============================================================================
# Functions used for survival analysis
# =============================================================================

def kmAnalysis(survivalDf,durationCol,statusCol,saveFile=None):
    
    from lifelines import KaplanMeierFitter
    from scipy import stats
    

    kmf = KaplanMeierFitter()
    kmf.fit(survivalDf.loc[:,durationCol],survivalDf.loc[:,statusCol])
    survFunc = kmf.survival_function_
    
    m, b, r, p, e = stats.linregress(list(survFunc.index),survFunc.iloc[:,0])
    
    survivalDf = survivalDf.sort_values(by=durationCol)
    ttpfs = np.array(survivalDf.loc[:,durationCol])
    survTime = np.array(survFunc.index)
    survProb = []
    
    for i in range(len(ttpfs)):
        date = ttpfs[i]
        if date in survTime:
            survProb.append(survFunc.loc[date,"KM_estimate"])
        elif date not in survTime:
            lbix = np.where(np.array(survFunc.index)<date)[0][-1]
            est = 0.5*(survFunc.iloc[lbix,0]+survFunc.iloc[lbix+1,0])
            survProb.append(est)
            
    kmEstimate = pd.DataFrame(survProb)
    kmEstimate.columns = ["kmEstimate"]
    kmEstimate.index = survivalDf.index
    
    pfsDf = pd.concat([survivalDf,kmEstimate],axis=1)
    
    if saveFile is not None:
        pfsDf.to_csv(saveFile)
        
    return pfsDf

def guanRank(kmSurvival,saveFile=None):

    gScore = []
    for A in range(kmSurvival.shape[0]):
        aScore = 0
        aPfs = kmSurvival.iloc[A,0]
        aStatus = kmSurvival.iloc[A,1]
        aProbPFS = kmSurvival.iloc[A,2]
        if aStatus == 1:
            for B in range(kmSurvival.shape[0]):
                if B == A:
                    continue
                bPfs = kmSurvival.iloc[B,0]
                bStatus = kmSurvival.iloc[B,1]
                bProbPFS = kmSurvival.iloc[B,2]            
                if bPfs > aPfs:
                    aScore+=1
                if bPfs <= aPfs:
                    if bStatus == 0:
                        aScore+=aProbPFS/bProbPFS
                if bPfs == aPfs:
                    if bStatus == 1:
                        aScore+=0.5
        elif aStatus == 0:
            for B in range(kmSurvival.shape[0]):
                if B == A:
                    continue
                bPfs = kmSurvival.iloc[B,0]
                bStatus = kmSurvival.iloc[B,1]
                bProbPFS = kmSurvival.iloc[B,2]            
                if bPfs >= aPfs:
                    if bStatus == 0:
                        tmp = 1-0.5*bProbPFS/aProbPFS
                        aScore+=tmp
                    elif bStatus == 1:
                        tmp = 1-bProbPFS/aProbPFS
                        aScore+=tmp                    
                if bPfs < aPfs:
                    if bStatus == 0:
                        aScore+=0.5*aProbPFS/bProbPFS
        gScore.append(aScore)
                        
    GuanScore = pd.DataFrame(gScore)  
    GuanScore = GuanScore/float(max(gScore))
    GuanScore.index = kmSurvival.index
    GuanScore.columns = ["GuanScore"] 
    survivalData = pd.concat([kmSurvival,GuanScore],axis=1)
    survivalData.sort_values(by="GuanScore",ascending=False,inplace=True)
    
    if saveFile is not None:
        survivalData.to_csv(saveFile)
        
    return survivalData

def survivalMembershipAnalysis(task):

    from lifelines import CoxPHFitter

    start, stop = task[0]
    membershipDf,SurvivalDf = task[1]
    
    overlapPatients = list(set(membershipDf.columns)&set(SurvivalDf.index))
    if len(overlapPatients) == 0:
        print("samples are not represented in the survival data")
        return 
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]
       
    coxResults = {}
    keys = membershipDf.index[start:stop]
    ct=0
    for key in keys:
        ct+=1
        if ct%10==0:
            print(ct)
        try:
            memberVector = pd.DataFrame(membershipDf.loc[key,overlapPatients])
            Survival2 = pd.concat([Survival,memberVector],axis=1)
            Survival2.sort_values(by=Survival2.columns[0],inplace=True)
            
            cph = CoxPHFitter()
            cph.fit(Survival2, duration_col=Survival2.columns[0], event_col=Survival2.columns[1])
            
            tmpcph = cph.summary
            
            cox_hr = tmpcph.loc[key,"z"]
            cox_p = tmpcph.loc[key,"p"]  
            coxResults[key] = (cox_hr, cox_p)
        except:
            coxResults[key] = (0, 1)
    return coxResults

def survivalMembershipAnalysisDirect(membershipDf,SurvivalDf,key):

    from lifelines import CoxPHFitter
    
    overlapPatients = list(set(membershipDf.columns)&set(SurvivalDf.index))
    if len(overlapPatients) == 0:
        print("samples are not represented in the survival data")
        return 
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]
       
    coxResults = {}

    memberVector = pd.DataFrame(membershipDf.loc[key,overlapPatients])
    Survival2 = pd.concat([Survival,memberVector],axis=1)
    Survival2.sort_values(by=Survival2.columns[0],inplace=True)
    
    cph = CoxPHFitter()
    cph.fit(Survival2, duration_col=Survival2.columns[0], event_col=Survival2.columns[1])
    
    tmpcph = cph.summary
    
    cox_hr = tmpcph.loc[key,"z"]
    cox_p = tmpcph.loc[key,"p"]  
    coxResults[key] = (cox_hr, cox_p)
        
    return coxResults

def parallelMemberSurvivalAnalysis(membershipDf,numCores=5,survivalPath=None,survivalData=None):

    if survivalData is None:
        survivalData = pd.read_csv(survivalPath,index_col=0,header=0)
    taskSplit = splitForMultiprocessing(membershipDf.index,numCores)
    taskData = (membershipDf,survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = multiprocess(survivalMembershipAnalysis,tasks)
    survivalAnalysis = condenseOutput(coxOutput)

    return survivalAnalysis

def kmplot(srv,groups,labels,xlim_=None,filename=None):
    plt.figure()
    for group in groups:
        patients = list(set(srv.index)&set(group))
        kmDf = kmAnalysis(survivalDf=srv.loc[patients,["duration","observed"]],durationCol="duration",statusCol="observed")
        subset = kmDf[kmDf.loc[:,"observed"]==1]
        duration = np.concatenate([np.array([0]),np.array(subset.loc[:,"duration"])])
        kme = np.concatenate([np.array([1]),np.array(subset.loc[:,"kmEstimate"])])
        plt.step(duration,kme)

    axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    axes.set_ylim([0,1.09])
    if xlim_ is not None:
        axes.set_xlim([xlim_[0],xlim_[1]])
    axes.set_title("Progression-free survival",FontSize=16)
    axes.set_ylabel("Surviving proportion",FontSize=14)
    axes.set_xlabel("Timeline",FontSize=14)
    axes.legend(labels = labels)
    if filename is not None:
        plt.savefig(filename,bbox_inches="tight")
    return
 
# =============================================================================
# Functions used for causal inference
# =============================================================================

def biclusterTfIncidence(mechanisticOutput,regulons=None):

    import pandas as pd
    import numpy as np
    
    if regulons is not None:
    
        allTfs = regulons.keys()
        
        tfCount = []
        ct=0
        for tf in regulons.keys():
            tfCount.append([])
            for key in regulons[tf].keys():
                tfCount[-1].append(str(ct))
                ct+=1
        
        allBcs = np.hstack(tfCount)
        
        bcTfIncidence = pd.DataFrame(np.zeros((len(allBcs),len(allTfs))))
        bcTfIncidence.index = allBcs
        bcTfIncidence.columns = allTfs
        
        for i in range(len(allTfs)):
            tf = allTfs[i]
            bcs = tfCount[i]
            bcTfIncidence.loc[bcs,tf] = 1
            
        index = np.sort(np.array(bcTfIncidence.index).astype(int))
        if type(bcTfIncidence.index[0]) is str:
            bcTfIncidence = bcTfIncidence.loc[index.astype(str),:]
        elif type(bcTfIncidence.index[0]) is unicode:
            bcTfIncidence = bcTfIncidence.loc[index.astype(unicode),:]
        else:
            bcTfIncidence = bcTfIncidence.loc[index,:]  
        
        return bcTfIncidence

    allBcs = mechanisticOutput.keys()
    allTfs = list(set(np.hstack([mechanisticOutput[i].keys() for i in mechanisticOutput.keys()])))
    
    bcTfIncidence = pd.DataFrame(np.zeros((len(allBcs),len(allTfs))))
    bcTfIncidence.index = allBcs
    bcTfIncidence.columns = allTfs
    
    for bc in mechanisticOutput.keys():
        bcTfs = mechanisticOutput[bc].keys()
        bcTfIncidence.loc[bc,bcTfs] = 1
        
    index = np.sort(np.array(bcTfIncidence.index).astype(int))
    if type(bcTfIncidence.index[0]) is str:
        bcTfIncidence = bcTfIncidence.loc[index.astype(str),:]
    elif type(bcTfIncidence.index[0]) is unicode:
        bcTfIncidence = bcTfIncidence.loc[index.astype(unicode),:]
    else:
        bcTfIncidence = bcTfIncidence.loc[index,:]
    
    return bcTfIncidence

def tfExpression(expressionData,motifPath=os.path.join("..","data","all_tfs_to_motifs.pkl")):

    allTfsToMotifs = read_pkl(motifPath)
    tfs = list(set(allTfsToMotifs.keys())&set(expressionData.index))
    tfExp = expressionData.loc[tfs,:]
    return tfExp

def filterMutations(mutationPath,mutationFile,minNumMutations=None):

    filePath = os.path.join(mutationPath,mutationFile)
    mutations = pd.read_csv(filePath,index_col=0,header=0)
    if minNumMutations is None:
        minNumMutations = min(np.ceil(mutations.shape[1]*0.01),4)
    freqMuts = list(mutations.index[np.where(np.sum(mutations,axis=1)>=minNumMutations)[0]])
    filteredMutations = mutations.loc[freqMuts,:]
    
    return filteredMutations

def mutationMatrix(mutationPath,mutationFiles,minNumMutations=None):

    if type(mutationFiles) is str:
        mutationFiles = [mutationFiles]
    matrices = []
    for mutationFile in mutationFiles:
        matrix = filterMutations(mutationPath=mutationPath,mutationFile=mutationFile,minNumMutations=minNumMutations)
        matrices.append(matrix)
    filteredMutations = pd.concat(matrices,axis=0)
    
    return filteredMutations

def getMutations(mutationString,mutationMatrix):   
    return mutationMatrix.columns[np.where(mutationMatrix.loc[mutationString,:]>0)[0]]

def mutationRegulatorStratification(mutationDf,tfDf,threshold=0.05,dictionary_=False):
    
    incidence = pd.DataFrame(np.zeros((tfDf.shape[0],mutationDf.shape[0])))
    incidence.index = tfDf.index
    incidence.columns = mutationDf.index
    
    stratification = {}
    tfCols = set(tfDf.columns)
    mutCols = set(mutationDf.columns)
    for mutation in mutationDf.index:
        mut = getMutations(mutation,mutationDf)
        wt = list(mutCols-set(mut))
        mut = list(set(mut)&tfCols)
        wt = list(set(wt)&tfCols)
        tmpMut = tfDf.loc[:,mut]
        tmpWt = tfDf.loc[:,wt]
        ttest = stats.ttest_ind(tmpMut,tmpWt,axis=1,equal_var=False)
        significant = np.where(ttest[1]<=threshold)[0]
        hits = list(tfDf.index[significant])
        if len(hits) > 0:
            incidence.loc[hits,mutation] = 1
            if dictionary_ is not False:
                stratification[mutation] = {}
                for i in range(len(hits)):
                    stratification[mutation][hits[i]] = [ttest[0][significant[i]],ttest[1][significant[i]]]
    
    if dictionary_ is not False:
        return incidence, stratification
    return incidence

def generateCausalInputs(expressionData,mechanisticOutput,coexpressionModules,saveFolder,mutationFile="filteredMutationsIA12.csv",regulon_dict=None):
    import os
    import numpy as np
    
    if not os.path.isdir(saveFolder):
        os.mkdir(saveFolder)
    
    # set working directory to results folder
    os.chdir(saveFolder)
    # identify the data folder
    os.chdir(os.path.join("..","data"))
    dataFolder = os.getcwd()   
    # write csv files for input into causal inference module
    os.chdir(os.path.join("..","src"))
    
    #bcTfIncidence      
    bcTfIncidence = biclusterTfIncidence(mechanisticOutput,regulons=regulon_dict)
    bcTfIncidence.to_csv(os.path.join(saveFolder,"bcTfIncidence.csv"))
    
    #eigengenes
    eigengenes = principalDf(coexpressionModules,expressionData,subkey=None,regulons=regulon_dict,minNumberGenes=1)
    eigengenes = eigengenes.T
    index = np.sort(np.array(eigengenes.index).astype(int))
    eigengenes = eigengenes.loc[index.astype(str),:]
    eigengenes.to_csv(os.path.join(saveFolder,"eigengenes.csv"))
    
    #tfExpression
    tfExp = tfExpression(expressionData)
    tfExp.to_csv(os.path.join(saveFolder,"tfExpression.csv"))
    
    #filteredMutations:
    filteredMutations = filterMutations(dataFolder,mutationFile)
    filteredMutations.to_csv(os.path.join(saveFolder,"filteredMutations.csv"))    
    
    #regStratAll
    tfStratMutations = mutationRegulatorStratification(filteredMutations,tfDf=tfExp,threshold=0.01)                
    keepers = list(set(np.arange(tfStratMutations.shape[1]))-set(np.where(np.sum(tfStratMutations,axis=0)==0)[0]))
    tfStratMutations = tfStratMutations.iloc[:,keepers]    
    tfStratMutations.to_csv(os.path.join(saveFolder,"regStratAll.csv"))
    
    return

def processCausalResults(causalPath=os.path.join("..","results","causal"),causalDictionary=False):

    causalFiles = []           
    for root, dirs, files in os.walk(causalPath, topdown=True):
       for name in files:
          if name.split(".")[-1] == 'DS_Store':
              continue
          causalFiles.append(os.path.join(root, name))
          
    if causalDictionary is False:
        causalDictionary = {}
    for csv in causalFiles:
        tmpcsv = pd.read_csv(csv,index_col=False,header=None)
        for i in range(1,tmpcsv.shape[0]):
            score = float(tmpcsv.iloc[i,-2])
            if score <1:
                break
            bicluster = int(tmpcsv.iloc[i,-3].split(":")[-1].split("_")[-1])
            if bicluster not in causalDictionary.keys():
                causalDictionary[bicluster] = {}
            regulator = tmpcsv.iloc[i,-5].split(":")[-1]
            if regulator not in causalDictionary[bicluster].keys():
                causalDictionary[bicluster][regulator] = []
            mutation = tmpcsv.iloc[i,1].split(":")[-1]
            if mutation not in causalDictionary[bicluster][regulator]:
                causalDictionary[bicluster][regulator].append(mutation)
    
    return causalDictionary

def analyzeCausalResults(task):

    start, stop = task[0]
    preProcessedCausalResults,mechanisticOutput,filteredMutations,tfExp,eigengenes = task[1]
    postProcessed = {}
    if mechanisticOutput is not None:
        mechOutKeyType = type(mechanisticOutput.keys()[0])
    allPatients = set(filteredMutations.columns)
    keys = preProcessedCausalResults.keys()[start:stop]
    ct=-1
    for bc in keys:
        ct+=1
        if ct%10 == 0:
            print(ct)
        postProcessed[bc] = {}
        for tf in preProcessedCausalResults[bc].keys():
            for mutation in preProcessedCausalResults[bc][tf]:
                mut = getMutations(mutation,filteredMutations)
                wt = list(allPatients-set(mut))
                mutTfs = tfExp.loc[tf,mut][tfExp.loc[tf,mut]>-4.01]
                if len(mutTfs) <=1:
                    mutRegT = 0
                    mutRegP = 1
                elif len(mutTfs) >1:
                    wtTfs = tfExp.loc[tf,wt][tfExp.loc[tf,wt]>-4.01]
                    mutRegT, mutRegP = stats.ttest_ind(list(mutTfs),list(wtTfs),equal_var=False)
                mutBc = eigengenes.loc[bc,mut][eigengenes.loc[bc,mut]>-4.01]
                if len(mutBc) <=1:
                    mutBcT = 0
                    mutBcP = 1
                    mutCorrR = 0
                    mutCorrP = 1
                elif len(mutBc) >1:
                    wtBc = eigengenes.loc[bc,wt][eigengenes.loc[bc,wt]>-4.01]
                    mutBcT, mutBcP = stats.ttest_ind(list(mutBc),list(wtBc),equal_var=False)
                    if len(mutTfs) <=2:
                        mutCorrR = 0
                        mutCorrP = 1
                    elif len(mutTfs) >2:
                        nonzeroPatients = list(set(np.array(mut)[tfExp.loc[tf,mut]>-4.01])&set(np.array(mut)[eigengenes.loc[bc,mut]>-4.01]))
                        mutCorrR, mutCorrP = stats.pearsonr(list(tfExp.loc[tf,nonzeroPatients]),list(eigengenes.loc[bc,nonzeroPatients]))
                signMutTf = 1
                if mutRegT < 0:
                    signMutTf = -1
                elif mutRegT == 0:
                    signMutTf = 0
                signTfBc = 1
                if mutCorrR < 0:
                    signTfBc = -1
                elif mutCorrR == 0:
                    signTfBc = 0
                if mechanisticOutput is not None:
                    if mechOutKeyType is int:              
                        phyper = mechanisticOutput[bc][tf][0]
                    elif mechOutKeyType is not int:
                        phyper = mechanisticOutput[str(bc)][tf][0]
                elif mechanisticOutput is None:
                    phyper = 1e-10
                pMutRegBc = 10**-((-np.log10(mutRegP)-np.log10(mutBcP)-np.log10(mutCorrP)-np.log10(phyper))/4.)
                pWeightedTfBc = 10**-((-np.log10(mutCorrP)-np.log10(phyper))/2.)
                mutFrequency = len(mut)/float(filteredMutations.shape[1])
                postProcessed[bc][tf] = {}
                postProcessed[bc][tf]["regBcWeightedPValue"] = pWeightedTfBc
                postProcessed[bc][tf]["edgeRegBc"] = signTfBc
                postProcessed[bc][tf]["regBcHyperPValue"] = phyper
                if "mutations" not in postProcessed[bc][tf].keys():
                    postProcessed[bc][tf]["mutations"] = {}
                postProcessed[bc][tf]["mutations"][mutation] = {}
                postProcessed[bc][tf]["mutations"][mutation]["mutationFrequency"] = mutFrequency
                postProcessed[bc][tf]["mutations"][mutation]["mutRegBcWeightedPValue"] = pMutRegBc
                postProcessed[bc][tf]["mutations"][mutation]["edgeMutReg"] = signMutTf
                postProcessed[bc][tf]["mutations"][mutation]["mutRegPValue"] = mutRegP
                postProcessed[bc][tf]["mutations"][mutation]["mutBcPValue"] = mutBcP
                postProcessed[bc][tf]["mutations"][mutation]["regBcCorrPValue"] = mutCorrP
                postProcessed[bc][tf]["mutations"][mutation]["regBcCorrR"] = mutCorrR
         
    return postProcessed

def postProcessCausalResults(preProcessedCausalResults,filteredMutations,tfExp,eigengenes,mechanisticOutput=None,numCores=5):
    
    taskSplit = splitForMultiprocessing(preProcessedCausalResults.keys(),numCores)
    taskData = (preProcessedCausalResults,mechanisticOutput,filteredMutations,tfExp,eigengenes)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    Output = multiprocess(analyzeCausalResults,tasks)
    postProcessedAnalysis = condenseOutput(Output)

    return postProcessedAnalysis

def causalMechanisticNetworkDictionary(postProcessedCausalAnalysis,biclusterRegulatorPvalue=0.05,regulatorMutationPvalue=0.05,mutationFrequency = 0.025,requireCausal=False):
    
    tabulatedResults = []
    ct=-1
    for key in postProcessedCausalAnalysis.keys():
        ct+=1 
        if ct%10==0:
            print(ct)
        lines = []
        regs = postProcessedCausalAnalysis[key].keys()
        for reg in regs:
            bcid = key
            regid = reg
            bcRegEdgeType = int(postProcessedCausalAnalysis[key][reg]['edgeRegBc'])
            bcRegEdgePValue = postProcessedCausalAnalysis[key][reg]['regBcWeightedPValue']
            bcTargetEnrichmentPValue = postProcessedCausalAnalysis[key][reg]['regBcHyperPValue']
            if bcRegEdgePValue <= biclusterRegulatorPvalue:
                if len(postProcessedCausalAnalysis[key][reg]['mutations'])>0:
                    for mut in postProcessedCausalAnalysis[key][reg]['mutations'].keys():                        
                        mutFrequency = postProcessedCausalAnalysis[key][reg]['mutations'][mut]['mutationFrequency']
                        mutRegPValue = postProcessedCausalAnalysis[key][reg]['mutations'][mut]['mutRegPValue']
                        if mutFrequency >= mutationFrequency:
                            if mutRegPValue <= regulatorMutationPvalue:                                
                                mutid = mut
                                mutRegEdgeType = int(postProcessedCausalAnalysis[key][reg]['mutations'][mut]['edgeMutReg'])                             
                            elif mutRegPValue > regulatorMutationPvalue:
                                mutid = np.nan #"NA"
                                mutRegEdgeType = np.nan #"NA"
                                mutRegPValue = np.nan #"NA"
                                mutFrequency = np.nan #"NA"                                  
                        elif mutFrequency < mutationFrequency:
                            mutid = np.nan #"NA"
                            mutRegEdgeType = np.nan #"NA"
                            mutRegPValue = np.nan #"NA"
                            mutFrequency = np.nan #"NA"    
                elif len(postProcessedCausalAnalysis[key][reg]['mutations'])==0:
                    mutid = np.nan #"NA"
                    mutRegEdgeType = np.nan #"NA"
                    mutRegPValue = np.nan #"NA"
                    mutFrequency = np.nan #"NA"
            elif bcRegEdgePValue > biclusterRegulatorPvalue:
                continue 
            line = [bcid,regid,bcRegEdgeType,bcRegEdgePValue,bcTargetEnrichmentPValue,mutid,mutRegEdgeType,mutRegPValue,mutFrequency]
            lines.append(line)
        if len(lines) == 0:
            continue
        stack = np.vstack(lines)
        df = pd.DataFrame(stack)
        df.columns = ["Cluster","Regulator","RegulatorToClusterEdge","RegulatorToClusterPValue","RegulatorBindingSiteEnrichment","Mutation","MutationToRegulatorEdge","MutationToRegulatorPValue","FrequencyOfMutation"]
        tabulatedResults.append(df)
        
    resultsDf = pd.concat(tabulatedResults,axis=0)
    resultsDf = resultsDf[resultsDf["RegulatorToClusterEdge"]!='0']
    resultsDf.index = np.arange(resultsDf.shape[0])

    if requireCausal is True:
        resultsDf = resultsDf[resultsDf["Mutation"]!="nan"]
        
    return resultsDf

def clusterInformation(causalMechanisticNetwork,key):
    return causalMechanisticNetwork[causalMechanisticNetwork["Cluster"]==key]
  
def showCluster(expressionData,coexpressionModules,key):
    plt.figure(figsize=(10,10))
    plt.imshow(expressionData.loc[coexpressionModules[key],:],vmin=-1,vmax=1)
    plt.title("Cluster Expression",FontSize=16)
    plt.xlabel("Patients",FontSize=14)
    plt.ylabel("Genes",FontSize=14)     
    return     
