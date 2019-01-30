#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:43:29 2018

@author: MattWall
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy import random as rd
import os
import json
from sklearn.decomposition import PCA
import multiprocessing, multiprocessing.pool
from collections import Counter
import matplotlib.pyplot as plt
import time
#%%

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

def annotatedGenes(filepath):
    important_genes = pd.read_table(filepath)
    importantEnsembl = list(set(list(important_genes.iloc[:,0])))
    return importantEnsembl
        
    
def loadMiner(location="Desktop",expression=True):
    import pandas as pd
    import sys
    import os
    
    home = os.path.expanduser("~")
    minerPath = home+"/"+location+"/miner/src"
    sys.path.append(minerPath)
    os.chdir(minerPath)
    
    from miner import read_json
    config = read_json("../configuration_file.json")    

    important_genes_path = config["important_genes_path"]
    important_genes = pd.read_table(important_genes_path)
    importantEnsembl = list(set(list(important_genes.iloc[:,0])))
    
    if expression is True:
        expressionPath = config["expressionFile"]
        Zscore = pd.read_csv(expressionPath,index_col=0,header=0)
        return Zscore, importantEnsembl
        
    return importantEnsembl

def hyper(population,set1,set2,overlap):
    
    b = max(set1,set2)
    c = min(set1,set2)
    hyp = stats.hypergeom(population,b,c)
    prb = sum([hyp.pmf(l) for l in range(overlap,c+1)])
    
    return prb 


def bh_correction(dict_,threshold = 0.05):
    from statsmodels.sandbox.stats.multicomp import multipletests
    p_motifs = [key for key in dict_.keys()]
    p_vals = [dict_[key] for key in dict_.keys()]

    if len(p_vals) == 0:
        return {}
    
    bh_correction = multipletests(p_vals, alpha=threshold, method='fdr_tsbh', is_sorted=False, returnsorted=False)

    ps = bh_correction[1][bh_correction[0]]
    motifs_ = np.array(p_motifs)[bh_correction[0]]
    
    pass_dict = {}
    for i in range(0,len(motifs_)):
        pass_dict[motifs_[i]] = ps[i]
    
    return pass_dict

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

def getMutations(mutationString,mutationMatrix):   
    return mutationMatrix.columns[np.where(mutationMatrix.loc[mutationString,:]>0)[0]]

def pcaClustering(task,expressionArray,pcaArray,indexConversion):
    
    pcaClusters = {}
    start = task[0]
    stop = task[1]
    
    for i in range(start,stop):
        if i == round((stop-start)/2.):
            print("halfway done")
        tmpPearson = pearson_array(expressionArray,pcaArray[i,:])
        tmpClusterIndices = np.where(tmpPearson>0.35)[0]
        tmpClusterIndices = tmpClusterIndices[np.argsort(-tmpPearson[tmpClusterIndices])]
        if len(tmpClusterIndices) > 4:
            pcaClusters[i] = list(indexConversion[tmpClusterIndices])
        
    return pcaClusters

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

def principalDecomposition(geneset=None,df=None,override=False,threshold = 1):
    
    if override is not False:
        pca0fm = override
        
    elif override is False:
        pca0fm = FrequencyMatrix(matrix=df.loc[geneset,:],overExpThreshold = threshold)
    
    pretest = pca0fm.copy()
    pretest[pretest<0.5] = 0
    pretest[pretest>0] = 1
    test = pretest*pretest.T
    
    matrix = np.array(test)
    poolClusters = set([])
    unClustered = set(range(matrix.shape[0]))
    sumAxis0 = np.sum(matrix,axis=0)
    clusterList = []
    
    for iteration in range(10):
        remainder = np.array(list(unClustered))
        if len(remainder) == 0:
            break
        probeIx = remainder[np.argmax(sumAxis0[remainder])]
        cluster = np.where(matrix[:,probeIx]>0)[0]
        clusterList.append(cluster)
        poolClusters = poolClusters|set(cluster)
        unClustered = unClustered - poolClusters    
    
    plotSortedClusters = clusterList
    clusterLengths = [len(plotSortedClusters[i]) for i in range(len(plotSortedClusters))]
    plotSortedClusters.sort(key = lambda s: len(s))
    
    for i in range(len(plotSortedClusters)-1):
        for j in plotSortedClusters[i]:
            for k in range(i+1,len(plotSortedClusters)):
                if j in plotSortedClusters[k]:
                    plotSortedClusters[i] = np.delete(plotSortedClusters[i],np.where(plotSortedClusters[i]==j)[0])
                    break
    
    plotSortedClusters.sort(key = lambda s: -len(s))
    
    #flattened = np.hstack(plotSortedClusters)
    #plt.imshow(np.array(pca0fm)[flattened,:][:,flattened],vmin=-0,vmax=1,cmap="viridis")
    
    highPass = [0.4,0.3,0.25]
    refinedClusters = []
    for threshold in highPass:
        preCombine = np.array(pca0fm.copy())
        preCombine[preCombine<threshold] = 0
        preCombine[preCombine>0] = 1
    
        for clustIx in range(len(plotSortedClusters)):
            if len(plotSortedClusters[clustIx])<5:
                break
            coreCluster = preCombine[:,plotSortedClusters[clustIx]]
            tmp1 = np.where(np.sum(coreCluster,axis=1)/float(len(plotSortedClusters[clustIx]))>0.8)[0]
            tmp2 = np.where(np.sum(preCombine[tmp1,:][:,tmp1],axis=0)/float(len(tmp1))>0.8)[0]
            revisedCluster = tmp1[tmp2]
            refinedClusters.append(revisedCluster)
    
    #flattened = np.hstack(refinedClusters)
    #plt.imshow(np.array(pca0fm)[flattened,:][:,flattened],vmin=0.3,vmax=0.31,cmap="viridis")
    
    #final step to merge clusters
    if len(refinedClusters)>0:
        refinedGeneset = np.array(list(set(np.hstack(refinedClusters))))
        clusterIncidence = pd.DataFrame(np.zeros((len(refinedClusters),len(refinedGeneset))))
        clusterIncidence.columns = refinedGeneset
    
        for clusterIx in range(len(refinedClusters)):
            clusterIncidence.loc[clusterIx,refinedClusters[clusterIx]] = 1

    elif len(refinedClusters)==0:
        return []
    #go through rows, mask others, if >= 30% disappears, merge (or delete?)
    preSum = np.sum(clusterIncidence,axis=1)
    dropouts = []
    for row in range(clusterIncidence.shape[0]):
        if row in dropouts:
            continue
        post = clusterIncidence-clusterIncidence.iloc[row,:]
        post[post<1] = 0
        postSum = np.sum(post,axis=1)
        postSum[row] = preSum[row]
        fractionRemaining = postSum/preSum
        dropouts = list(set(dropouts)|set(np.where(fractionRemaining<0.7)[0]))
    keepers = list(set(range(clusterIncidence.shape[0]))-set(dropouts)) 
    
    finalClusters = []
    for ix in keepers:
        cluster = list(clusterIncidence.columns[np.where(clusterIncidence.loc[ix,:]>0)[0]])
        finalClusters.append(list(pca0fm.index[np.array(cluster)]))
    
    return finalClusters   

def principalBiclustering(df,outputDirectory = os.path.join(os.path.expanduser("~"),"Desktop"),genepool=None,maxMinutes=5):
    
    import time
    
    biclusterDict = {}
    if genepool is None:
        genepool = list(df.index)
    
    initialTime = time.time()
    
    accountedFor = set([])
    fixed=True   
    ct = 0
    for iteration in range(500):
        relevantGenes = list(set(df.index)&set(genepool)-accountedFor)
        expressionData = df.loc[relevantGenes,:] #ratios_df.loc[geneset,:] #normal
        indexVector = np.arange(expressionData.shape[1])
        
        startTime = time.time()
        principalBiclusters = []
        
        if fixed is True:
            rd.seed(12) #12 
        expressionDataClean = expressionData[expressionData.iloc[:,1].notnull()] #normal
        bootstrapIndices = rd.choice(indexVector,1000,replace=True) #1000
        bootstrapExpression = expressionDataClean.iloc[:,bootstrapIndices]
        
        #pca = PCA(0.95)
        pca = PCA(10)
        principalComponents = pca.fit_transform(bootstrapExpression)
        principalDf = pd.DataFrame(principalComponents)
        
        bootstrapExpressionArray = np.array(bootstrapExpression)
        pcaComponents = pca.components_
        expressionIndex = np.array(relevantGenes) 
        
        pcaClusters = pcaClustering(task=[0,len(pcaComponents)],expressionArray=bootstrapExpressionArray,pcaArray=pcaComponents,indexConversion=expressionIndex)
        
        for i in range(10):
            try:
                finalClusters = principalDecomposition(geneset = pcaClusters[i],df = df)
                principalBiclusters.append(finalClusters)
            except:
                continue
            
        stopTime = time.time()
        timeElapsed = (stopTime-startTime)/60.
        print("completed clustering in " +str(timeElapsed)+ " minutes!")
        
        #% exploration        
        if len(principalBiclusters) == 0:
            fixed=False
            ct+=1
            if ct>25:
                write_json(biclusterDict,outputDirectory+"principalBiclusters.json")
                return biclusterDict        
            
            seed = rd.choice(np.arange(10000),1)
            rd.seed(seed)
            continue
        
        finalClusters = []
        for i in range(len(principalBiclusters)):
            for j in range(len(principalBiclusters[i])):
                finalClusters.append(principalBiclusters[i][j])
        
        #%
        finalClusters.sort(key = lambda s: len(s))
        
        dropouts = []
        for i in range(len(finalClusters)-1):  
            for j in range(i+1,len(finalClusters)):
                overlap = len(set(finalClusters[i])&set(finalClusters[j]))/float(min(len(finalClusters[i]),len(finalClusters[j])))
                if overlap>0.8:
                    if len(finalClusters[j])>=len(finalClusters[i]):
                        dropouts.append(i)
                    else:
                        dropouts.append(j)
                       
        keep = list(set(np.arange(len(finalClusters)))-set(dropouts))  
        finalKeepers = np.array(finalClusters)[np.array(keep)]
        
        accountedFor = accountedFor|set(np.hstack(finalKeepers))
                
        for i in range(len(finalKeepers)): 
            biclusterDict[len(biclusterDict)] = {}
            biclusterDict[len(biclusterDict)-1]['genes'] = list(finalKeepers[i])


        finalTime = time.time()
        if (finalTime-initialTime)/60. > maxMinutes:
            write_json(biclusterDict,outputDirectory+"principalBiclusters.json")
            return biclusterDict  
           
    write_json(biclusterDict,outputDirectory+"principalBiclusters.json")
    return biclusterDict  

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

def transformBiclusters(expressionData,biclusters=None,override=False):
    
    if override is False:
        PCs = principalDf(biclusters,expressionData)
    elif override is not False:
        PCs = override
    pcArray = np.array(PCs)
    pi = np.linalg.pinv(pcArray)
    
    coefficients = np.matmul(pi,np.array(expressionData.T)).T
    transformed = np.matmul(coefficients,pcArray.T)
    transformed = pd.DataFrame(transformed)
    transformed.index = expressionData.index
    transformed.columns = expressionData.columns

    return transformed, coefficients, PCs

def getMembership(clusterDict,expressionData,minNumberGenes=8,overExpressionThreshold=80):
    
    zero = np.percentile(expressionData,0)
    expressionThreshold = np.mean([np.percentile(expressionData.iloc[:,i][expressionData.iloc[:,i]>zero],overExpressionThreshold) for i in range(expressionData.shape[1])])
    genesetDict = {i:clusterDict[i]['genes'] for i in clusterDict.keys()}
    membersDict = {}
    
    for i in genesetDict.keys():
    
        genes = list(set(genesetDict[i])&set(expressionData.index))
        if len(genes) < minNumberGenes:
            continue
        
        testCluster = expressionData.loc[genes,:]
        testCluster[testCluster<expressionThreshold] = 0
        testCluster[testCluster>0] = 1
        
        clusterSum = np.sum(testCluster,axis=0)/float(testCluster.shape[0])

        mask = np.where(clusterSum>=0.333)[0]
        
        biclusterMembers = list(expressionData.columns[mask])
        
        if len(biclusterMembers) < 0.01*expressionData.shape[1]:
            continue
        
        membersDict[i] = biclusterMembers
        
    return membersDict, list(expressionData.columns)

def getMembershipDf(membersDict,memberList):
    
    index = membersDict.keys()
    memberDf = pd.DataFrame(np.zeros((len(index),len(memberList))))
    memberDf.index = index
    memberDf.columns = memberList
    
    for i in membersDict.keys():
        memberDf.loc[i,membersDict[i]] = 1
        
    return memberDf

def read_json(filename):
    with open(filename) as data:
        dict_ = json.load(data)    
    return dict_

def dictionaryFromDirectory(directory):
    import os
     
    collectedClusters = {}
    # Set the directory you want to start from
    rootDir = directory
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: {:.1000s}'.format(dirName))
        for fname in fileList:
            if fname.split(".")[-1] == "json":
                tmpClusters = read_json(os.path.join(dirName,fname))
                for ix in range(len(tmpClusters)):
                    key = tmpClusters.keys()[ix]
                    collectedClusters[len(collectedClusters)] = tmpClusters[key]
    return collectedClusters

def decompositionDictionary(geneList):
    fpcs = {}
    for i in range(len(geneList)):
        fpcs[i] = {}
        genes = geneList[i]
        if len(genes) < 4:
            continue
        fpcs[i]['genes'] = genes
        pca = PCA(1)
        principalComponents = pca.fit_transform(Zscore.loc[genes,:].T)
        principalComponents = principalComponents.flatten()
        normPC = np.linalg.norm(principalComponents)
        pearson = stats.pearsonr(principalComponents,np.median(Zscore.loc[genes,:],axis=0))
        signCorrection = pearson[0]/np.abs(pearson[0])
        firstPrincipalComponent = signCorrection*principalComponents/normPC
        fpcs[i]['FPC'] = firstPrincipalComponent
    return fpcs

def iterateClusters(task,expDf,componentsDf,clusterDict):

    startIndex,stopIndex = task
    clusterIndexNames = componentsDf.columns
    outputDictionary = {}
    
    expArray = np.array(expDf)
    expColumns = expDf.columns
    expIndex = expDf.index

    for i in range(startIndex,stopIndex):
        try:
            clusterIx = clusterIndexNames[i]
            if (i-startIndex)%5 == 0:
                print(round(100*float(i-startIndex)/(stopIndex-startIndex)))
            if len(clusterDict[int(clusterIx)]['genes']) <4:
                continue
            
            correlations = pearson_array(expArray,list(componentsDf.loc[expColumns,clusterIx]))
            highlyCorrelated = np.where(correlations>=0.4)[0]
            expansion = expIndex[highlyCorrelated]
            expansionDecomposition = principalDecomposition(geneset=expansion,df=expDf,threshold=1)
            remainder = list(set(clusterDict[int(clusterIx)]['genes'])-set(np.hstack(expansionDecomposition)))
            expansionDecomposition.append(remainder)
            processDecomposition = decompositionDictionary(expansionDecomposition)
            outputDictionary[clusterIx] = processDecomposition
        except:
            print(i)
            continue
        
    return outputDictionary

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

def overlapAnalysis(task):
    
    start, stop = task[0]
    crossrefDf = task[1]
    overlapDictionary = {}        
    for ix in range(start,stop):
        key = crossrefDf.columns[ix]
        refSum = np.array(np.sum(crossrefDf,axis=0)).astype(float)
        prod = np.multiply(np.array(crossrefDf.loc[:,key]),np.array(crossrefDf).T)
        numOverlap = np.sum(prod,axis=1)
        combine = np.where(numOverlap/refSum[ix]>=0.9)[0]
        overlapDictionary[key] = list(crossrefDf.columns[combine])
    return overlapDictionary

def getAxes(clusters,expressionData):
    axes = {}
    for key in clusters.keys():
        genes = clusters[key]
        fpc = PCA(1)
        principalComponents = fpc.fit_transform(expressionData.loc[genes,:].T)
        axes[key] = principalComponents.ravel()
    return axes

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

def decompose(geneset,expressionData,minNumberGenes=8):
    
    fm = FrequencyMatrix(expressionData.loc[geneset,:])
    tst = np.multiply(fm,fm.T)
    tst[tst<np.percentile(tst,85)]=0
    tst[tst>0]=1
    unmix_tst = unmix(tst)
    unmixedFiltered = [i for i in unmix_tst if len(i)>=minNumberGenes]
    return unmixedFiltered

def recursiveDecomposition(geneset,expressionData,minNumberGenes=8):
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

def reconstruction(decomposedList,expressionData,threshold=0.925):
    clusters = {i:decomposedList[i] for i in range(len(decomposedList))}
    axes = getAxes(clusters,expressionData)
    recombine = combineClusters(axes,clusters,threshold)
    return recombine

def recursiveAlignment(geneset,expressionData,minNumberGenes=8):
    recDecomp = recursiveDecomposition(geneset,expressionData,minNumberGenes)
    if len(recDecomp) == 0:
        return []
    reconstructed = reconstruction(recDecomp,expressionData)
    reconstructedList = [reconstructed[i] for i in reconstructed.keys() if reconstructed[i]>minNumberGenes]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList

def processCoexpressionLists(lists,expressionData,threshold=0.925):
    reconstructed = reconstruction(lists,expressionData,threshold)
    reconstructedList = [reconstructed[i] for i in reconstructed.keys()]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList
    
def mergeKeys(expansionSets,numCores):

    clusteredGenes = list(set(np.hstack([expansionSets[i] for i in expansionSets.keys()])))
    
    crossrefDf = pd.DataFrame(np.zeros((len(clusteredGenes),len(expansionSets))))
    crossrefDf.index = clusteredGenes
    crossrefDf.columns = expansionSets.keys()
    for key in expansionSets.keys():
        crossrefDf.loc[expansionSets[key],key] = 1

    #multiprocess overlapDictionary step
    taskSplit = splitForMultiprocessing(crossrefDf.columns,numCores)
    taskData = crossrefDf
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    overlapOutput = multiprocess(overlapAnalysis,tasks)
    overlapDictionary = condenseOutput(overlapOutput)
    overlapList = [overlapDictionary[i] for i in overlapDictionary.keys()]
    overlapList.sort(key = lambda s: len(s))
    
    probe = 0
    for iteration in range(int(0.5*len(overlapList)**2)):
        for i in range(probe+1,len(overlapList)):
            tmp = set(overlapList[probe])&set(overlapList[i])
            if len(tmp)>0:
                overlapList[probe] = list(set(overlapList[probe])|set(overlapList[i]))
                del overlapList[i]
                break
            if i == len(overlapList)-1:
                probe+=1
            if probe == len(overlapList)-1:
                return overlapList
            elif i < len(overlapList)-1:
                continue
            break
        
    return overlapList
                   

def mergeGenes(geneDict,mergeList):
    
    mergeDict = {}
    for i in range(len(mergeList)):
        genes = set([])
        for j in mergeList[i]:
            genes = genes|set(geneDict[j])
        mergeDict[len(mergeDict)] = list(genes)
    return mergeDict

def axisGenesets(clusters,components,expressionData):
    axisSets = {}
    for i in range(components.shape[1]):
        key = clusters.keys()[i]
        componentKey = components.columns[i]
        fpc = np.array(components[componentKey])
        pa = pearson_array(np.array(expressionData.loc[clusters[key],:]),fpc)
        paSort = np.argsort(pa)
        if len(paSort) > 100:
            genes = np.array(clusters[key])[paSort[-100:]]
        else:
            genes = np.array(clusters[key])
        axisSets[key] = list(genes)
    return axisSets

def postProcessClusters(clusters,expressionData,numCores=5,minNumberGenes=5):
    import time
    start = time.time()
    keyMerge = mergeKeys(clusters,numCores)
    stop = time.time()
    print('\nCompleted step 1 of 5 in {:.3f} minutes'.format((stop-start)/60.))

    start = time.time()    
    revisedClusters = mergeGenes(clusters,keyMerge)
    stop = time.time()
    print('\nCompleted step 2 of 5 in {:.3f} minutes'.format((stop-start)/60.))

    start = time.time()
    revisedComponents = principalDf(revisedClusters,expressionData,subkey=None,minNumberGenes=minNumberGenes)        
    stop = time.time()
    print('\nCompleted step 3 of 5 in {:.3f} minutes'.format((stop-start)/60.))

    start = time.time()
    axisSets = axisGenesets(revisedClusters,revisedComponents,expressionData)
    stop = time.time()
    print('\nCompleted step 4 of 5 in {:.3f} minutes'.format((stop-start)/60.))

    start = time.time()
    axes = principalDf(axisSets,expressionData,subkey=None,minNumberGenes=1)
    stop = time.time()
    print('\nCompleted step 5 of 5 in {:.3f} minutes'.format((stop-start)/60.))

    return axes, revisedClusters

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

def motifEnrichmentAnalysis(task):
    
    start, stop = task[0]
    motifEnrichment = {}
    mechanisticOutput, primary_tf_to_motif, tfbsdb_ensembl, coexpressionModules, populationLength = task[1]
    keys = coexpressionModules.keys()[start:stop]
    
    ct = 0
    for key in keys:
        ct+=1
        if ct%10 == 0:
            print(ct)            
        for tfkey in mechanisticOutput[key].keys():
            motifs = primary_tf_to_motif[tfkey]
            for motif in motifs:
                tmp = motif.split("$")
                if len(tmp)>1:
                    motif = ("_").join(tmp)
                try:
                    motifGenes = tfbsdb_ensembl[motif]
                except:
                    continue
                overlap = len(set(motifGenes)&set(coexpressionModules[key]))
                phyper = hyper(populationLength,len(motifGenes),len(coexpressionModules[key]),overlap)
                if phyper < 0.05:
                    if key not in motifEnrichment.keys():
                        motifEnrichment[key] = {}
                    if tfkey not in motifEnrichment[key].keys():
                        motifEnrichment[key][tfkey] = {}
                    motifEnrichment[key][tfkey][motif] = phyper
    
    return motifEnrichment

def condenseOutput(output):
    
    results = {}
    for i in range(len(output)):
        resultsDict = output[i]
        keys = resultsDict.keys()
        for j in range(len(resultsDict)):
            key = keys[j]
            results[key] = resultsDict[key]
    return results

def parallelMotifAnalysis(mechanisticOutput,coexpressionModules,expressionData,numCores=5,networkPath=os.path.join("..","data","network_dictionaries")):

    primary_tf_to_motif = read_pkl(os.path.join(networkPath,"primary_tf_to_motif.pkl"))
    tfbsdb_ensembl = read_pkl(os.path.join(networkPath,"tfbsdb_ensembl.pkl"))
    tfbsdb_ensembl_reciprocal = read_pkl(os.path.join(networkPath,"reciprocal_tfbsdb_ensembl.pkl"))
    populationLength = len(set(tfbsdb_ensembl_reciprocal.keys())&set(expressionData.index))
    taskSplit = splitForMultiprocessing(coexpressionModules.keys(),numCores)
    taskData = (mechanisticOutput, primary_tf_to_motif, tfbsdb_ensembl, coexpressionModules, populationLength)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    motifOutput = multiprocess(motifEnrichmentAnalysis,tasks)
    motifAnalysis = condenseOutput(motifOutput)

    return motifAnalysis

def expandClusters(task):
    
    import time    
    ini = time.time()
    
    start, stop = task[0]
    clusters,expressionData = task[1]
    keys = clusters.keys()[start:stop]
    index = expressionData.index
    
    zero = np.percentile(expressionData,0)
    expressionThreshold = np.mean([np.percentile(expressionData.iloc[:,i][expressionData.iloc[:,i]>zero],80) for i in range(expressionData.shape[1])])
    
    binaryZ = np.array(expressionData.copy())
    binaryZ[binaryZ<expressionThreshold] = 0
    binaryZ[binaryZ>0] = 1
    
    expansionSets = {}
    
    for key in keys:
        genes = clusters[key]
        
        clust = np.array(expressionData.loc[genes,:])
        clust[clust<expressionThreshold] = 0
        clust[clust>0] = 1
        scores = np.sum(clust,axis=0)/float(clust.shape[0])
        members08 = np.where(scores>=0.5)[0]
        
        slice08 = binaryZ[:,members08]
        rowScores = np.sum(slice08,axis=1)/float(len(members08))
        keepLib = np.where(rowScores>0.45)[0]
        
        cluster1 = {}
        cluster1['0'] = list(index[keepLib])
        pca1 = principalDf(cluster1,expressionData=expressionData,subkey=None)
        pa1 = pearson_array(np.array(expressionData.loc[cluster1['0'],:]),np.array(pca1['0']))
        corrFilter = np.where(pa1>=0.3)[0]
        
        expandedGenelist = list(index[keepLib[corrFilter]])
        expansionSets[key] = expandedGenelist    
    
    fini = time.time()
    print('\nminer expansion completed in {:.5f} seconds'.format(fini-ini))
    
    return expansionSets

def parallelExpansion(dataframe,clusters,numCores,resultsDirectory,components=None,deleteTmpFiles="../tmp"):  

    if not os.path.isdir(resultsDirectory):
        os.mkdir(resultsDirectory)        
        
    try:
        split = splitForMultiprocessing(clusters.keys(),numCores)
        tasks = [[split[i],(clusters,dataframe)] for i in range(numCores)]
        expandedOutput = multiprocess(expandClusters,tasks)
        expandedGenesets = condenseOutput(expandedOutput)
    except:
        singleTask = [(0,len(clusters)),(clusters,dataframe)]
        expandedGenesets = expandClusters(singleTask)
            
    write_json(expandedGenesets,os.path.join(resultsDirectory,"expandedGenesets.json"))            
    
    if deleteTmpFiles is not False:
        if deleteTmpFiles is not None:
            import shutil
            shutil.rmtree(deleteTmpFiles)
        
    if components is not None:                   
        expandedFPCdf = principalDf(expandedGenesets,expressionData=dataframe,subkey=None,minNumberGenes=1)
        expandedFPCdf.to_csv(os.path.join(resultsDirectory,"expandedFPCs.csv"))   
        return expandedGenesets, expandedFPCdf

    return expandedGenesets

def filterNetwork(clusters,minNumberGenes=8):
    filteredClusters = {}
    for key in clusters:
        if len(clusters[key])>minNumberGenes:
            filteredClusters[key] = clusters[key]
    return filteredClusters

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

def coexpressionMedianMatrix(expressionData,coexpressionModules):
    keys = coexpressionModules.keys()
    coexpressionMedians = [np.median(expressionData.loc[list(set(coexpressionModules[key])&set(expressionData.index)),:],axis=0) for key in coexpressionModules.keys()]
    coexpressionMatrix = pd.DataFrame(np.vstack(coexpressionMedians))
    coexpressionMatrix.columns = expressionData.columns
    coexpressionMatrix.index = keys
    reorderKeys = np.argsort(np.array(keys).astype(int))
    return coexpressionMatrix.iloc[reorderKeys,:]

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

def f1Decomposition(sampleMembers,thresholdSFM=0.333,sampleFrequencyMatrix=None):
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

def getSimilarityClusters(sampleDictionary,fequencyThreshold=0.333,similarityThreshold=0.15,highResolution=False):   
    sampleFrequencyMatrix = sampleCoincidenceMatrix(sampleDictionary,freqThreshold=fequencyThreshold,frequencies=True)
    similarityClusters = f1Decomposition(sampleDictionary,thresholdSFM=similarityThreshold)

    if highResolution is True:
        highResolutionClusters = []
        lowConfidenceClusters = []
        for subset in similarityClusters:
            if len(subset) <= 5:
                lowConfidenceClusters.append(subset)
                continue
            similarityMatrixSubset = sampleFrequencyMatrix.loc[subset,subset]
            subsetSimilarityClusters = f1Decomposition(sampleDictionary,thresholdSFM=0.333,sampleFrequencyMatrix=similarityMatrixSubset)
            highResolutionClusters.append(subsetSimilarityClusters)

        return highResolutionClusters, lowConfidenceClusters
    return similarityClusters

def classesFromClusters(highResolutionClusters):
    classes = []
    for i in highResolutionClusters:
        for j in i:
            if len(j)>5:
                classes.append(j)
    return classes

def convertDictionary(dictionary,expressionData):
    convertedDictionary = {key:list(set(dictionary[key])&set(expressionData.index)) for key in dictionary.keys()}
    return convertedDictionary
  
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

def confusionMatrix(predictor,SurvivalDf,hrFlag="HR_FLAG"):
    predictor = pd.DataFrame(predictor)
    risk = pd.DataFrame(SurvivalDf.loc[:,hrFlag])
    try:
        risk[risk=="CENSORED"] = 0
        risk[risk=="FALSE"] = 0
        risk[risk=="TRUE"] = 1
    except:
        pass
    overlapPatients = list(set(predictor.index)&set(risk.index))
    predictionVsResult = pd.concat([predictor.loc[overlapPatients,:],risk.loc[overlapPatients,:]],axis=1)
    predictionVsResult

    vector1 = np.array(predictionVsResult.iloc[:,1]).astype(int)
    vector2 = np.array(predictionVsResult.iloc[:,0]).astype(int)
    members = set(np.where(vector1==1)[0])
    nonMembers = set(np.where(vector1==0)[0])
    predictedMembers = set(np.where(vector2==1)[0])
    predictedNonMembers = set(np.where(vector2==0)[0])

    crosstab = np.array([[len(members&predictedMembers),len(predictedNonMembers&members)],[len(predictedMembers&nonMembers),len(predictedNonMembers&nonMembers)]])
    crosstab = pd.DataFrame(crosstab)
    crosstab.index = ["Actual Members","Actual Non-members"]
    crosstab.columns = ["Predicted Members","Predicted Non-members"]
    return crosstab

def f1Matrix(vector1,matrix):
    
    members = np.where(vector1==1)[0]
    nonMembers = np.where(vector1==0)[0]
    
    matrix = np.array(matrix)
    class1 = matrix[:,members]
    class0 = matrix[:,nonMembers]
    
    TP = np.sum(class1,axis=1)
    FN = class1.shape[1]-TP
    FP = np.sum(class0,axis=1)
    F1 = TP.astype(float)/(TP+FN+FP)
    
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

def mapExpressionToNetwork(centroidMatrix,membershipMatrix,threshold = 0.2):
    
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

def expandRegulonDictionary(regulons):
    regulonDictionary = {}
    for tf in regulons.keys():
        for key in regulons[tf].keys():
            regulonKey = ("_").join([tf,key])
            regulonDictionary[regulonKey] = regulons[tf][key]
    return regulonDictionary

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
    ax.imshow(ordered_matrix,cmap='viridis')
    try:
        asp = ordered_matrix.shape[1]/float(ordered_matrix.shape[0])
        ax.set_aspect(asp)
    except:
        pass
    ax.grid(False)
        
    plt.title(ylabel.split("s")[0]+"Activation",FontSize=16)
    plt.xlabel("Samples",FontSize=14)
    plt.ylabel(ylabel,FontSize=14)
    if resultsDirectory is not None:
        plt.savefig(os.path.join(resultsDirectory,"patientClassesByRevisedClusters.pdf"))
    return ordered_matrix

def removeIncoherentClusters(dataframe,collectedClusters,collectedComponents):
    
    incoherent = []    
    for key in collectedClusters.keys():
        try:
            tmpArray = np.array(dataframe.loc[collectedClusters[key]['genes'],:])
        except:
               tmpArray = np.array(dataframe.loc[collectedClusters[key],:])
        try: 
            correlations = pearson_array(tmpArray,np.array(collectedComponents.loc[:,str(key)]))
            if float(len(correlations[correlations>0.3]))/len(correlations) <= 0.5:
                incoherent.append(key)
        except:
            incoherent.append(key)
            
    for delKey in incoherent:
        del collectedClusters[delKey]
    
    try:
        clusters = {key:collectedClusters[key]['genes'] for key in collectedClusters.keys()}
    except:
        clusters = {key:collectedClusters[key] for key in collectedClusters.keys()}
    
    return clusters

def processInitialClusters(clusterList,expressionData):
    initialClusters = {i:clusterList[i] for i in range(len(clusterList))}
    icpdf = principalDf(initialClusters,expressionData,subkey=None)
    initialClusters = {i:clusterList[i] for i in np.array(icpdf.columns).astype(int)}
    
    filteredInitialClusters = removeIncoherentClusters(expressionData,initialClusters,icpdf)
    if len(filteredInitialClusters) < icpdf.shape[1]:
        icpdf = principalDf(filteredInitialClusters,expressionData,subkey=None)
        filteredInitialClusters = {i:clusterList[i] for i in np.array(icpdf.columns).astype(int)}
    return filteredInitialClusters
    
def mergeClusters(clusters,components,correlationThreshold=0.925):
    import numpy as np
    
    processedClusters = {}
    componentArray = np.array(components.T)
    fusedComponents = []
    integer = True
    if type(clusters.keys()[0]) is not int:
        integer = False
    
    for i in range(components.shape[1]):
        if i == 100:
            print(i)
        if i in fusedComponents:
            continue
        paTmp = pearson_array(componentArray,np.array(components[str(i)]))
        overlap = np.where(paTmp>correlationThreshold)[0]
        fusedComponents = list(set(fusedComponents)|set(overlap))
        if integer is False:
                genes = list(set(np.hstack([clusters[str(j)] for j in overlap])))
        elif integer is True:
            genes = list(set(np.hstack([clusters[j] for j in overlap])))
        processedClusters[i] = genes
    return processedClusters

def biclusterMembership(coexpressionModules,expressionData):
    import pandas as pd
    samples = {}
    ct = 0
    for key in coexpressionModules.keys():
        ct+=1
        if ct%50 == 0:
            print(ct)
        genes = coexpressionModules[key]
        dta = plotBicluster(genes,expressionData,pValue=0.15,dataOnly=True)
        members = expressionData.columns[dta[0]]
        samples[key] = list(members)
    return samples

def mechanisticInference(axes,revisedClusters,expressionData,correlationThreshold=0.3,dataFolder=os.path.join(os.path.expanduser("~"),"Desktop","miner","data")):
    import os
    tfToGenesPath = os.path.join(dataFolder,"network_dictionaries","tfbsdb_tf_to_genes.pkl")
    genesToTfPath = os.path.join(dataFolder,"network_dictionaries","tfbsdb_genes_to_tf.pkl")
    tfToGenes = read_pkl(tfToGenesPath)
    genesToTf = read_pkl(genesToTfPath)
     
    tfs = tfToGenes.keys()     
    tfMap = axisTfs(axes,tfs,expressionData,correlationThreshold=correlationThreshold) 
    taskSplit = splitForMultiprocessing(revisedClusters.keys(),5)
    tasks = [[taskSplit[i],(expressionData,revisedClusters,tfMap,tfToGenes)] for i in range(len(taskSplit))]
    tfbsdbOutput = multiprocess(tfbsdbEnrichment,tasks)
    mechanisticOutput = condenseOutput(tfbsdbOutput)
    
    return mechanisticOutput

def gene_conversion(gene_list,input_type="ensembl.gene", output_type="symbol",list_symbols=None):

    if input_type =="ensembl":
        input_type = "ensembl.gene"
    if output_type =="ensembl":
        output_type = "ensembl.gene"
    #kwargs = symbol,ensembl, entrezgene
    import mygene #requires pip install beyond anaconda
    mg = mygene.MyGeneInfo()
    gene_query = mg.querymany(gene_list, scopes=input_type, fields=[output_type], species="human", as_dataframe=True)

    if list_symbols is not None:
        if output_type == "ensembl.gene":
            list_ = list(gene_query[output_type])
            #print(list_)
            output = []
            for dict_ in list_:
                if type(dict_) is dict:
                    output.append(dict_["gene"])
                else:
                    for subdict in dict_:
                        output.append(subdict["gene"])
        else:
            output = list(gene_query[output_type])
        return output
    
    dict_ = {}
    try:
        trimmed_df = gene_query[gene_query.iloc[:,2]>0]
        for i in range(0,trimmed_df.shape[0]):
            tmp = trimmed_df.index[i]
            tmp1 = trimmed_df.iloc[i,2]
            dict_[tmp] = []
            lencheck = len(tmp1)
            if lencheck == 1:
                dict_[tmp].append(str(tmp1).split("'")[3])
            if lencheck > 1:
                for j in range(0,len(tmp1)):            
                    dict_[tmp].append(str(tmp1[j]).split("'")[3])
    except:
        return gene_query
    
    return dict_

def cluster(expressionData,minNumberGenes = 8,minNumberOverExpSamples=4,maxExclusion=0.275,random_state=12,iterations=5,overExpressionThreshold=80):
    
    try:
        df = expressionData.copy()
        maxStep = int(np.round(10*maxExclusion))
        axisSets = []
        allGenesMapped = []
        bestHits = []
        
        zero = np.percentile(expressionData,0)
        expressionThreshold = np.mean([np.percentile(expressionData.iloc[:,i][expressionData.iloc[:,i]>zero],overExpressionThreshold) for i in range(expressionData.shape[1])])
        
        startTimer = time.time()
        trial = -1
        for step in range(maxStep):
            trial+=1
            itr=-1
            for iteration in range(iterations):
                itr+=1.
                progress = 33.3333*trial+33.3333*(itr/iterations)
                print('{:.2f} percent complete'.format(progress))
                genesMapped = []
                bestMapped = []
                axesDiscovered = []
                
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
                        pdc = recursiveAlignment(clst,expressionData=df)
                        if len(pdc)==0:
                            continue
                                
                        elif len(pdc)>0:
                            for j in range(len(pdc)):
                                fpc = PCA(1,random_state=random_state)
                                principalComponents = fpc.fit_transform(df.loc[pdc[j],:].T)
                                egngn = pd.DataFrame(principalComponents)
                                egngn.index = df.columns
                                clusterArray = np.array(df.loc[pdc[j],:])
                                pearsonCoherence = pearson_array(clusterArray,np.array(egngn[0]))
                                normalPC = np.array(egngn[0])/np.linalg.norm(np.array(egngn[0]))
                                gns = np.array(pdc[j])[np.where(np.abs(pearsonCoherence)>0.5)[0]]
                                if len(gns) > minNumberGenes:
                                    genesMapped.append(gns)
                                    axesDiscovered.append(normalPC)
                                    
                axisSets.append(axesDiscovered)
                allGenesMapped.extend(genesMapped)
                try:
                    stackGenes = np.hstack(genesMapped)
                except:
                    stackGenes = []
                    print(len(genesMapped), len(allGenesMapped))
                residualGenes = list(set(df.index)-set(stackGenes))
                df = df.loc[residualGenes,:]
            
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
            
        stopTimer = time.time()
        
        print('\ncoexpression clustering completed in {:.2f} minutes'.format((stopTimer-startTimer)/60.))
    except:
        print('\nClustering failed. Ensure that expression data is formatted with genes as rows and samples as columns.')
        print('Consider transposing data (expressionData = expressionData.T) and retrying')
        
    return bestHits

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

def confidence_interval(n,p=0.1):
    
    df = n-1
    chi_sq = stats.chi2(df)
    coarse_x = np.linspace(0,10*n,10000)
    upper = min(np.where(chi_sq.cdf(coarse_x)>min(2*p,0.99))[0])
    x = np.linspace(0,coarse_x[upper],10000)
    lb = min(np.where(chi_sq.cdf(x)>p)[0])
    xp = 0.5*(x[lb]+x[lb+1])
    
    s2p = xp/float(n)
    
    return s2p

def varianceBin(n,p=0.15):
    variance = confidence_interval(n,p)
    vbin = np.sqrt(12.*variance/(1-(1./n**2)))
    return vbin

def plotBicluster(genesIndices,expressionMatrix,pValue=0.15,ymin=-7,ymax=7,title=None,savePlot=None,dataOnly=False,highlight=None,plotOnly=False):
    import matplotlib.pyplot as plt
    
    if type(genesIndices[0]) is int:
        cluster = expressionMatrix[genesIndices,:]

    if type(genesIndices[0]) is not int:
        cluster = np.array(expressionMatrix.loc[genesIndices,:])

    catchDropouts = []    
    splitCluster = []
    for i in range(cluster.shape[1]):
        tmp = cluster[:,i][cluster[:,i]>-4.01]
        if len(tmp) <= 3:
            catchDropouts.append(i)
        elif len(tmp) > 3:
            splitCluster.append(tmp)
    
    
    percentiles = np.vstack([np.percentile(splitCluster[i],[25,50,75]) for i in range(len(splitCluster))])
    
    variance = np.array([np.var(splitCluster[i]) for i in range(len(splitCluster))])
    lens = np.array([len(splitCluster[i]) for i in range(len(splitCluster))])
    
    uniqueLens = list(set(lens))
    
    cutoffs = {}
    for length in uniqueLens:
        cutoffs[length] = confidence_interval(length,pValue)
        
    inside = []
    outside = []
    
    for i in range(len(splitCluster)):
        if variance[i] <= cutoffs[lens[i]]:
            inside.append(i)
        else:
            outside.append(i)
            
    members = np.array(inside)
    nonMembers = np.array(outside)
    
    if len(nonMembers) <1:
        nonMembers = np.array([members[-1]])
        splitCluster.append(1)
        
    
    argsortMembers = np.argsort(percentiles[:,1][members])
    membersSpread = variance[members[argsortMembers]]**0.5
    membersSorted25th = percentiles[:,0][members[argsortMembers]]
    membersSortedMedian = percentiles[:,1][members[argsortMembers]]       
    membersSorted75th = percentiles[:,2][members[argsortMembers]]    
     
    argsortNonMembers = np.argsort(percentiles[:,1][nonMembers])   
    nonMembersSpread = variance[nonMembers[argsortNonMembers]]**0.5
    nonMembersSorted25th = percentiles[:,0][nonMembers[argsortNonMembers]]
    nonMembersSortedMedian = percentiles[:,1][nonMembers[argsortNonMembers]]       
    nonMembersSorted75th = percentiles[:,2][nonMembers[argsortNonMembers]]    
    
    iqrMembers = []
    for i in range(len(members)):
        iqrMembers.append(membersSorted75th[i])
        iqrMembers.append(membersSorted25th[i])
    
    iqrNonMembers = []
    for i in range(len(nonMembers)):
        iqrNonMembers.append(nonMembersSorted75th[i])
        iqrNonMembers.append(nonMembersSorted25th[i])    
        
    boundaryMembers = []
    for i in range(len(members)):
        boundaryMembers.append(membersSortedMedian[i]+2*membersSpread[i])
        boundaryMembers.append(membersSortedMedian[i]-2*membersSpread[i])
    
    boundaryNonMembers = []
    for i in range(len(nonMembers)):
        boundaryNonMembers.append(nonMembersSortedMedian[i]+2*nonMembersSpread[i])
        boundaryNonMembers.append(nonMembersSortedMedian[i]-2*nonMembersSpread[i])
    
    doubleRangeMembers = []
    for i in range(len(members)):
        doubleRangeMembers.append(i)
        doubleRangeMembers.append(i)
    
    doubleRangeNonMembers = []
    for i in range(len(members),len(splitCluster)):
        doubleRangeNonMembers.append(i)
        doubleRangeNonMembers.append(i)    
    
    doubleRange = np.concatenate([doubleRangeMembers,doubleRangeNonMembers])
    
    boundary = np.concatenate([boundaryMembers,boundaryNonMembers])   
    middle = np.concatenate([membersSortedMedian,nonMembersSortedMedian])
    
    if dataOnly:
        return members[argsortMembers], nonMembers[argsortNonMembers], variance[members[argsortMembers]], variance[nonMembers[argsortNonMembers]]
        
        
    plt.figure()
    plt.plot(doubleRange,boundary,color="gray",LineWidth=2,alpha=0.25)
    
    plt.plot(doubleRangeMembers,iqrMembers,color="blue",LineWidth=2,alpha=0.5)        
    plt.plot(doubleRangeNonMembers,iqrNonMembers,color="gray",LineWidth=2,alpha=0.5)  
    
    plt.plot(range(0,len(members)),middle[0:len(members)],color="white",LineWidth=3)        
    plt.plot(range(len(members),len(middle)),middle[len(members):],color="white",LineWidth=3)  
    
    plt.plot([len(members),len(members)],[ymin,ymax],color="red",LineWidth = 2,LineStyle="--")
    
    plt.ylim(ymin,ymax)
    plt.ylabel("Relative expression",FontSize=16)
    plt.xlabel("Sample Index",FontSize=16)   
    if title is not None:
        plt.title(title,FontSize=16) 
    plt.yticks(FontSize=14)
    plt.xticks(FontSize=14)
    plt.tight_layout()
    if savePlot is not None:
        plt.savefig(savePlot)
        
    if plotOnly is True:
        return
    return members[argsortMembers], nonMembers[argsortNonMembers], variance[members[argsortMembers]], variance[nonMembers[argsortNonMembers]]

def survivalMedianAnalysis(task):
    import lifelines
    #from lifelines import KaplanMeierFitter
    from lifelines import CoxPHFitter

    start, stop = task[0]
    dict_,expressionData,SurvivalDf = task[1]
    
    overlapPatients = list(set(expressionData.columns)&set(SurvivalDf.index))
    if len(overlapPatients) == 0:
        print("samples are not represented in the survival data")
        return 
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]
       
    coxResults = {}
    keys = dict_.keys()[start:stop]
    ct=0
    for key in keys:
        ct+=1
        if ct%10==0:
            print(ct)
        geneset = dict_[key]
        cluster = expressionData.loc[geneset,overlapPatients]
        nz = np.count_nonzero(cluster+4.01,axis=0)
        
        medians = []
        for i in range(cluster.shape[1]):
            if nz[i] >= 3:
                median = np.median(cluster.iloc[:,i][cluster.iloc[:,i]>-4.01])
            elif nz[i] < 3:
                median = np.median(cluster.iloc[:,i])
            medians.append(median)
            
        medianDf = pd.DataFrame(medians)
        medianDf.index = overlapPatients
        medianDf.columns = ["median"]
        Survival2 = pd.concat([Survival,medianDf],axis=1)
        Survival2.sort_values(by=Survival2.columns[0],inplace=True)
        
        cph = CoxPHFitter()
        cph.fit(Survival2, duration_col=Survival2.columns[0], event_col=Survival2.columns[1])
        
        tmpcph = cph.summary
        
        cox_hr = tmpcph.loc["median","z"]
        cox_p = tmpcph.loc["median","p"]  
        coxResults[key] = (cox_hr, cox_p)
        
    return coxResults

def survivalMembershipAnalysis(task):
    import lifelines
    #from lifelines import KaplanMeierFitter
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
    import lifelines
    #from lifelines import KaplanMeierFitter
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

def parallelSurvivalAnalysis(coexpressionModules,expressionData,numCores=5,survivalPath=os.path.join("..","data","survivalIA12.csv"),survivalData=None):

    if survivalData is None:
        survivalData = pd.read_csv(survivalPath,index_col=0,header=0)
    taskSplit = splitForMultiprocessing(coexpressionModules.keys(),numCores)
    taskData = (coexpressionModules,expressionData,survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = multiprocess(survivalMedianAnalysis,tasks)
    survivalAnalysis = condenseOutput(coxOutput)

    return survivalAnalysis

def parallelMemberSurvivalAnalysis(membershipDf,numCores=5,survivalPath=None,survivalData=None):

    if survivalData is None:
        survivalData = pd.read_csv(survivalPath,index_col=0,header=0)
    taskSplit = splitForMultiprocessing(membershipDf.index,numCores)
    taskData = (membershipDf,survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = multiprocess(survivalMembershipAnalysis,tasks)
    survivalAnalysis = condenseOutput(coxOutput)

    return survivalAnalysis

def getStratifiers(membershipSurvival,threshold):
    stratifiers = {}
    for key in membershipSurvival.keys():
        if membershipSurvival[key][1]<threshold:
            stratifiers[key] = membershipSurvival[key]
    return stratifiers

def collectResults(coexpressionModules,samples,motifAnalysis,mechanisticOutput,survivalAnalysis):
    minerCollectedResults = {}
    for key in coexpressionModules.keys():
        minerCollectedResults[key] = {}
        minerCollectedResults[key]["BiclusterID"] = key
        minerCollectedResults[key]["ClusterIncluded"] = 1
        minerCollectedResults[key]["Genes"] = (";").join(coexpressionModules[key])
        minerCollectedResults[key]["Samples"] = (";").join(samples[key])    
        if key in motifAnalysis.keys():
            minerCollectedResults[key]["Motifs"] = (";").join([(":").join(motifAnalysis[key][i].keys()) for i in motifAnalysis[key].keys()])
            tmp1 = []
            for i in motifAnalysis[key].keys():
                tmp2 = []
                for j in motifAnalysis[key][i].keys():
                    tmp2.append(str(motifAnalysis[key][i][j]))
                tmp1.append((":").join(tmp2))
            minerCollectedResults[key]["MotifWeights"] = (";").join(tmp1)
            minerCollectedResults[key]["MotifRegulators"] = (";").join(motifAnalysis[key].keys())        
        minerCollectedResults[key]["Regulators"] = (";").join(mechanisticOutput[key].keys())
        minerCollectedResults[key]["RegulatorWeights"] = (";").join([str(mechanisticOutput[key][i][0]) for i in mechanisticOutput[key].keys()])
        minerCollectedResults[key]["RegulatorReason"] = "Motifs;Correlation"
        minerCollectedResults[key]["CoxHR"] = survivalAnalysis[key][0] 
        minerCollectedResults[key]["CoxPV"] = survivalAnalysis[key][1]
    return minerCollectedResults

def resultsToCSV(minerCollectedResults,saveFile):
    template = np.zeros((len(minerCollectedResults),len(minerCollectedResults[minerCollectedResults.keys()[0]].keys())))
    templateDf = pd.DataFrame(template)
    templateDf.index = minerCollectedResults.keys()
    templateDf.columns = minerCollectedResults[minerCollectedResults.keys()[0]].keys()
    for i in minerCollectedResults.keys():
        for j in minerCollectedResults[i].keys():
            tmp = minerCollectedResults[i][j]
            templateDf.loc[i,j] = tmp
    
    sortColumn = np.array(templateDf["BiclusterID"]).astype(int)
    templateDf["BiclusterID"] = sortColumn
    templateDf.sort_values(by="BiclusterID",inplace=True)
    orderedColumns = ['BiclusterID','Genes','Samples','Regulators','RegulatorWeights','RegulatorReason','Motifs','MotifWeights','MotifRegulators','CoxHR','CoxPV','ClusterIncluded']
    templateDf = templateDf.loc[:,orderedColumns]    
    templateDf.to_csv(saveFile)
    return templateDf

def regulatorsToCSV(coregulationModules,saveFile):    
    regulatorTemplate = gene_conversion(coregulationModules.keys())
    regulatorTemplate.iloc[:,1] = regulatorTemplate.index
    regulatorTemplate.columns = ["RegulatorEntrez","RegulatorID","RegulatorSymbol"]
    typeArray = ["TF" for i in range(regulatorTemplate.shape[0])]
    typeColumn = pd.DataFrame(typeArray)
    typeColumn.index = regulatorTemplate.index
    typeColumn.columns = ["RegulatorType"]
    
    regulatorCSV = pd.concat([regulatorTemplate,typeColumn],axis=1)
    regulatorCSV = regulatorCSV.loc[:,["RegulatorID","RegulatorEntrez","RegulatorSymbol","RegulatorType"]]
    regulatorCSV.to_csv(saveFile)
    return regulatorCSV

def clustersForFIRM(clusters,conversionDf,savedf=None):
    import pandas as pd
    import numpy as np
    
    conversionGenes = set(list(conversionDf.index))
    entrezClusters = {}
    for key in clusters:
        ensemblGenes = clusters[key]
        overlapGenes = list(set(ensemblGenes)&conversionGenes)
        entrezGenes = list(conversionDf.loc[overlapGenes,"_id"])
        entrezClusters[key] = entrezGenes
    
    gene = []
    group = []
    geneAppend = gene.append
    groupAppend = group.append

    for key in entrezClusters:
        for element in entrezClusters[key]:
            geneAppend(element)
            groupAppend(key)
            
    stack = np.vstack([gene,group]).T
    df = pd.DataFrame(stack)
    df.columns = ["Gene","Group"]
    
    if savedf is not None:
        df.to_csv(savedf,index=False)
    
    return df

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

def getCoexpressionModules(mechanisticOutput):
    coexpressionModules = {}
    for i in mechanisticOutput.keys():
        genes = list(set(np.hstack([mechanisticOutput[i][key][1] for key in mechanisticOutput[i].keys()])))
        coexpressionModules[i] = genes
    return coexpressionModules

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

def generateCausalInputs(expressionData,mechanisticOutput,coexpressionModules,saveFolder,mutationFile="filteredMutationsIA12.csv",regulon_dict=None):
    import os
    import numpy as np
    import pandas as pd
    
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
    
def convertRegulons(df,conversionTable):
    reconvertedRegulators = np.array(conversionTable[np.array(df.Regulator)])
    reconvertedGenes = np.array(conversionTable[np.array(df.Gene)])
    reconvertedIDs = np.array(df.Regulon_ID)
    regulonTable = pd.DataFrame(np.vstack([reconvertedIDs,reconvertedRegulators,reconvertedGenes]).T)
    regulonTable.columns = ["Regulon_ID","Regulator","Gene"]
    return regulonTable

def convertDictionary(dict_,conversionTable):
    converted = {}
    for i in dict_.keys():
        genes = dict_[i]
        conv_genes = list(conversionTable[genes])
        converted[i] = conv_genes
    return converted

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

def laplacian(regulons,resultsDirectory=None):
    lens = [len(regulons[i]) for i in regulons.keys()]
    genes = list(set(np.hstack([np.hstack([regulons[i][j] for j in regulons[i].keys()]) for i in regulons.keys()])))
    degree = np.hstack([lens,np.ones(sum(lens))])

    features = list(set(regulons.keys())|set(genes))
    laplacianIncidence = pd.DataFrame(np.zeros((len(features),len(features))))
    laplacianIncidence.index = features
    laplacianIncidence.columns = features
    for tf in regulons.keys():
        tfGenes = list(set(np.hstack([regulons[tf][i] for i in regulons[tf].keys()])))
        laplacianIncidence.loc[tf,tfGenes] = -1
        laplacianIncidence.loc[tfGenes,tf] = -1
    trace = -np.sum(laplacianIncidence,axis=1)
    for i in range(len(trace)):
        laplacianIncidence.iloc[i,i] = trace[i]

    Tneg = pd.DataFrame(np.zeros((len(features),len(features))))
    Tneg.index = features
    Tneg.columns = features
    for i in range(len(trace)):
        Tneg.iloc[i,i] = 1./np.sqrt(trace[i])

    Laplacian = np.dot(np.array(Tneg),np.dot(np.array(laplacianIncidence),np.array(Tneg)))

    Laplacian = pd.DataFrame(Laplacian)
    Laplacian.index = features
    Laplacian.columns = features
    if resultsDirectory is not None:
        Laplacian.to_csv(os.path.join(resultsDirectory,"regulonsLaplacian.csv"))

    listLaplacian = [list(Laplacian.iloc[i,:]) for i in range(Laplacian.shape[0])]

    from scipy import sparse
    sparseLaplacian = sparse.csr_matrix(listLaplacian)
    vals, vecs = sparse.linalg.eigsh(sparseLaplacian, k=30)

    eigenvectors = pd.DataFrame(vecs)
    eigenvectors.index = features
    eigenvectors.columns = range(vecs.shape[1])
    if resultsDirectory is not None:
        eigenvectors.to_csv(os.path.join(resultsDirectory,"regulonsLaplacianEigenvectors.csv"))

    eigenvalues = pd.DataFrame(vals)
    eigenvalues.index = range(len(vals))
    eigenvalues.columns = ["eigenvalues"]
    if resultsDirectory is not None:
        eigenvalues.to_csv(os.path.join(resultsDirectory,"regulonsLaplacianEigenvalues.csv"))

    return Laplacian, eigenvectors, eigenvalues
    
def groupVector(centroidClusters,expressionData,index=0):
    group = centroidClusters[index]
    converter = pd.DataFrame(np.zeros(expressionData.shape[1]))
    converter.index = expressionData.columns
    converter.columns = ["index"]

    converter.loc[group,"index"] = 1
    groupVector = np.array(converter.iloc[:,0])
    return groupVector

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

def plotDifferentialMatrix(overExpressedMembersMatrix,underExpressedMembersMatrix,orderedOverExpressedMembers,cmap="viridis",saveFile=None):
    differentialActivationMatrix = overExpressedMembersMatrix-underExpressedMembersMatrix
    fig = plt.figure(figsize=(7,7))  
    ax = fig.add_subplot(111)
    try:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    except:
        pass
    orderedDM = differentialActivationMatrix.loc[orderedOverExpressedMembers.index,orderedOverExpressedMembers.columns]
    ax.imshow(orderedDM,cmap=cmap,vmin=-1,vmax=1)
    asp = differentialActivationMatrix.shape[1]/float(differentialActivationMatrix.shape[0])
    ax.set_aspect(asp)
    ax.grid(False)
    if saveFile is not None:
        plt.ylabel("Modules",FontSize=14)
        plt.xlabel("Samples",FontSize=14)
        ax.set_aspect(asp)
        ax.grid(False)        
        plt.savefig(saveFile,bbox_inches="tight")
    return orderedDM

def stateReduction(df,clusterList,stateThreshold=0.75):
    
    statesDf = pd.DataFrame(np.zeros((len(clusterList),df.shape[1])))
    statesDf.index = range(len(clusterList))
    statesDf.columns = df.columns

    for i in range(len(clusterList)):
        patients = clusterList[i]
        subset = df.loc[:,patients]

        filter_o8 = np.where(subset.sum(axis=1)/float(subset.shape[1])>=0.8)[0]
        filtered_regulons = subset.index[filter_o8]

        state_df = df.loc[filtered_regulons,df.columns]

        keep_high_o8 = np.where(state_df.sum(axis=0)/float(state_df.shape[0])>=stateThreshold)[0]
        keep_low_o8 = np.where(state_df.sum(axis=0)/float(state_df.shape[0])<=-1*stateThreshold)[0]
        hits_high = np.array(df.columns)[keep_high_o8]
        hits_low = np.array(df.columns)[keep_low_o8]

        statesDf.loc[i,hits_high] = 1
        statesDf.loc[i,hits_low] = -1
        
    return statesDf

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

def plotStates(statesDf,tsneDf,numCols=3,saveFile=None,size=10):
    number_figure_columns = numCols
    number_figure_rows = int(np.ceil(float(statesDf.shape[0])/number_figure_columns))
    fig = plt.figure(figsize=(10,10))
    for ix in range(statesDf.shape[0]):
        ax = fig.add_subplot(number_figure_rows,number_figure_columns,ix+1)
        # overlay single state onto tSNE plot
        stateIndex = ix

        group = pd.DataFrame(np.zeros(statesDf.shape[1]))
        group.index = statesDf.columns
        group.columns = ["status"]
        group.loc[statesDf.columns,"status"] = list(statesDf.iloc[stateIndex,:])
        group = np.array(group.iloc[:,0])
        ax.scatter(tsneDf.iloc[:,0],tsneDf.iloc[:,1],cmap="bwr",c=group,vmin=-1,vmax=1,s=size)

    if saveFile is not None:
        plt.savefig(saveFile,bbox_inches="tight")

    return

def kaplanMeier(SurvivalDf, hr_ids, lr_ids, resultsDirectory, xmax=None, filename="kaplanMeier.pdf", title="Progression_free survival", xlabel="Timeline", ylabel="Surviving proportion"):
    from lifelines import KaplanMeierFitter

    SurvivalDf.columns = ['duration','observed']

    T_hr = SurvivalDf.loc[hr_ids,"duration"]
    E_hr = SurvivalDf.loc[hr_ids,"observed"]
    kmf_hr = KaplanMeierFitter()
    ax = plt.subplot(111)

    ax_hr = kmf_hr.fit(T_hr, event_observed=E_hr)

    T_lr = SurvivalDf.loc[lr_ids,"duration"]
    E_lr = SurvivalDf.loc[lr_ids,"observed"]
    kmf_lr = KaplanMeierFitter()
    ax_lr = kmf_lr.fit(T_lr, event_observed=E_lr)


    kmf_hr.survival_function_.plot(ax=ax)
    if xmax is not None:
        plt.xlim(0,xmax)
    kmf_lr.survival_function_.plot(ax=ax,label="High-risk")
    plt.legend(["High-risk","Low-risk"])
    if xmax is not None:
        plt.xlim(0,xmax)

    plt.title(title,Fontsize=14)
    plt.ylabel(ylabel,FontSize=14)
    plt.xlabel(xlabel,FontSize=14)
    plt.savefig(os.path.join(resultsDirectory,filename))
    return

def logicalPredictor(train_membership,test_membership,train_survival,test_survival,numCores=5,maxPvalue = None,iterations = 7):

    train_membershipSurvival = parallelMemberSurvivalAnalysis(membershipDf = train_membership,numCores=numCores,survivalPath="",survivalData=train_survival)
    hrs = np.array([train_membershipSurvival[key][0] for key in train_membershipSurvival.keys()])
    best_key = np.array(train_membershipSurvival.keys())[np.argmax(hrs)]

    predictors = []
    predictors.append(best_key)
    best_score = 0
    
    test_membershipSurvival = parallelMemberSurvivalAnalysis(membershipDf = test_membership,numCores=numCores,survivalPath="",survivalData=test_survival)

    pairsTrain = [(i,train_membershipSurvival[i][1]) for i in train_membershipSurvival.keys()]
    pairsTrain.sort(key = lambda s: s[1])
    pairsTrainKept = pairsTrain[0:int(len(pairsTrain)/2.)]
    ptk1 = [i[0] for i in pairsTrainKept]
    
    pairsTest = [(i,test_membershipSurvival[i][1]) for i in test_membershipSurvival.keys()]
    pairsTest.sort(key = lambda s: s[1])
    pairsTestKept = pairsTest[0:int(len(pairsTest)/2.)]
    ptk2 = [i[0] for i in pairsTestKept]
    ptk2ps = [i[1] for i in pairsTestKept]
    percentileCutoff = ptk2ps[1] #np.percentile(ptk2ps,2)
    
    if maxPvalue is None:
        maxPvalue = 10**(-np.ceil(-np.log10(pairsTest[0][1])))

    topKeys = set(ptk1)&set(ptk2)
    topKeys = list(topKeys|set(predictors))
    
    train_membership = train_membership.loc[topKeys,:]
    test_membership = test_membership.loc[topKeys,:]
    
    for iteration in range(iterations):

        train_membership = train_membership*train_membership.loc[predictors[-1],:]
        test_membership = test_membership*test_membership.loc[predictors[-1],:]
        train_membershipSurvival = parallelMemberSurvivalAnalysis(membershipDf = train_membership,numCores=numCores,survivalPath="",survivalData=train_survival)
        test_membershipSurvival = parallelMemberSurvivalAnalysis(membershipDf = test_membership,numCores=numCores,survivalPath="",survivalData=test_survival)

        train_hrs = np.array([train_membershipSurvival[key][0] for key in train_membershipSurvival.keys()])
        test_hrs = np.array([test_membershipSurvival[key][0] for key in test_membershipSurvival.keys()])
        train_ps = np.array([train_membershipSurvival[key][1] for key in train_membershipSurvival.keys()])
        test_ps = np.array([test_membershipSurvival[key][1] for key in test_membershipSurvival.keys()])

        pFilter = np.where(test_ps<maxPvalue)[0]
        if len(pFilter) == 0:
            maxPvalue = percentileCutoff
        pFilter = np.where(test_ps<maxPvalue)[0]
        if len(pFilter) == 0:
            print(maxPvalue)
            print(test_ps)            
            return predictors
        
        average_hrs = 0.5*(train_hrs[pFilter]+test_hrs[pFilter])
        tmp_keys = np.array(train_membershipSurvival.keys())[pFilter]
        tmp_best = tmp_keys[np.argmax(average_hrs)]
        score = max(average_hrs)
        if score<=best_score:
            break
        best_score = score
        best_key = tmp_best
        predictors.append(best_key)
        print(predictors)

    return predictors

def labelVector(hr,lr):
    labels = np.concatenate([np.ones(len(hr)),np.zeros(len(lr))]).astype(int)
    labelsDf = pd.DataFrame(labels)
    labelsDf.index = np.concatenate([hr,lr])
    labelsDf.columns = ["label"]
    return labelsDf

def kmAnalysis(survivalDf,durationCol,statusCol,saveFile=None):
    
    import lifelines
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

def precision(matrix, labels):
    
    vector = labels.iloc[:,0]
    vectorMasked = (matrix*vector).T
    TP = np.array(np.sum(vectorMasked,axis=0)).astype(float)
    FP = np.array(np.sum(matrix,axis=1)-TP).astype(float)
    prec = TP/(TP+FP)
    prec[np.where(TP<=5)[0]]=0
    return prec

def areaScore(highRisk,lowRisk,survivalDf,durationCol="duration",censorCol = "observed",kmCol = "kmEstimate",plot=False):
    hr_pats = list(set(highRisk)&set(survivalDf.index))
    hr_survival = survivalDf.loc[hr_pats,:]
    hr_survival.sort_values(by=durationCol,inplace=True)
    hr_survival.head(5)
    hr_pfs = kmAnalysis(hr_survival,durationCol,censorCol,saveFile=None)

    lr_pats = list(set(lowRisk)&set(survivalDf.index))
    lr_survival = survivalDf.loc[lr_pats,:]
    lr_survival.sort_values(by=durationCol,inplace=True)
    lr_survival.head(5)
    lr_pfs = kmAnalysis(lr_survival,durationCol,censorCol,saveFile=None)

    hrDuration = np.array(hr_pfs.loc[:,durationCol])
    hrDurationShift = np.concatenate([np.array([0]),np.array([hrDuration[i] for i in range(len(hrDuration)-1)])])
    lrDuration = np.array(lr_pfs.loc[:,durationCol])
    lrDurationShift = np.concatenate([np.array([0]),np.array([lrDuration[i] for i in range(len(lrDuration)-1)])])

    hrKME = np.array(hr_pfs.loc[:,kmCol])
    hrKMEShift = np.concatenate([np.array([1]),np.array([hrKME[i] for i in range(len(hrKME)-1)])])
    lrKME = np.array(lr_pfs.loc[:,kmCol])
    lrKMEShift = np.concatenate([np.array([1]),np.array([lrKME[i] for i in range(len(lrKME)-1)])])

    hr_widths = hrDuration - hrDurationShift
    hr_area = np.sum(hr_widths*hrKMEShift)
    lr_widths = lrDuration - lrDurationShift
    lr_area = np.sum(lr_widths*lrKMEShift)
    area_statistic = max(hr_area,lr_area)/float(min(hr_area,lr_area))
    
    if plot is True:
        plt.step(hrDuration,hrKME)
        plt.step(lrDuration,lrKME)
        plt.ylim(0,1)
        
    return hr_area, lr_area, area_statistic

def tscores(riskGroups,srvs,numDatasets=None):
    if numDatasets is None:
        numDatasets = len(srvs)
    hr_tvals = []
    for i in range(len(riskGroups)):
        tmp_tval = []
        for iteration in range(numDatasets):
            srv = srvs[iteration]
            hr_samples = riskGroups[i][iteration]
            lr_samples = list(set(srv.index)-set(hr_samples))

            gHR = np.array(srv.loc[hr_samples,"GuanScore"])
            gLR = np.array(srv.loc[lr_samples,"GuanScore"])
            ttest = stats.ttest_ind(gHR,gLR,equal_var=False)
            tmp_tval.append(ttest[0])
            #print(ttest[0])
        hr_tvals.append(np.mean(tmp_tval))
    nans = np.where(np.abs(np.array(hr_tvals).astype(int))<0)[0]
    tscores = np.array(hr_tvals)
    tscores[nans] = 0
    return tscores

def predictor(Dfs,Srvs,levels=50,numTrials=10,threshold=0.05,override=None,predictorFile=None):

    numDatasets = len(Dfs)
    if override is not None:
        numDatasets = override

    start = time.time()

    tmpDfs = []
    for i in range(numDatasets):
        tmpDfs.append(Dfs[i].copy())
    #tmpDfs.append(Dfs[0].copy())
    #tmpDfs.append(Dfs[1].copy())

    srvs = []
    for i in range(numDatasets):
        srvs.append(Srvs[i].copy())
    #srvs.append(Srvs[0].copy())
    #srvs.append(Srvs[1].copy())

    pvals = []
    branches = []
    riskGroups = []
    hr_proportions = [0.1,0.15,0.2,0.25,0.3]
    lr_proportions = [0.65,0.7,0.75,0.8,0.85]
    proportions = np.concatenate([hr_proportions,lr_proportions])

    for level in range(levels):
        try:
            bestHits = []
            for i in range(len(proportions)):
                hits = []
                proportion = proportions[i]
                for trial in range(numTrials):
                    if trial>0:
                        for i in range(numDatasets):
                            tmpDfs[i] = tmpDfs[i]*tmpDfs[i].loc[hit,:]
                            
                        #tmpDfs = [tmpDfs[0]*tmpDfs[0].loc[hit,:],tmpDfs[1]*tmpDfs[1].loc[hit,:]]
                    tmpScores = []
                    precisionValues = []
                    for iteration in range(numDatasets):
                        srv = srvs[iteration]
                        tmpDf = tmpDfs[iteration]           
                        num_hr = int(proportion*srv.shape[0])

                        if proportion in hr_proportions:
                            hr_samples = srv.index[0:num_hr]
                            lr_samples = srv.index[num_hr:]
                        elif proportion in lr_proportions:
                            hr_samples = srv.index[num_hr:] #reversed for lowest-risk prediction
                            lr_samples = srv.index[0:num_hr] #reversed for lowest-risk prediction

                        restrictedLabels = np.concatenate([hr_samples,lr_samples])
                        restrictedMatrix = tmpDf.loc[:,restrictedLabels]
                        labels = labelVector(hr_samples,lr_samples)
                        prec = precision(restrictedMatrix, labels)
                        weightedPrecision = prec*np.array(np.sum(tmpDf,axis=1))**0.333
                        precisionValues.append(weightedPrecision)

                    precisionMatrix = np.vstack(precisionValues)
                    scores = np.mean(precisionMatrix,axis=0)
                    hit = restrictedMatrix.index[np.argmax(scores)]
                    if hit in hits:
                        bestHits.append(hits)
                        break
                    hits.append(hit)
                    if trial == numTrials-1:
                        bestHits.append(hits)                

                tmpDfs = []
                for i in range(numDatasets):
                    tmpDfs.append(Dfs[i].copy())

                #tmpDfs[0] = Dfs[0]
                #tmpDfs[1] = Dfs[1]

            finalScores = []
            for bh in range(len(bestHits)):
                hits = bestHits[bh]
                tmpScores = []
                for iteration in range(numDatasets):
                    tmpDf = Dfs[iteration].copy()
                    srv = srvs[iteration]
                    for hit in hits:
                        tmpDf = tmpDf*tmpDf.loc[hit,:]
                    hr_samples = list(set(tmpDf.columns[tmpDf.loc[hits[-1],:]==1])&set(srv.index))
                    lr_samples = list(set(tmpDf.columns[tmpDf.loc[hits[-1],:]==0])&set(srv.index))
                    if min(len(hr_samples),len(lr_samples)) < 0.01*srv.shape[0]:
                        tmpScores.append(0)
                        continue
                    gHR = np.array(srv.loc[hr_samples,"GuanScore"])
                    gLR = np.array(srv.loc[lr_samples,"GuanScore"])
                    ttest = stats.ttest_ind(gHR,gLR,equal_var=False)
                    tmpScores.append(ttest[0])

                combinedScore = np.mean(tmpScores)
                finalScores.append(combinedScore)

            selection = np.argmax(np.abs(np.array(finalScores)))

            hits = bestHits[selection]
            tmpPvals = []
            tmpHrs = []
            tmpLrs = []
            for iteration in range(numDatasets):
                tmpDf = Dfs[iteration].copy()
                srv = srvs[iteration]
                for hit in hits:
                    tmpDf = tmpDf*tmpDf.loc[hit,:]
                hr_samples = list(set(tmpDf.columns[tmpDf.loc[hits[-1],:]==1])&set(srv.index))
                lr_samples = list(set(tmpDf.columns[tmpDf.loc[hits[-1],:]==0])&set(srv.index))

                gHR = np.array(srv.loc[hr_samples,"GuanScore"])
                gLR = np.array(srv.loc[lr_samples,"GuanScore"])
                ttest = stats.ttest_ind(gHR,gLR,equal_var=False)
                tmpPvals.append(ttest[1])

                if iteration == 0:
                    if len(hr_samples) <= len(lr_samples):
                        tag = "hr"
                    elif len(hr_samples) > len(lr_samples):
                        tag = "lr"

                if tag == "hr":
                    tmpHrs.append(hr_samples)
                    tmpLrs.append(lr_samples)
                if tag == "lr":
                    tmpHrs.append(lr_samples) 
                    tmpLrs.append(hr_samples)

            print(min(tmpPvals))
            if min(tmpPvals) > threshold:
                break
            pvals.append(min(tmpPvals))
            branches.append(hits)
            lrs = tmpLrs
            riskGroups.append(tmpHrs)

            for iteration in range(numDatasets):
                tmpDfs[iteration] = Dfs[iteration].loc[:,lrs[iteration]]
                srvs[iteration] = srvs[iteration].loc[list(set(lrs[iteration])&set(srvs[iteration].index)),:]
                srvs[iteration].sort_values(by="GuanScore",ascending=False,inplace=True)

            print(hits)  
        except:
            break

    stop = time.time()
    print("completed in {:.3f} minutes".format((stop-start)/60.))

    ts = tscores(riskGroups,Srvs,numDatasets)
    predictor_dictionary = {}
    for i in range(len(branches)):
        predictor_dictionary[i] = []
        predictor_dictionary[i].append(branches[i])
        predictor_dictionary[i].append(ts[i])

    if predictorFile is not None:
        write_json(dict_=predictor_dictionary,output_file=predictorFile)

    return predictor_dictionary

def predictorScores(df,srv,pred_dict):
    scoresDf = pd.DataFrame(np.zeros(srv.shape[0]))
    scoresDf.index = srv.index
    scoresDf.columns = ["t-score"]

    scores = []

    labeled = set([])
    for key in pred_dict.keys():
        signature = pred_dict[key][0]
        t = pred_dict[key][1]

        tmpHrs = []
        tmpDf = df.copy()
        for hit in signature:
            tmpDf = tmpDf*tmpDf.loc[hit,:]
        hr_samples = list(set(tmpDf.columns[tmpDf.loc[signature[-1],:]==1])&set(srv.index)-labeled)
        lr_samples = list(set(tmpDf.columns[tmpDf.loc[signature[-1],:]==0])&set(srv.index)-labeled)

        if len(hr_samples) <= len(lr_samples):
            tag = "hr"
        elif len(hr_samples) > len(lr_samples):
            tag = "lr"

        if tag == "hr":
            tmpHrs.append(hr_samples)
            labeled = labeled|set(hr_samples)
        if tag == "lr":
            tmpHrs.append(lr_samples) 
            labeled = labeled|set(lr_samples)

        scores.append(tmpHrs)
        scoresDf.loc[tmpHrs[0],"t-score"] = t

    return scoresDf

def predictClasses(predictor_scores,thresholds = None):
    
    if thresholds is None:
        pcts = np.percentile(np.array(predictor_scores.iloc[:,0]),[10,90])
        if pcts[0] < 0:
            lowCut = np.ceil(pcts[0])
        if pcts[0] >= 0:
            print("thresholds could not be inferred. Please define thresholds and try again.")
        if pcts[1] < 0:
            print("thresholds could not be inferred. Please define thresholds and try again.")
        if pcts[1] >= 0:
            highCut = np.floor(pcts[1])
        
        modLow = -1
        modHigh = 1
        
        if lowCut >= -1:
            modLow = lowCut/2.
        if highCut <= 1:
            modHigh = highCut/2.
            
        thresholds = [highCut,modHigh,modLow,lowCut]
        print("thresholds:", thresholds)

    group0 = list(predictor_scores.index[np.where(predictor_scores.iloc[:,0]>=thresholds[0])[0]])
    group1 = list(predictor_scores.index[np.where(predictor_scores.iloc[:,0]>=thresholds[1])[0]])
    group2 = list(predictor_scores.index[np.where(predictor_scores.iloc[:,0]>=thresholds[2])[0]])
    group3 = list(predictor_scores.index[np.where(predictor_scores.iloc[:,0]>thresholds[3])[0]])
    group4 = list(predictor_scores.index[np.where(predictor_scores.iloc[:,0]<=thresholds[3])[0]])
    
    outputDf = predictor_scores.copy()
    outputDf.columns = ["class"]
    veryHigh = group0
    veryLow = group4
    modHigh = list(set(group1)-set(group0))
    modLow = list(set(group3)-set(group2))
    grouped = list(set(veryHigh)|set(veryLow)|set(modHigh)|set(modLow))
    average = list(set(predictor_scores.index)-set(grouped))
    
    outputDf.loc[veryHigh,"class"] = 5
    outputDf.loc[modHigh,"class"] = 4
    outputDf.loc[average,"class"] = 3
    outputDf.loc[modLow,"class"] = 2
    outputDf.loc[veryLow,"class"] = 1
    
    return outputDf

def predictorThresholds(predictor_scores):
    
    pcts = np.percentile(np.array(predictor_scores.iloc[:,0]),[10,90])
    if pcts[0] < 0:
        lowCut = np.ceil(pcts[0])
    if pcts[0] >= 0:
        print("thresholds could not be inferred. Please define thresholds and try again.")
    if pcts[1] < 0:
        print("thresholds could not be inferred. Please define thresholds and try again.")
    if pcts[1] >= 0:
        highCut = np.floor(pcts[1])
    
    modLow = -1
    modHigh = 1
    
    if lowCut >= -1:
        modLow = lowCut/2.
    if highCut <= 1:
        modHigh = highCut/2.
        
    thresholds = [highCut,modHigh,modLow,lowCut]
    return thresholds
    
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
    
def splitRiskSignatures(pred_dict,thresholds):
    
    thresholds = np.array(thresholds)
    scores = thresholds[np.argsort(-thresholds)]
    splitRisk = {"very high":[],"high":[],"average":[],"low":[],"very low":[]}
    
    for i in pred_dict.keys():
        signature = pred_dict[i][0]
        score = pred_dict[i][1]
        if score >= scores[0]:
            splitRisk["very high"].append(signature)
        elif score < scores[0]:
            if score >= scores[1]:
                splitRisk["high"].append(signature)  
            elif score < scores[1]:
                if score >= scores[2]:
                    splitRisk["average"].append(signature) 
                elif score < scores[2]:
                    if score >= scores[3]:
                        splitRisk["low"].append(signature) 
                    elif score < scores[3]:
                        splitRisk["very low"].append(signature)
    return splitRisk  

def plotPCA(df):
    df = df.T
    pca = PCA(2)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(df)
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    return projected[:, 0], projected[:, 1]
