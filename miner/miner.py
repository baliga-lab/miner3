#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:20:40 2019

@author: MattWall
"""
import numpy as np
from numpy.random import choice
from scipy import stats
from scipy.stats import rankdata
from scipy.stats import chi2_contingency

import pandas as pd

import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.model_selection import train_test_split


from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

import multiprocessing, multiprocessing.pool
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from collections import Counter
import seaborn as sns
import mygene #requires pip install beyond anaconda
import pickle
import json
import time
import warnings
import os
import logging
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import traceback

from tqdm.notebook import tqdm, trange
from .progressbar import printProgressBar


def getMutations(mutationString, mutationMatrix):
    return mutationMatrix.columns[np.where(mutationMatrix.loc[mutationString,:]>0)[0]]

# =============================================================================
# Functions used for reading and writing files
# =============================================================================

def read_pkl(input_file):
    import pickle
    with open(input_file, 'rb') as f:
        dict_ = pickle.load(f)
    return dict_

def write_pkl(dictionary,output_file):
    import pickle
    output = open(output_file, 'wb')
    pickle.dump(dictionary, output)
    output.close()

def read_json(filename):
    with open(filename) as data:
        dict_ = json.load(data)
    return dict_

def write_json(dict_, output_file):
    import json
    output_file = output_file
    with open(output_file, 'w') as fp:
        json.dump(dict_, fp)

def read_file_to_df(filename: str) -> pd.DataFrame:
    """
    reads a dataframe from a file, attempting to guess the separator
    """
    extension = filename.split(".")[-1]
    if extension == "csv":
        df = pd.read_csv(filename,index_col=0,header=0)
        shape = df.shape
        if shape[1] == 0:
            df = pd.read_csv(filename,index_col=0,header=0,sep="\t")
    elif extension == "txt":
        df = pd.read_csv(filename,index_col=0,header=0,sep="\t")
        shape = df.shape
        if shape[1] == 0:
            df = pd.read_csv(filename,index_col=0,header=0)
    return df

def fileToReferenceDictionary(filename,dictionaryName,index_col=0):
    read_reference_db = pd.read_csv(filename,index_col=0,header=0)

    if list(read_reference_db.iloc[:,0]) == range(len(read_reference_db.iloc[:,0])):
        read_reference_db = read_reference_db.iloc[:,1:]
        print("deleted")
    read_reference_db.index = read_reference_db.iloc[:,0]
    read_reference_db = read_reference_db.iloc[:,1:]
    read_reference_db.head()

    reference_dic = {}
    for key in list(set(read_reference_db.index)):
        tmp_df = read_reference_db.loc[key,:]
        if type(tmp_df) is not pd.core.frame.DataFrame:
            tmp_df = pd.DataFrame(tmp_df)
        reference_dic[key] = list(tmp_df.iloc[:,0])

    write_pkl(reference_dic,dictionaryName)

    return reference_dic

def download_file_from_google_drive(id, destination):
    #taken from stackoverflow:
    #https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
    import requests

    print('downloading',id)
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)
    print('download complete')

    return

# =============================================================================
# Functions used for pre-processing data
# =============================================================================

def remove_null_rows(df: pd.DataFrame):
    minimum = np.percentile(df,0)
    if minimum == 0:
        filteredDf = df.loc[df.sum(axis=1)>0,:]
    else:
        filteredDf = df
    return filteredDf

def convertToEnsembl(df, conversionTable, input_format=None):
    from collections import Counter
    # Index Conversion table on ENSG notation
    conversionTableEnsg = conversionTable.copy()
    conversionTableEnsg.index = conversionTableEnsg.iloc[:,0]

    # Use Counter to count redundancies in ENSG row
    ensg = np.array(conversionTableEnsg.iloc[:,0])
    ensgCount = Counter(ensg)
    ensgCountMC = ensgCount.most_common()

    # Use Counter results to identify which genes need special conversion
    special_conversion = np.where(np.vstack(ensgCountMC)[:,1].astype(int)>1)[0]
    normal_conversion = np.where(np.vstack(ensgCountMC)[:,1].astype(int)==1)[0]

    # Pull out of loop for increased efficiency
    if input_format is None:
        input_format = conversionTableEnsg.columns[1]

    index_set = set(df.index)

    # Convert identifiers with >1 alias
    conversionEnsg = []
    conversionAffy = []
    for i in special_conversion:
        tmp_ensg = ensgCountMC[i][0]
        tmp_genes = list(set(list(conversionTableEnsg.loc[tmp_ensg,input_format]))&index_set)
        conv = np.argmax(np.mean(df.loc[tmp_genes,:],axis=1))
        conversionEnsg.append(tmp_ensg)
        conversionAffy.append(conv)

    # Convert identifiers with exactly 1 match
    for j in normal_conversion:
        tmp_ensg = ensgCountMC[j][0]
        conv = conversionTableEnsg.loc[tmp_ensg,input_format]
        conversionEnsg.append(tmp_ensg)
        conversionAffy.append(conv)

    # Prepare results dataframe
    conversion_df = pd.DataFrame(conversionEnsg)
    conversion_df.index = conversionAffy
    conversion_df.columns = ["Ensembl"]

    return conversion_df

def AffyToEnsemblDf(validation_path,expressionData_file,conversionTable_file,reference_index,output_file):
    expressionData_matrix = read_file_to_df(expressionData_file)
    conversionTable = read_file_to_df(conversionTable_file)
    expressionData_ensembl = convertToEnsembl(expressionData_matrix,conversionTable,input_format=None)
    expressionData_ensembl.head()

    index = intersect(expressionData_ensembl.index,expressionData_matrix.index)
    ensembl_overlap = intersect(reference_index,expressionData_ensembl.loc[index,"Ensembl"])
    expressionData_matrix = expressionData_matrix.loc[index,:]

    converted_array = np.zeros((len(ensembl_overlap),expressionData_matrix.shape[1]))
    converted_expression = pd.DataFrame(converted_array)
    converted_expression.index = ensembl_overlap
    converted_expression.columns = expressionData_matrix.columns

    for gene in index:
        ensembl = expressionData_ensembl.loc[gene,"Ensembl"]
        tmp = np.array(expressionData_matrix.loc[gene,:])
        if type(ensembl) is not str:
            ensembl = list(ensembl)
            for j in range(len(ensembl)):
                tmp_ensembl = ensembl[j]
                converted_expression.loc[tmp_ensembl,:] = tmp
            continue
        converted_expression.loc[ensembl,:] = tmp

    converted_expression = converted_expression.loc[ensembl_overlap,:]
    converted_expression.to_csv(output_file)
    return converted_expression


def convert_ids_orig(exp_data: pd.DataFrame, conversion_file_path: str):
    """
    Original table based conversion. This is needlessly complicated and
    just kept here for legacy purposes.
    It attempts to find out if the genes are specified in the first column or
    the first row of the input matrix and what type of identifier it uses
    """
    id_map = pd.read_csv(conversion_file_path, sep="\t")
    genetypes = sorted(set(id_map.iloc[:, 2]))
    exp_index = np.array(exp_data.index).astype(str)
    exp_columns = np.array(exp_data.columns).astype(str)
    mapped_genes = []
    transposed = False

    for geneType in genetypes:
        subset = id_map[id_map.iloc[:, 2] == geneType]
        subset.index = subset.iloc[:, 1]
        matching_rows = list(set(exp_index) & set(subset.index))
        matching_cols = list(set(exp_columns) & set(subset.index))

        if len(matching_rows) > len(mapped_genes):
            mapped_genes = matching_rows
            transposed = False
            gtype = geneType
            continue

        if len(matching_cols) > len(mapped_genes):
            mapped_genes = matching_cols
            transposed = True
            gtype = geneType
            continue

    if len(mapped_genes) == 0:
        print("Error: Gene identifiers not recognized")

    if transposed:
        exp_data = exp_data.T

    converted_data = exp_data.loc[mapped_genes, :]

    # build a conversion table based on the actual existing identifiers in the
    # input data by taking the subset of input identifier map
    subset = id_map[id_map.iloc[:,2] == gtype]
    subset.index = subset.iloc[:, 1]
    conv_table = subset.loc[mapped_genes, :]
    conv_table.index = conv_table.iloc[:, 0]
    conv_table = conv_table.iloc[:, 1]
    conv_table.columns = ["Name"]

    new_index = list(subset.loc[mapped_genes, "Preferred_Name"])
    converted_data.index = new_index

    # conv_table is a data frame ||preferred name | name ||
    # make the conversion table into a dictionary for the unmapped check
    id_dict = {}
    for idx, item in conv_table.items():
        id_dict[item] = idx

    # check for unmapped data
    comp_new_index = set(new_index)
    dropped_genes = []
    for i, idx in enumerate(exp_data.index):
        if not idx in id_dict:
            dropped_genes.append((i, idx))

    # Check for duplicates
    duplicates = [item for item, count in Counter(new_index).items() if count > 1]
    singles = list(set(converted_data.index) - set(duplicates))

    corrections = []
    for duplicate in duplicates:
        dup_data = converted_data.loc[duplicate, :]
        first_choice = pd.DataFrame(dup_data.iloc[0, :]).T
        corrections.append(first_choice)

    if len(corrections) == 0:
        print("completed identifier conversion.\n%d genes were converted. %d genes were dropped due to identifier mismatch" % (converted_data.shape[0], len(dropped_genes)))
        return converted_data, conv_table

    # The corrections handling handles duplications in the data
    # Technically, this should not be done in identifier conversion
    corrections_df = pd.concat(corrections, axis=0)
    uncorrected_data = converted_data.loc[singles, :]
    converted_data = pd.concat([uncorrected_data, corrections_df], axis=0)

    print("completed identifier conversion.\n%d genes were converted. %d genes were dropped due to identifier mismatch" % (converted_data.shape[0], len(dropped_genes)))
    return converted_data, conv_table


"""
IDENTIFIER CONVERSION START
THIS IS JUST FOR REFERENCE !!!! DON'T USE !!!!!
"""
def identifierConversion(expressionData, conversionTable=os.path.join("..","data","identifier_mappings.txt")):
    idMap = pd.read_csv(conversionTable,sep="\t")
    genetypes = list(set(idMap.iloc[:,2]))
    previousIndex = np.array(expressionData.index).astype(str)
    previousColumns = np.array(expressionData.columns).astype(str)
    bestMatch = []
    for geneType in genetypes:
        subset = idMap[idMap.iloc[:,2]==geneType]
        subset.index = subset.iloc[:,1]
        mappedGenes = list(set(previousIndex)&set(subset.index))
        mappedSamples = list(set(previousColumns)&set(subset.index))
        #if len(mappedGenes)>=max(10,0.01*expressionData.shape[0]):
        if len(mappedGenes)>len(bestMatch):
                bestMatch = mappedGenes
                state = "original"
                gtype = geneType
                continue
        #if len(mappedSamples)>=max(10,0.01*expressionData.shape[1]):
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

"""
IDENTIFIER CONVERSION END
"""
def readExpressionFromGZipFiles(directory):

    rootDir = directory
    sample_dfs = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            #print('\t%s' % fname)
            extension = fname.split(".")[-1]
            if extension == 'gz':
                path = os.path.join(rootDir,dirName,fname)
                df = pd.read_csv(path, compression='gzip', index_col=0,header=None, sep='\t', quotechar='"')
                df.columns = [fname.split(".")[0]]
                sample_dfs.append(df)

    expressionData = pd.concat(sample_dfs,axis=1)
    return expressionData

def entropy(vector):
    data = np.array(vector)
    #hist = np.histogram(data,bins=50,)[0]
    hist = np.histogram(data[~np.isnan(data)],bins=50,)[0]
    length = len(hist)

    if length <= 1:
        return 0

    counts = np.bincount(hist)
    probs = [float(i)/length for i in counts]
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute standard entropy.
    for i in probs:
        if i >0:
            ent -= float(i)*np.log(i)
    return ent

def quantile_norm(df,axis=1):
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    if axis == 1:
        array = np.array(df)

        ranked_array = np.zeros(array.shape)
        for i in range(0,array.shape[0]):
            ranked_array[i,:] = rankdata(array[i,:],method='min') - 1

        sorted_array = np.zeros(array.shape)
        for i in range(0,array.shape[0]):
            sorted_array[i,:] = np.sort(array[i,:])

        qn_values = np.nanmedian(sorted_array,axis=0)

        quant_norm_array = np.zeros(array.shape)
        for i in range(0,array.shape[0]):
            for j in range(0,array.shape[1]):
                #quant_norm_array[i,j] = qn_values[int(ranked_array[i,j])]
                quant_norm_array[i,j] = qn_values[ranked_array[i,j].astype(np.int64)]

        quant_norm = pd.DataFrame(quant_norm_array)
        quant_norm.columns = list(df.columns)
        quant_norm.index = list(df.index)

    if axis == 0:
        array = np.array(df)

        ranked_array = np.zeros(array.shape)
        for i in range(0,array.shape[1]):
            ranked_array[:,i] = rankdata(array[:,i],method='min') - 1

        sorted_array = np.zeros(array.shape)
        for i in range(0,array.shape[1]):
            sorted_array[:,i] = np.sort(array[:,i])

        qn_values = np.nanmedian(sorted_array,axis=1)

        quant_norm_array = np.zeros(array.shape)
        for i in range(0,array.shape[0]):
            for j in range(0,array.shape[1]):
                #quant_norm_array[i,j] = qn_values[int(ranked_array[i,j])]
                quant_norm_array[i,j] = qn_values[ranked_array[i,j].astype(np.int64)]

        quant_norm = pd.DataFrame(quant_norm_array)
        quant_norm.columns = list(df.columns)
        quant_norm.index = list(df.index)

    return quant_norm

def transformFPKM(expressionData,fpkm_threshold=1,minFractionAboveThreshold=0.5,highlyExpressed=False,quantile_normalize=False):

    median = np.median(np.median(expressionData,axis=1))
    expDataCopy = expressionData.copy()
    expDataCopy[expDataCopy<fpkm_threshold]=0
    expDataCopy[expDataCopy>0]=1
    cnz = np.count_nonzero(expDataCopy,axis=1)
    keepers = np.where(cnz>=int(minFractionAboveThreshold*expDataCopy.shape[1]))[0]
    threshold_genes = expressionData.index[keepers]
    expDataFiltered = expressionData.loc[threshold_genes,:]

    if highlyExpressed is True:
        median = np.median(np.median(expDataFiltered,axis=1))
        expDataCopy = expDataFiltered.copy()
        expDataCopy[expDataCopy<median]=0
        expDataCopy[expDataCopy>0]=1
        cnz = np.count_nonzero(expDataCopy,axis=1)
        keepers = np.where(cnz>=int(0.5*expDataCopy.shape[1]))[0]
        median_filtered_genes = expDataFiltered.index[keepers]
        expDataFiltered = expressionData.loc[median_filtered_genes,:]

    if quantile_normalize is True:
        expDataFiltered = quantile_norm(expDataFiltered,axis=0)

    finalExpData = pd.DataFrame(np.log2(expDataFiltered+1))
    finalExpData.index = expDataFiltered.index
    finalExpData.columns = expDataFiltered.columns

    return finalExpData

def preProcessTPM(tpm):
    cutoff = stats.norm.ppf(0.00001)

    tmp_array_raw = np.array(tpm)
    keep = []
    keepappend = keep.append
    for i in range(0,tmp_array_raw.shape[0]):
        if np.count_nonzero(tmp_array_raw[i,:]) >= round(float(tpm.shape[1])*0.5):
            keepappend(i)

    tpm_zero_filtered = tmp_array_raw[keep,:]
    tpm_array = np.array(tpm_zero_filtered)
    positive_medians = []

    for i in range(0,tpm_array.shape[1]):
        tmp1 = tpm_array[:,i][tpm_array[:,i]>0]
        positive_medians.append(np.median(tmp1))

    # 2^10 - 1 = 1023
    scale_factors = [float(1023)/positive_medians[i] for i in range(0,len(positive_medians))]

    tpm_scale = np.zeros(tpm_array.shape)
    for i in range(0,tpm_scale.shape[1]):
        tpm_scale[:,i] = tpm_array[:,i]*scale_factors[i]

    tpm_scale_log2 = np.zeros(tpm_scale.shape)
    for i in range(0,tpm_scale_log2.shape[1]):
        tpm_scale_log2[:,i] = np.log2(tpm_scale[:,i]+1)

    tpm_filtered_df = pd.DataFrame(tpm_scale_log2)
    tpm_filtered_df.columns = list(tpm.columns)
    tpm_filtered_df.index = list(np.array(tpm.index)[keep])

    qn_tpm_filtered = quantile_norm(tpm_filtered_df,axis=0)
    qn_tpm = quantile_norm(qn_tpm_filtered,axis=1)

    qn_tpm_array = np.array(qn_tpm)

    tpm_z = np.zeros(qn_tpm_array.shape)
    for i in range(0,tpm_z.shape[0]):
        tmp = qn_tpm_array[i,:][qn_tpm_array[i,:]>0]
        mean = np.mean(tmp)
        std = np.std(tmp)
        for j in range(0,tpm_z.shape[1]):
            tpm_z[i,j] = float(qn_tpm_array[i,j] - mean)/std
            if tpm_z[i,j] < -4:
                tpm_z[i,j] = cutoff

    tpm_entropy = []
    for i in range(0,tpm_z.shape[0]):
        tmp = entropy(tpm_z[i,:])
        tpm_entropy.append(tmp)

    tpmz_df = pd.DataFrame(tpm_z)
    tpmz_df.columns = list(tpm.columns)
    tpmz_df.index = list(np.array(tpm.index)[keep])


    ent = pd.DataFrame(tpm_entropy)
    ent.index = list(tpmz_df.index)
    ent.columns = ['entropy']

    tpm_ent_df = pd.concat([tpmz_df,ent],axis=1)

    tpm_entropy_sorted = tpm_ent_df.sort_values(by='entropy',ascending=False)

    tmp = tpm_entropy_sorted[tpm_entropy_sorted.loc[:,'entropy']>=0]
    tpm_select = tmp.iloc[:,0:-1]

    return tpm_select

def standardizeData(df):
    zscoreDf = zscore(df)
    qn_tpm_0 = quantile_norm(zscoreDf,axis=0)
    qn_tpm_1 = quantile_norm(qn_tpm_0,axis=1)
    return qn_tpm_1

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

def correct_batch_effects(df: pd.DataFrame, do_preprocess_tpm: bool):
    zscoredExpression = zscore(df)
    means = []
    stds = []
    for i in range(zscoredExpression.shape[1]):
        mean = np.mean(zscoredExpression.iloc[:,i])
        std = np.std(zscoredExpression.iloc[:,i])
        means.append(mean)
        stds.append(std)

    if do_preprocess_tpm and np.std(means) >= 0.15:
        zscoredExpression = preProcessTPM(df)

    return zscoredExpression


def preprocess(filename: str, mapfile_path: str, do_preprocess_tpm: bool=True):
    raw_expression = read_file_to_df(filename)
    raw_expression_zero_filtered = remove_null_rows(raw_expression)
    zscored_expression = correct_batch_effects(raw_expression_zero_filtered, do_preprocess_tpm)

    if mapfile_path is not None:
        exp_data, conversion_table = convert_ids_orig(zscored_expression, mapfile_path)
        return exp_data, conversion_table
    else:
        return zscored_expression

# =============================================================================
# Functions used for clustering
# =============================================================================

def pearson_array(array, vector):
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


def get_axes(clusters, expressionData, random_state):
    axes = {}
    for key in list(clusters.keys()):
        genes = clusters[key]
        fpc = PCA(1)
        principalComponents = fpc.fit_transform(expressionData.loc[genes,:].T)
        axes[key] = principalComponents.ravel()
    return axes

def FrequencyMatrix(matrix,overExpThreshold = 1):

    final_index = None
    if type(matrix) == pd.core.frame.DataFrame:
        final_index = matrix.index
        matrix = np.array(matrix)

    index = np.arange(matrix.shape[0])

    matrix[matrix<overExpThreshold] = 0
    matrix[matrix>0] = 1

    frequency_dictionary = {name:[] for name in index}

    for column in range(matrix.shape[1]):
        hits = np.where(matrix[:,column]>0)[0]
        geneset = index[hits]
        for name in geneset:
            frequency_dictionary[name].extend(geneset)

    fm = np.zeros((len(index),len(index)))
    for key in list(frequency_dictionary.keys()):
        tmp = sorted(frequency_dictionary[key])
        if len(tmp) == 0:
            continue
        count = Counter(tmp)
        results_ = np.vstack(list(count.items()))
        fm[key,results_[:,0]] = results_[:,1]/float(count[key])

    fm_df = pd.DataFrame(fm)

    if final_index is not None:
        fm_df.index = final_index
        fm_df.columns = final_index

    return fm_df


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
        maxSum = df.index[np.argmax(np.array(sumDf1))]
        hits = np.where(df.loc[maxSum]>0)[0]
        hitIndex = sorted(df.index[hits])
        block = df.loc[hitIndex,hitIndex]
        blockSum = block.sum(axis=1)
        coreBlock = sorted(blockSum.index[np.where(blockSum>=np.median(blockSum))[0]])
        remainder = sorted(set(df.index)-set(coreBlock))
        frequencyClusters.append(coreBlock)
        if len(remainder) == 0 or len(coreBlock) == 1:
            return sorted(frequencyClusters)
        df = df.loc[remainder,remainder]
    if returnAll:
        frequencyClusters.append(remainder)
    return sorted(frequencyClusters)

def remix(df,frequencyClusters):
    finalClusters = []
    for cluster in frequencyClusters:
        sliceDf = df.loc[cluster,:]
        sumSlice = sliceDf.sum(axis=0)
        cut = min(0.8,np.percentile(sumSlice.loc[cluster]/float(len(cluster)),90))
        minGenes = max(4,cut*len(cluster))
        keepers = list(sliceDf.columns[np.where(sumSlice>=minGenes)[0]])
        keepers = list(set(keepers)|set(cluster))
        finalClusters.append(sorted(keepers))
        finalClusters.sort(key = lambda s: -len(s))
    return finalClusters

def decompose(geneset,expressionData,minNumberGenes=6,pct_threshold=80):
    fm = FrequencyMatrix(expressionData.loc[geneset,:])
    tst = np.multiply(fm,fm.T)
    tst[tst<np.percentile(tst,pct_threshold)]=0
    tst[tst>0]=1
    unmix_tst = unmix(tst)
    unmixedFiltered = [i for i in unmix_tst if len(i)>=minNumberGenes]
    return unmixedFiltered

def recursive_decomposition(geneset,expressionData,minNumberGenes=6,pct_threshold=80):

    unmixedFiltered = decompose(geneset,expressionData,minNumberGenes,pct_threshold)
    if len(unmixedFiltered) == 0:
        return []
    shortSets = [i for i in unmixedFiltered if len(i)<50]
    longSets = [i for i in unmixedFiltered if len(i)>=50]
    if len(longSets)==0:
        return unmixedFiltered
    for ls in longSets:
        unmixedFiltered = decompose(ls,expressionData,minNumberGenes,pct_threshold)
        if len(unmixedFiltered)==0:
            continue
        shortSets.extend(unmixedFiltered)
    return shortSets

def merge_cluster_list(cix,dist_mat,min_corr = 0.5):
    corr_mat = []
    for i in range(len(cix)):
        corr_mat.append(list(dist_mat.loc[cix[i],:].mean(axis=0)))

    corr_mat = np.vstack(corr_mat)

    pairs = []
    for i in range(corr_mat.shape[0]):
        tmp_pairs = [i]
        for j in range(i,corr_mat.shape[0]):
            if len(pairs) > 0:
                if j in np.hstack(pairs):
                    continue
            r, _ = stats.pearsonr(corr_mat[i,:],corr_mat[j,:])
            if r>min_corr:
                if i != j:
                    tmp_pairs.append(j)

        if len(pairs) > 0:
            if i in np.hstack(pairs):
                continue
        pairs.append(tmp_pairs)

    merged_clusters = []
    for pair in pairs:
        tmp_ext = []
        for ix in pair:
            tmp_ext.extend(cix[ix])
        merged_clusters.append(tmp_ext)

    corr_mat = []
    for i in range(len(merged_clusters)):
        corr_mat.append(list(dist_mat.loc[merged_clusters[i],:].mean(axis=0)))
    corr_mat = np.vstack(corr_mat)

    best_matches = []
    tf_ix = []
    for i in range(dist_mat.shape[0]):
        tmp_corr = pearson_array(corr_mat,np.array(dist_mat.iloc[i,:]))
        if np.amax(tmp_corr) < 0.1:
            continue
        if np.isnan(np.amax(tmp_corr)):
            continue
        best_matches.append(np.argmax(tmp_corr))
        tf_ix.append(dist_mat.index[i])

    best_matches_df = pd.DataFrame(best_matches)
    best_matches_df.index = tf_ix
    best_matches_df.columns = ['cluster']

    final_merged_clusters = []
    for i in list(range(len(best_matches_df.cluster.unique()))):
        tmp_hits = list(best_matches_df.index[np.where(best_matches_df.cluster==i)[0]])
        final_merged_clusters.append(tmp_hits)

    final_merged_clusters.sort(key = lambda s: -len(s))

    return final_merged_clusters


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
    for key in list(dict_.keys()):
        newSet = iterativeCombination(dict_,key,iterations=25)
        if newSet not in decomposedSets:
            decomposedSets.append(newSet)
    return decomposedSets

def combineClusters(axes,clusters,threshold=0.925):

    if len(axes) <=1:
        return clusters

    combineAxes = {}
    filterKeys = np.array(list(axes.keys()))
    axesMatrix = np.vstack([axes[i] for i in filterKeys])

    for key in filterKeys:
        axis = axes[key]
        pearson = pearson_array(axesMatrix,axis)
        combine = np.where(pearson>threshold)[0]
        combineAxes[key] = filterKeys[combine]

    revisedClusters = {}
    combinedKeys = decomposeDictionaryToLists(combineAxes)
    for keyList in combinedKeys:
        genes = sorted(set(np.hstack([clusters[i] for i in keyList])))
        revisedClusters[len(revisedClusters)] = genes

    return revisedClusters

def reconstruction(decomposedList,expressionData, random_state, threshold=0.925):

    if len(decomposedList) == 0:
        return decomposedList
    if type(decomposedList[0]) is not list:
        if type(decomposedList[0]) is not np.ndarray:
            return decomposedList

    clusters = {i:decomposedList[i] for i in range(len(decomposedList))}
    axes = get_axes(clusters, expressionData, random_state)
    return combineClusters(axes,clusters,threshold)


def recursive_alignment(geneset,expressionData,minNumberGenes=6,
                        pct_threshold=80, random_state=12):
    recDecomp = recursive_decomposition(geneset,expressionData,minNumberGenes,pct_threshold)
    if len(recDecomp) == 0:
        return []

    reconstructed = reconstruction(recDecomp,expressionData, random_state)
    reconstructedList = [reconstructed[i] for i in list(reconstructed.keys()) if len(reconstructed[i]) > minNumberGenes]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList


NUM_PCA_COMPONENTS = 10

def cluster(expressionData, minNumberGenes=6, minNumberOverExpSamples=4, maxSamplesExcluded=0.50, svd_solver="arpack",
            random_state=12, overExpressionThreshold=80, pct_threshold=80):
    df = expressionData.copy()
    maxStep = int(np.round(10*maxSamplesExcluded))
    bestHits = []

    zero = np.percentile(expressionData,0)
    expressionThreshold = np.mean([np.percentile(expressionData.iloc[:, i][expressionData.iloc[:, i]>zero], overExpressionThreshold)
                                   for i in range(expressionData.shape[1])])

    startTimer = time.time()
    #trial = -1
    pca = PCA(NUM_PCA_COMPONENTS, random_state=random_state)

    printProgressBar(0, maxStep, prefix='Progress:', suffix='Complete', length=50)
    for step in range(maxStep):
        #trial += 1
        #progress = (100. / maxStep) * trial
        #print('{:.2f} percent complete'.format(progress))
        genesMapped = []
        bestMapped = []

        #Get the first 10 PCs of df (i.e., the expression subset)
        principalComponents = pca.fit_transform(df.T)
        principalDf = pd.DataFrame(principalComponents)
        principalDf.index = df.columns

        for i in range(NUM_PCA_COMPONENTS):
            pearson = pearson_array(np.array(df), np.array(principalDf[i]))
            if len(pearson) == 0:
                continue
            highpass = max(np.percentile(pearson,95), 0.1)
            lowpass = min(np.percentile(pearson,5), -0.1)
            cluster1 = np.array(sorted(df.index[np.where(pearson > highpass)[0]]))
            cluster2 = np.array(sorted(df.index[np.where(pearson < lowpass)[0]]))

            for clst in [cluster1, cluster2]:
                pdc = recursive_alignment(clst, expressionData=df, minNumberGenes=minNumberGenes,
                                          pct_threshold=pct_threshold, random_state=random_state)
                if len(pdc) == 0:
                    continue
                elif len(pdc) == 1:
                    genesMapped.append(sorted(pdc[0]))
                elif len(pdc) > 1:
                    for j in range(len(pdc) - 1):
                        if len(pdc[j]) > minNumberGenes:
                            genesMapped.append(sorted(pdc[j]))

        genesMapped = sorted(genesMapped)
        try:
            stackGenes = np.hstack(genesMapped)
        except:
            stackGenes = []

        # filter out best-represented genes prior to next iteration
        residualGenes = sorted(set(df.index) - set(stackGenes))
        df = df.loc[residualGenes, :]

        # computationally fast surrogate for passing the overexpressed significance test:
        for ix in range(len(genesMapped)):
            tmpCluster = expressionData.loc[genesMapped[ix],:]
            tmpCluster[tmpCluster < expressionThreshold] = 0
            tmpCluster[tmpCluster > 0] = 1
            sumCluster = np.array(np.sum(tmpCluster, axis=0))
            numHits = np.where(sumCluster > 0.333 * len(genesMapped[ix]))[0]
            bestMapped.append(numHits)
            if len(numHits) > minNumberOverExpSamples:
                bestHits.append(genesMapped[ix])

        # filter out the best-represented samples prior to next iteration
        if len(bestMapped) > 0:
            countHits = Counter(np.hstack(bestMapped))
            ranked = countHits.most_common()
            dominant = [i[0] for i in ranked[0:int(np.ceil(0.1*len(ranked)))]]
            remainder = [i for i in np.arange(df.shape[1]) if i not in dominant]
            df = df.iloc[:, remainder]

        # update progress
        printProgressBar(step + 1, maxStep, prefix='Progress:', suffix='Complete', length=50)

    bestHits.sort(key=lambda s: -len(s))

    stopTimer = time.time()
    print('\ncoexpression clustering completed in {:.2f} minutes'.format((stopTimer-startTimer)/60.))
    return bestHits

def background_df(expressionData):

    low = np.percentile(expressionData,100./3,axis=0)
    high = np.percentile(expressionData,200./3,axis=0)
    evenCuts = zipper([low,high])

    bkgd = expressionData.copy()
    for i in range(bkgd.shape[1]):
        lowCut = evenCuts[i][0]
        highCut = evenCuts[i][1]
        tmp = bkgd.iloc[:,i]
        tmp[tmp >= highCut] = 1
        tmp[tmp <= lowCut] = -1
        tmp[np.abs(tmp) != 1] = 0

    return bkgd

def assignMembership(geneset,background,p=0.05):

    cluster = np.array(background.loc[geneset,:])
    classNeg1 = len(geneset)-np.count_nonzero(cluster+1,axis=0)
    class0 = len(geneset)-np.count_nonzero(cluster,axis=0)
    class1 = len(geneset)-np.count_nonzero(cluster-1,axis=0)
    observations = zipper([classNeg1,class0,class1])

    highpass = stats.binom.ppf(1-p/3.0,len(geneset),1./3)
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

def getClusterScores(regulonModules,background,p=0.05):
    clusterScores = {}
    for key in list(regulonModules.keys()):
        members = assignMembership(regulonModules[key],background,p)
        score = clusterScore(members)
        clusterScores[key] = score
    return clusterScores

def filterCoexpressionDict(coexpressionDict,clusterScores,threshold=0.01):
    filterPoorClusters = np.where(clusterScores>threshold)[0]
    for x in filterPoorClusters:
        del coexpressionDict[x]
    keys = coexpressionDict.keys()
    filteredDict = {str(i):coexpressionDict[keys[i]] for i in range(len(coexpressionDict))}
    return filteredDict

def biclusterMembershipDictionary(revisedClusters,background,label=2,p=0.05):

    background_genes = set(background.index)
    if label == "excluded":
        members = {}
        for key in list(revisedClusters.keys()):
            tmp_genes = list(set(revisedClusters[key])&background_genes)
            if len(tmp_genes)>1:
                assignments = assignMembership(tmp_genes,background,p=p)
            else:
                assignments = [np.array([]) for i in range(background.shape[1])]
            nonMembers = np.array([i for i in range(len(assignments)) if len(assignments[i])==0])
            if len(nonMembers) == 0:
                members[key] = []
                continue
            members[key] = list(background.columns[nonMembers])
        return members

    if label == "included":
        members = {}
        for key in list(revisedClusters.keys()):
            tmp_genes = list(set(revisedClusters[key])&background_genes)
            if len(tmp_genes)>1:
                assignments = assignMembership(tmp_genes,background,p=p)
            else:
                assignments = [np.array([]) for i in range(background.shape[1])]
            included = np.array([i for i in range(len(assignments)) if len(assignments[i])!=0])
            if len(included) == 0:
                members[key] = []
                continue
            members[key] = list(background.columns[included])
        return members

    members = {}
    for key in list(revisedClusters.keys()):
        tmp_genes = list(set(revisedClusters[key])&background_genes)
        if len(tmp_genes)>1:
            assignments = assignMembership(tmp_genes,background,p=p)
        else:
            members[key] = []
            continue
        overExpMembers = np.array([i for i in range(len(assignments)) if label in assignments[i]])
        if len(overExpMembers) ==0:
            members[key] = []
            continue
        members[key] = list(background.columns[overExpMembers])
    return members

def membershipToIncidence(membershipDictionary,expressionData):

    incidence = np.zeros((len(membershipDictionary),expressionData.shape[1]))
    incidence = pd.DataFrame(incidence)
    incidence.index = membershipDictionary.keys()
    incidence.columns = expressionData.columns
    for key in list(membershipDictionary.keys()):
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

def processCoexpressionLists(lists,expressionData, random_state, threshold=0.925):
    reconstructed = reconstruction(lists,expressionData, random_state, threshold)
    reconstructedList = [reconstructed[i] for i in reconstructed.keys()]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList


def reviseInitialClusters(clusterList, expressionData, random_state=12, threshold=0.925):
    coexpressionLists = processCoexpressionLists(clusterList, expressionData, random_state, threshold)
    coexpressionLists.sort(key= lambda s: -len(s))

    for iteration in range(5):
        previousLength = len(coexpressionLists)
        coexpressionLists = processCoexpressionLists(coexpressionLists, expressionData,
                                                     random_state, threshold)
        newLength = len(coexpressionLists)
        if newLength == previousLength:
            break

    coexpressionLists.sort(key= lambda s: -len(s))
    coexpressionDict = {str(i):list(coexpressionLists[i]) for i in range(len(coexpressionLists))}

    return coexpressionDict

# =============================================================================
# Functions used for mechanistic inference
# =============================================================================

def regulonDictionary(regulons):
    regulonModules = {}
    #str(i):[regulons[key][j]]}
    df_list = []

    for tf in list(regulons.keys()):
        for key in list(regulons[tf].keys()):
            genes = regulons[tf][key]
            id_ = str(len(regulonModules))
            regulonModules[id_] = regulons[tf][key]
            for gene in genes:
                df_list.append([id_,tf,gene])

    array = np.vstack(df_list)
    df = pd.DataFrame(array)
    df.columns = ["Regulon_ID","Regulator","Gene"]

    return regulonModules, df

def regulonIdToRegulator(regulonDf):

    idIndexedRegulonDf = regulonDf.copy()
    idIndexedRegulonDf.index = regulonDf["Regulon_ID"]
    reg_index = idIndexedRegulonDf.index
    unique_ids = []
    unique_indices = []
    for i in range(idIndexedRegulonDf.shape[0]):
        tmp_ix = reg_index[i]
        if tmp_ix not in unique_ids:
            unique_ids.append(tmp_ix)
            unique_indices.append(i)

    regulonIDtoRegulator = idIndexedRegulonDf.loc[:,"Regulator"]
    regulonIDtoRegulator = pd.DataFrame(regulonIDtoRegulator.iloc[unique_indices])

    return(regulonIDtoRegulator)

def regulonDictToDf(expandedRegulons,regulonIDtoRegulator):

    df_list = []
    for id_ in list(expandedRegulons.keys()):
        genes = expandedRegulons[id_]
        tf = regulonIDtoRegulator.loc[id_,"Regulator"]
        for gene in genes:
            df_list.append([id_,tf,gene])

    array = np.vstack(df_list)
    df = pd.DataFrame(array)
    df.columns = ["Regulon_ID","Regulator","Gene"]

    return df

def regulonExpansion(task):

    from sklearn.metrics import roc_auc_score

    start, stop = task[0]
    eigengenes,regulonModules,regulonDf,expressionData,tfbsdbGenes,overExpressedMembersMatrix,corrThreshold,auc_threshold = task[1]
    eigenarray = np.array(eigengenes)
    regulonIDtoRegulator = regulonIdToRegulator(regulonDf)

    reference_index = np.array(eigengenes.index).astype(str)
    expanded_modules = {key:regulonModules[key] for key in regulonModules.keys()}
    ct = -1
    genes = list(set(list(tfbsdbGenes.keys()))&set(expressionData.index))[start:stop]
    printProgressBar(0, len(genes), prefix='Progress:', suffix='Complete',
                     length=50)
    for gene in genes:
        ct += 1
        #if ct%1000 == 0:
        #    print("Completed {:d} of {:d} iterations".format(ct,stop-start))
        printProgressBar(ct, len(genes), prefix='Progress:', suffix='Complete',
                         length=50)
        pa = pearson_array(eigenarray,np.array(expressionData.loc[gene,:]))
        tfbs = tfbsdbGenes[gene]
        hits = np.where(pa>corrThreshold)[0]
        regulon_hits = reference_index[hits]
        tf_hits = regulonIDtoRegulator.loc[regulon_hits,"Regulator"]
        tf_hits_overlap = list(set(tf_hits)&set(tfbs))

        regulon_id_hits = []
        for i in regulon_hits:
            if regulonIDtoRegulator.loc[i,"Regulator"] in tf_hits_overlap:
                regulon_id_hits.append(i)

        tmp_overX = overExpressedMembersMatrix.loc[np.array(regulon_id_hits).astype(str),:]

        gene_array = np.array(expressionData.loc[gene,:])
        for i in tmp_overX.index:
            class_labels = np.array(tmp_overX.loc[i,:])
            sum_ = sum(class_labels)
            auc = 0
            if sum_ > 0:
                auc = roc_auc_score(np.array(tmp_overX.loc[i,:]),gene_array)
            if auc >= auc_threshold:
                expanded_modules[i].append(gene)

    return expanded_modules

def parallelRegulonExpansion(eigengenes,regulonModules,regulonDf,expressionData,tfbsdbGenes_file,overExpressedMembersMatrix,corrThreshold = 0.25,auc_threshold = 0.70,numCores=5):

    tfbsdbGenes = read_pkl(tfbsdbGenes_file)
    genes = list(set(list(tfbsdbGenes.keys()))&set(expressionData.index))
    taskSplit = splitForMultiprocessing(genes,numCores)
    taskData = (eigengenes, regulonModules, regulonDf, expressionData, tfbsdbGenes, overExpressedMembersMatrix,corrThreshold,auc_threshold)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    output = multiprocess(regulonExpansion,tasks)
    expandedRegulons = condenseOutput(output)
    expandedRegulons = {key:list(set(expandedRegulons[key])) for key in expandedRegulons.keys()}
    return expandedRegulons

def principal_df(dict_, expressionData, regulons=None, subkey='genes',
                 minNumberGenes=8, random_state=12,
                 svd_solver="arpack"):

    pcDfs = []
    setIndex = set(expressionData.index)

    if regulons is not None:
        dict_, df = regulonDictionary(regulons)
    for i in sorted(dict_.keys()):
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
    # returns with sorted index and columns for consistency
    #return principalMatrix.sort_index(axis=0).sort_index(axis=1)
    return principalMatrix


def axisTfs(axesDf,tfList,expressionData,correlationThreshold=0.3):

    axesArray = np.array(axesDf.T)
    if correlationThreshold > 0:
        #print(tfList)
        tfArray = np.array(expressionData.reindex(tfList))
    axes = np.array(axesDf.columns)
    tfDict = {}

    if type(tfList) is list:
        tfs = np.array(tfList)
    elif type(tfList) is not list:
        tfs = np.array(list(tfList))

    if correlationThreshold == 0:
        for axis in range(axesArray.shape[0]):
            tfDict[axes[axis]] = tfs

        return tfDict

    for axis in range(axesArray.shape[0]):
        tfDict_key = axes[axis]
        tfCorrelation = pearson_array(tfArray,axesArray[axis,:])
        tfDict[tfDict_key] = tfs[np.where(np.abs(tfCorrelation)>=correlationThreshold)[0]]

    return tfDict


def zipper(ls):
    zipped = []
    for i in range(len(ls[0])):
        vals = []
        for j in range(len(ls)):
            vals.append(ls[j][i])
        zipped.append(tuple(vals))
    return zipped

def splitForMultiprocessing(vector,cores):

    partition = int(len(vector)/cores)
    remainder = len(vector) - cores*partition
    starts = np.arange(0,len(vector),partition)[0:cores]
    for i in range(remainder):
        starts[cores-remainder+i] = starts[cores-remainder+i] + i

    stops = starts+partition
    for i in range(remainder):
        stops[cores-remainder+i] = stops[cores-remainder+i] + 1

    zipped = zipper([starts,stops])

    return zipped

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


def condenseOutput(output,output_type = dict):

    if output_type is dict:
        results = {}
        for i in range(len(output)):
            resultsDict = output[i]
            keys = list(resultsDict.keys())
            for j in range(len(resultsDict)):
                key = keys[j]
                results[key] = resultsDict[key]
        return results

    elif output_type is not dict:
        import pandas as pd
        results = pd.concat(output,axis=0)

    return results

def tfbsdbEnrichment(task):

    start, stop = task[0]
    allGenes,revisedClusters,tfMap,tfToGenes,p = task[1]
    keys = list(revisedClusters.keys())[start:stop]

    if len(allGenes) == 1:
        population_size = int(allGenes[0])
        clusterTfs = {}
        for key in keys:
            for tf in tfMap[str(key)]:
                hits0TfTargets = tfToGenes[tf]
                hits0clusterGenes = revisedClusters[key]
                overlapCluster = list(set(hits0TfTargets)&set(hits0clusterGenes))
                if len(overlapCluster) <= 1:
                    continue
                pHyper = hyper(population_size,len(hits0TfTargets),len(hits0clusterGenes),len(overlapCluster))
                if pHyper < p:
                    if key not in list(clusterTfs.keys()):
                        clusterTfs[key] = {}
                    clusterTfs[key][tf] = [pHyper,overlapCluster]

    elif len(allGenes) > 1:
        population_size = len(allGenes)
        clusterTfs = {}
        for key in keys:
            for tf in tfMap[str(key)]:
                hits0TfTargets = list(set(tfToGenes[tf])&set(allGenes))
                hits0clusterGenes = revisedClusters[key]
                overlapCluster = list(set(hits0TfTargets)&set(hits0clusterGenes))
                if len(overlapCluster) <= 1:
                    continue
                pHyper = hyper(population_size,len(hits0TfTargets),len(hits0clusterGenes),len(overlapCluster))
                if pHyper < p:
                    if key not in list(clusterTfs.keys()):
                        clusterTfs[key] = {}
                    clusterTfs[key][tf] = [pHyper,overlapCluster]

    return clusterTfs

def mechanisticInference(axes,revisedClusters,expressionData,correlationThreshold=0.3,numCores=5,p=0.05,
                         database_path=None):
    #print('Running mechanistic inference')
    """
    if override_database is None:
        tfToGenesPath = os.path.join(dataFolder,"network_dictionaries",database)
        tfToGenes = read_pkl(tfToGenesPath)
    elif override_database is not None:
        tfToGenes = override_database
    """
    if database_path.endswith(".pkl"):
        tfToGenes = read_pkl(database_path)
    elif database_path.endswith(".json"):
        with open(database_path) as infile:
            tfToGenes = json.load(infile)
    else:
        raise("unknown database format for tfs_to_genes")

    if correlationThreshold <= 0:
        allGenes = [int(len(expressionData.index))]
    elif correlationThreshold > 0:
        allGenes = list(expressionData.index)

    tfs = list(tfToGenes.keys())
    tfMap = axisTfs(axes,tfs,expressionData,correlationThreshold=correlationThreshold)
    taskSplit = splitForMultiprocessing(list(revisedClusters.keys()),numCores)
    tasks = [[taskSplit[i],(allGenes,revisedClusters,tfMap,tfToGenes,p)] for i in range(len(taskSplit))]
    tfbsdbOutput = multiprocess(tfbsdbEnrichment,tasks)
    mechanisticOutput = condenseOutput(tfbsdbOutput)

    return mechanisticOutput

def coincidenceMatrix(coregulationModules,key,freqThreshold = 0.333):

    tf = list(coregulationModules.keys())[key]
    subRegulons = coregulationModules[tf]
    srGenes = list(set(np.hstack([subRegulons[i] for i in subRegulons.keys()])))

    template = pd.DataFrame(np.zeros((len(srGenes),len(srGenes))))
    template.index = srGenes
    template.columns = srGenes
    for key in list(subRegulons.keys()):
        genes = subRegulons[key]
        template.loc[genes,genes]+=1
    trace = np.array([template.iloc[i,i] for i in range(template.shape[0])]).astype(float)
    normDf = ((template.T)/trace).T
    normDf[normDf<freqThreshold]=0
    normDf[normDf>0]=1

    return normDf

def getCoregulationModules(mechanisticOutput):

    coregulationModules = {}
    for i in list(mechanisticOutput.keys()):
        for key in list(mechanisticOutput[i].keys()):
            if key not in list(coregulationModules.keys()):
                coregulationModules[key] = {}
            genes = mechanisticOutput[i][key][1]
            coregulationModules[key][i] = genes
    return coregulationModules


#Changed > to >= in minNumberGenes
def getRegulons(coregulationModules,minNumberGenes=5,freqThreshold = 0.333):

    regulons = {}
    keys = list(coregulationModules.keys())
    for i in range(len(keys)):
        tf = keys[i]
        normDf = coincidenceMatrix(coregulationModules,key=i,freqThreshold = freqThreshold)
        unmixed = unmix(normDf)
        remixed = remix(normDf,unmixed)
        if len(remixed)>0:
            for cluster in remixed:
                if len(cluster)>=minNumberGenes:
                    if tf not in list(regulons.keys()):
                        regulons[tf] = {}
                    regulons[tf][len(regulons[tf])] = cluster
    return regulons


def getCoexpressionModules(mechanisticOutput):
    coexpressionModules = {}
    for i in list(mechanisticOutput.keys()):
        genes = list(set(np.hstack([mechanisticOutput[i][key][1] for key in mechanisticOutput[i].keys()])))
        coexpressionModules[i] = genes
    return coexpressionModules

def f1Regulons(coregulationModules,minNumberGenes=5,freqThreshold = 0.1):

    regulons = {}
    keys = list(coregulationModules.keys())
    for i in range(len(keys)):
        tf = keys[i]
        normDf = coincidenceMatrix(coregulationModules,key=i,freqThreshold = freqThreshold)
        remixed = f1Binary(normDf)
        if len(remixed)>0:
            for cluster in remixed:
                if len(cluster)>minNumberGenes:
                    if tf not in list(regulons.keys()):
                        regulons[tf] = {}
                    regulons[tf][len(regulons[tf])] = cluster
    return regulons

# =============================================================================
# Functions used for post-processing mechanistic inference
# =============================================================================

def convertDictionary(dict_,conversionTable):
    converted = {}
    for i in list(dict_.keys()):
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

def generateInputForFIRM(revisedClusters,saveFile):

    identifier_mapping = pd.read_csv(os.path.join(os.path.split(os.getcwd())[0],"data","identifier_mappings.txt"),sep="\t")
    identifier_mapping_entrez = identifier_mapping[identifier_mapping.Source == "Entrez Gene ID"]
    identifier_mapping_entrez.index = identifier_mapping_entrez.iloc[:,0]

    Gene = []
    Group = []
    identified_genes = set(identifier_mapping_entrez.index)
    for key in list(revisedClusters.keys()):
        cluster = revisedClusters[key]
        tmp_genes = list(set(cluster)&identified_genes)
        tmp_entrez = list(identifier_mapping_entrez.loc[tmp_genes,"Name"])
        tmp_group = [key for i in range(len(tmp_entrez))]
        Gene.extend(tmp_entrez)
        Group.extend(tmp_group)

    firm_df = pd.DataFrame(np.vstack([Gene,Group]).T)
    firm_df.columns = ["Gene","Group"]
    firm_df.to_csv(saveFile,index=None,sep="\t")

    return firm_df

# =============================================================================
# Functions used for inferring sample subtypes
# =============================================================================

def sampleCoincidenceMatrix(dict_,freqThreshold = 0.333,frequencies=False):

    keys = list(dict_.keys())
    lists = [dict_[key] for key in keys]
    samples = list(set(np.hstack(lists)))

    frequency_dictionary = {name:[] for name in samples}
    for key in keys:
        hits = dict_[key]
        for name in hits:
            frequency_dictionary[name].extend(hits)

    labels = list(frequency_dictionary.keys())
    fm = pd.DataFrame(np.zeros((len(labels),len(labels))))
    fm.index = labels
    fm.columns = labels

    for i in range(len(labels)):
        key = labels[i]
        tmp = frequency_dictionary[key]
        if len(tmp) == 0:
            continue
        count = Counter(tmp)
        results_ = np.vstack(list(count.items()))
        fm.loc[key,results_[:,0]] = np.array(results_[:,1]).astype(float)/int(count[key])

    if frequencies is not False:
        return fm

    fm[fm<freqThreshold]=0
    fm[fm>0]=1

    return fm

def matrix_to_dictionary(matrix,threshold=0.5):
    primaryDictionary = {key:matrix.columns[np.where(matrix.loc[key,:]>=threshold)[0]] for key in matrix.index}
    return primaryDictionary


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
    probeSample = similarityMatrix.index[np.argmax(np.array(similarityMatrix.sum(axis=1)))]
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
        pred_non_membs_list = sorted(predictedNonMembers)

        tmp_overlap = intersect(predictedMembers,similarityMatrix.columns)
        sumSlice = np.sum(similarityMatrix.loc[:,tmp_overlap],axis=1)/float(len(tmp_overlap))
        members = set(similarityMatrix.index[np.where(sumSlice>0.8)[0]])

        if members==predictedMembers:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers)==0:
                break
            #similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            similarityMatrix = similarityMatrix.loc[pred_non_membs_list, pred_non_membs_list]
            probeSample = similarityMatrix.sum(axis=1).idxmax()
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
            #similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            similarityMatrix = similarityMatrix.loc[pred_non_membs_list, pred_non_membs_list]
            probeSample = similarityMatrix.sum(axis=1).idxmax()
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
    plt.title(title,fontsize=fontsize+2)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    if savefig is not None:
        plt.savefig(savefig,bbox_inches="tight")
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


def centroidExpansion(classes,sampleMatrix,f1Threshold = 0.3,returnCentroids=None):
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

def getCentroids(classes,sampleMatrix):
    centroids = []
    for i in range(len(classes)):
        clusterComponents = sampleMatrix.loc[:,classes[i]]
        class1 = np.mean(clusterComponents,axis=1)
        centroid = pd.DataFrame(class1)
        centroid.columns = [i]
        centroid.index = sampleMatrix.index
        centroids.append(centroid)
    return pd.concat(centroids,axis=1)

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

def orderMembership(centroidMatrix,membershipMatrix,mappedClusters,ylabel="",resultsDirectory=None,showplot=False):

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

    if showplot is False:
        return ordered_matrix

    if showplot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except:
            pass
        ax.imshow(ordered_matrix,cmap='viridis',aspect="auto")
        ax.grid(False)

        plt.title(ylabel.split("s")[0]+"Activation",fontsize=16)
        plt.xlabel("Samples",fontsize=14)
        plt.ylabel(ylabel,fontsize=14)
        if resultsDirectory is not None:
            plt.savefig(os.path.join(resultsDirectory,"binaryActivityMap.pdf"))
    return ordered_matrix

def plotDifferentialMatrix(overExpressedMembersMatrix,underExpressedMembersMatrix,orderedOverExpressedMembers,cmap="viridis",aspect="auto",saveFile=None,showplot=False):
    differentialActivationMatrix = overExpressedMembersMatrix-underExpressedMembersMatrix
    orderedDM = differentialActivationMatrix.loc[orderedOverExpressedMembers.index,orderedOverExpressedMembers.columns]

    if showplot is False:

        return orderedDM

    elif showplot is True:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        try:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except:
            pass

        ax.imshow(orderedDM,cmap=cmap,vmin=-1,vmax=1,aspect=aspect)
        ax.grid(False)
        if saveFile is not None:
            plt.ylabel("Regulons",fontsize=14)
            plt.xlabel("Samples",fontsize=14)
            ax.grid(False)
            plt.savefig(saveFile,bbox_inches="tight")

    return orderedDM

def kmeans(df,numClusters,random_state=None):

    if random_state is not None:
        # Number of clusters
        kmeans = KMeans(n_clusters=numClusters,random_state=random_state, n_init=10)

    elif random_state is None:
        # Number of clusters
        kmeans = KMeans(n_clusters=numClusters, n_init=10)

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
    # sklearn.Kmeans() can throw a lot of warnings, which can be ignored
    warnings.simplefilter("ignore")

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
        if len(patients)<= minClusterSize_x:
            x_clusters.append(patients)
            continue

        if allow_singletons is not True:
            if len(patients)<= 2*minClusterSize_x:
                x_clusters.append(patients)
                continue

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
            clusters_x, labels_x, centroids_x = kmeans(df,numClusters=top_hit,random_state=random_state)

            clusters_x.sort(key=lambda s: -len(s))
            x_clusters.append(list(clusters_x))
        elif len(sil_scores) == 0:
            x_clusters.append(patients)
    try:
        micro_states = []
        for i in range(len(x_clusters)):
            if len(x_clusters[i])>0:
                if type(x_clusters[i][0]) is not str:
                    for j in range(len(x_clusters[i])):
                        micro_states.append(x_clusters[i][j])
                elif type(x_clusters[i][0]) is str:
                    micro_states.append(x_clusters[i])

        order_x = np.hstack(micro_states)
        fig = plt.figure(figsize=(7,7))
        ax = fig.gca()
        ax.imshow(dfr.loc[order_y,order_x],cmap="bwr",vmin=-1,vmax=1)
        ax.set_aspect(dfr.shape[1]/float(dfr.shape[0]))
        ax.grid(False)
        ax.set_ylabel("Regulons",fontsize=14)
        ax.set_xlabel("Samples",fontsize=14)
        if saveFile is not None:
            plt.savefig(saveFile,bbox_inches="tight")

        return y_clusters, micro_states

    except:
        pass

    return y_clusters, x_clusters

def cluster_features(dfr,clusterList,minClusterSize_x = 5,minClusterSize_y = 5,
                    max_groups = 50,allow_singletons = False,random_state = 12):
    t1 = time.time()

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
            clusters_y, labels_y, centroids_y = kmeans(df,numClusters=top_hit,random_state=random_state)
            clusters_y.sort(key=lambda s: -len(s))
            y_clusters.append(list(clusters_y))

        elif len(sil_scores) == 0:
            y_clusters.append(regs)

    # order cluster groups for visual appeal in heatmap
    ordered_groups = []
    for s in range(len(clusterList)):
        group_sums = []
        for g in range(len(y_clusters)):
            sumsum = dfr.loc[np.hstack(y_clusters[g]),clusterList[s]].sum().sum()
            group_sums.append(sumsum)
        ogs = np.argsort(-np.array(group_sums))
        for o in ogs:
            if o not in ordered_groups:
                ordered_groups.append(o)
                break

    arranged_groups = [y_clusters[i] for i in ordered_groups]

    # convert complex list into simple list
    extracted_lists = []
    for gr in range(len(arranged_groups)):
        g_type = type(arranged_groups[gr][0])
        if g_type is not str:
            for lst in arranged_groups[gr]:
                extracted_lists.append(list(lst))
        elif g_type is str:
            extracted_lists.append(arranged_groups[gr])

    extracted_lists

    t2 = time.time()
    print("Completed clustering in {:.2f} minutes".format(float(t2-t1)/60))

    return extracted_lists, y_clusters

def intersect(x,y):
    return list(set(x)&set(y))

def setdiff(x,y):
    return list(set(x)-set(y))

def union(x,y):
    return list(set(x)|set(y))

def sample(x,n,replace=True):
    from numpy.random import choice
    return choice(x,n,replace=replace)

def train_test(x,y,names=None):

    # identify class labels
    class_0 = np.where(y<=0)[0]
    class_1 = np.where(y>0)[0]

    # define class lengths
    n_class_0 = len(class_0)
    n_class_1 = len(class_1)

    # bootstrap class labels
    bootstrap_train_0 = list(set(sample(class_0,n_class_0,replace = True)))
    bootstrap_test_0 = setdiff(class_0,bootstrap_train_0)
    if len(bootstrap_test_0) == 0:
        bootstrap_train_0 = list(set(sample(class_0,n_class_0,replace = True)))
        bootstrap_test_0 = setdiff(class_0,bootstrap_train_0)

    bootstrap_train_1 = list(set(sample(class_1,n_class_1,replace = True)))
    bootstrap_test_1 = setdiff(class_1,bootstrap_train_1)
    if len(bootstrap_test_1) == 0:
        bootstrap_train_1 = list(set(sample(class_1,n_class_1,replace = True)))
        bootstrap_test_1 = setdiff(class_1,bootstrap_train_1)

    # prepare bootstrap training and test sets
    train_rows = np.hstack([bootstrap_train_0,
                    bootstrap_train_1])

    test_rows = np.hstack([bootstrap_test_0,
                   bootstrap_test_1])

    x_train = x[:,train_rows]
    x_test = x[:,test_rows]
    y_train = y[train_rows]
    y_test = y[test_rows]

    if names is None:
        split = {"x_train":x_train,
                 "x_test":x_test,
                 "y_train":y_train,
                 "y_test":y_test
                }

    elif names is not None:
        train_names = np.array(names)[train_rows]
        test_names = np.array(names)[test_rows]
        split = {"x_train":x_train,
                 "x_test":x_test,
                 "y_train":y_train,
                 "y_test":y_test,
                 "names_train":train_names,
                 "names_test":test_names
                }
    return split


def univariate_comparison(subtypes,srv,expressionData,network_activity_diff,n_iter = 500,hr_prop = 0.30,lr_prop = 0.70, results_directory = None):

    # Instantiate results dictionary
    boxplot_data = {name:{"expression":[],"activity":[]} for name in subtypes.keys()}

    boxplot_data = {name:{"expression":{"aucs":[],"genes":[]},"activity":{"aucs":[],"genes":[]}} for name in subtypes.keys()}

    # Define subtype of patients
    for name in subtypes.keys():
        #print("Iterating subtype "+name)
        # Define subtype patients
        subtype = subtypes[name]

        # Arrange subtype patients by their response status
        ordered_patients = [pat for pat in srv.index if pat in subtype]

        # Define high- and low-risk groups
        risk_groups = [ordered_patients[0:round(len(ordered_patients)*hr_prop)],
                       ordered_patients[round(len(ordered_patients)*(1-lr_prop)):]]

        # Define gene expression and network activity arrays
        x_expression = np.array(expressionData.loc[
            network_activity_diff.index,np.hstack(risk_groups)])

        x_activity = np.array(network_activity_diff.loc[
            network_activity_diff.index,np.hstack(risk_groups)])

        # Arrange response classes
        y = np.hstack([
            np.ones(len(risk_groups[0])),
            np.zeros(len(risk_groups[1]))
        ]).astype(int)

        # Arrange response names
        names = np.hstack(risk_groups)

        # Bootstrap analysis using ROC AUC of individual features (gene expression)
        results_expression = univariate_predictor(x_expression,y,names,
                                            n_iter=n_iter,gene_labels=network_activity_diff.index)

        # Bootstrap analysis using ROC AUC of individual features (network activity)
        results_activity = univariate_predictor(x_activity,y,names,
                                            n_iter=n_iter,gene_labels=network_activity_diff.index)

        # Expression AUCs
        expression_aucs = np.array(results_expression["AUC"]).astype(float)

        # Activity AUCs
        activity_aucs = np.array(results_activity["AUC"]).astype(float)

        # Expression predictors
        prediction_df_exp = pd.DataFrame(np.vstack(Counter(list(results_expression.Gene)).most_common()))
        prediction_df_exp.columns = ["Gene","Frequency"]
        prediction_df_exp.iloc[:,-1] = np.array(prediction_df_exp.iloc[:,-1]).astype(float)/n_iter

        # Activity predictors
        prediction_df_act = pd.DataFrame(np.vstack(Counter(list(results_activity.Gene)).most_common()))
        prediction_df_act.columns = ["Gene","Frequency"]
        prediction_df_act.iloc[:,-1] = np.array(prediction_df_act.iloc[:,-1]).astype(float)/n_iter

        # Save AUCs
        boxplot_data[name]["expression"]["aucs"] = expression_aucs
        boxplot_data[name]["activity"]["aucs"] = activity_aucs

        # Save genes
        boxplot_data[name]["expression"]["genes"] = prediction_df_exp
        boxplot_data[name]["activity"]["genes"] = prediction_df_act

    # Format subtype AUC data for seaborn plotting
    rows = []
    for name in subtypes.keys():
        for i in range(n_iter):
            tmp_exp = [name,"Expression",boxplot_data[name]["expression"]["aucs"][i]]
            tmp_act = [name,"Activity",boxplot_data[name]["activity"]["aucs"][i]]
            rows.append(tmp_exp)
            rows.append(tmp_act)

    boxplot_dataframe = pd.DataFrame(np.vstack(rows))
    boxplot_dataframe.columns = ["Subtype", "Method", "AUC"]
    boxplot_dataframe.loc[:,"AUC"] = pd.to_numeric(boxplot_dataframe.loc[:,"AUC"],errors='coerce')

    sns.set(font_scale=1.5, style="whitegrid")
    fig = plt.figure(figsize=(16,4))

    p = sns.stripplot(data=boxplot_dataframe, x='Subtype', y='AUC',
                      hue="Method",
                      dodge=True,jitter=0.25,size=3,palette='viridis')
    ax = sns.boxplot(data=boxplot_dataframe, x='Subtype', y='AUC',
                     hue="Method",
                      dodge=True,fliersize=0,palette='viridis')

    # Add transparency to colors
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.3))

    handles, labels = p.get_legend_handles_labels()
    l = plt.legend(handles[2:], labels[2:],fontsize=14)

    if results_directory is not None:
        plt.savefig(os.path.join(results_directory,"UnivariateComparison.pdf"),bbox_inches="tight")

    return boxplot_dataframe, boxplot_data, fig

def univariate_survival(subtypes,optimized_survival_parameters,network_activity_diff,srv,results_directory=None,font_scale=1.5):

    sns.set(font_scale=font_scale,style="whitegrid")
    ncols=len(subtypes.keys())
    fig = plt.figure(figsize=(16, 4))
    for s in range(ncols):
        subtype_name = list(optimized_survival_parameters.keys())[s]
        most_predictive_gene = optimized_survival_parameters[subtype_name]['gene']
        threshold = optimized_survival_parameters[subtype_name]['threshold']
        subtype = subtypes[subtype_name]
        ordered_patients = [pat for pat in srv.index if pat in subtype]
        timeline = list(srv.loc[ordered_patients,srv.columns[0]])
        max_time = max(timeline)

        idcs_plus = np.where(network_activity_diff.loc[most_predictive_gene,ordered_patients]>threshold)[0]
        idcs_minus = np.where(network_activity_diff.loc[most_predictive_gene,ordered_patients]<=threshold)[0]
        groups = [np.array(ordered_patients)[idcs_plus],
                 np.array(ordered_patients)[idcs_minus]]

        ax = fig.add_subplot(1, 5, s+1)   #top and bottom left
        kmplot(srv,groups,labels = ["Activated","Inactivated"],
                     xlim_=None,filename=None,color=["r","b"],lw=3,alpha=1,fs=20,subplots=True)

        ax.set_ylim(-0.05,100.05)
        ax.grid(color='w', linestyle='--', linewidth=1)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        #tit = gene_conversion(most_predictive_gene,list_symbols=True)[0]
        #ax.set_title(tit)

        timeline = list(srv.loc[ordered_patients,srv.columns[0]])
        max_time = max(timeline)
        ax.set_xticks(np.arange(0, max_time, 500))
        ax.set_xlabel(subtype_name)
        if s == 0:
            ax.set_ylabel("Progression-free (%)")
            #ax.set_yticklabels(np.arange(-20, 120, 20))
            yticks = np.arange(-20, 120, 20)
            yticklabels = list(map(str, yticks))
            ax.set_yticks(yticks, labels=yticklabels)

        if s>0:
            ax.set_yticklabels("")

    if results_directory is not None:
        plt.savefig(os.path.join(results_directory,"UnivariateSurvival.pdf"),bbox_inches="tight")

    return fig


def composite_survival_figure(univariate_comparison_df,subtypes,
                             optimized_survival_parameters,network_activity_diff,
                             expressionData,srv,gene_clusters,states,
                             results_directory=None,gene_names=None):

    import seaborn as sns
    # Instantiate figure
    fig3 = plt.figure(constrained_layout=True,figsize=(16,12))
    sns.set(font_scale=1.5,style="whitegrid")

    # Heatmaps
    gs = fig3.add_gridspec(4, 5)
    f3_ax0 = gs[0:2,:].subgridspec(1, 2)

    # gene expression
    f3_ax00 = fig3.add_subplot(f3_ax0[0, 0])
    f3_ax00.imshow(expressionData.loc[np.hstack(gene_clusters),np.hstack(states)],
               cmap='bwr',aspect=0.1,vmin=-2,vmax=2,interpolation='none')
    yticks = list(range(-1,7))
    yticklabels = list(map(str, yticks))
    f3_ax00.set_yticks(yticks, labels=yticklabels)
    f3_ax00.set_ylabel("Genes (thousands)")
    f3_ax00.set_title("Gene expression")
    f3_ax00.set_xlabel("Patients")
    plt.grid(False)

    # network activity
    f3_ax01 = fig3.add_subplot(f3_ax0[0, 1])
    f3_ax01.imshow(network_activity_diff.loc[np.hstack(gene_clusters),np.hstack(states)],
               cmap='bwr',aspect=0.1,interpolation='none')
    f3_ax01.set_xlabel("Patients")
    f3_ax01.set_title("Network activity")
    f3_ax01.set_yticks(yticks, labels=yticklabels)
    f3_ax01.set_ylabel("Genes (thousands)")
    plt.grid(False)

    # Boxplots
    f3_ax1 = fig3.add_subplot(gs[2, :])
    f3_ax1.set_xlabel("")

    sns.stripplot(data=univariate_comparison_df, x='Subtype', y='AUC',hue="Method",
                      dodge=True,jitter=0.25,size=3,palette={"Expression":'#0055b3',"Activity":'#00C65E'})#,palette='viridis'
    sns.boxplot(data=univariate_comparison_df, x='Subtype', y='AUC',hue="Method",
                      dodge=True,fliersize=0,palette={"Expression":'#0055b3',"Activity":'#00C65E'})

    #Black and white boxplots
    plt.setp(f3_ax1.artists, edgecolor = 'k', facecolor='w')
    plt.setp(f3_ax1.lines, color='k')

    handles, labels = f3_ax1.get_legend_handles_labels()
    plt.legend(handles[2:], labels[2:],fontsize=14,loc='best')

    # Survival plots
    f3_ax2 = fig3.add_subplot(gs[3, :])
    f3_ax2.set_yticklabels("")
    f3_ax2.set_xticklabels("")
    f3_ax2.grid(False)
    f3_ax2.spines['left'].set_visible(False)
    f3_ax2.spines['top'].set_visible(False)
    f3_ax2.spines['right'].set_visible(False)
    f3_ax2.spines['bottom'].set_visible(False)

    ncols=len(subtypes.keys())
    hr_groups = []
    lr_groups = []

    for s in range(ncols):
        subtype_name = list(optimized_survival_parameters.keys())[s]
        most_predictive_gene = optimized_survival_parameters[subtype_name]['gene']
        threshold = optimized_survival_parameters[subtype_name]['threshold']
        subtype = subtypes[subtype_name]
        ordered_patients = [pat for pat in srv.index if pat in subtype]
        timeline = list(srv.loc[ordered_patients,srv.columns[0]])
        max_time = max(timeline)

        idcs_plus = np.where(network_activity_diff.loc[most_predictive_gene,ordered_patients]>threshold)[0]
        idcs_minus = np.where(network_activity_diff.loc[most_predictive_gene,ordered_patients]<=threshold)[0]
        groups = [np.array(ordered_patients)[idcs_plus],
                 np.array(ordered_patients)[idcs_minus]]

        hr_groups = list(set(hr_groups)|set(np.array(ordered_patients)[idcs_plus]))
        lr_groups = list(set(lr_groups)|set(np.array(ordered_patients)[idcs_minus]))

        ax = fig3.add_subplot(gs[3,s])
        kmplot(srv,groups,labels = ["Activated","Inactivated"],
                     xlim_=None,filename=None,color=['red','blue'],lw=3,alpha=1,fs=20,subplots=True)

        ax.set_ylim(-0.05,100.05)
        ax.grid(color='w', linestyle='--', linewidth=1)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        timeline = list(srv.loc[ordered_patients,srv.columns[0]])
        max_time = max(timeline)
        #ax.set_xticklabels(np.arange(0, round(max_time/30.5), round(500/30.5)))
        xticks = np.arange(0, round(max_time/30.5), round(500/30.5))
        xticklabels = list(map(str, xticks))
        ax.set_xticks(xticks, labels=xticklabels)
        ax.set_xlabel("Weeks")
        if s==0:
            ax.set_ylabel("Progression-free (%)")
            #ax.set_yticklabels(np.arange(-20, 120, 20))

        if s>0:
            ax.set_ylabel("")

        if gene_names is None:
            gene_name = most_predictive_gene
        elif gene_names is not None:
            gene_name = gene_names[s]
        ax.text(0.33, 1.05, subtype_name+" +\n"+gene_name, transform=ax.transAxes,
                size=16)

    #Add letters
    import string
    axs = [f3_ax00,f3_ax01,f3_ax1,f3_ax2]
    for n, ax in enumerate(axs):
        ax.text(-0.1, 1.00, string.ascii_lowercase[n], transform=ax.transAxes,
                size=24, weight='bold')

    if results_directory is not None:
        plt.savefig(os.path.join(results_directory,"UnivariateSurvival.pdf"),bbox_inches="tight")

    lr_groups = list(set(lr_groups)-set(hr_groups))

    return fig3, hr_groups, lr_groups

def optimize_threshold(most_predictive_gene,ordered_patients,network_activity_diff,srv,abs_threshold=None,pct_threshold=None):

    if abs_threshold is None:
        threshold = srv.iloc[int(0.3*srv.shape[0]),:]["GuanScore"]

    elif abs_threshold is not None:
        threshold = srv.iloc[int(abs_threshold*srv.shape[0]),:]["GuanScore"]

    if pct_threshold is not None:
        threshold = np.percentile(np.array(srv.loc[ordered_patients,:]["GuanScore"]),pct_threshold)

    y_true = np.array(np.array(srv.loc[ordered_patients,"GuanScore"])>=threshold).astype(int)

    opt_thr = []
    for thr in np.arange(-1,1,0.01):
        f1_tmp = f1(y_true,np.array(np.array(network_activity_diff.loc[most_predictive_gene,ordered_patients])>thr).astype(int))
        opt_thr.append(f1_tmp)

    #print("F1 score: {:.2f}".format(max(opt_thr)))

    threshold = np.arange(-1,1,0.01)[np.argsort(np.array(opt_thr))[-1]]
    return threshold

def optimize_survival_parameters(univariate_comparison_dict,
                                  network_activity_diff,subtypes,srv,abs_threshold=0.25):

    optimized_survival_parameters = {subtype_name:{'gene':[],'threshold':[]} for subtype_name in subtypes.keys()}

    for subtype_name in list(subtypes.keys()):
        subtype = subtypes[subtype_name]
        ordered_patients = [pat for pat in srv.index if pat in subtype]
        most_predictive_gene = univariate_comparison_dict[subtype_name]["activity"]["genes"].iloc[0,0]

        optimized_survival_parameters[subtype_name]['gene'] = most_predictive_gene
        optimized_survival_parameters[subtype_name]['threshold'] = optimize_threshold(most_predictive_gene,
                       ordered_patients,
                       network_activity_diff,
                       srv,abs_threshold)

    return optimized_survival_parameters


def optimize_parameters_ridge(x,y,names,srv,n_iter=10,show=True,results_directory=None):
    """
    Function to test a range of regularization parameters for Ridge regression.
    """

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import Ridge

    ranges = [np.array(list(range(25,250100,5000))),
              np.array(list(range(25,25100,500))),
              np.array(list(range(1,502,10))),
              np.arange(0.001,1.002,0.02)
             ]
    means = []
    stds = []

    printProgressBar(0, len(ranges), prefix='Progress:', suffix='Complete',
                     length=50)
    for ar in range(len(ranges)):
        a_range = ranges[ar]

        #print("Iteration {:d} of {:d}".format(ar+1,len(ranges)))
        printProgressBar(ar, len(ranges), prefix='Progress:', suffix='Complete',
                         length=50)
        all_curves = []
        for iteration in range(n_iter):
            train_test_dict = train_test(x,y,names)
            curve_mmrf = []
            for a in a_range:

                X = train_test_dict["x_train"].T
                y_gs = np.array(srv.loc[train_test_dict["names_train"],"GuanScore"])
                clf = Ridge(random_state=0,alpha=a,fit_intercept=True)
                clf.fit(X, y_gs)

                y_ = train_test_dict["y_test"]
                decision_function_score = clf.predict(train_test_dict["x_test"].T)
                curve_mmrf.append(roc_auc_score(y_,decision_function_score))

            all_curves.append(curve_mmrf)

        ac_array = np.vstack(all_curves)
        means.append(np.mean(ac_array,axis=0))
        stds.append(np.std(ac_array,axis=0))

    naive_opt = [max(means[i]) for i in range(len(means))]
    max_arg = np.argsort(naive_opt)[-1]
    max_max = max(naive_opt)
    arg_opt = np.where(means[max_arg]==max_max)[0]
    if len(arg_opt) >1:
        arg_opt = arg_opt[0]
    par_opt = float(ranges[max_arg][arg_opt])

    if show is True:
        fig1, axs1 = plt.subplots(nrows=2, ncols=2,sharey=True,figsize=(8,8))
        for i in range(len(ranges)):
            if i < 2:
                j = 0
            elif i >= 2:
                j = 1
            top_curve = means[i]+stds[i]
            mid_curve = means[i]
            bottom_curve = means[i]-stds[i]

            axs1[j,i%2].fill_between(ranges[i],top_curve,bottom_curve,alpha=0.3)
            axs1[j,i%2].plot(ranges[i],mid_curve)
            fig1.text(0.5, 0.06, "Ridge parameter", ha='center',fontsize=14)
            fig1.text(0.02, 0.5, "AUC", va='center', rotation='vertical',fontsize=14)
        if results_directory is not None:
            plt.savefig(os.path.join(results_directory,"Ridge_parameter_optimization.pdf"),bbox_inches="tight")


    print("Optimized parameter: a = {:.3f}\nMean AUC with optimized parameter: {:.3f}".format(par_opt,max_max))
    return par_opt, max_max, means, stds

def ridge(x,y,names,lambda_min,srv,n_iter = 100,plot_label = "Ridge",results_directory = None):
    """
    Return random test set aucs of n_iter bootstraps using Ridge regression.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import Ridge
    aucs = []
    for iteration in range(n_iter):
        if iteration%50 == 0:
            print("Iteration {:d} of {:d}".format(iteration,n_iter))
        train_test_dict = train_test(x,y,names)
        X = train_test_dict["x_train"].T
        y_gs = np.array(srv.loc[train_test_dict["names_train"],"GuanScore"])
        clf = Ridge(random_state=0,alpha=lambda_min,fit_intercept=True) #C=15 MMRF, C=0.5 GSE24080UAMS, C=0.3 GSE19784HOVON65, C=2.5 EMTAB4032
        clf.fit(X, y_gs)

        y_ = train_test_dict["y_test"]
        decision_function_score = clf.predict(train_test_dict["x_test"].T)
        aucs.append(roc_auc_score(y_,decision_function_score))

    plt.figure(figsize=(4,4))
    plt.boxplot(aucs)
    plt.ylabel("AUC",fontsize=20)
    plt.title(plot_label,fontsize=20)

    if results_directory is not None:
        plt.savefig(os.path.join(results_directory,"Ridge_AUC.pdf"),bbox_inches="tight")

    return aucs

def optimize_ridge_model(feature_matrix,survival_matrix,n_iter=250,train_cut_high = 0.20,train_cut_low = 0.50,max_range=5000,range_step=100,savefile=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import Ridge

    # Split MMRF training
    high_risk = survival_matrix.index[0:round(survival_matrix.shape[0]*train_cut_high)]
    low_risk = survival_matrix.index[round(survival_matrix.shape[0]*train_cut_low):]

    test_curves = []

    for iteration in range(n_iter):
        if iteration%min(100,n_iter-1) == 0:
            print('Optimization is {:.2f}% complete'.format(100*float(iteration+1)/n_iter))
        smpl_hr = high_risk
        training_hr = np.random.choice(smpl_hr,len(smpl_hr),replace=True)#np.random.choice(smpl_hr,int(0.8*len(smpl_hr)),replace=False)
        test_hr = setdiff(smpl_hr,training_hr)

        smpl_lr = low_risk
        training_lr = np.random.choice(smpl_lr,len(smpl_lr),replace=True)#np.random.choice(smpl_lr,int(0.8*len(smpl_lr)),replace=False)
        test_lr = setdiff(smpl_lr,training_lr)

        train_set = np.hstack([training_hr,training_lr])
        test_set = np.hstack([test_hr,test_lr])

        curve_test = []
        a_range = np.arange(1,max_range,range_step)

        for a in a_range:
            X = np.array(feature_matrix.loc[:,train_set]).T
            y = np.zeros(len(train_set))
            y[0:len(training_hr)] = 1
            clf = Ridge(random_state=0,alpha=a,fit_intercept=True)
            clf.fit(X, y)

            y = np.zeros(len(test_set))
            y[0:len(test_hr)] = 1
            decision_function_score = clf.predict(np.array(feature_matrix.loc[:,test_set]).T)
            curve_test.append(roc_auc_score(y,decision_function_score))

        test_curves.append(curve_test)

    mean_result = np.mean(np.vstack(test_curves),axis=0)
    sd_result = np.std(np.vstack(test_curves),axis=0)
    alpha_opt = a_range[np.argmax(mean_result)]

    plt.figure(figsize=(8,8))
    plt.plot(a_range,mean_result)
    plt.xlabel("Regularization parameter",fontsize=20)
    plt.ylabel("AUC",fontsize=20)
    plt.title("Parameter optimization",fontsize=20)

    if savefile is not None:
        plt.savefig(savefile,bbox_inches="tight")

    return alpha_opt, mean_result, sd_result, a_range

def optimize_combinatorial_fit(feature_matrix,response,
                               n_iter = 25,a_range=None,savefile=None):
    from sklearn.linear_model import Ridge
    samples = feature_matrix.columns
    test_curves = []
    for iteration in range(n_iter):
        if iteration%min(100,n_iter-1) == 0:
            print('Optimization is {:.2f}% complete'.format(100*float(iteration+1)/n_iter))
        smpl_set = samples
        train_set = np.random.choice(smpl_set,len(smpl_set),replace=True)
        test_set = setdiff(smpl_set,train_set)

        curve_test = []

        for a in a_range:

            X = np.array(feature_matrix.loc[:,train_set]).T
            y = np.array(response.loc[train_set])
            clf = Ridge(random_state=0,alpha=a,fit_intercept=True)
            clf.fit(X, y)

            y = np.array(response.loc[test_set])

            decision_function_score = clf.predict(np.array(feature_matrix.loc[:,test_set]).T)
            curve_test.append(stats.spearmanr(decision_function_score,y)[0])

        test_curves.append(curve_test)

    mean_result = np.mean(np.vstack(test_curves),axis=0)
    sd_result = np.std(np.vstack(test_curves),axis=0)
    alpha_opt = a_range[np.argmax(mean_result)]

    plt.figure(figsize=(8,8))
    plt.fill_between(np.log10(a_range),mean_result+sd_result,mean_result-sd_result,alpha=0.3)
    plt.plot(np.log10(a_range),mean_result,lw=3)
    plt.ylim(min(mean_result-3*sd_result),max(mean_result+3*sd_result))
    plt.xlabel("log10(Regularization parameter)",fontsize=20)
    plt.ylabel("Correlation",fontsize=20)
    plt.title("Parameter optimization",fontsize=20)

    if savefile is not None:
            plt.savefig(savefile,bbox_inches="tight")

    return alpha_opt, mean_result, sd_result, a_range

def infer_combinatorial_regulation(df,regulonDf,target_gene,a_range,id_table,
                               n_iter = 25,regulators=None,target=None,resultsDirectory=None):
    from sklearn.linear_model import Ridge
    if regulonDf is not None:
        regulators = list(regulonDf[regulonDf.Gene==target_gene]["Regulator"])

    feature_matrix = df.loc[regulators,:]
    response = df.loc[target_gene,:]

    samples = feature_matrix.columns
    test_curves = []
    for iteration in range(n_iter):
        if iteration%min(100,n_iter-1) == 0:
            print('Optimization is {:.2f}% complete'.format(100*float(iteration+1)/n_iter))
        smpl_set = samples
        train_set = np.random.choice(smpl_set,len(smpl_set),replace=True)
        test_set = setdiff(smpl_set,train_set)

        curve_test = []

        for a in a_range:

            X = np.array(feature_matrix.loc[:,train_set]).T
            y = np.array(response.loc[train_set])
            clf = Ridge(random_state=0,alpha=a,fit_intercept=True)
            clf.fit(X, y)

            y = np.array(response.loc[test_set])

            decision_function_score = clf.predict(np.array(feature_matrix.loc[:,test_set]).T)
            curve_test.append(stats.spearmanr(decision_function_score,y)[0])

        test_curves.append(curve_test)

    mean_result = np.mean(np.vstack(test_curves),axis=0)
    sd_result = np.std(np.vstack(test_curves),axis=0)
    alpha_opt = a_range[np.argmax(mean_result)]

    plt.figure(figsize=(8,8))
    plt.fill_between(np.log10(a_range),mean_result+sd_result,mean_result-sd_result,alpha=0.3)
    plt.plot(np.log10(a_range),mean_result,lw=3)
    plt.ylim(min(mean_result-3*sd_result),max(mean_result+3*sd_result))
    plt.xlabel("log10(Regularization parameter)",fontsize=20)
    plt.ylabel("Correlation",fontsize=20)
    plt.title("Parameter optimization",fontsize=20)


    if target is None:
        target = gene_conversion(target_gene,id_table = id_table,list_symbols=True)

    if resultsDirectory is not None:
        savefile = os.path.join(resultsDirectory,target[0]+"_predictor_opt.pdf")
        plt.savefig(savefile,bbox_inches="tight")

    # Apply predictor
    X = np.array(feature_matrix).T
    y = np.array(response)
    clf = Ridge(random_state=0,alpha=alpha_opt,fit_intercept=True)
    clf.fit(X, y)
    decision_function_score = clf.predict(np.array(feature_matrix).T)
    stats.spearmanr(decision_function_score,y)

    # Save predictor
    coef_df = pd.DataFrame(clf.coef_)
    coef_df.index = gene_conversion(feature_matrix.index,id_table = id_table,list_symbols=True)
    coef_df.columns = target

    if resultsDirectory is not None:
        savefile = os.path.join(resultsDirectory,target[0]+"_predictor_expression.csv")
        coef_df.to_csv(savefile)

    return coef_df

def gene_aucs(x,y,return_aucs=False):
    from sklearn.metrics import roc_auc_score

    if len(x.shape) == 1:
        auc = roc_auc_score(y,x)
        return  auc, 0

    # t-test sorting
    from scipy import stats
    t, p = stats.ttest_ind(x[:,np.where(y==1)[0]], x[:,np.where(y==0)[0]],axis=1,equal_var=False)
    args = np.argsort(t)
    if len(args) > 100:
        args = args[-100:]

    # ROC AUC
    aucs = []
    for i in args:
        aucs.append(roc_auc_score(y,x[i,:]))

    if return_aucs is True:
        return aucs, args

    return max(aucs), args[np.argmax(aucs)]

def univariate_risk_stratifiers(data_df,srv,samples,lr_prop=0.7):
    risk_subset = pd.DataFrame(srv.loc[samples,"GuanScore"])
    risk_subset.sort_values(by="GuanScore",inplace=True,ascending=True)

    n_lr = int(lr_prop*risk_subset.shape[0])
    lr = risk_subset.index[0:n_lr]
    hr = risk_subset.index[n_lr:]
    risk_sorted = np.hstack([lr,hr])

    y = np.zeros(len(risk_sorted))
    y[-len(hr):] = 1

    x = np.array(data_df.loc[:,risk_sorted])
    aucs_, args_ = gene_aucs(x,y,return_aucs=True)
    genes_ = data_df.index[args_]

    return aucs_, genes_

def risk_aucs_from_genes(args,exp_df,act_df,y,risk_samples):

    from sklearn.metrics import roc_auc_score
    aucs_x = []
    aucs_n = []
    aucs_r = []
    for i in args:
        if i not in act_df.index:
            continue
        if i not in exp_df.index:
            continue
        aucs_x.append(roc_auc_score(y,np.array(exp_df.loc[i,risk_samples])))
        aucs_n.append(roc_auc_score(y,np.array(act_df.loc[i,risk_samples])))
        r_choice = np.random.choice(act_df.index,1,replace=False)[0]
        aucs_r.append(roc_auc_score(y,np.array(act_df.loc[r_choice,risk_samples])))

    return aucs_r, aucs_x, aucs_n

def risk_auc_wrapper(args,exp_df,act_df,srv_df,samples,lr_prop = 0.7):
    risk_subset_ = pd.DataFrame(srv_df.loc[samples,"GuanScore"])
    risk_subset_.sort_values(by="GuanScore",inplace=True,ascending=True)

    n_lr = int(lr_prop*risk_subset_.shape[0])
    lr_ = risk_subset_.index[0:n_lr]
    hr_ = risk_subset_.index[n_lr:]
    risk_samples = np.hstack([lr_,hr_])

    y = np.zeros(len(risk_samples))
    y[-len(hr_):] = 1

    aucs_r, aucs_x, aucs_n = risk_aucs_from_genes(args,exp_df,act_df,y,risk_samples)

    return aucs_r, aucs_x, aucs_n

def rank_risk_samples(srv_df,samples,lr_prop = 0.7):
    risk_subset_ = pd.DataFrame(srv_df.loc[samples,"GuanScore"])
    risk_subset_.sort_values(by="GuanScore",inplace=True,ascending=True)

    n_lr = int(lr_prop*risk_subset_.shape[0])
    lr_ = risk_subset_.index[0:n_lr]
    hr_ = risk_subset_.index[n_lr:]
    risk_samples = np.hstack([lr_,hr_])

    y = np.zeros(len(risk_samples))
    y[-len(hr_):] = 1

    return risk_samples, y

def univariate_predictor(x,y,names,n_iter=200,gene_labels=None):
    """
    Return results using single features to predict response.
    """
    if gene_labels is None:
        gene_labels = np.arange(x.shape[0])

    auc_tests = []
    gene_test = []
    for iteration in range(n_iter):
        train_test_dict = train_test(x,y,names)

        x_train = train_test_dict["x_train"]
        x_test = train_test_dict["x_test"]
        y_train = train_test_dict["y_train"]
        y_test = train_test_dict["y_test"]

        auc_train, ix_train = gene_aucs(x_train,y_train)
        auc_test, ix_test = gene_aucs(x_test[ix_train,:],y_test)

        auc_tests.append(auc_test)
        gene_test.append(gene_labels[ix_train])

    results = pd.DataFrame(np.vstack([auc_tests,gene_test]).T)
    results.columns = ["AUC","Gene"]

    return results


def transcriptionalPrograms(programs,reference_dictionary):
    transcriptionalPrograms = {}
    programRegulons = {}
    p_stack = []
    programs_flattened = programs #np.array(programs).flatten()
    for i in range(len(programs_flattened)):
        if len(np.hstack(programs_flattened[i]))>len(programs_flattened[i]):
            for j in range(len(programs_flattened[i])):
                p_stack.append(list(programs_flattened[i][j]))
        else:
            p_stack.append(list(programs_flattened[i]))

    for j in range(len(p_stack)):
        key = ("").join(["TP",str(j)])
        regulonList = [i for i in p_stack[j]]
        programRegulons[key] = regulonList
        tmp = [reference_dictionary[i] for i in p_stack[j]]
        transcriptionalPrograms[key] = list(set(np.hstack(tmp)))
    return transcriptionalPrograms, programRegulons


def reduceModules(df,programs,states,stateThreshold=0.75,saveFile=None):

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
        ax.set_ylabel("Transcriptional programs",fontsize=14)
        ax.set_xlabel("Samples",fontsize=14)
        plt.savefig(saveFile,bbox_inches="tight")

    return statesDf

def programsVsStates(statesDf,states,filename=None, csvpath=None, showplot=False):
    pixel = np.zeros((statesDf.shape[0],len(states)))
    for i in range(statesDf.shape[0]):
        for j in range(len(states)):
            pixel[i,j] = np.mean(statesDf.loc[statesDf.index[i],states[j]])

    pixel = pd.DataFrame(pixel)
    pixel.index = statesDf.index
    pixel.to_csv(csvpath, sep='\t')

    if showplot is False:
        return pixel

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pixel,cmap="bwr",vmin=-1,vmax=1,aspect="auto")
    ax.grid(False)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Transcriptional programs",fontsize=14)
    plt.xlabel("Transcriptional states",fontsize=14)
    if filename is not None:
        plt.savefig(filename,bbox_inches="tight")

    return pixel

def getStratifyingRegulons(states_list_1,states_list_2,reference_matrix,p=0.05,plot=True):

    if type(states_list_1[0]) == 'str':
        states_list_1 = [states_list_1]

    if type(states_list_2[0]) == 'str':
        states_list_2 = [states_list_2]

    matrix1 = reference_matrix.loc[:,np.hstack(states_list_1)]
    matrix2 = reference_matrix.loc[:,np.hstack(states_list_2)]

    ttest = stats.ttest_ind(matrix1,matrix2,axis=1,equal_var=False)

    if min(ttest[1]) > p:
        print("No hits detected. Cutoff p-value is too strict")
        return []

    results = pd.DataFrame(np.vstack(ttest).T)
    results.index = reference_matrix.index
    results.columns = ["t-statistic","p-value"]
    results = results[results["p-value"]<=p]
    results.sort_values(by="t-statistic",ascending=False,inplace=True)
    print(results)

    if plot:
        ttest_data_source = reference_matrix.loc[results.index,np.hstack([np.hstack(states_list_1),np.hstack(states_list_2)])]
        figure = plt.figure()
        ax = figure.gca()
        ax.imshow(ttest_data_source,cmap="bwr",aspect="auto")
        ax.grid(False)

    return results

# =============================================================================
# Functions used for cluster analysis
# =============================================================================

def getEigengenes(coexpressionModules,expressionData,regulon_dict=None,saveFolder=None):
    eigengenes = principal_df(coexpressionModules,expressionData,subkey=None,regulons=regulon_dict,minNumberGenes=1)
    eigengenes = eigengenes.T
    index = np.sort(np.array(eigengenes.index).astype(int))
    eigengenes = eigengenes.loc[index.astype(str),:]
    if saveFolder is not None:
        eigengenes.to_csv(os.path.join(saveFolder,"eigengenes.csv"))
    return eigengenes

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

        above_basline_ps = {key:basline_ps[key] for key in list(basline_ps.keys()) if basline_ps[key]<threshold}
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
    for key in list(dict_.keys()):
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
    for module in list(goBio_enriched.keys()):
        conv = {}
        for key in list(goBio_enriched[module].keys()):
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

def tsneStateLabels(tsneDf,states):
    labelsDf = pd.DataFrame(1000*np.ones(tsneDf.shape[0]))
    labelsDf.index = tsneDf.index
    labelsDf.columns = ["label"]

    for i in range(len(states)):
        tagged = states[i]
        labelsDf.loc[tagged,"label"] = i
    state_labels = np.array(labelsDf.iloc[:,0])
    return state_labels

def plotStates(statesDf,tsneDf,numCols=None,numRows=None,saveFile=None,size=10,aspect=1,scale=2):

    if numRows is None:
        if numCols is None:
            numRows = int(round(np.sqrt(statesDf.shape[0])))
            rat = np.floor(statesDf.shape[0]/float(numRows))
            rem = statesDf.shape[0]-numRows*rat
            numCols = int(rat+rem)
        elif numCols is not None:
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


def survivalMedianAnalysisDirect(median_df,SurvivalDf):

    from lifelines import CoxPHFitter

    k = median_df.columns[0]
    combinedSurvival = pd.concat([SurvivalDf.loc[:,["duration","observed"]],median_df],axis=1)

    try:
        coxResults = {}
        cph = CoxPHFitter()
        cph.fit(combinedSurvival, duration_col=SurvivalDf.columns[0], event_col=SurvivalDf.columns[1])

        tmpcph = cph.summary

        cox_hr = tmpcph.loc[k,"z"]
        cox_p = tmpcph.loc[k,"p"]
        coxResults[k] = (cox_hr, cox_p)

    except:
        coxResults[k] = (0,1)

    return coxResults

def survivalMedianAnalysis(task):

    start, stop = task[0]
    referenceDictionary,expressionData,SurvivalDf = task[1]

    overlapPatients = list(set(expressionData.columns)&set(SurvivalDf.index))
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]
    Survival.sort_values(by=Survival.columns[0],inplace=True)
    sorted_patients = Survival.index

    cox_regulons = []
    cox_keys = []
    keys = list(referenceDictionary.keys())[start:stop]
    for i in range(len(keys)):
        if i%100==0:
            print("Completed {:d} of {:d} iterations".format(i,len(keys)))
        key = keys[i]
        cluster = np.array(expressionData.loc[referenceDictionary[key],sorted_patients])
        median_ = np.mean(cluster,axis=0)
        median_df = pd.DataFrame(median_)
        median_df.index = sorted_patients
        median_df.columns = [key]

        cox_results_ = survivalMedianAnalysisDirect(median_df,Survival)
        cox_keys.append(key)
        cox_regulons.append([cox_results_[key][0],cox_results_[key][1]])

    cox_regulons_output = pd.DataFrame(np.vstack(cox_regulons))
    cox_regulons_output.index = cox_keys
    cox_regulons_output.columns = ['HR','p-value']

    return cox_regulons_output

def parallelMedianSurvivalAnalysis(referenceDictionary,expressionDf,survivalData,numCores=5):

    taskSplit = splitForMultiprocessing(list(referenceDictionary.keys()),numCores)
    taskData = (referenceDictionary,expressionDf,survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = multiprocess(survivalMedianAnalysis,tasks)
    survivalAnalysis = condenseOutput(coxOutput,output_type="df")

    return survivalAnalysis


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
        if ct % 100 == 0:
            print("completed {:d} iterations on thread".format(ct))
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

def survivalMembershipAnalysisDirect(membership_df,SurvivalDf):

    from lifelines import CoxPHFitter

    k = membership_df.columns[0]
    survival_patients = list(set(membership_df.index)&set(SurvivalDf.index))
    combinedSurvival = pd.concat([SurvivalDf.loc[survival_patients,SurvivalDf.columns[0:2]],
                                  membership_df.loc[survival_patients,:]],axis=1)
    combinedSurvival.sort_values(by=combinedSurvival.columns[0],inplace=True)

    try:
        cph = CoxPHFitter()
        cph.fit(combinedSurvival, duration_col=combinedSurvival.columns[0], event_col=combinedSurvival.columns[1])

        tmpcph = cph.summary

        cox_hr = tmpcph.loc[k,"z"]
        cox_p = tmpcph.loc[k,"p"]
    except:
        cox_hr, cox_p = (0,1)

    return cox_hr, cox_p

def parallelMemberSurvivalAnalysis(membershipDf,numCores=5,survivalPath=None,survivalData=None):

    if survivalData is None:
        survivalData = pd.read_csv(survivalPath,index_col=0,header=0)
    taskSplit = splitForMultiprocessing(membershipDf.index,numCores)
    taskData = (membershipDf,survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = multiprocess(survivalMembershipAnalysis,tasks)
    survivalAnalysis = condenseOutput(coxOutput)

    return survivalAnalysis

def survivalAnalysis(task):

    from lifelines import CoxPHFitter

    start, stop = task[0]
    expressionDf,SurvivalDf = task[1]

    overlapPatients = list(set(expressionDf.columns)&set(SurvivalDf.index))
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]

    coxResults = {}
    keys = expressionDf.index[start:stop]

    for key in keys:
        values = np.array(expressionDf.loc[key,overlapPatients])
        try:
            medianDf = pd.DataFrame(values)
            medianDf.index = overlapPatients
            medianDf.columns = ["value"]
            Survival2 = pd.concat([Survival,medianDf],axis=1)
            Survival2.sort_values(by=Survival2.columns[0],inplace=True)

            cph = CoxPHFitter()
            cph.fit(Survival2, duration_col=Survival2.columns[0], event_col=Survival2.columns[1])

            tmpcph = cph.summary

            cox_hr = tmpcph.loc["value","z"]
            cox_p = tmpcph.loc["value","p"]
            coxResults[key] = (cox_hr, cox_p)
        except:
            coxResults[key] = (0, 1)

    return coxResults

def parallelSurvivalAnalysis(expressionDf,survivalData,numCores=5):

    taskSplit = splitForMultiprocessing(expressionDf.index,numCores)
    taskData = (expressionDf,survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = multiprocess(survivalAnalysis,tasks)
    survivalResults = condenseOutput(coxOutput)

    return survivalResults


def kmplot(srv,groups,labels,legend=None,title=None,xlim_=None,filename=None,color=None,lw=1,alpha=1,fs=None,subplots=False,csvFile=None):

    for i in range(len(groups)):
        group = groups[i]
        patients = list(set(srv.index)&set(group))
        if len(patients) < 2:
            continue
        kmDf = kmAnalysis(survivalDf=srv.loc[patients,["duration","observed"]],durationCol="duration",statusCol="observed",saveFile = csvFile)
        subset = kmDf[kmDf.loc[:,"observed"]==1]
        duration = np.concatenate([np.array([0]),np.array(subset.loc[:,"duration"])])
        kme = 100*np.concatenate([np.array([1]),np.array(subset.loc[:,"kmEstimate"])])
        label = labels[i]
        if color is not None:
            if subplots is True:
                ax = plt.gca()
                ax.step(duration,kme,color=color[i],linewidth=lw,alpha=alpha,label=label)
            elif subplots is False:
                plt.step(duration,kme,color=color[i],linewidth=lw,alpha=alpha,label=label)
        elif color is None:
            if subplots is True:
                ax = plt.gca()
                ax.step(duration,kme,linewidth=lw,alpha=alpha,label=label)
            elif subplots is False:
                plt.step(duration,kme,linewidth=lw,alpha=alpha,label=label)

    plt.xlabel("Time (days)")
    plt.ylabel("Progression-free (%)")

    if fs is not None:
        plt.xlabel("Time (days)",fontsize=fs)
        plt.ylabel("Progression-free (%)",fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)

    if title is not None:
        plt.title(title)
        if fs is not None:
            plt.title(title,fontsize=fs)

    if legend is not None:
        plt.legend(loc='upper right')

    if filename is not None:
        plt.savefig(filename,bbox_inches="tight")



def regulon_survival(key,df,srv,filename=None,title=None,min_samples=5,primary_tag="Over",other_tag="Neither"):

    over = df.columns[np.where(df.loc[key,:]==1)[0]]
    if len(intersect(over,srv.index))>min_samples:
        membership_over = pd.DataFrame(np.zeros(df.shape[1]))
        membership_over.index = df.columns
        membership_over.columns = ["Overexpressed"]
        membership_over.loc[over,"Overexpressed"] = 1
        cox_hr_over, cox_p_over = survivalMembershipAnalysisDirect(membership_over,srv)
    else:
        over = []
        cox_hr_over = "NA"
        cox_p_over = "NA"

    under = df.columns[np.where(df.loc[key,:]==-1)[0]]
    if len(intersect(under,srv.index))>min_samples:
        membership_under = pd.DataFrame(np.zeros(df.shape[1]))
        membership_under.index = df.columns
        membership_under.columns = ["Underexpressed"]
        membership_under.loc[under,"Underexpressed"] = 1
        cox_hr_under, cox_p_under = survivalMembershipAnalysisDirect(membership_under,srv)
    else:
        under = []
        cox_hr_under = "NA"
        cox_p_under = "NA"

    neither = df.columns[np.where(df.loc[key,:]==0)[0]]
    if len(intersect(neither,srv.index))>min_samples:
        membership_neither = pd.DataFrame(np.zeros(df.shape[1]))
        membership_neither.index = df.columns
        membership_neither.columns = [other_tag]
        membership_neither.loc[neither,other_tag] = 1
        cox_hr_neither, cox_p_neither = survivalMembershipAnalysisDirect(membership_neither,srv)
    else:
        neither = []
        cox_hr_neither = "NA"
        cox_p_neither = "NA"

    groups = [neither,under,over]
    labels = [other_tag,"Under",primary_tag]
    colors = ["gray","blue","red"]
    kmplot(srv=srv,groups=groups,labels=labels,legend=True,title=title,xlim_=(-100,1750),
                 filename=None,lw=2,color=colors,alpha=0.8)

    haz_rat = [cox_hr_over,cox_hr_neither,cox_hr_under]
    haz_p = [cox_p_over,cox_p_neither,cox_p_under]
    cox_results = pd.DataFrame(np.vstack([haz_rat,haz_p]).T)
    cox_results.columns = ["HR","p-value"]
    cox_results.index = ["Overexpressed",other_tag,"Underexpressed"]

    if filename is not None:
        plt.savefig(filename,bbox_inches="tight")

    return cox_results


def composite_regulon_survival(key,df,guanSurvivalDfMMRF,cytogenetics,translocations,
                               remainder,plotsDirectory=None,dataDirectory=None,
                               nametag="regulon_",title_tag="R-",status_tag=" activity",
                               primary_tag="Over",other_tag="Neither",ext="pdf"):

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Instantiate figure
    sns.set(font_scale=1.2,style="whitegrid")
    fig = plt.figure(constrained_layout=True,figsize=(16,8))
    gs = fig.add_gridspec(2, 4)

    # All
    fig.add_subplot(gs[0, 0])
    srv = guanSurvivalDfMMRF
    regulon_cox_results_all = regulon_survival(key,df,srv,filename=None,
                                               title="Combined survival vs. "+title_tag+key+status_tag,
                                               primary_tag=primary_tag,other_tag=other_tag)

    # remainder
    fig.add_subplot(gs[0, 1])
    ranked_samples = remainder
    ranked_samples = intersect(ranked_samples,guanSurvivalDfMMRF.index)
    srv = guanSurvivalDfMMRF.loc[ranked_samples,:]
    regulon_cox_results_other = regulon_survival(key,df,srv,filename=None,
                                                 title="Hyperdiploid survival vs. "+title_tag+key+status_tag,
                                                 primary_tag=primary_tag,other_tag=other_tag)

    # Del 17
    fig.add_subplot(gs[0, 2])
    ranked_samples = getMutations("del17",cytogenetics)
    ranked_samples = intersect(ranked_samples,guanSurvivalDfMMRF.index)
    srv = guanSurvivalDfMMRF.loc[ranked_samples,:]
    regulon_cox_results_del17 = regulon_survival(key,df,srv,filename=None,
                                                 title="Del 17 survival vs. "+title_tag+key+status_tag,
                                                 primary_tag=primary_tag,other_tag=other_tag)

    # Amp 1q
    fig.add_subplot(gs[0, 3])
    ranked_samples = getMutations("amp1q",cytogenetics)
    ranked_samples = intersect(ranked_samples,guanSurvivalDfMMRF.index)
    srv = guanSurvivalDfMMRF.loc[ranked_samples,:]
    regulon_cox_results_amp1q = regulon_survival(key,df,srv,filename=None,
                                                 title="Amp 1q survival vs. "+title_tag+key+status_tag,
                                                 primary_tag=primary_tag,other_tag=other_tag)

    # t(11;14)
    fig.add_subplot(gs[1, 0])
    ranked_samples = getMutations("RNASeq_CCND1_Call",translocations)
    ranked_samples = intersect(ranked_samples,guanSurvivalDfMMRF.index)
    srv = guanSurvivalDfMMRF.loc[ranked_samples,:]
    regulon_cox_results_t1114 = regulon_survival(key,df,srv,filename=None,
                                                 title="t(11;14) survival vs. "+title_tag+key+status_tag,
                                                 primary_tag=primary_tag,other_tag=other_tag)

    # t(4;14)
    fig.add_subplot(gs[1, 1])
    ranked_samples = getMutations("RNASeq_WHSC1_Call",translocations)
    ranked_samples = intersect(ranked_samples,guanSurvivalDfMMRF.index)
    srv = guanSurvivalDfMMRF.loc[ranked_samples,:]
    regulon_cox_results_t414 = regulon_survival(key,df,srv,filename=None,
                                                title="t(4;14) survival vs. "+title_tag+key+status_tag,
                                                primary_tag=primary_tag,other_tag=other_tag)

    # t(14;16)
    fig.add_subplot(gs[1, 2])
    ranked_samples = getMutations("RNASeq_MAF_Call",translocations)
    ranked_samples = intersect(ranked_samples,guanSurvivalDfMMRF.index)
    srv = guanSurvivalDfMMRF.loc[ranked_samples,:]
    regulon_cox_results_t1416 = regulon_survival(key,df,srv,filename=None,
                                                 title="t(14;16) survival vs. "+title_tag+key+status_tag,
                                                 primary_tag=primary_tag,other_tag=other_tag)

    # MYC
    fig.add_subplot(gs[1, 3])
    ranked_samples = getMutations("RNASeq_MYC_Call",translocations)
    ranked_samples = intersect(ranked_samples,guanSurvivalDfMMRF.index)
    srv = guanSurvivalDfMMRF.loc[ranked_samples,:]
    regulon_cox_results_myc = regulon_survival(key,df,srv,filename=None,
                                               title="High-MYC survival vs. "+title_tag+key+status_tag,
                                               primary_tag=primary_tag,other_tag=other_tag)

    if plotsDirectory is not None:
        output_file = os.path.join(plotsDirectory,nametag+key+"_survival."+ext)
        plt.savefig(output_file,bbox_inches="tight")

    # Cox results
    cox_out = pd.concat([
        regulon_cox_results_all,
        regulon_cox_results_other,
        regulon_cox_results_del17,
        regulon_cox_results_amp1q,
        regulon_cox_results_t1114,
        regulon_cox_results_t414,
        regulon_cox_results_t1416,
        regulon_cox_results_myc
    ],axis=1)

    cox_out.columns = [
        "Combined HR",
        "Combined p-value",
        "Hyperdiploid HR",
        "Hyperdiploid p-value",
        "Del 17 HR",
        "Del 17 p-value",
        "Amp 1q HR",
        "Amp 1q p-value",
        "t(11;14) HR",
        "t(11;14) p-value",
        "t(4;14) HR",
        "t(4;14) p-value",
        "t(14;16) HR",
        "t(14;16) p-value",
        "High-MYC HR",
        "High-MYC p-value",
    ]

    if dataDirectory is not None:
        output_csv = os.path.join(dataDirectory,nametag+key+"_survival.csv")
        cox_out.to_csv(output_csv)

    return cox_out

def subtype_survival(subtype,data_dir,pval_threshold = 0.1):
    regulon_survival_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    reg_name = []
    rsd = []
    for i in range(len(regulon_survival_files)):
        filename = regulon_survival_files[i]
        reg_sur_table = pd.read_csv(regulon_survival_files[i],index_col=0,header=0)
        subtype_p = (" ").join([subtype,"p-value"])
        reg_sub_p = reg_sur_table.loc[:,subtype_p]
        p_min = min(np.array(reg_sub_p))
        if p_min <= pval_threshold:
            regname = ("-").join(["R",os.path.split(filename)[-1].split("_")[1]])
            reg_name.append(regname)
            neglogp_rel = -np.log10(p_min)+np.log10(min(np.array(reg_sur_table.iloc[:,1])))
            subtype_hr = (" ").join([subtype,"HR"])
            reg_sub_hr = reg_sur_table.loc[:,subtype_hr]
            subtype_data = np.hstack([list(reg_sub_hr),list(reg_sub_p),neglogp_rel,filename])
            rsd.append(subtype_data)
    if len(rsd) > 0:
        results = pd.DataFrame(np.vstack(rsd))
        results.columns = ["HR Over","HR Neither","HR Under","p-value Over","p-value Neither","p-value Under","Rel.Sig","filename"]
        results.index = reg_name
        results.sort_values(by="Rel.Sig",ascending=False,inplace=True)

        return results

    return []

def rank_risk(subtype,subtype_df,srv,override=None):

    if override is None:
        samples = getMutations(subtype,subtype_df)
    elif override is not None:
        samples = override

    tmp_srv = srv.loc[list(set(samples)&set(srv.index))]
    tmp_srv.sort_values(by="GuanScore",ascending=True,inplace=True)
    samples_ranked = tmp_srv.index
    return samples_ranked

def rank_activity(subtype,subtype_df,sort_df,key,subset=None,override=None):

    if override is None:
        samples = getMutations(subtype,subtype_df)
    elif override is not None:
        samples = override

    if subset is None:
        tmp_sort = pd.DataFrame(sort_df.loc[key,list(set(samples)&set(sort_df.columns))])
    if subset is not None:
        tmp_sort = pd.DataFrame(sort_df.loc[key,list(set(samples)&set(sort_df.columns)&set(subset))])

    tmp_sort.columns = [key]
    tmp_sort.sort_values(by=key,inplace=True,ascending=True)
    samples_ranked = tmp_sort.index
    return samples_ranked

def combinedStates(groups,ranked_groups,survivalDf,minSamples=4,maxStates=7):
    high_risk_indices = []
    for i in range(1,len(ranked_groups)+1):
        tmp_group = ranked_groups[-i]
        tmp_len = len(set(survivalDf.index)&set(groups[tmp_group]))
        if tmp_len >= minSamples:
            high_risk_indices.append(tmp_group)
        if len(high_risk_indices) >=maxStates:
            break

    combinations_high = []
    for i in range(len(high_risk_indices)-1):
        combinations_high.append(high_risk_indices[0:i+1])

    low_risk_indices = []
    for i in range(len(ranked_groups)):
        tmp_group = ranked_groups[i]
        tmp_len = len(set(survivalDf.index)&set(groups[tmp_group]))
        if tmp_len >= minSamples:
            low_risk_indices.append(tmp_group)
        if len(low_risk_indices) >=maxStates:
            break

    combinations_low = []
    for i in range(len(low_risk_indices)-1):
        combinations_low.append(low_risk_indices[0:i+1])

    combined_states_high = []
    for i in range(len(combinations_high)):
        tmp = []
        for j in range(len(combinations_high[i])):
            tmp.append(groups[combinations_high[i][j]])
        combined_states_high.append(np.hstack(tmp))

    combined_states_low = []
    for i in range(len(combinations_low)):
        tmp = []
        for j in range(len(combinations_low[i])):
            tmp.append(groups[combinations_low[i][j]])
        combined_states_low.append(np.hstack(tmp))

    combined_states = np.concatenate([combined_states_high,combined_states_low])
    combined_indices_high = ["&".join(np.array(combinations_high[i]).astype(str)) for i in range(len(combinations_high))]
    combined_indices_low = ["&".join(np.array(combinations_low[i]).astype(str)) for i in range(len(combinations_low))]
    combined_indices = np.concatenate([combined_indices_high,combined_indices_low])

    return combined_states, combined_indices

def align_to_reference(reference,new):
    tmp_alignment = []
    alignment = []
    for i in range(len(reference)):
        for j in range(len(new)):
            overlap = len(intersect(reference[i],new[j]))/float(min(len(reference[i]),len(new[j])))
            if overlap >= 0.5:
                if j not in tmp_alignment:
                    tmp_alignment.append(j)
                    alignment.append(new[j])

    for k in range(len(new)):
        if len(intersect(new[k],np.hstack(alignment))) == 0:
            alignment.append(new[k])

    return alignment

# =============================================================================
# Functions used for logic-based predictor
# =============================================================================

def precision(matrix, labels):
    vector = labels.iloc[:,0]
    vectorMasked = (matrix*vector).T
    TP = np.array(np.sum(vectorMasked,axis=0)).astype(float)
    FP = np.array(np.sum(matrix,axis=1)-TP).astype(float)
    prec = TP/(TP+FP)
    prec[np.where(TP<=5)[0]]=0
    return prec

def labelVector(hr,lr):
    labels = np.concatenate([np.ones(len(hr)),np.zeros(len(lr))]).astype(int)
    labelsDf = pd.DataFrame(labels)
    labelsDf.index = np.concatenate([hr,lr])
    labelsDf.columns = ["label"]
    return labelsDf

def predictRisk(expressionDf, regulonModules, model_filename, mapfile_path):
    expressionDf, _ = convert_ids_orig(expressionDf, mapfile_path)
    expressionDf = zscore(expressionDf)
    bkgdDf = background_df(expressionDf)
    overExpressedMembers = biclusterMembershipDictionary(regulonModules,bkgdDf,label=2,p=0.1)
    overExpressedMembersMatrix = membershipToIncidence(overExpressedMembers,expressionDf)

    labels = overExpressedMembersMatrix.columns
    predictor_formatted = np.array(overExpressedMembersMatrix).T

    loaded_model = pickle.load(open(model_filename, 'rb'))
    prediction = loaded_model.predict(predictor_formatted)

    hr_dt = labels[prediction.astype(bool)]
    lr_dt = labels[(1-prediction).astype(bool)]

    return hr_dt, lr_dt

def gene_conversion(gene_list,input_type="ensembl.gene", output_type="symbol",list_symbols=None,id_table=None):

    if input_type =="ensembl":
        input_type = "ensembl.gene"
    if output_type =="ensembl":
        output_type = "ensembl.gene"
    #kwargs = symbol,ensembl, entrezgene

    if id_table is None:
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

    if input_type == "ensembl.gene":
        conv_ = id_table.loc[gene_list,"Name"]

    if input_type == "symbol":
        tmp_table = id_table.copy()
        tmp_table.index = list(id_table.loc[:,"Name"])
        conv_ = tmp_table.loc[gene_list,"Preferred_Name"]
        del tmp_table

    if list_symbols is True:
        if type(conv_)==pd.core.series.Series:
            conv_ = list(conv_)
        else:
            conv_ = [conv_]

    return conv_

def swarmplot(samples,survival,savefile,ylabel="Relative risk",labels = None):

    import seaborn as sns
    allSamples = samples
    try:
        allSamples = np.hstack(samples)
    except:
        pass

    survival_samples = list(set(survival.index)&set(allSamples))
    srv = survival.loc[survival_samples,:]
    guan_srv = pd.DataFrame(srv.loc[:,"GuanScore"])
    guan_srv.columns = ["value"]
    guan_srv_group = pd.DataFrame(-np.ones(guan_srv.shape[0]))
    guan_srv_group.index = guan_srv.index
    guan_srv_group.columns = ["group"]
    guan_srv_df = pd.concat([guan_srv,guan_srv_group],axis=1)

    if len(samples[0][0]) > 1:
        groups = samples

    elif len(samples[0][0]) == 1:
        groups = []
        groups.append(samples)

    if labels is None:
        labels = range(len(groups))

    label_dfs = []
    for i in range(len(groups)):
        group = list(set(srv.index)&set(groups[i]))
        if len(group)>=1:
            label = labels[i]
            tmp_df = guan_srv_df.loc[group,:]
            tmp_df.loc[:,"group"] = label
            label_dfs.append(tmp_df)
    if len(label_dfs)>1:
        guan_srv_df = pd.concat(label_dfs,axis=0)
    elif len(label_dfs)==1:
        guan_srv_df = label_dfs[0]

    plt.figure(figsize=(12,8))
    ax = sns.boxplot(x='group', y='value', data=guan_srv_df)
    for patch in ax.artists:
        patch.set_edgecolor('black')
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.8))

    sns.swarmplot(x='group', y='value',data=guan_srv_df,size=7, color=[0.15,0.15,0.15],edgecolor="black")

    plt.ylabel(ylabel,fontsize=24)
    plt.xlabel("",fontsize=0)
    plt.ylim(-0.05,1.05)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(savefile,bbox_inches="tight")

    return guan_srv_df

def generatePredictionMatrix(srv,mtrx,high_risk_cutoff = 0.2):

    srv = srv.copy()
    srv.sort_values(by='GuanScore',ascending=False,inplace=True)

    highRiskSamples = list(srv.index[0:int(high_risk_cutoff*srv.shape[0])])
    lowRiskSamples = list(srv.index[int(high_risk_cutoff*srv.shape[0]):])

    hrFlag = pd.DataFrame(np.ones((len(highRiskSamples),1)).astype(int))
    hrFlag.index = highRiskSamples
    hrFlag.columns = ["HR_FLAG"]

    lrFlag = pd.DataFrame(np.zeros((len(lowRiskSamples),1)).astype(int))
    lrFlag.index = lowRiskSamples
    lrFlag.columns = ["HR_FLAG"]

    hrMatrix = pd.concat([mtrx.loc[:,highRiskSamples].T,hrFlag],axis=1)
    hrMatrix.columns = np.array(hrMatrix.columns).astype(str)
    lrMatrix = pd.concat([mtrx.loc[:,lowRiskSamples].T,lrFlag],axis=1)
    lrMatrix.columns = np.array(lrMatrix.columns).astype(str)
    #predictionMatrix = pd.concat([hrMatrix,lrMatrix],axis=0)

    return hrMatrix, lrMatrix

def plotRiskStratification(lbls,mtrx,srv,survival_tag,resultsDirectory=None):
    warnings.filterwarnings("ignore")

    hr_dt = mtrx.columns[lbls.astype(bool)]
    lr_dt = mtrx.columns[(1-lbls).astype(bool)]

    kmTag = "decision_tree"
    kmFilename = ("_").join([survival_tag,kmTag,"high-risk",".pdf"])

    groups = [hr_dt,lr_dt]
    labels = ["High-risk","Low-risk"]

    cox_vectors = []
    srv_set = set(srv.index)
    for i in range(len(groups)):
        group = groups[i]
        patients = list(set(group)&srv_set)
        tmp_df = pd.DataFrame(np.zeros(srv.shape[0]))
        tmp_df.index = srv.index
        tmp_df.columns = [labels[i]]
        tmp_df.loc[patients,labels[i]] = 1
        cox_vectors.append(tmp_df)

    pre_cox = pd.concat(cox_vectors,axis=1).T
    pre_cox.head(5)

    cox_dict = parallelMemberSurvivalAnalysis(membershipDf = pre_cox,numCores=1,survivalPath="",survivalData=srv)
    print('Risk stratification of '+survival_tag+' has Hazard Ratio of {:.2f}'.format(cox_dict['High-risk'][0]))

    if resultsDirectory is not None:
        plotName = os.path.join(resultsDirectory,kmFilename)
        kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=plotName)
        plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))

    elif resultsDirectory is None:
        kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=None)
        plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))


def iAUC(srv,mtrx,classifier,plot_all=False):
    from sklearn import metrics

    predicted_probabilities = classifier.predict_proba(np.array(mtrx.T))[:,1]
    predicted_probabilities_df = pd.DataFrame(predicted_probabilities)
    predicted_probabilities_df.index = mtrx.columns
    predicted_probabilities_df.columns = ["probability_high_risk"]

    srv_observed = srv[srv.iloc[:,1]==1]
    srv_unobserved = srv[srv.iloc[:,1]==0]

    aucs = []
    cutoffs = []
    tpr_list = []
    fpr_list = []
    for cutoff in 30.5*np.arange(12,25,2):#range(0,max(list(srv.iloc[:,0]))+interval,interval):
        srv_extended = srv_unobserved[srv_unobserved.iloc[:,0]>=cutoff]
        if srv_extended.shape[0] > 0:
            srv_total = pd.concat([srv_observed,srv_extended],axis=0)
        elif srv_extended.shape[0] == 0:
            srv_total = srv_observed.copy()
        pass_index = np.where(srv_total.iloc[:,0]<cutoff)[0]
        true_hr = []
        if len(pass_index) > 0:
            true_hr = srv_total.index[pass_index]
        true_lr = list(set(srv_total.index)-set(true_hr))

        tpr = []
        fpr = []
        for threshold in np.arange(0,1.02,0.01):
            model_pp = predicted_probabilities_df.copy()
            model_pp = model_pp.loc[list(set(model_pp.index)&set(srv_total.index)),:]
            predicted_hr = model_pp.index[np.where(model_pp.iloc[:,0]>=threshold)[0]]
            predicted_lr = list(set(model_pp.index)-set(predicted_hr))

            tp = set(true_hr)&set(predicted_hr)
            allpos = set(true_hr)
            tn = set(true_lr)&set(predicted_lr)
            allneg = set(true_lr)

            if len(allpos) == 0:
                tp_rate = 0
            elif len(allpos) > 0:
                tp_rate = len(tp)/float(len(allpos))

            if len(allneg) == 0:
                tn_rate = 0
            elif len(allneg) > 0:
                tn_rate = len(tn)/float(len(allneg))

            tpr.append(tp_rate)
            fpr.append(1-tn_rate)

        if plot_all is True:
            plt.figure()
            plt.plot(fpr,tpr)
            plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--")
            plt.ylim(-0.05,1.05)
            plt.xlim(-0.05,1.05)
            plt.title('ROC curve, cutoff = {:d}'.format(int(cutoff)))

        area = metrics.auc(fpr,tpr)
        aucs.append(area)
        cutoffs.append(cutoff)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    integrated_auc = np.mean(aucs)

    print('classifier has integrated AUC of {:.3f}'.format(integrated_auc))

    tpr_stds = np.std(np.vstack(tpr_list),axis=0)
    tpr_means = np.mean(np.vstack(tpr_list),axis=0)
    fpr_means = np.mean(np.vstack(fpr_list),axis=0)

    plt.figure()
    plt.plot(fpr_means,tpr_means+tpr_stds,'-',color="blue",linewidth=1)
    plt.plot(fpr_means,tpr_means-tpr_stds,'-',color="blue",linewidth=1)
    plt.fill_between(fpr_means, tpr_means-tpr_stds, tpr_means+tpr_stds,color="blue",alpha=0.2)
    plt.plot(fpr_means,tpr_means,'-k',linewidth=1.5)
    plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--r")
    plt.ylim(-0.05,1.05)
    plt.xlim(-0.05,1.05)
    plt.title('Integrated AUC = {:.2f}'.format(integrated_auc))
    plt.ylabel('Sensitivity',fontsize=14)
    plt.xlabel('1-Specificity',fontsize=14)

    return aucs, cutoffs, tpr_list, fpr_list

def predictionMatrix(membership_datasets,survival_datasets,high_risk_cutoff=0.20):
    hr_matrices = []
    lr_matrices = []

    for i in range(len(membership_datasets)):
        hrmatrix, lrmatrix = generatePredictionMatrix(survival_datasets[i],membership_datasets[i],high_risk_cutoff = high_risk_cutoff)
        hr_matrices.append(hrmatrix)
        lr_matrices.append(lrmatrix)

    hrMatrixCombined = pd.concat(hr_matrices,axis=0)
    lrMatrixCombined = pd.concat(lr_matrices,axis=0)
    predictionMat = pd.concat([hrMatrixCombined,lrMatrixCombined],axis=0)

    return predictionMat


def riskStratification(lbls,mtrx,guan_srv,survival_tag,classifier,resultsDirectory=None,plot_all=False,guan_rank=False,high_risk_cutoffs=None,plot_any=True):
    warnings.filterwarnings("ignore")

    from sklearn import metrics

    guan_srv = guan_srv.loc[list(set(guan_srv.index)&set(mtrx.columns)),:]
    if plot_any is True:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
        f.tight_layout(pad=1.08)
        f.set_figwidth(10)
        f.set_figheight(4)

    predicted_probabilities = classifier.predict_proba(np.array(mtrx.T))[:,1]
    predicted_probabilities_df = pd.DataFrame(predicted_probabilities)
    predicted_probabilities_df.index = mtrx.columns
    predicted_probabilities_df.columns = ["probability_high_risk"]

    srv = guan_srv.iloc[:,0:2]
    srv_observed = guan_srv[guan_srv.iloc[:,1]==1]
    srv_unobserved = guan_srv[guan_srv.iloc[:,1]==0]

    if high_risk_cutoffs is None:
        high_risk_cutoffs = np.percentile(list(srv_observed.iloc[:,0]),[10,15,20,25,30])

    aucs = []
    cutoffs = []
    tpr_list = []
    fpr_list = []
    prec = []
    rec = []
    for i in range(len(high_risk_cutoffs)):#range(0,max(list(srv.iloc[:,0]))+interval,interval):
        if guan_rank is True:
            percentile = 10+i*(20.0/(len(high_risk_cutoffs)-1))
            number_samples = int(np.ceil(guan_srv.shape[0]*(percentile/100.0)))
            cutoff = guan_srv.iloc[number_samples,0]
            true_hr = guan_srv.index[0:number_samples]
            true_lr = guan_srv.index[number_samples:]
            srv_total = guan_srv.copy()

        elif guan_rank is not True:
            cutoff = high_risk_cutoffs[i]
            srv_extended = srv_unobserved[srv_unobserved.iloc[:,0]>=cutoff]
            if srv_extended.shape[0] > 0:
                srv_total = pd.concat([srv_observed,srv_extended],axis=0)
            elif srv_extended.shape[0] == 0:
                srv_total = srv_observed.copy()
            pass_index = np.where(srv_total.iloc[:,0]<cutoff)[0]
            true_hr = []
            if len(pass_index) > 0:
                true_hr = srv_total.index[pass_index]
            true_lr = list(set(srv_total.index)-set(true_hr))


        #use predicted_probabilities_df against true_hr, true_lr to compute precision and recall from sklearn.metrics

        tpr = []
        fpr = []
        precisions = []
        recalls = []
        for threshold in np.arange(0,1.02,0.01):
            model_pp = predicted_probabilities_df.copy()
            model_pp = model_pp.loc[list(set(model_pp.index)&set(srv_total.index)),:]
            predicted_hr = model_pp.index[np.where(model_pp.iloc[:,0]>=threshold)[0]]
            predicted_lr = list(set(model_pp.index)-set(predicted_hr))

            tp = set(true_hr)&set(predicted_hr)
            fp = set(true_lr)&set(predicted_hr)
            allpos = set(true_hr)
            tn = set(true_lr)&set(predicted_lr)
            fn = set(true_hr)&set(predicted_lr)
            allneg = set(true_lr)

            if len(allpos) == 0:
                tp_rate = 0
                precision = 0
                recall=0
            elif len(allpos) > 0:
                tp_rate = len(tp)/float(len(allpos))

                if len(tp) + len(fp) > 0:
                    precision = len(tp)/float(len(tp) + len(fp))
                elif len(tp) + len(fp) == 0:
                    precision = 0

                if len(tp) +len(fn) > 0:
                    recall = len(tp)/float(len(tp) +len(fn))
                elif len(tp) +len(fn) == 0:
                    recall = 0
            if len(allneg) == 0:
                tn_rate = 0
            elif len(allneg) > 0:
                tn_rate = len(tn)/float(len(allneg))

            tpr.append(tp_rate)
            fpr.append(1-tn_rate)

            precisions.append(precision)
            recalls.append(recall)

        if plot_all is True:
            plt.figure()
            plt.plot(fpr,tpr)
            plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--")
            plt.ylim(-0.05,1.05)
            plt.xlim(-0.05,1.05)
            plt.title('ROC curve, cutoff = {:d}'.format(int(cutoff)))

        area = metrics.auc(fpr,tpr)
        aucs.append(area)
        cutoffs.append(cutoff)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        prec.append(precisions)
        rec.append(recalls)

    integrated_auc = np.mean(aucs)

    #print('classifier has integrated AUC of {:.3f}'.format(integrated_auc))
    tpr_stds = np.std(np.vstack(tpr_list),axis=0)
    tpr_means = np.mean(np.vstack(tpr_list),axis=0)
    fpr_means = np.mean(np.vstack(fpr_list),axis=0)

    if plot_any is True:
        ax1.fill_between(fpr_means, tpr_means-tpr_stds, tpr_means+tpr_stds,color=[0,0.4,0.6],alpha=0.3)
        ax1.plot(fpr_means,tpr_means,color=[0,0.4,0.6],linewidth=1.5)
        ax1.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--",color=[0.2,0.2,0.2])
        ax1.set_ylim(-0.05,1.05)
        ax1.set_xlim(-0.05,1.05)
        ax1.set_title('Integrated AUC = {:.2f}'.format(integrated_auc))
        ax1.set_ylabel('Sensitivity',fontsize=14)
        ax1.set_xlabel('1-Specificity',fontsize=14)

    hr_dt = mtrx.columns[lbls.astype(bool)]
    lr_dt = mtrx.columns[(1-lbls).astype(bool)]

    kmTag = "decision_tree"
    kmFilename = ("_").join([survival_tag,kmTag,"high-risk",".pdf"])

    groups = [hr_dt,lr_dt]
    labels = ["High-risk","Low-risk"]

    cox_vectors = []
    srv_set = set(srv.index)
    for i in range(len(groups)):
        group = groups[i]
        patients = list(set(group)&srv_set)
        tmp_df = pd.DataFrame(np.zeros(srv.shape[0]))
        tmp_df.index = srv.index
        tmp_df.columns = [labels[i]]
        tmp_df.loc[patients,labels[i]] = 1
        cox_vectors.append(tmp_df)

    pre_cox = pd.concat(cox_vectors,axis=1).T
    pre_cox.head(5)

    cox_dict = parallelMemberSurvivalAnalysis(membershipDf = pre_cox,numCores=1,survivalPath="",survivalData=srv)
    #print('Risk stratification of '+survival_tag+' has Hazard Ratio of {:.2f}'.format(cox_dict['High-risk'][0]))

    hazard_ratio = cox_dict['High-risk'][0]
    if plot_any is True:
        if resultsDirectory is not None:
            plotName = os.path.join(resultsDirectory,kmFilename)
            kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=plotName)
            plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))

        elif resultsDirectory is None:
            kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=None)
            plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))

    return aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec


def generatePredictor(membership_datasets,survival_datasets,dataset_labels,iterations=20,method='xgboost',n_estimators=100,output_directory=None,best_state=None,test_only=True,separate_results=True,metric='roc_auc',class1_proportion=0.20, test_proportion=0.35,colsample_bytree=1,subsample=1):

    from sklearn.model_selection import train_test_split

    if method=='xgboost':
        os.environ['KMP_DUPLICATE_LIB_OK']='True' #prevents kernel from dying when running XGBClassifier
        from xgboost import XGBClassifier

    elif method=='decisionTree':
        from sklearn.tree import DecisionTreeClassifier

    predictionMat = predictionMatrix(membership_datasets,survival_datasets,high_risk_cutoff=class1_proportion)

    X = np.array(predictionMat.iloc[:,0:-1])
    Y = np.array(predictionMat.iloc[:,-1])
    X = X.astype('int')
    Y = Y.astype('int')

    samples_ = np.array(predictionMat.index)

    if best_state is None:
        mean_aucs = []
        mean_hrs = []
        pct_labeled = []
        for rs in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_proportion, random_state = rs)
            X_train_columns, X_test_columns, y_train_samples, y_test_samples = train_test_split(X, samples_, test_size = test_proportion, random_state = rs)

            train_datasets = []
            test_datasets = []
            for td in range(len(membership_datasets)):
                dataset = membership_datasets[td]
                train_members = list(set(dataset.columns)&set(y_train_samples))
                test_members = list(set(dataset.columns)&set(y_test_samples))
                train_datasets.append(dataset.loc[:,train_members])
                test_datasets.append(dataset.loc[:,test_members])

            if method=='xgboost':
                eval_set = [(X_train, y_train), (X_test, y_test)]
                clf = XGBClassifier(n_jobs=1,random_state=12,n_estimators=n_estimators,colsample_bytree=colsample_bytree,subsample=subsample)
                clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=False)
            elif method=='decisionTree':
                clf = DecisionTreeClassifier(criterion = "gini", random_state = 12, max_depth=6, min_samples_leaf=5)
                clf.fit(X_train, y_train)

            train_predictions = []
            test_predictions = []
            for p in range(len(membership_datasets)):
                tmp_train_predictions = clf.predict(np.array(train_datasets[p].T))
                tmp_test_predictions = clf.predict(np.array(test_datasets[p].T))
                train_predictions.append(tmp_train_predictions)
                test_predictions.append(tmp_test_predictions)

                #tmp_train_predictions_floats = clf.predict_proba(np.array(train_datasets[p].T))
                #tmp_test_predictions_floats = clf.predict_proba(np.array(test_datasets[p].T))

            if test_only is True:
                scores = []
                hrs = []
                for j in range(len(test_datasets)):
                    mtrx = test_datasets[j]
                    guan_srv = survival_datasets[j]
                    survival_tag = dataset_labels[j]
                    lbls = test_predictions[j]
                    aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = riskStratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
                    score = np.mean(aucs)
                    scores.append(score)
                    hrs.append(hazard_ratio)
                    pct_labeled.append(100*sum(lbls)/float(len(lbls)))

                mean_auc = np.mean(scores)
                mean_hr = np.mean(hrs)
                mean_aucs.append(mean_auc)
                mean_hrs.append(mean_hr)
                print(rs,mean_auc,mean_hr)

            elif test_only is False:
                scores = []
                hrs = []
                for j in range(len(test_datasets)):
                    mtrx = test_datasets[j]
                    guan_srv = survival_datasets[j]
                    survival_tag = dataset_labels[j]
                    lbls = test_predictions[j]
                    aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = riskStratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
                    score = np.mean(aucs)
                    scores.append(score)
                    hrs.append(hazard_ratio)

                    mtrx = train_datasets[j]
                    lbls = train_predictions[j]
                    aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = riskStratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
                    score = np.mean(aucs)
                    scores.append(score)
                    hrs.append(hazard_ratio)

                mean_auc = np.mean(scores)
                mean_hr = np.mean(hrs)
                mean_aucs.append(mean_auc)
                mean_hrs.append(mean_hr)
                print(rs,mean_auc,mean_hr)

        if metric == 'roc_auc':
            best_state = np.argsort(np.array(mean_aucs))[-1]
        elif metric == 'hazard_ratio':
            best_state = np.argsort(np.array(mean_hrs))[-1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_proportion, random_state = best_state)
    X_train_columns, X_test_columns, y_train_samples, y_test_samples = train_test_split(X, samples_, test_size = test_proportion, random_state = best_state)

    train_datasets = []
    test_datasets = []
    for td in range(len(membership_datasets)):
        dataset = membership_datasets[td]
        train_members = list(set(dataset.columns)&set(y_train_samples))
        test_members = list(set(dataset.columns)&set(y_test_samples))
        train_datasets.append(dataset.loc[:,train_members])
        test_datasets.append(dataset.loc[:,test_members])

    if method=='xgboost':
        eval_set = [(X_train, y_train), (X_test, y_test)]
        clf = XGBClassifier(n_jobs=1,random_state=12,n_estimators=n_estimators,colsample_bytree=colsample_bytree,subsample=subsample)
        clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=False)
    elif method=='decisionTree':
        clf = DecisionTreeClassifier(criterion = "gini", random_state = 12, max_depth=6, min_samples_leaf=5)
        clf.fit(X_train, y_train)

    train_predictions = []
    test_predictions = []
    for p in range(len(membership_datasets)):
        tmp_train_predictions = clf.predict(np.array(train_datasets[p].T))
        tmp_test_predictions = clf.predict(np.array(test_datasets[p].T))
        train_predictions.append(tmp_train_predictions)
        test_predictions.append(tmp_test_predictions)

    mean_aucs = []
    mean_hrs = []
    if test_only is True:
        scores = []
        hrs = []
        pct_labeled = []
        for j in range(len(test_datasets)):
            mtrx = test_datasets[j]
            guan_srv = survival_datasets[j]
            survival_tag = dataset_labels[j]
            lbls = test_predictions[j]
            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = riskStratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
            score = np.mean(aucs)
            scores.append(score)
            hrs.append(hazard_ratio)
            pct_labeled.append(100*sum(lbls)/float(len(lbls)))

        mean_auc = np.mean(scores)
        mean_hr = np.mean(hrs)
        mean_aucs.append(mean_auc)
        mean_hrs.append(mean_hr)
        precision_matrix = np.vstack(prec)
        recall_matrix = np.vstack(rec)
        print(best_state,mean_auc,mean_hr)

    elif test_only is False:
        scores = []
        hrs = []
        pct_labeled = []
        for j in range(len(test_datasets)):
            mtrx = test_datasets[j]
            guan_srv = survival_datasets[j]
            survival_tag = dataset_labels[j]
            lbls = test_predictions[j]
            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = riskStratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
            score = np.mean(aucs)
            scores.append(score)
            hrs.append(hazard_ratio)
            pct_labeled.append(100*sum(lbls)/float(len(lbls)))

            mtrx = train_datasets[j]
            lbls = train_predictions[j]
            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = riskStratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
            score = np.mean(aucs)
            scores.append(score)
            hrs.append(hazard_ratio)

        mean_auc = np.mean(scores)
        mean_hr = np.mean(hrs)
        mean_aucs.append(mean_auc)
        mean_hrs.append(mean_hr)
        precision_matrix = np.vstack(prec)
        recall_matrix = np.vstack(rec)
        print(best_state,mean_auc,mean_hr)

    train_predictions = []
    test_predictions = []
    predictions = []
    #add print for percent labeled high-risk
    for p in range(len(membership_datasets)):
        tmp_train_predictions = clf.predict(np.array(train_datasets[p].T))
        tmp_test_predictions = clf.predict(np.array(test_datasets[p].T))
        tmp_predictions = clf.predict(np.array(membership_datasets[p].T))
        train_predictions.append(tmp_train_predictions)
        test_predictions.append(tmp_test_predictions)
        predictions.append(tmp_predictions)

    if separate_results is False:
        for j in range(len(membership_datasets)):
            mtrx = membership_datasets[j]
            guan_srv = survival_datasets[j]
            survival_tag = dataset_labels[j]
            lbls = predictions[j]

            percent_classified_hr = 100*sum(lbls)/float(len(lbls))
            print('classified {:.1f} percent of population as high-risk'.format(percent_classified_hr))

            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = riskStratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=True)
            if output_directory is not None:
                plt.savefig(os.path.join(output_directory,('_').join([survival_tag,method,metric,'survival_predictions.pdf'])),bbox_inches='tight')

    elif separate_results is True:
        for j in range(len(membership_datasets)):
            guan_srv = survival_datasets[j]
            survival_tag = dataset_labels[j]

            mtrx = train_datasets[j]
            lbls = train_predictions[j]

            percent_classified_hr = 100*sum(lbls)/float(len(lbls))
            print('classified {:.1f} percent of training population as high-risk'.format(percent_classified_hr))

            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = riskStratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=True)
            if output_directory is not None:
                plt.savefig(os.path.join(output_directory,('_').join([survival_tag,method,metric,'training_survival_predictions.pdf'])),bbox_inches='tight')

            mtrx = test_datasets[j]
            lbls = test_predictions[j]

            percent_classified_hr = 100*sum(lbls)/float(len(lbls))
            print('classified {:.1f} percent of test population as high-risk'.format(percent_classified_hr))

            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = riskStratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=True)
            if output_directory is not None:
                plt.savefig(os.path.join(output_directory,('_').join([survival_tag,method,metric,'test_survival_predictions.pdf'])),bbox_inches='tight')

    nextIteration = []
    class0 = []
    class1 = []
    for p in range(len(membership_datasets)):
        tmp_predictions = clf.predict(np.array(membership_datasets[p].T))
        tmp_class_0 = membership_datasets[p].columns[(1-np.array(tmp_predictions)).astype(bool)]
        tmp_class_1 = membership_datasets[p].columns[np.array(tmp_predictions).astype(bool)]
        nextIteration.append(membership_datasets[p].loc[:,tmp_class_0])
        class0.append(tmp_class_0)
        class1.append(tmp_class_1)

    print(best_state)

    if best_state is not None:
        return clf, class0, class1, mean_aucs, mean_hrs, pct_labeled, precision_matrix, recall_matrix

    return clf, class0, class1, mean_aucs, mean_hrs, pct_labeled, precision_matrix, recall_matrix

def differentialActivity(regulon_matrix,reference_matrix,baseline_patients,relapse_patients,minRegulons = 5,useAllRegulons = False,maxRegulons = 5,highlight=None,savefile = None):

    reference_matrix.index = np.array(reference_matrix.index).astype(str)

    genes = []
    mean_baseline_frequency = []
    mean_relapse_frequency = []
    mean_significance = []
    skipped = []

    t1 = time.time()
    for gene in list(set(regulon_matrix["Gene"])):
        regulons_ = np.array(regulon_matrix[regulon_matrix.Gene==gene]["Regulon_ID"]).astype(str)
        if len(regulons_)<minRegulons:
            skipped.append(gene)
            continue

        baseline_freq = []
        relapse_freq = []
        neglogps = []

        for regulon_ in regulons_:

            baseline_values = reference_matrix.loc[regulon_,baseline_patients]
            relapse_values = reference_matrix.loc[regulon_,relapse_patients]

            indicator = len(set(reference_matrix.iloc[0,:]))

            if indicator>2:
                t, p = stats.ttest_ind(relapse_values,baseline_values)

            elif indicator ==2:
                # chi square
                rpos = np.sum(relapse_values)
                if np.sum(rpos) == 0:
                    continue
                rneg = len(relapse_values)-rpos

                bpos = np.sum(baseline_values)
                if np.sum(bpos) == 0:
                    continue
                bneg = len(baseline_values)-bpos

                obs = np.array([[rpos,rneg],[bpos,bneg]])
                chi2, p, dof, ex = stats.chi2_contingency(obs, correction=False)

            if useAllRegulons is True:
                neglogps.append(-np.log10(p))
                baseline_freq.append(np.mean(baseline_values))
                relapse_freq.append(np.mean(relapse_values))

            elif useAllRegulons is False:
                if len(neglogps)<=maxRegulons:
                    neglogps.append(-np.log10(p))
                    baseline_freq.append(np.mean(baseline_values))
                    relapse_freq.append(np.mean(relapse_values))

                if len(neglogps)>maxRegulons:
                    tmp_nlp = -np.log10(p)
                    if min(neglogps) < tmp_nlp:
                        argmin = np.argmin(neglogps)
                        neglogps[argmin] = tmp_nlp
                        tmp_baseline = np.mean(baseline_values)
                        baseline_freq[argmin] = tmp_baseline
                        tmp_relapse = np.mean(relapse_values)
                        relapse_freq[argmin] = tmp_relapse

        mean_relapse_frequency.append(np.mean(relapse_freq))
        mean_baseline_frequency.append(np.mean(baseline_freq))
        mean_significance.append(np.mean(neglogps))
        genes.append(gene)

    relapse_over_baseline = np.log2(np.array(mean_relapse_frequency).astype(float)/np.array(mean_baseline_frequency))
    volcano_data_ = pd.DataFrame(np.vstack([mean_baseline_frequency,mean_relapse_frequency,relapse_over_baseline,mean_significance]).T)
    volcano_data_.index = genes
    volcano_data_.columns = ["phenotype1_frequency","phenotype2_frequency","log2(phenotype2/phenotype1)","-log10(p)"]
    volcano_data_.sort_values(by="-log10(p)",ascending=False,inplace=True)
    volcano_data_

    t2 = time.time()

    print('completed in {:.2f} minutes'.format((t2-t1)/60.))

    insigvoldata_patients = volcano_data_.index[volcano_data_["-log10(p)"]<=-np.log10(0.05)]
    sigvoldata_patients = volcano_data_.index[volcano_data_["-log10(p)"]>-np.log10(0.05)]

    insigvoldata = volcano_data_.loc[insigvoldata_patients,:]
    sigvoldata = volcano_data_.loc[sigvoldata_patients,:]

    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(sigvoldata["phenotype2_frequency"],sigvoldata["log2(phenotype2/phenotype1)"],color = [0.3,0.4,1],edgecolor = [0,0,1],alpha=0.4,s=5)
        ax.scatter(insigvoldata["phenotype2_frequency"],insigvoldata["log2(phenotype2/phenotype1)"],color = [0.5,0.5,0.5],edgecolor = [0.25,0.25,0.25],alpha=0.4,s=5)

        if highlight is not None:
            if type(highlight) is not str:
                highlight = list(set(highlight)-set(skipped))
            else:
                highlight = list(set([highlight])-set(skipped))

            if len(highlight)>0:
                ax.scatter(volcano_data_.loc[highlight,"phenotype2_frequency"],volcano_data_.loc[highlight,"log2(phenotype2/phenotype1)"],color = "red",edgecolor="red",alpha=0.4,s=5)

        plt.ylim(-0.4+min(list(sigvoldata["log2(phenotype2/phenotype1)"])),0.4+max(list(sigvoldata["log2(phenotype2/phenotype1)"])))
        plt.ylabel("log2(phenotype2/phenotype1)",fontsize=14)
        plt.xlabel("log2(phenotype2/phenotype1)",fontsize=14)

        if savefile is not None:
            plt.savefig(savefile,bbox_inches="tight")
    except:
        print('Error: Analysis was successful, but could not generate plot')

    return volcano_data_

def chiSquareTest(risk_status,membership_array):
    from scipy.stats import chi2_contingency
    ps = []
    for i in range(membership_array.shape[0]):
        obs = pd.crosstab(risk_status,membership_array[i,:])
        chi2, p, dof, ex = chi2_contingency(obs, correction=False)
        ps.append(p)
    return ps

def networkActivity(reference_matrix,regulon_matrix,minRegulons = 2):

    reference_columns = reference_matrix.columns
    reference_regulonDf = regulon_matrix.copy()
    reference_regulonDf.index = list(regulon_matrix.loc[:,"Regulon_ID"])

    genes = []
    activities = []
    for gene in reference_regulonDf.Gene.unique():
        regulon_list = np.array(reference_regulonDf[reference_regulonDf.Gene==gene]["Regulon_ID"])
        if len(regulon_list) >= minRegulons:
            activity = list(reference_matrix.loc[regulon_list.astype(str),:].mean(axis=0))
            genes.append(gene)
            activities.append(activity)

    activity_df = pd.DataFrame(np.vstack(activities))
    activity_df.index = genes
    activity_df.columns = reference_columns

    return activity_df

def sortedHeatmap(features,samples,data_df,sort_df,sort_column,num_breaks=10,override=False):

    if override is False:
        tmp_srv = sort_df.loc[list(set(samples)&set(sort_df.index))]
        tmp_srv.sort_values(by=sort_column,ascending=True,inplace=True)
        index = tmp_srv.index

    elif override is not False:
        index = np.array(samples)

    splits = splitForMultiprocessing(index,num_breaks)

    partial_means = []
    for tpl in splits:
        tmp_means = data_df.loc[features,index[tpl[0]:tpl[1]]].mean(axis=1)
        partial_means.append(tmp_means)

    final_df = pd.concat(partial_means,axis=1)

    return final_df

def stitchHeatmaps(heatmap_list):

    heatmaps = []
    for h in range(len(heatmap_list)-1):
        tmp_hmap = heatmap_list[h]
        tmp_spacer = pd.DataFrame(np.zeros(tmp_hmap.shape[0]))
        tmp_spacer.index = tmp_hmap.index
        tmp_spacer.columns = ["n/a"]
        hmap = pd.concat([tmp_hmap,tmp_spacer],axis=1)
        heatmaps.append(hmap)
    heatmaps.append(heatmap_list[-1])

    final_df = pd.concat(heatmaps,axis=1)

    return final_df

def stiched_heatmap2(heatmap_list,cmap = "Blues",results_directory=None):

    import seaborn as sns
    # Instantiate figure
    fig = plt.figure(constrained_layout=True,figsize=(16,3))

    # Set figure axes
    gs = fig.add_gridspec(1, len(heatmap_list))

    # Fill first subplot
    fig.add_subplot(gs[0,0])
    sns.heatmap(np.asarray(heatmap_list[0]),cmap = cmap,square=False,
               yticklabels=heatmap_list[0].index,xticklabels="",cbar=False)

    for h in range(1,len(heatmap_list)-1):
        subset = np.asarray(heatmap_list[h])
        fig.add_subplot(gs[0,h])
        sns.heatmap(subset,cmap = cmap,square=False,
                   yticklabels="",xticklabels="",cbar=False)

    subset = np.asarray(heatmap_list[h])
    fig.add_subplot(gs[0,h])
    sns.heatmap(subset,cmap = cmap,square=False,
               yticklabels="",xticklabels="")

    if results_directory is not None:
        plt.savefig(os.path.join(results_directory,"stitched_heatmap.pdf"))

    return

def composite_figure_4(stitched_list,cmaps,id_table=None,results_directory=None,font_scale=1.5,figsize=(16,15)):
    #warnings.filterwarnings("ignore")
    import seaborn as sns
    # Instantiate figure
    sns.set(font_scale = font_scale)
    fig = plt.figure(constrained_layout=True,figsize=figsize)

    # Set figure axes
    num_plots = len(stitched_list)
    gs = fig.add_gridspec(num_plots, 1)

    for h in range(num_plots):
        # Fill first subplot
        fig.add_subplot(gs[h,0])
        labels = stitched_list[h].index
        if id_table is not None:
            labels = list(id_table.loc[labels,"Name"])
        sh = sns.heatmap(np.asarray(stitched_list[h]),cmap = cmaps[h],square=False,
                   yticklabels=labels,xticklabels="")
        sh.set_yticklabels(labels,rotation=0)

    if results_directory is not None:
        plt.savefig(os.path.join(results_directory,"Figure4.pdf"))

    return

def boxplot_figure(boxplot_data,labels):
    formatted_data = []
    formatted_labels = []
    for i in range(len(boxplot_data)):
        tmp_data = np.array(list(boxplot_data[i])).astype(float)
        tmp_labels = [labels[i] for iteration in range(len(tmp_data))]
        formatted_data.extend(tmp_data)
        formatted_labels.extend(tmp_labels)

    formatted_boxplot_data = pd.DataFrame(np.vstack([formatted_data,formatted_labels]).T)
    formatted_boxplot_data.columns = ["data","label"]
    #formatted_boxplot_data.iloc[:,0] = formatted_boxplot_data.iloc[:,0].convert_objects(convert_numeric=True)
    formatted_boxplot_data.iloc[:,0] = formatted_boxplot_data.iloc[:,0]

    return formatted_boxplot_data

def correlate_risk(gene_list,subtype_pats,ref_df,srv,subtype_labels,gene_labels=None):
    risk_correlations = np.zeros((len(gene_list),len(subtype_pats)))
    risk_ps = np.zeros((len(gene_list),len(subtype_pats)))

    for i in range(len(gene_list)):
        gene = gene_list[i]
        for j in range(len(subtype_pats)):
            pats = subtype_pats[j]
            pats = intersect(pats,srv.index)
            corr = stats.spearmanr(ref_df.loc[gene,pats],srv.loc[pats,"GuanScore"])
            risk_correlations[i,j] = corr[0]
            risk_ps[i,j] = corr[1]

    risk_correlations_df = pd.DataFrame(risk_correlations)
    risk_correlations_df.index = gene_labels
    if gene_labels is None:
        gene_labels = gene_conversion(gene_list,list_symbols=True)
        risk_correlations_df.index = gene_labels
    risk_correlations_df.columns = subtype_labels

    risk_ps_df = pd.DataFrame(risk_ps)
    risk_ps_df.index = gene_labels
    risk_ps_df.columns = subtype_labels

    return risk_correlations_df, risk_ps_df

def regulon_variance(validation_df,regulonModules):

    from sklearn.decomposition import PCA
    validation_index = validation_df.index

    n = []
    var = []
    pca_var = []
    modules = []
    for r in regulonModules.keys():
        genes = intersect(regulonModules[r],validation_index)
        tmp_df = validation_df.loc[genes,:]
        #Variance of genes
        var_ = np.mean(list(np.var(tmp_df,axis=0)))
        #Variance explained by PC1
        pca = PCA(1,random_state=12)
        principalComponents = pca.fit_transform(tmp_df.T)
        varx = pca.explained_variance_ratio_[0]
        #Store results
        n.append(len(genes))
        var.append(var_)
        pca_var.append(varx)
        modules.append(r)

    results_df = pd.DataFrame(np.vstack([n,var,pca_var,modules]).T)
    results_df.columns = ["N","Variance","Variance_expained","Modules"]
    results_df.index = list(regulonModules.keys())

    return results_df

def random_regulon_variance(validation_df,ns,n_iter=100,seed=True):

    from sklearn.decomposition import PCA
    validation_index = validation_df.index

    num = []
    var = []
    pca_var = []
    ct = 0
    sd = 0
    for n in ns:
        ct+=1
        if ct%10==0:
            print(ct)
        for iteration in range(n_iter):
            if seed is True:
                sd+=1
                np.random.seed(sd)
            genes = np.random.choice(validation_index,n,replace=False)
            tmp_df = validation_df.loc[genes,:]
            #Variance of genes
            var_ = np.mean(list(np.var(tmp_df,axis=0)))
            #Variance explained by PC1
            pca = PCA(1,random_state=12)
            principalComponents = pca.fit_transform(tmp_df.T)
            varx = pca.explained_variance_ratio_[0]
            #Store results
            num.append(len(genes))
            var.append(var_)
            pca_var.append(varx)

    results_df = pd.DataFrame(np.vstack([num,var,pca_var]).T)
    results_df.columns = ["N","Variance","Variance_explained"]
    results_df.index = num

    return results_df

def random_significance(random_results,variable="Variance_explained",p=0.05):

    pct = 100*p
    if variable=="Variance_explained":
        pct = 100-pct
    ns = []
    pcts = []
    for i in random_results.N.unique().astype(int):
        tmp_pct = np.percentile(list(random_results.loc[i,variable]),pct)
        pcts.append(tmp_pct)
        ns.append(i)

    results = pd.DataFrame(pcts)
    results.index = ns
    results.columns = ["p="+str(p)]

    return results

def significant_regulons(coherence_df,variance_cutoffs,variance_explained_cutoffs):

    var_pass = []
    var_exp_pass = []
    modules_pass = []
    modules_exp_pass = []

    #for i in range(coherence_df.shape[0]):
    for i in coherence_df.index:
        tmp_n, tmp_var, tmp_var_exp, modules = list(coherence_df.iloc[i,:])
        tmp_var_cutoff = variance_cutoffs.loc[int(tmp_n),"p=0.05"]
        tmp_var_exp_cutoff = variance_explained_cutoffs.loc[int(tmp_n),"p=0.05"]

        if tmp_var <= tmp_var_cutoff:
            var_pass.append(1)
            modules_pass.append(i)
        elif tmp_var > tmp_var_cutoff:
            var_pass.append(0)
            modules_pass.append(i)

        if tmp_var_exp < tmp_var_exp_cutoff:
            var_exp_pass.append(0)
            modules_exp_pass.append(i)
        elif tmp_var_exp >= tmp_var_exp_cutoff:
            var_exp_pass.append(1)
            modules_exp_pass.append(i)

    var_passDF = pd.DataFrame(np.vstack([var_pass,modules_pass]).T)
    var_exp_passDF = pd.DataFrame(np.vstack([var_exp_pass,modules_exp_pass]).T)
    var_passDF.columns = ["Pass_Fail","Module"]
    var_exp_passDF.columns = ["Pass_Fail","Module"]

    return var_pass, var_exp_pass, var_passDF, var_exp_passDF

def wilcox_df(df,phenotype1,phenotype2):
    import pandas as pd

    # Perform wilcoxon rank-sum test for all pathway nes values across phenotypes
    wilcoxon_list = [stats.ranksums(df.loc[p,phenotype1],df.loc[p,phenotype2]) for p in df.index]
    wilcoxon_results = pd.DataFrame(np.vstack(wilcoxon_list))
    wilcoxon_results.columns = ["Statistic","p-value"]
    wilcoxon_results.index = df.index

    # Order dataframe by test statistic
    wilcoxon_results.sort_values(by="Statistic",ascending=False,inplace=True)

    return wilcoxon_results

def wilcox_binary_df(bin_df,cont_df,pr_key):
    wilcox_stats = []
    wilcox_ps = []
    for i in bin_df.index:
        phenotype1 = getMutations(i,bin_df)
        phenotype2 = setdiff(bin_df.columns,phenotype1)
        wilcox = stats.ranksums(cont_df.loc[pr_key,phenotype1],cont_df.loc[pr_key,phenotype2])

        wilcox_stats.append(wilcox[0])
        wilcox_ps.append(wilcox[1])

    wilcox_results = pd.DataFrame(np.vstack([wilcox_stats,wilcox_ps]).T)
    wilcox_results.columns = ["statistic","p-value"]
    wilcox_results.index = bin_df.index
    wilcox_results.sort_values(by="statistic",ascending=False,inplace=True)

    return wilcox_results

# =============================================================================
#
# =============================================================================

def get_downstream_tfs(tf,tf_network_ensembl):
    '''Create dictionary of TF keys and target tf elements'''
    tmp_df = tf_network_ensembl[tf_network_ensembl.Source==tf]
    tmp_act_tf = list(tmp_df[tmp_df.Edge==1].Target)
    tmp_rep_tf = list(tmp_df[tmp_df.Edge==-1].Target)
    return {"activated":tmp_act_tf,"repressed":tmp_rep_tf}

def downstream_tf_analysis(regulators,tf_network,causal_subset):
    '''Create dictionary of TF keys and target tf elements, restricted to
    align with causal perturbation of target tfs.'''
    #Generate dictionary of primary downstream tfs
    tf_tf_dictionary = {tf:get_downstream_tfs(tf,tf_network) for tf in regulators}

    #Separate tfs that are activated from those that are repressed in the causal subset
    mut_act = causal_subset[causal_subset.MutationRegulatorEdge==1].Regulator.unique()
    mut_rep = causal_subset[causal_subset.MutationRegulatorEdge==-1].Regulator.unique()

    aligned_tfs = {}
    #Create dictionary of primary downstream tfs that match causal subset direction
    for tf in mut_act:
        tmp_act_tfs = intersect(mut_act,tf_tf_dictionary[tf]["activated"])
        tmp_rep_tfs = intersect(mut_rep,tf_tf_dictionary[tf]["repressed"])
        aligned_tfs[tf] = union(tmp_act_tfs,tmp_rep_tfs)

    for tf in mut_rep:
        tmp_act_tfs = intersect(mut_rep,tf_tf_dictionary[tf]["activated"])
        tmp_rep_tfs = intersect(mut_act,tf_tf_dictionary[tf]["repressed"])
        aligned_tfs[tf] = union(tmp_act_tfs,tmp_rep_tfs)

    return aligned_tfs

def propagate_network(regulators,aligned_tfs_primary,n_propagate=2):
    '''Create dictionary of TF keys and aligned target tf elements.
    n-propagate sets the number of steps to propagate from TF source
    when collecting targets. If n_propagate=2, the direct targets of
    a TF, and the direct targets of each of the TF's direct targets
    are included in the dictionary. coverage_dict is the TF-targets
    dictionary. span_dict is the number of propagation steps needed to
    reach every tf in the network (only useful if n_propagate>2).'''

    import time
    t1 = time.time()

    span_dict = {}
    coverage_dict = {}
    for tf in regulators:
        tf_list = [tf]

        for iteration in range(n_propagate):
            more_tfs = []
            for reg in tf_list:
                next_tfs = aligned_tfs_primary[reg]
                unique_tfs = setdiff(next_tfs,tf_list)
                more_tfs.extend(unique_tfs)

            if len(more_tfs) == 0:
                break

            #Extend tf_list to include tfs discovered in this iteration
            tf_list.extend(more_tfs)

            if len(tf_list)>1:
                tf_list = list(set(tf_list))

            if len(tf_list) == len(regulators):
                break

        coverage_dict[tf] = tf_list
        if len(tf_list) == len(regulators):
            span_dict[tf] = iteration

        elif len(tf_list) < len(regulators):
            span_dict[tf] = len(regulators)

    t2 = time.time()
    print("completed network propagation in {:.3e} seconds".format(t2-t1))

    return coverage_dict, span_dict

def dict_to_jaccard(coverage_dict):

    import numpy as np
    coverage_keys = list(coverage_dict.keys())

    #Generate pairwise matrix of Jaccard distance between TFs in network
    jaccard_matrix = np.zeros((len(coverage_keys),len(coverage_keys)))
    for i in range(len(coverage_keys)):
        key1 = coverage_keys[i]
        tgts1 = coverage_dict[key1]
        for j in range(i,len(coverage_keys)):
            key2 = coverage_keys[j]
            tgts2 = coverage_dict[key2]
            len_intersection = len(intersect(tgts1,tgts2))
            len_union = len(union(tgts1,tgts2))
            if len_union == 0:
                jaccard = 0
            if len_intersection == 0:
                jaccard = 0
            if len_union>0:
                jaccard = float(len_intersection)/len_union

            jaccard_matrix[i,j] = jaccard
            jaccard_matrix[j,i] = jaccard

    #Format as dataframe
    jaccard_matrix = pd.DataFrame(jaccard_matrix)
    jaccard_matrix.index = coverage_keys
    jaccard_matrix.columns = coverage_keys

    return jaccard_matrix

def df_to_jaccard(df):

    index = list(df.index)

    #Generate pairwise matrix of Jaccard distance between TFs in network
    jaccard_matrix = np.zeros((len(index),len(index)))
    for i in range(len(index)):
        key1 = index[i]
        tgts1 = df.columns[df.loc[key1,:]==1]
        for j in range(i,len(index)):
            key2 = index[j]
            tgts2 = df.columns[df.loc[key2,:]==1]
            len_intersection = len(intersect(tgts1,tgts2))
            len_union = len(union(tgts1,tgts2))
            if len_union == 0:
                jaccard = 0
            if len_intersection == 0:
                jaccard = 0
            if len_union>0:
                jaccard = float(len_intersection)/len_union

            jaccard_matrix[i,j] = jaccard
            jaccard_matrix[j,i] = jaccard

    #Format as dataframe
    jaccard_matrix = pd.DataFrame(jaccard_matrix)
    jaccard_matrix.index = index
    jaccard_matrix.columns = index

    return jaccard_matrix

def cluster_distance_matrix(dist_matrix,pct_threshold=80):
    import numpy as np

    #Set parameters for binarization
    tst = dist_matrix.copy()
    values = np.hstack(tst.values)
    thresh = np.percentile(values[values>0],pct_threshold)
    tst[tst<thresh]=0
    tst[tst>0]=1

    #Cluster binarized data
    unmix_tst = unmix(tst)

    #Match missing tfs to the best available group
    missing_tfs = setdiff(dist_matrix.index,np.hstack(unmix_tst))
    if len(missing_tfs) > 0:
        for tmp_tf in missing_tfs:
            tmp_jacc = dist_matrix.loc[setdiff(dist_matrix.index,[tmp_tf]),tmp_tf]
            tmp_argmax = tmp_jacc.index[np.argmax(list(tmp_jacc))]
            if dist_matrix.loc[tmp_tf,tmp_argmax] > 0:
                tmp_match = [i for i in range(len(unmix_tst)) if tmp_argmax in unmix_tst[i]]
                if len(tmp_match) == 0:
                    unmix_tst.append([tmp_tf])
                elif len(tmp_match) > 0:
                    unmix_tst[tmp_match[0]].append(tmp_tf)
            elif dist_matrix.loc[tmp_tf,tmp_argmax] == 0:
                unmix_tst.append([tmp_tf])

    return unmix_tst

def select_optimal_tfs(unmix_tst,coverage_dict):
    import numpy as np
    max_coverage_tfs = []
    for l in unmix_tst:
        maxcov_tf = [len(coverage_dict[tf]) for tf in l]
        max_tf = np.array(l)[np.argmax(maxcov_tf)]
        max_coverage_tfs.append(max_tf)
    return max_coverage_tfs

def infer_master_regulators(coverage_dict, network_degrees, unmix_tst):
    #Select TF with greatest coverage from each cluster
    max_coverage_tfs = select_optimal_tfs(unmix_tst,coverage_dict)

    #Order selected TFs by out/in-degree ratio (highest ratio of direct targets)
    selected_regulator_degrees = network_degrees.loc[
        [network_degrees.index[np.where(network_degrees.Alt_ID==tf)[0]][0] for tf in max_coverage_tfs],:]
    selected_regulator_degrees.dropna(inplace=True)
    selected_regulator_degrees.sort_values(by="out/in",ascending=False,inplace=True)
    selected_regulator_degrees

    #Optimize proprtion of network recovered for the smallest number of TFs.
    #Use cut-off of at least 5% coverage increase to include TF.
    mapped_regs = []
    network_coverage = []
    for i in range(selected_regulator_degrees.shape[0]):
        tmp_regs = coverage_dict[selected_regulator_degrees.iloc[i,0]]
        mapped_regs = union(mapped_regs,tmp_regs)
        network_coverage.append(len(mapped_regs)/float(network_degrees.shape[0]))

    #Apply cutoff for calling regulator "master regulator"
    tol = 0.05 #0.05 means don't count regulator if it contributes < 5%
    diff_ = np.diff(network_coverage)
    cut = len(diff_)
    for j in range(len(diff_)):
        if diff_[j]<tol:
            cut=j
            break

    #Select master regulators
    master_regulator_list = list(selected_regulator_degrees.index[0:cut+1])
    print("Master regulators:",master_regulator_list)

    #Make dictionary of master regulators and downstream targets
    master_regulator_dict = {tf:coverage_dict[network_degrees.loc[tf,"Alt_ID"]] for tf in master_regulator_list}

    return master_regulator_list, master_regulator_dict, selected_regulator_degrees

def master_partners(master_regulator_list,tf_network):
    loops = []
    for mstr in master_regulator_list:
        tmp_sources = list(tf_network[tf_network.Target==mstr].Source)
        tmp_targets = list(tf_network[tf_network.Source==mstr].Target)
        pot_hits = list(set(tmp_sources)&set(tmp_targets))

        if len(pot_hits) > 0:
            for hit in pot_hits:
                tmp_df1 = tf_network[tf_network.Target==mstr]
                tmp_df1.index = tmp_df1.Source
                tmp_hit_mstr_edge = float(tmp_df1.loc[hit,"Edge"])

                tmp_df2 = tf_network[tf_network.Source==mstr]
                tmp_df2.index = tmp_df2.Target
                tmp_hit_edge_mstr = float(tmp_df2.loc[hit,"Edge"])

                if tmp_hit_mstr_edge>0:
                    if tmp_hit_edge_mstr>0:
                        loops.append([mstr,hit])

    if len(loops) == 0:
        print("No master partners discovered")

    elif len(loops) > 0:
        print("Discovered master partners:",loops)

    return loops

def causal_significance_test(causal_results,m_matrix,mut,
                             regulonDf,eigengenes,expressionData,network_activity_diff,
                             n_iter=10):

    t1 = time.time()
    causal_results_subset = causal_results[causal_results.Mutation==mut]
    regulon_subset = list(causal_results_subset.index)

    aligned_results = []
    diff_aligned_results = []
    for iteration in range(n_iter):
        #identify subtypes
        mut_pats = getMutations(mut,m_matrix)

        #randomize patients
        mut_pats = sample(m_matrix.columns,n=len(mut_pats),replace=False)
        wt_pats = setdiff(m_matrix.columns,mut_pats)

        tmp_aligned_results = []
        tmp_diff_aligned_results = []
        for reg in regulon_subset:

            #get regulator
            regulator = regulonDf[regulonDf.Regulon_ID==int(reg)].Regulator.unique()[0]
            #get regulons downstream of regulator
            downstream_regulons = regulonDf[regulonDf.Regulator==regulator].Regulon_ID.unique().astype(str)

            #downstream differential expression
            diff_exp_stat = []
            diff_exp_p = []
            regreg_r = []
            regreg_p = []
            for dr in downstream_regulons:
                #differential expression
                s, p = stats.ranksums(eigengenes.loc[dr,mut_pats],
                               eigengenes.loc[dr,wt_pats])
                diff_exp_stat.append(s)
                diff_exp_p.append(p)
                #correlation to regulator
                if regulator in network_activity_diff.index:
                    r, pval = stats.spearmanr(network_activity_diff.loc[regulator,:],
                                             eigengenes.loc[dr,:])
                else:
                    r, pval = stats.spearmanr(expressionData.loc[regulator,:],
                                             eigengenes.loc[dr,:])
                regreg_r.append(r)
                regreg_p.append(pval)

            #mutation regulator edge
            if regulator in network_activity_diff.index:
                mutreg_stat, mutreg_p = stats.ranksums(network_activity_diff.loc[regulator,mut_pats],
                                   network_activity_diff.loc[regulator,wt_pats])
            else:
                mutreg_stat, mutreg_p = stats.ranksums(expressionData.loc[regulator,mut_pats],
                                   expressionData.loc[regulator,wt_pats])

            mutreg_edge = np.sign(mutreg_stat)

            #regulator regulon edge
            regreg_edge = np.sign(regreg_r)

            #mutation-regulator-regulon edge
            mut_reg_reg = mutreg_edge*regreg_edge

            #alignment mask
            aligned = np.array(mut_reg_reg==np.sign(diff_exp_stat))

            #diff exp mask
            diff_mask = np.array(np.array(diff_exp_p)<0.05)

            #diff exp and aligned mask
            diff_aligned = aligned*diff_mask

            #Fraction differentially expressed and aligned
            fraction_diff_aligned = np.mean(diff_aligned)

            #Fraction aligned, regardless of differential expression
            fraction_aligned = np.mean(aligned)

            #append tmp results
            tmp_aligned_results.append(fraction_aligned)
            tmp_diff_aligned_results.append(fraction_diff_aligned)

        #append results
        aligned_results.append(tmp_aligned_results)
        diff_aligned_results.append(tmp_diff_aligned_results)


    aligned_results_df = pd.DataFrame(np.vstack(aligned_results).T)
    aligned_results_df.index = regulon_subset
    diff_aligned_results_df = pd.DataFrame(np.vstack(diff_aligned_results).T)
    diff_aligned_results_df.index = regulon_subset

    t2 = time.time()

    print((t2-t1)/60.)

    #calculate permutation statistics
    perm_mean = diff_aligned_results_df.mean(axis=1)
    perm_std = diff_aligned_results_df.std(axis=1)
    perm_95 = perm_mean+2*perm_std

    #collect results
    results = pd.concat([perm_mean,perm_std,perm_95,causal_results_subset.Fraction_of_aligned_and_diff_exp_edges],axis=1)
    results.columns = ["mean","std","upper_95","Fraction_of_aligned_and_diff_exp_edges"]

    return results

def make_data_df(results,cutoffs):
    #instantiate lists
    lbls = []
    vals = []
    thresh = []
    sig = []
    flows = []

    for cutoff in cutoffs:
        #subset results by cutoff
        tmp_results = results[results.Fraction_of_aligned_and_diff_exp_edges>cutoff]
        tmp_significant_calls = tmp_results[tmp_results.Fraction_of_aligned_and_diff_exp_edges>tmp_results.upper_95]

        #retrieve permutation and observed values
        tmp_results.upper_95.values
        tmp_results.Fraction_of_aligned_and_diff_exp_edges.values

        #calculate proportion of significant hits
        passed = tmp_significant_calls.shape[0]/tmp_results.shape[0]

        #stack labels
        kind = np.hstack([["Permutation" for i in range(len(tmp_results.upper_95.values))],
                         ["Observed" for i in range(len(tmp_results.Fraction_of_aligned_and_diff_exp_edges.values))]])
        lbls.extend(kind)

        #stack values
        values = np.hstack([tmp_results.upper_95.values,
                            tmp_results.Fraction_of_aligned_and_diff_exp_edges.values])
        vals.extend(values)

        #stack cutoff
        cut = [cutoff for i in range(len(values))]
        thresh.extend(cut)

        #stack number of flows
        n_flow = tmp_results.shape[0]
        num_flows = [n_flow for i in range(len(values))]
        flows.extend(num_flows)

        #stack fraction passed
        sig_fraction = [passed for i in range(len(values))]
        sig.extend(sig_fraction)


    data_df = pd.DataFrame(np.vstack([vals,lbls,thresh,flows,sig]).T)
    data_df.columns = ["value","type","cutoff","flows","significant"]
    data_df["value"] = pd.to_numeric(data_df["value"])
    #data_df["cutoff"] = pd.to_numeric(data_df["cutoff"])
    data_df["flows"] = pd.to_numeric(data_df["flows"])
    data_df["significant"] = pd.to_numeric(data_df["significant"])

    return data_df
# =============================================================================
# Functions related to pathway enrichments
# =============================================================================

def chi_square(binary_1,binary_2,table=False):
    from scipy.stats import chi2_contingency
    obs = pd.crosstab(binary_1,binary_2)
    chi2, p, dof, ex = chi2_contingency(obs, correction=False)
    #Add directionality to chisq statistic
    sum1 = obs.iloc[0,0]+obs.iloc[1,1]
    sum2 = obs.iloc[0,1]+obs.iloc[1,0]

    sign = 1
    if sum2>sum1:
        sign=-1
    chi2 = sign*chi2

    if table is True:
        return chi2, p, obs

    return chi2, p

def chisquare_binary_df(bin_df,ref_df,pr_key):
    chisq_stats = []
    chisq_ps = []
    for i in bin_df.index:
        phenotype1 = getMutations(i,bin_df)
        phenotype2 = setdiff(bin_df.columns,phenotype1)

        a = ref_df.loc[pr_key,phenotype1]>0
        bin_vector = (1*a)
        count_ones_a = sum(bin_vector)
        count_zeros_a = len(bin_vector)-count_ones_a

        b = ref_df.loc[pr_key,phenotype2]>0
        bin_vector = (1*b)
        count_ones_b = sum(bin_vector)
        count_zeros_b = len(bin_vector)-count_ones_b

        cont_tbl = np.array([[count_ones_a,count_zeros_a],[count_ones_b,count_zeros_b]])
        chi2, p, dof, ex = stats.chi2_contingency(cont_tbl)

        sign = ref_df.loc[pr_key,phenotype1].mean()-ref_df.loc[pr_key,phenotype2].mean()
        if sign==0:
            sign = 1
        else:
            sign = sign/abs(sign)

        chi2 = sign*chi2
        chisq_stats.append(chi2)
        chisq_ps.append(p)

    chisq_results = pd.DataFrame(np.vstack([chisq_stats,chisq_ps]).T)
    chisq_results.columns = ["statistic","p-value"]
    chisq_results.index = bin_df.index
    chisq_results.sort_values(by="statistic",ascending=False,inplace=True)

    return chisq_results

def decision_tree_predictor(ref_df,target,test_proportion=0.35,rs=12,depth=2,criterion="entropy"):
    #import functions
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    #Format for prediction
    predictionMat = pd.concat([ref_df.T,target],axis=1)

    #Convert to numpy arrays for sklearn
    X = np.array(predictionMat.iloc[:,0:-1])
    Y = np.array(predictionMat.iloc[:,-1])
    X = X.astype(float)
    Y = Y.astype('int')

    #List sample names so that the training and test labels can be retrieved
    samples_ = np.array(predictionMat.index)

    #Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_proportion, random_state = rs)
    X_train_columns, X_test_columns, y_train_samples, y_test_samples = train_test_split(X, samples_, test_size = test_proportion, random_state = rs)

    #Generate decision tree classifier
    clf = DecisionTreeClassifier(criterion = criterion,
                                 random_state = rs,
                                 max_depth=depth,
                                 min_samples_leaf=3)
    clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test, y_train_samples, y_test_samples

def visualize_decision_tree(clf,filename=None):
    #Visualize decision tree
    from sklearn import tree
    from sklearn.externals.six import StringIO
    import pydot
    from graphviz import Source

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())

    # visualize decision tree
    graph[0].set_graph_defaults(size = "\"8,8\"")
    decisionTree = Source(graph[0].to_string())

    if filename is not None:
        graph[0].write_pdf(filename)

    return decisionTree

def combinatorial_risk(target,reference_matrix,covariate_matrix,primary_variable,
                       min_group_size=5,min_mean_size=10,sort=True,target_type="continuous"):

    #Determine primary variable type
    indicator = len(set(reference_matrix.loc[primary_variable,:]))
    if indicator <= 2:
        primary_type = "binary"
    elif indicator >2:
        primary_type = "categorical"

    #Get target variable names outside of loop to save time
    target_index = target.index
    target_mean = target.mean()

    #Instantiate output lists
    kw_p_list = []
    index_list = []
    mean_list = []

    phenotypes = []
    #Split population according to primary variable
    if primary_type == "binary":
        phenotype1 = getMutations(primary_variable,reference_matrix)
        phenotypes.append(phenotype1)
        phenotypes.append(setdiff(reference_matrix.columns,phenotype1))

    if primary_type == "categorical":
        for cat in list(set(reference_matrix.loc[primary_variable,:])):
            phenotypes.append(reference_matrix.columns[
                np.where(reference_matrix.loc[primary_variable,:]==cat)[0]])

    if primary_type == "continuous":
        phenotypes.append(reference_matrix.columns)

    #Split population into subtypes based upon primary and categorical variables
    for covariate in covariate_matrix.index:
        interaction_subtypes = []
        cats = list(set(covariate_matrix.loc[covariate,:]))
        for cat in cats:
            subtype = covariate_matrix.columns[
                np.where(covariate_matrix.loc[covariate,:]==cat)[0]]
            for phenotype in phenotypes:
                cat_subtype = intersect(phenotype,subtype)
                interaction_subtypes.append(cat_subtype)

        if covariate == primary_variable:
            interaction_subtypes = phenotypes

        # Perform statistical test on subtypes
        if target_type == "continuous":
            #kruskal-wallis of subtypes
            kw_subtypes = []
            lenkw = 0
            for group in interaction_subtypes:
                kw_subtype = intersect(target_index,group)
                if len(kw_subtype) >= min_group_size:
                    kw_subtypes.append(target[kw_subtype])
                    lenkw+=1
            if lenkw >= 2:
                kw_stat, kw_p = stats.kruskal(*kw_subtypes)
                kw_p_list.append(kw_p)
                index_list.append(covariate)
                means = [np.mean(i) for i in kw_subtypes if len(intersect(i.index,target_index))>=min_mean_size]
                max_mean = max(means)
                mean_list.append(max_mean)
            elif lenkw < 2:
                kw_p_list.append(1)
                index_list.append(covariate)
                mean_list.append(target_mean)
                continue


    results = pd.DataFrame(np.vstack([kw_p_list,mean_list]).T)
    results.index = index_list
    results.columns = ["p-value","highest_risk"]
    if sort is True:
        results.sort_values(by="p-value",ascending=True,inplace=True)

    return results

def subset_survival(patients,srv):
    intersection = list(set(patients)&set(srv.index))
    tmp_srv = srv.loc[intersection,:]
    tmp_srv.sort_values(by="GuanScore",ascending=False,inplace=True)
    return tmp_srv

def check_nans(lst,pass_only=True):
    nans = [i for i in range(len(lst)) if type(lst[i])==float]
    pass_ = list(set(list(range(len(lst))))-set(nans))

    if pass_only is False:
        return (pass_,nans)
    return pass_

def scenic_regulons_to_df(regulons):
    import pandas as pd
    import numpy as np

    # Write scenic regulon information into miner format
    index_col = []
    tf_col = []
    gene_col = []
    for i in range(len(regulons)):
        tmp_index = regulons[i].name
        tmp_tf = tmp_index.split("(")[0]
        tmp_regulon = list(dict(regulons[i].gene2weight).keys())
        for j in range(len(tmp_regulon)):
            tf_id = tmp_tf
            gene_id = tmp_regulon[j]
            index_col.append(tmp_index)
            tf_col.append(tf_id)
            gene_col.append(gene_id)

    #Convert output to dataframe
    scenic_regulonDf = pd.DataFrame(np.vstack([index_col,tf_col,gene_col]).T)
    scenic_regulonDf.columns = ["Regulon_ID","Regulator","Gene"]

    return scenic_regulonDf

def pairwise_distance(df):
    from sklearn.metrics import pairwise_distances
    sample_distances = pairwise_distances(np.array(df.T),metric="euclidean")
    sample_distances = pd.DataFrame(sample_distances)
    sample_distances.columns = df.columns
    sample_distances.index = df.columns

    return sample_distances

def permute_matrix(unscrambled):
    import pandas as pd
    import numpy as np
    #Generate permutation matrix
    permuted_mutations = pd.DataFrame(np.zeros(unscrambled.shape))
    permuted_mutations.index = unscrambled.index
    permuted_mutations.columns = unscrambled.columns
    num_muts = list(unscrambled.sum(axis=1))

    for i in range(permuted_mutations.shape[0]):
        ix = permuted_mutations.index[i]
        rand_muts = sample(unscrambled.columns,int(num_muts[i]),replace=False)
        permuted_mutations.loc[ix,rand_muts] = 1
    return permuted_mutations


def optimize_causal_flows(task):

    causal_results,mutation_matrix,regulonDf,eigengenes,expressionData,network_activity_diff,n_iter,cutoff=task[1]

    #input task list, output dataframe
    all_muts = causal_results.Mutation.unique()
    tmp_mut_list = intersect(all_muts,mutation_matrix.index)
    mut_list = tmp_mut_list[task[0][0]:task[0][1]]

    mut_names = []
    sig_flow_results = []
    for i in range(len(mut_list)):
        mut = mut_list[i]
        tmp_causal = causal_significance_test(causal_results,mutation_matrix,mut,
                                 regulonDf,eigengenes,expressionData,network_activity_diff,
                                 n_iter=n_iter)

        #test significance against optimized threshold
        tmp_results = tmp_causal[tmp_causal.Fraction_of_aligned_and_diff_exp_edges>cutoff]
        if tmp_results.shape[0]>0:
            tmp_significant_calls = tmp_results[tmp_results.Fraction_of_aligned_and_diff_exp_edges>tmp_results.upper_95]
            if tmp_significant_calls.shape[0]==0:
                continue

            mut_names.append(mut)
            #list regulators
            regulators = [
                regulonDf[regulonDf.Regulon_ID==int(reg)]["Regulator"].unique()[0]
                for reg in tmp_significant_calls.index
            ]
            #format significant causal flows
            tmp_cm_df = pd.DataFrame(list(zip(
                [mut for i in range(tmp_significant_calls.shape[0])],
                regulators,
                tmp_significant_calls.index)))
            tmp_cm_df.columns = ["Mutation","Regulator","Regulon"]

            #add dataframe to list for concatenation
            sig_flow_results.append(tmp_cm_df)

    if len(sig_flow_results)==0:
        sig_flow_results = pd.DataFrame([[],[],[]]).T
        sig_flow_results.columns = ["Mutation","Regulator","Regulon"]

    final_cm_results = pd.concat(sig_flow_results,axis=0)

    return final_cm_results


def parallel_causal_significance(causal_results,
                                 mutation_matrix,
                                 regulonDf,
                                 eigengenes,
                                 expressionData,
                                 network_activity_diff,
                                 n_iter=10,
                                 cutoff=0.2,
                                 n_cores = 5,
                                 savefile=None):

    #start timer
    import time
    t_start = time.time()
    #filter causal flows by optimized threshold
    all_muts = causal_results.Mutation.unique()
    tmp_mut_list = intersect(all_muts,mutation_matrix.index)
    print("Analyzing {:d} mutations across {:d} threads".format(len(tmp_mut_list),n_cores))

    taskSplit = splitForMultiprocessing(tmp_mut_list,n_cores)
    taskData = (causal_results,mutation_matrix,regulonDf,eigengenes,
                expressionData,network_activity_diff,n_iter,cutoff)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    output = multiprocess(optimize_causal_flows,tasks)
    results = pd.concat(output,axis=0)

    t_end = time.time()

    print("completed analysis in {:.2f} minutes".format((t_end-t_start)/60.))

    if savefile is not None:
        results.to_csv(savefile)

    return results


## generate regulon activity from gene expression data
## regulonModules - dictionary Completed 100 of 760 iterationsdefining genes for each regulon
## expressionData - table of  normalized rnaseq read counts
##
def generateRegulonActivity(regulonModules, expressionData, p=0.05, returnBkgd="no"):

    # select reference dictionary for downstream analysis (pr_genes, revisedClusters, coexpressionModules, or regulonModules)
    referenceDictionary = regulonModules

    # create a background matrix used for statistical hypothesis testing
    bkgd = background_df(expressionData)

    # for each cluster, give samples that show high coherent cluster activity
    ## label 0 -down active, 1 - inactive, 2 - up active
    overExpressedMembers = biclusterMembershipDictionary(referenceDictionary,bkgd,label=2,p=p)
    # for each clus|ter, give samples that show low coherent cluster activity
    underExpressedMembers = biclusterMembershipDictionary(referenceDictionary,bkgd,label=0,p=p)
    # convert overExpressedMembers dictionary to binary matrix
    overExpressedProgramsMatrix = membershipToIncidence(overExpressedMembers,expressionData)
    # convert underExpressedMembers dictionary to binary matrix
    underExpressedProgramsMatrix = membershipToIncidence(underExpressedMembers,expressionData)

    # Create program matrix with values of {-1,0,1}
    dfr_programs = overExpressedProgramsMatrix-underExpressedProgramsMatrix

    if returnBkgd == "yes":
        out = bkgd
    else:
        out=dfr_programs

    return out


def make_entrez_map(conversion_table_path):
    result = {}
    idMap = pd.read_csv(conversion_table_path, sep="\t")
    genetypes = list(set(idMap.iloc[:,2]))
    for index, row in idMap.iterrows():
        preferred, name, source = row
        if source.startswith('Entrez'):
            result[preferred] = name
    return result
