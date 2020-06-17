#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:58:16 2020

@author: mwall
"""
# =============================================================================
# Import libraries
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Ridge LOOCV
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, roc_curve
# =============================================================================
# Create directory to save output
# =============================================================================

# Path to the miner directory
input_path = os.path.join(os.path.expanduser('~'),'Desktop','GitHub','miner','miner')

# create name for results folder where output files will be saved
resultsFolder = "results_miner3_stabilized"

# name of the folder containing the miner network
#networkFolder = "miner_network_results"
networkFolder = "results_minCorrelation_0o2_50_allFiles"

# create results directory
resultsDirectory = os.path.join(os.path.split(os.getcwd())[0],resultsFolder)
if not os.path.isdir(resultsDirectory):
    os.mkdir(resultsDirectory)
 
# =============================================================================
# Import miner
# =============================================================================

os.chdir(os.path.join(input_path,'src'))
import miner

# =============================================================================
# Load primary data
# =============================================================================

# Load expression Data
expressionFile = os.path.join(input_path,"data","expression","IA12Zscore.csv")
#expressionData = pd.read_csv(expressionFile,index_col=0,header=0)
expressionData, conversionTable = miner.preprocess(expressionFile)

# Load mutations
common_mutations = pd.read_csv(os.path.join(input_path,'data','mutations','commonMutations.csv'),index_col=0,header=0)
translocations = pd.read_csv(os.path.join(input_path,'data','mutations','translocationsIA12.csv'),index_col=0,header=0)
cytogenetics = pd.read_csv(os.path.join(input_path,'data','mutations','cytogenetics.csv'),index_col=0,header=0)
cytogenetics = cytogenetics.loc[:,list(set(cytogenetics.columns)&set(expressionData.columns))]
common_patients_mutations_translocations = list(set(translocations.columns)&set(common_mutations.columns))
mutation_matrix = pd.concat([common_mutations.loc[:,common_patients_mutations_translocations],translocations.loc[:,common_patients_mutations_translocations]],axis=0)

#If previous results exist, use miner.read_json to load them
revisedClusters = miner.read_json(os.path.join(input_path,networkFolder,"coexpressionDictionary.json"))
coexpressionModules = miner.read_json(os.path.join(input_path,networkFolder,"coexpressionModules.json"))
regulonModules = miner.read_json(os.path.join(input_path,networkFolder,"regulons.json"))
mechanisticOutput = miner.read_json(os.path.join(input_path,networkFolder,"mechanisticOutput.json"))
regulonDf = pd.read_csv(os.path.join(input_path,networkFolder,"regulonDf.csv"),index_col=0,header=0)
overExpressedMembersMatrix = pd.read_csv(os.path.join(input_path,networkFolder,"overExpressedMembers.csv"),index_col=0,header=0)
overExpressedMembersMatrix.index = np.array(overExpressedMembersMatrix.index).astype(str)
underExpressedMembersMatrix = pd.read_csv(os.path.join(input_path,networkFolder,"underExpressedMembers.csv"),index_col=0,header=0)
underExpressedMembersMatrix.index = np.array(underExpressedMembersMatrix.index).astype(str)
eigengenes = pd.read_csv(os.path.join(input_path,networkFolder,"eigengenes.csv"),index_col=0,header=0)
eigengenes.index = np.array(underExpressedMembersMatrix.index).astype(str)
diff_matrix_MMRF = overExpressedMembersMatrix-underExpressedMembersMatrix

# load primary survival data (i.e., corresponding to expression data training set)
survivalMMRF = pd.read_csv(os.path.join(input_path,"data","survival","survivalIA12.csv"),index_col=0,header=0)
survivalDfMMRF = survivalMMRF.iloc[:,0:2]
survivalDfMMRF.columns = ["duration","observed"]
overExpressedMembersMatrixMMRF = overExpressedMembersMatrix
kmDfMMRF = miner.kmAnalysis(survivalDf=survivalDfMMRF,durationCol="duration",statusCol="observed")
guanSurvivalDfMMRF= miner.guanRank(kmSurvival=kmDfMMRF)

# Load transcriptional programs
transcriptional_programs = miner.read_json(os.path.join(input_path,networkFolder,'transcriptional_programs.json'))
program_list = [transcriptional_programs[str(key)] for key in range(len(transcriptional_programs.keys()))]

# Create dictionary of program genes
# make dictionary of genes by program
pr_genes = {}
for i in range(len(program_list)):
    rgns = program_list[i]
    genes = []
    for r in rgns:
        genes.append(regulonModules[r])
    genes = list(set(np.hstack(genes)))
    pr_genes[i] = genes

# =============================================================================
# Load and map validation data
# =============================================================================

# define modules to interrogate in test sets
reference_modules = regulonModules

# GSE24080UAMS - test set 1
expressionDataGSE24080UAMS = pd.read_csv(os.path.join(input_path,"data","expression","GSE24080UAMSentrezIDlevel.csv"),index_col=0,header=0)
expressionDataGSE24080UAMS, _ = miner.identifierConversion(expressionData=expressionDataGSE24080UAMS)
expressionDataGSE24080UAMS = miner.zscore(expressionDataGSE24080UAMS)
bkgdGSE24080UAMS = miner.backgroundDf(expressionDataGSE24080UAMS)
overExpressedMembersGSE24080UAMS = miner.biclusterMembershipDictionary(reference_modules,bkgdGSE24080UAMS,label=2,p=0.1)
overExpressedMembersMatrixGSE24080UAMS = miner.membershipToIncidence(overExpressedMembersGSE24080UAMS,expressionDataGSE24080UAMS)
underExpressedMembersGSE24080UAMS = miner.biclusterMembershipDictionary(reference_modules,bkgdGSE24080UAMS,label=0,p=0.1)
underExpressedMembersMatrixGSE24080UAMS = miner.membershipToIncidence(underExpressedMembersGSE24080UAMS,expressionDataGSE24080UAMS)

# GSE19784HOVON65 - test set 2
expressionDataGSE19784HOVON65 = pd.read_csv(os.path.join(input_path,"data","expression","GSE19784HOVON65entrezIDlevel.csv"),index_col=0,header=0)
expressionDataGSE19784HOVON65, _ = miner.identifierConversion(expressionData=expressionDataGSE19784HOVON65)
expressionDataGSE19784HOVON65 = miner.zscore(expressionDataGSE19784HOVON65)
bkgdGSE19784HOVON65 = miner.backgroundDf(expressionDataGSE19784HOVON65)
overExpressedMembersGSE19784HOVON65 = miner.biclusterMembershipDictionary(reference_modules,bkgdGSE19784HOVON65,label=2,p=0.1)
overExpressedMembersMatrixGSE19784HOVON65 = miner.membershipToIncidence(overExpressedMembersGSE19784HOVON65,expressionDataGSE19784HOVON65)
underExpressedMembersGSE19784HOVON65 = miner.biclusterMembershipDictionary(reference_modules,bkgdGSE19784HOVON65,label=0,p=0.1)
underExpressedMembersMatrixGSE19784HOVON65 = miner.membershipToIncidence(underExpressedMembersGSE19784HOVON65,expressionDataGSE19784HOVON65)

# EMTAB4032 - test set 3
expressionDataEMTAB4032 = pd.read_csv(os.path.join(input_path,"data","expression","EMTAB4032entrezIDlevel.csv"),index_col=0,header=0)
expressionDataEMTAB4032, _ = miner.identifierConversion(expressionData=expressionDataEMTAB4032)
expressionDataEMTAB4032 = miner.zscore(expressionDataEMTAB4032)
bkgdEMTAB4032 = miner.backgroundDf(expressionDataEMTAB4032)
overExpressedMembersEMTAB4032 = miner.biclusterMembershipDictionary(reference_modules,bkgdEMTAB4032,label=2,p=0.1)
overExpressedMembersMatrixEMTAB4032 = miner.membershipToIncidence(overExpressedMembersEMTAB4032,expressionDataEMTAB4032)
underExpressedMembersEMTAB4032 = miner.biclusterMembershipDictionary(reference_modules,bkgdEMTAB4032,label=0,p=0.1)
underExpressedMembersMatrixEMTAB4032 = miner.membershipToIncidence(underExpressedMembersEMTAB4032,expressionDataEMTAB4032)

# generate discrete network mapping matrices for each dataset
dfr = overExpressedMembersMatrix-underExpressedMembersMatrix
dfrGSE24080UAMS = overExpressedMembersMatrixGSE24080UAMS-underExpressedMembersMatrixGSE24080UAMS
dfrGSE19784HOVON65 = overExpressedMembersMatrixGSE19784HOVON65-underExpressedMembersMatrixGSE19784HOVON65
dfrEMTAB4032 = overExpressedMembersMatrixEMTAB4032-underExpressedMembersMatrixEMTAB4032

# =============================================================================
# Load and map validation data
# =============================================================================

survival = pd.read_csv(os.path.join(input_path,"data","survival","globalClinTraining.csv"),index_col=0,header=0)

# GSE24080UAMS
survivalGSE24080UAMS = survival[survival.index=='GSE24080UAMS']
survivalGSE24080UAMS.index = survivalGSE24080UAMS.iloc[:,0]
survivalDfGSE24080UAMS = survivalGSE24080UAMS.loc[:,["D_PFS","D_PFS_FLAG"]]
survivalDfGSE24080UAMS.columns = ["duration","observed"]
kmDfGSE24080UAMS = miner.kmAnalysis(survivalDf=survivalDfGSE24080UAMS,durationCol="duration",statusCol="observed")
guanSurvivalDfGSE24080UAMS = miner.guanRank(kmSurvival=kmDfGSE24080UAMS)

# GSE19784HOVON65 
survivalGSE19784HOVON65 = survival[survival.index=='HOVON65']
survivalGSE19784HOVON65.index = survivalGSE19784HOVON65.iloc[:,0]
survivalDfGSE19784HOVON65 = survivalGSE19784HOVON65.loc[:,["D_PFS","D_PFS_FLAG"]]
survivalDfGSE19784HOVON65.columns = ["duration","observed"]
kmDfGSE19784HOVON65 = miner.kmAnalysis(survivalDf=survivalDfGSE19784HOVON65,durationCol="duration",statusCol="observed")
guanSurvivalDfGSE19784HOVON65 = miner.guanRank(kmSurvival=kmDfGSE19784HOVON65)

# EMTAB4032
survivalEMTAB4032 = survival[survival.index=='EMTAB4032']
survivalEMTAB4032.index = survivalEMTAB4032.iloc[:,0]
survivalDfEMTAB4032 = survivalEMTAB4032.loc[:,["D_PFS","D_PFS_FLAG"]]
survivalDfEMTAB4032.columns = ["duration","observed"]
kmDfEMTAB4032 = miner.kmAnalysis(survivalDf=survivalDfEMTAB4032,durationCol="duration",statusCol="observed")
guanSurvivalDfEMTAB4032 = miner.guanRank(kmSurvival=kmDfEMTAB4032)

# =============================================================================
# MMRF program significance
# =============================================================================

# Regulon coherence MMRF (training set)
validation_df = expressionData.copy()
# Coherence measures of regulons
program_coherence_MMRF = miner.regulon_variance(validation_df,pr_genes)
ns = np.array(list(set(list(program_coherence_MMRF.iloc[:,0])))).astype(int)
# Coherence measures of random permutations
random_results_MMRF_programs = miner.random_regulon_variance(validation_df,ns,n_iter=500)
variance_explained_cutoffs_MMRF_programs = miner.random_significance(random_results_MMRF_programs,variable="Variance_explained",p=0.05)
variance_cutoffs_MMRF_programs = miner.random_significance(random_results_MMRF_programs,variable="Variance",p=0.05)
#Comparison of regulon variance to random variance
var_pass_MMRF_programs, var_exp_pass_MMRF_programs = miner.significant_regulons(program_coherence_MMRF,variance_cutoffs_MMRF_programs,
                                              variance_explained_cutoffs_MMRF_programs)
MMRF_percent_significant_p_0o05 = 100*sum(var_pass_MMRF_programs)/float(program_coherence_MMRF.shape[0])
print("{:.2f}% of programs are significantly coherent at p = 0.05".format(MMRF_percent_significant_p_0o05))

# =============================================================================
# GSE24080 program significance
# =============================================================================

# Regulon coherence GSE19784
validation_df = expressionDataGSE24080UAMS.copy()
# Coherence measures of regulons
program_coherence_GSE24080 = miner.regulon_variance(validation_df,pr_genes)
ns = np.array(list(set(list(program_coherence_GSE24080.iloc[:,0])))).astype(int)
# Coherence measures of random permutations
random_results_GSE24080_programs = miner.random_regulon_variance(validation_df,ns,n_iter=500)
variance_explained_cutoffs_GSE24080_programs = miner.random_significance(random_results_GSE24080_programs,variable="Variance_explained",p=0.05)
variance_cutoffs_GSE24080_programs = miner.random_significance(random_results_GSE24080_programs,variable="Variance",p=0.05)
#Comparison of regulon variance to random variance
var_pass_GSE24080_programs, var_exp_pass_GSE24080_programs = miner.significant_regulons(program_coherence_GSE24080,variance_cutoffs_GSE24080_programs,
                                              variance_explained_cutoffs_GSE24080_programs)
GSE24080_percent_significant_p_0o05 = 100*sum(var_pass_GSE24080_programs)/float(program_coherence_GSE24080.shape[0])
print("{:.2f}% of programs are significantly coherent at p = 0.05".format(GSE24080_percent_significant_p_0o05))


# =============================================================================
# GSE19784 program significance
# =============================================================================

# Regulon coherence GSE19784
validation_df = expressionDataGSE19784HOVON65.copy()
# Coherence measures of regulons
program_coherence_GSE19784 = miner.regulon_variance(validation_df,pr_genes)
ns = np.array(list(set(list(program_coherence_GSE19784.iloc[:,0])))).astype(int)
# Coherence measures of random permutations
random_results_GSE19784_programs = miner.random_regulon_variance(validation_df,ns,n_iter=500)
variance_explained_cutoffs_GSE19784_programs = miner.random_significance(random_results_GSE19784_programs,variable="Variance_explained",p=0.05)
variance_cutoffs_GSE19784_programs = miner.random_significance(random_results_GSE19784_programs,variable="Variance",p=0.05)
#Comparison of regulon variance to random variance
var_pass_GSE19784_programs, var_exp_pass_GSE19784_programs = miner.significant_regulons(program_coherence_GSE19784,variance_cutoffs_GSE19784_programs,
                                              variance_explained_cutoffs_GSE19784_programs)
GSE19784_percent_significant_p_0o05 = 100*sum(var_pass_GSE19784_programs)/float(program_coherence_GSE19784.shape[0])
print("{:.2f}% of programs are significantly coherent at p = 0.05".format(GSE19784_percent_significant_p_0o05))

# =============================================================================
# MMRF network activity
# =============================================================================

# Infer network activity
network_activity_overexpressed_MMRF = miner.networkActivity(regulon_matrix=regulonDf.copy(),
                                                 reference_matrix=overExpressedMembersMatrix.copy(),
                                                 minRegulons = 2)
network_activity_underexpressed_MMRF = miner.networkActivity(regulon_matrix=regulonDf.copy(),
                                                 reference_matrix=underExpressedMembersMatrix.copy(),
                                                 minRegulons = 2)
network_activity_diff_MMRF = network_activity_overexpressed_MMRF-network_activity_underexpressed_MMRF

# Infer transcriptional states
minClusterSize = int(np.ceil(0.01*expressionData.shape[1]))
referenceMatrix = network_activity_diff_MMRF
primaryMatrix = network_activity_overexpressed_MMRF
primaryDictionary = miner.matrix_to_dictionary(primaryMatrix,threshold=0.5)
secondaryMatrix = network_activity_underexpressed_MMRF
secondaryDictionary = miner.matrix_to_dictionary(secondaryMatrix,threshold=0.5)

np.random.seed(12)
inferred_states_MMRF, centroidClusters_MMRF = miner.inferSubtypes(referenceMatrix,primaryMatrix,secondaryMatrix,primaryDictionary,secondaryDictionary,minClusterSize = int(np.ceil(0.01*primaryMatrix.shape[1])),restricted_index=None)
states_dictionary_MMRF= {str(i):inferred_states_MMRF[i] for i in range(len(inferred_states_MMRF))}
print(len(inferred_states_MMRF),len(np.hstack(inferred_states_MMRF)))

# Infer gene clusters
states_MMRF = inferred_states_MMRF.copy() #states_list.copy()
dfr = network_activity_diff_MMRF.copy()
minClusterSize_x = minClusterSize
minClusterSize_y = 6
max_groups = 50
allow_singletons = False
random_state = 12

# Cluster genes using original transcriptional states
gene_clusters_MMRF, gene_groups_MMRF = miner.cluster_features(dfr,states_MMRF,minClusterSize_x,minClusterSize_y,
                    max_groups,allow_singletons,random_state)

# Plot clustered network activity data
plt.imshow(dfr.loc[np.hstack(gene_clusters_MMRF),np.hstack(states_MMRF)],
               cmap='bwr',aspect="auto",vmin=-1.25,vmax=1.25)
plt.grid(False)

# Plot clustered expression data
plt.imshow(expressionData.loc[np.hstack(gene_clusters_MMRF),np.hstack(states_MMRF)],
               cmap='bwr',aspect="auto",vmin=-2,vmax=2)
plt.grid(False)

# =============================================================================
# GSE24080 network activity
# =============================================================================

# Infer network activity
network_activity_overexpressed_GSE24080 = miner.networkActivity(regulon_matrix=regulonDf.copy(),
                                                 reference_matrix=overExpressedMembersMatrixGSE24080UAMS.copy(),
                                                 minRegulons = 2)
network_activity_underexpressed_GSE24080 = miner.networkActivity(regulon_matrix=regulonDf.copy(),
                                                 reference_matrix=underExpressedMembersMatrixGSE24080UAMS.copy(),
                                                 minRegulons = 2)
network_activity_diff_GSE24080 = network_activity_overexpressed_GSE24080-network_activity_underexpressed_GSE24080

# Infer transcriptional states
minClusterSize = int(np.ceil(0.01*expressionDataGSE24080UAMS.shape[1]))
referenceMatrix = network_activity_diff_GSE24080
primaryMatrix = network_activity_overexpressed_GSE24080
primaryDictionary = miner.matrix_to_dictionary(primaryMatrix,threshold=0.5)
secondaryMatrix = network_activity_underexpressed_GSE24080
secondaryDictionary = miner.matrix_to_dictionary(secondaryMatrix,threshold=0.5)

np.random.seed(12)
inferred_states_GSE24080, centroidClusters_GSE24080 = miner.inferSubtypes(referenceMatrix,primaryMatrix,secondaryMatrix,primaryDictionary,secondaryDictionary,minClusterSize = int(np.ceil(0.01*primaryMatrix.shape[1])),restricted_index=None)
states_dictionary_GSE24080 = {str(i):inferred_states_GSE24080[i] for i in range(len(inferred_states_GSE24080))}
print(len(inferred_states_GSE24080),len(np.hstack(inferred_states_GSE24080)))

# Infer gene clusters
states_GSE24080 = inferred_states_GSE24080.copy() #states_list.copy()
dfr = network_activity_diff_GSE24080.copy()
minClusterSize_x = minClusterSize
minClusterSize_y = 6
max_groups = 50
allow_singletons = False
random_state = 12

# Cluster genes using original transcriptional states
gene_clusters_GSE24080, gene_groups_GSE24080 = miner.cluster_features(dfr,states_GSE24080,minClusterSize_x,minClusterSize_y,
                    max_groups,allow_singletons,random_state)

# Plot clustered network activity data
plt.imshow(dfr.loc[np.hstack(gene_clusters_GSE24080),np.hstack(states_GSE24080)],
               cmap='bwr',aspect="auto",vmin=-1.25,vmax=1.25)
plt.grid(False)

# Plot clustered expression data
plt.imshow(expressionDataGSE24080UAMS.loc[np.hstack(gene_clusters_GSE24080),np.hstack(states_GSE24080)],
               cmap='bwr',aspect="auto",vmin=-2,vmax=2)
plt.grid(False)

# =============================================================================
# GSE19784 network activity
# =============================================================================

# Infer network activity
network_activity_overexpressed_GSE19784 = miner.networkActivity(regulon_matrix=regulonDf.copy(),
                                                 reference_matrix=overExpressedMembersMatrixGSE19784HOVON65.copy(),
                                                 minRegulons = 2)
network_activity_underexpressed_GSE19784 = miner.networkActivity(regulon_matrix=regulonDf.copy(),
                                                 reference_matrix=underExpressedMembersMatrixGSE19784HOVON65.copy(),
                                                 minRegulons = 2)
network_activity_diff_GSE19784 = network_activity_overexpressed_GSE19784-network_activity_underexpressed_GSE19784

# Infer transcriptional states
minClusterSize = int(np.ceil(0.01*expressionDataGSE19784HOVON65.shape[1]))
referenceMatrix = network_activity_diff_GSE19784
primaryMatrix = network_activity_overexpressed_GSE19784
primaryDictionary = miner.matrix_to_dictionary(primaryMatrix,threshold=0.5)
secondaryMatrix = network_activity_underexpressed_GSE19784
secondaryDictionary = miner.matrix_to_dictionary(secondaryMatrix,threshold=0.5)

np.random.seed(12)
inferred_states_GSE19784, centroidClusters_GSE19784 = miner.inferSubtypes(referenceMatrix,primaryMatrix,secondaryMatrix,primaryDictionary,secondaryDictionary,minClusterSize = int(np.ceil(0.01*primaryMatrix.shape[1])),restricted_index=None)
states_dictionary_GSE19784 = {str(i):inferred_states_GSE19784[i] for i in range(len(inferred_states_GSE19784))}
print(len(inferred_states_GSE19784),len(np.hstack(inferred_states_GSE19784)))

# Infer gene clusters
states_GSE19784 = inferred_states_GSE19784.copy() #states_list.copy()
dfr = network_activity_diff_GSE19784.copy()
minClusterSize_x = minClusterSize
minClusterSize_y = 6
max_groups = 50
allow_singletons = False
random_state = 12

# Cluster genes using original transcriptional states
gene_clusters_GSE19784, gene_groups_GSE19784 = miner.cluster_features(dfr,states_GSE19784,minClusterSize_x,minClusterSize_y,
                    max_groups,allow_singletons,random_state)

# Plot clustered network activity data
plt.imshow(dfr.loc[np.hstack(gene_clusters_GSE19784),np.hstack(states_GSE19784)],
               cmap='bwr',aspect="auto",vmin=-1.25,vmax=1.25)
plt.grid(False)

# Plot clustered expression data
plt.imshow(expressionDataGSE19784HOVON65.loc[np.hstack(gene_clusters_GSE19784),np.hstack(states_GSE19784)],
               cmap='bwr',aspect="auto",vmin=-2,vmax=2)
plt.grid(False)

# =============================================================================
# Gene expression vs network activity
# =============================================================================

# MMRF
net_genes = []
net_rs = []
net_ps = []
for gene in network_activity_diff_MMRF.index:
    r, p = stats.spearmanr(expressionData.loc[gene,:],network_activity_diff_MMRF.loc[gene,:])
    net_genes.append(gene)
    net_rs.append(r)
    net_ps.append(p)
    
_=plt.hist(net_rs)

# GSE24080
net_genes_GSE24080 = []
net_rs_GSE24080 = []
net_ps_GSE24080 = []

for gene in miner.intersect(network_activity_diff_GSE24080.index,expressionDataGSE24080UAMS.index):
    r, p = stats.spearmanr(expressionDataGSE24080UAMS.loc[gene,:],network_activity_diff_GSE24080.loc[gene,:])
    net_genes_GSE24080.append(gene)
    net_rs_GSE24080.append(r)
    net_ps_GSE24080.append(p)
    
_=plt.hist(net_rs_GSE24080)

# GSE19784
net_genes_GSE19784 = []
net_rs_GSE19784 = []
net_ps_GSE19784 = []

for gene in miner.intersect(network_activity_diff_GSE19784.index,expressionDataGSE19784HOVON65.index):
    r, p = stats.spearmanr(expressionDataGSE19784HOVON65.loc[gene,:],network_activity_diff_GSE19784.loc[gene,:])
    net_genes_GSE19784.append(gene)
    net_rs_GSE19784.append(r)
    net_ps_GSE19784.append(p)
    
_=plt.hist(net_rs_GSE19784)

# Comparison
_=plt.hist([net_rs,net_rs_GSE24080,net_rs_GSE19784],bins=20)

# =============================================================================
# Define essential genes for investigation
# =============================================================================

# subtype genes
nsd2 = "ENSG00000109685"
ccnd1 = "ENSG00000110092"
maf = "ENSG00000178573"
myc = "ENSG00000136997"

# first-line therapy targets
crbn = "ENSG00000113851"
ikzf1 = "ENSG00000185811"
ikzf3 = "ENSG00000161405"
nr3c1 = "ENSG00000113580"
psmb5 = "ENSG00000100804"

# proliferation genes
cks1b = "ENSG00000173207"
foxm1 = "ENSG00000111206"
e2f1 = "ENSG00000101412"
pcna = "ENSG00000132646"
phf19 = "ENSG00000119403"

# apoptosis genes
puma = "ENSG00000105327"
bim = "ENSG00000153094"
noxa = "ENSG00000141682"
bak = "ENSG00000030110"
bcl2a1 = "ENSG00000140379"
bcl2 = "ENSG00000171791"
bclxl = "ENSG00000171552"
mcl1 = "ENSG00000143384"

# targets of alternative therapies
hdac6 = "ENSG00000094631"
aph1a = "ENSG00000117362"
parp1 = "ENSG00000143799"
kif11 = "ENSG00000138160"
gsk3b = "ENSG00000082701"
aurka = "ENSG00000087586"
aurkb = "ENSG00000178999"
hdac2 = "ENSG00000196591"
anpep = "ENSG00000166825"

# =============================================================================
# Infer translocations from gene expression GSE24080
# =============================================================================

fig = plt.figure(constrained_layout=True,figsize=(12,4))
# Add subplots
gs = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

# Plot histograms
_= ax1.hist(expressionDataGSE24080UAMS.loc[ccnd1,:],bins=100)
_= ax2.hist(expressionDataGSE24080UAMS.loc[nsd2,:],bins=100)
_= ax3.hist(expressionDataGSE24080UAMS.loc[maf,:],bins=100)

t1114_threshold_GSE24080 = 1.1
ccnd1_GSE24080 = expressionDataGSE24080UAMS.columns[
    expressionDataGSE24080UAMS.loc[ccnd1,:]>t1114_threshold_GSE24080]

t414_threshold_GSE24080 = 1.0
nsd2_GSE24080 = expressionDataGSE24080UAMS.columns[
    expressionDataGSE24080UAMS.loc[nsd2,:]>t414_threshold_GSE24080]

t1416_threshold_GSE24080 = 2.0
maf_GSE24080 = expressionDataGSE24080UAMS.columns[
    expressionDataGSE24080UAMS.loc[maf,:]>t1416_threshold_GSE24080]

# =============================================================================
# Infer translocations from gene expression GSE19784
# =============================================================================

fig = plt.figure(constrained_layout=True,figsize=(12,4))
# Add subplots
gs = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

# Plot histograms
_= ax1.hist(expressionDataGSE19784HOVON65.loc[ccnd1,:],bins=100)
_= ax2.hist(expressionDataGSE19784HOVON65.loc[nsd2,:],bins=100)
_= ax3.hist(expressionDataGSE19784HOVON65.loc[maf,:],bins=100)

t1114_threshold_GSE19784 = 0.8
ccnd1_GSE19784 = expressionDataGSE19784HOVON65.columns[
    expressionDataGSE19784HOVON65.loc[ccnd1,:]>t1114_threshold_GSE19784]

t414_threshold_GSE19784 = 1.2
nsd2_GSE19784 = expressionDataGSE19784HOVON65.columns[
    expressionDataGSE19784HOVON65.loc[nsd2,:]>t414_threshold_GSE19784]

t1416_threshold_GSE19784 = 1.75
maf_GSE19784 = expressionDataGSE19784HOVON65.columns[
    expressionDataGSE19784HOVON65.loc[maf,:]>t1416_threshold_GSE19784]

# =============================================================================
# Visualize risk in GSE24080
# =============================================================================

test_gene = kif11
subtype_expression_df_GSE24080 = network_activity_diff_GSE24080
subtype_risk_df_GSE24080 = guanSurvivalDfGSE24080UAMS
spear_r_GSE24080, spear_p_GSE24080 = stats.spearmanr(subtype_expression_df_GSE24080.loc[test_gene,subtype_risk_df_GSE24080.index],
           subtype_risk_df_GSE24080.loc[:,"GuanScore"])
title_str_GSE24080 = "GSE24080: p="+"{:.2e}".format(spear_p_GSE24080)

plt.figure()
plt.scatter(subtype_expression_df_GSE24080.loc[test_gene,subtype_risk_df_GSE24080.index],
           subtype_risk_df_GSE24080.loc[:,"GuanScore"])
plt.title(title_str_GSE24080)

# =============================================================================
# Visualize risk in GSE19784
# =============================================================================

test_gene = kif11
subtype_expression_df_GSE19784 = network_activity_diff_GSE19784
subtype_risk_df_GSE19784 = guanSurvivalDfGSE19784HOVON65

plt.scatter(subtype_expression_df_GSE19784.loc[test_gene,subtype_risk_df_GSE19784.index],
           subtype_risk_df_GSE19784.loc[:,"GuanScore"])
plt.title("GSE19784")
stats.spearmanr(subtype_expression_df_GSE19784.loc[test_gene,subtype_risk_df_GSE19784.index],
           subtype_risk_df_GSE19784.loc[:,"GuanScore"])

# =============================================================================
# Visualize risk of specific genes
# =============================================================================

test_gene = aurka
gene_label = miner.gene_conversion(test_gene,list_symbols=True)[0]

subtype_expression_df_GSE24080 = expressionDataGSE24080UAMS# network_activity_diff_GSE24080
subtype_risk_df_GSE24080 = guanSurvivalDfGSE24080UAMS
subtype_GSE24080 = nsd2_GSE24080 #guanSurvivalDfGSE24080UAMS.index

subtype_expression_df_GSE19784 = expressionDataGSE19784HOVON65#network_activity_diff_GSE19784
subtype_risk_df_GSE19784 = guanSurvivalDfGSE19784HOVON65
subtype_GSE19784 = nsd2_GSE19784 #guanSurvivalDfGSE19784HOVON65.index

# Generate figures
fig = plt.figure(constrained_layout=True,figsize=(8,4))

# Add subplots
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

spear_r_GSE24080, spear_p_GSE24080 = stats.spearmanr(subtype_expression_df_GSE24080.loc[test_gene,subtype_GSE24080],
           subtype_risk_df_GSE24080.loc[subtype_GSE24080,"GuanScore"])
title_str_GSE24080 = "GSE24080: p="+"{:.2e}".format(spear_p_GSE24080)

spear_r_GSE19784, spear_p_GSE19784 = stats.spearmanr(subtype_expression_df_GSE19784.loc[test_gene,subtype_GSE19784],
           subtype_risk_df_GSE19784.loc[subtype_GSE19784,"GuanScore"])
title_str_GSE19784 = "GSE19784: p="+"{:.2e}".format(spear_p_GSE19784)

ax1.scatter(subtype_expression_df_GSE19784.loc[test_gene,subtype_GSE19784],
           subtype_risk_df_GSE19784.loc[subtype_GSE19784,"GuanScore"],alpha=0.5)
ax1.set_title(title_str_GSE19784)
ax1.set_ylabel(gene_label+" Activity",FontSize=14)

ax2.scatter(subtype_expression_df_GSE24080.loc[test_gene,subtype_GSE24080],
           subtype_risk_df_GSE24080.loc[subtype_GSE24080,"GuanScore"],alpha=0.5)
ax2.set_title(title_str_GSE24080)

# =============================================================================
# Validate network activity as improving risk prediction
# =============================================================================

# Identify risk-repdicting genes from MMRF training set
exp_df = expressionData.copy()
act_df = network_activity_diff_MMRF
srv_df = guanSurvivalDfMMRF.copy()
samples = guanSurvivalDfMMRF.index
lr_prop = 0.7

risk_all_MMRF, y = miner.rank_risk_samples(srv_df,samples,lr_prop = 0.7)

x = exp_df.loc[:,risk_all_MMRF]
aucs_all_MMRF, args_all_MMRF = miner.gene_aucs(np.array(x),y,return_aucs=True)
genes_all_MMRF_exp = x.index[args_all_MMRF]

act = act_df.loc[:,risk_all_MMRF]
aucs_all_MMRF, args_all_MMRF = miner.gene_aucs(np.array(act),y,return_aucs=True)
genes_all_MMRF_act = act.index[args_all_MMRF]

genes_all_MMRF = miner.intersect(genes_all_MMRF_exp,genes_all_MMRF_act)

# Calculate risk stratification AUCs in validation sets

#GSE24080
args = genes_all_MMRF
exp_df = expressionDataGSE24080UAMS
act_df = network_activity_diff_GSE24080
srv_df = guanSurvivalDfGSE24080UAMS.copy()
samples = guanSurvivalDfGSE24080UAMS.index
lr_prop = 0.7

aucs_r_all_GSE24080, aucs_x_all_GSE24080, aucs_n_all_GSE24080 = miner.risk_auc_wrapper(args,exp_df,act_df,srv_df,samples,lr_prop = 0.7)

#GSE19784
args = genes_all_MMRF
exp_df = expressionDataGSE19784HOVON65
act_df = network_activity_diff_GSE19784
srv_df = guanSurvivalDfGSE19784HOVON65.copy()
samples = guanSurvivalDfGSE19784HOVON65.index
lr_prop = 0.7

aucs_r_all_GSE19784, aucs_x_all_GSE19784, aucs_n_all_GSE19784 = miner.risk_auc_wrapper(args,exp_df,act_df,srv_df,samples,lr_prop = 0.7)

# Plot risk stratification AUCs in validation sets
bx_data = [aucs_r_all_GSE24080, aucs_x_all_GSE24080, aucs_n_all_GSE24080,
              aucs_r_all_GSE19784, aucs_x_all_GSE19784, aucs_n_all_GSE19784]
bx_names = ["Random\nGSE24080","Expression\nGSE24080","Activity\nGSE24080",
          "Random\nGSE19784","Expression\nGSE19784","Activity\nGSE19784"]

# figure related code
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(bx_data)

ax.set_title('Risk stratification AUCs', fontsize=14, fontweight='bold')
ax.set_ylabel('AUC',fontsize=14)
ax.set_xticklabels(bx_names, fontsize=10)
plt.savefig(os.path.join(resultsDirectory,"RiskExpressionActivity.pdf"),bbox_inches="tight")
# =============================================================================
# Ridge regression to predict risk from regulon activity
# =============================================================================

# Training and test set risk cut-offs
train_cut_high = 0.20
train_cut_low = 0.50
test_cut = 0.20

# Split MMRF training
diff_matrix_MMRF = overExpressedMembersMatrix-underExpressedMembersMatrix
high_risk_MMRF = guanSurvivalDfMMRF.index[0:round(guanSurvivalDfMMRF.shape[0]*train_cut_high)]
low_risk_MMRF = guanSurvivalDfMMRF.index[round(guanSurvivalDfMMRF.shape[0]*train_cut_low):]
pats_MMRF = np.hstack([high_risk_MMRF,low_risk_MMRF])
print('High-risk MMRF: {:d} samples\nLow-risk MMRF: {:d} samples'.format(
        len(high_risk_MMRF),len(low_risk_MMRF)))

# Split GSE24080 validation
diff_matrix_GSE24080UAMS = overExpressedMembersMatrixGSE24080UAMS-underExpressedMembersMatrixGSE24080UAMS
high_risk_GSE24080UAMS = guanSurvivalDfGSE24080UAMS.index[0:round(guanSurvivalDfGSE24080UAMS.shape[0]*test_cut)]
low_risk_GSE24080UAMS = guanSurvivalDfGSE24080UAMS.index[round(guanSurvivalDfGSE24080UAMS.shape[0]*test_cut):]
pats_GSE24080UAMS = np.hstack([high_risk_GSE24080UAMS,low_risk_GSE24080UAMS])
print('High-risk GSE24080: {:d} samples\nLow-risk GSE24080: {:d} samples'.format(
        len(high_risk_GSE24080UAMS),len(low_risk_GSE24080UAMS)))

# Split GSE19784 validation
diff_matrix_GSE19784HOVON65 = overExpressedMembersMatrixGSE19784HOVON65-underExpressedMembersMatrixGSE19784HOVON65
high_risk_GSE19784HOVON65 = guanSurvivalDfGSE19784HOVON65.index[0:round(guanSurvivalDfGSE19784HOVON65.shape[0]*test_cut)]
low_risk_GSE19784HOVON65 = guanSurvivalDfGSE19784HOVON65.index[round(guanSurvivalDfGSE19784HOVON65.shape[0]*test_cut):]
pats_GSE19784HOVON65 = np.hstack([high_risk_GSE19784HOVON65,low_risk_GSE19784HOVON65])
print('High-risk GSE19784: {:d} samples\nLow-risk GSE19784: {:d} samples'.format(
        len(high_risk_GSE19784HOVON65),len(low_risk_GSE19784HOVON65)))

# Split EMTAB4032 validation
diff_matrix_EMTAB4032 = overExpressedMembersMatrixEMTAB4032-underExpressedMembersMatrixEMTAB4032
high_risk_EMTAB4032 = guanSurvivalDfEMTAB4032.index[0:round(guanSurvivalDfEMTAB4032.shape[0]*test_cut)]
low_risk_EMTAB4032 = guanSurvivalDfEMTAB4032.index[round(guanSurvivalDfEMTAB4032.shape[0]*test_cut):]
pats_EMTAB4032 = np.hstack([high_risk_EMTAB4032,low_risk_EMTAB4032])
print('High-risk EMTAB4032: {:d} samples\nLow-risk EMTAB4032: {:d} samples'.format(
        len(high_risk_EMTAB4032),len(low_risk_EMTAB4032)))

# Optimize predictor parameters
feature_matrix = diff_matrix_MMRF
survival_matrix = guanSurvivalDfMMRF
savefile = os.path.join(resultsDirectory,"Ridge_optimization.pdf")

alpha_opt, mean_result, sd_result, a_range = miner.optimize_ridge_model(
        feature_matrix,survival_matrix,n_iter=250,train_cut_high = 0.20,
        train_cut_low = 0.50,max_range=5000,range_step=100,savefile=savefile)

#Note: alpha_opt = 1801 from 500 iteration optimization

# Instantiate Ridge model using optimized parameters
pats_MMRF = np.hstack([high_risk_MMRF,low_risk_MMRF])
pats_MMRF_full = guanSurvivalDfMMRF.index

X = np.array(feature_matrix.loc[:,pats_MMRF]).T
y = np.array(guanSurvivalDfMMRF.loc[pats_MMRF,"GuanScore"])
clf = Ridge(random_state=0,alpha=1801,fit_intercept=True) #alpha_opt = 1801
clf.fit(X, y) 

# Evaluate predictor on Training data using ROC AUC
pats_MMRF_full = guanSurvivalDfMMRF.index
y = np.zeros(len(pats_MMRF_full))
y[0:len(high_risk_MMRF)] = 1
decision_function_score_MMRF = clf.predict(np.array(diff_matrix_MMRF.loc[:,pats_MMRF_full]).T)
roc_MMRF = roc_auc_score(y,decision_function_score_MMRF)
print("MMRF AUC: {:.2f}".format(roc_MMRF))
fpr_MMRF, tpr_MMRF, thresholds_MMRF = roc_curve(y,decision_function_score_MMRF, pos_label=1)

# Evaluate predictor on GSE24080 data using ROC AUC
y = np.zeros(len(pats_GSE24080UAMS))
y[0:len(high_risk_GSE24080UAMS)] = 1
decision_function_score_GSE24080UAMS = clf.predict(np.array(diff_matrix_GSE24080UAMS.loc[:,pats_GSE24080UAMS]).T)
roc_GSE24080UAMS = roc_auc_score(y,decision_function_score_GSE24080UAMS)
print("GSE24080 AUC: {:.2f}".format(roc_GSE24080UAMS))
fpr_GSE24080UAMS, tpr_GSE24080UAMS, thresholds_GSE24080UAMS = roc_curve(y,decision_function_score_GSE24080UAMS, pos_label=1)

# Evaluate predictor on GSE19784 data using ROC AUC
y = np.zeros(len(pats_GSE19784HOVON65))
y[0:len(high_risk_GSE19784HOVON65)] = 1
decision_function_score_GSE19784HOVON65 = clf.predict(np.array(diff_matrix_GSE19784HOVON65.loc[:,pats_GSE19784HOVON65]).T)
roc_GSE19784HOVON65 = roc_auc_score(y,decision_function_score_GSE19784HOVON65)
print("GSE19784 AUC: {:.2f}".format(roc_GSE19784HOVON65))
fpr_GSE19784HOVON65, tpr_GSE19784HOVON65, thresholds_GSE19784HOVON65 = roc_curve(y,decision_function_score_GSE19784HOVON65, pos_label=1)

# Evaluate predictor on EMTAB4032 data using ROC AUC
y = np.zeros(len(pats_EMTAB4032))
y[0:len(high_risk_EMTAB4032)] = 1
decision_function_score_EMTAB4032 = clf.predict(np.array(diff_matrix_EMTAB4032.loc[:,pats_EMTAB4032]).T)
roc_EMTAB4032 = roc_auc_score(y,decision_function_score_EMTAB4032)
print("EMTAB4032 AUC: {:.2f}".format(roc_EMTAB4032))
fpr_EMTAB4032, tpr_EMTAB4032, thresholds_EMTAB4032 = roc_curve(y,decision_function_score_EMTAB4032, pos_label=1)

# Plot prediction results
plt.figure()
plt.plot(fpr_GSE24080UAMS,tpr_GSE24080UAMS)
plt.plot(fpr_GSE19784HOVON65,tpr_GSE19784HOVON65)
plt.plot(fpr_EMTAB4032,tpr_EMTAB4032)
plt.plot(fpr_EMTAB4032,fpr_EMTAB4032,'--k')
plt.ylabel("Sensitivity",FontSize=16)
plt.xlabel("1-Specificity",FontSize=16)
plt.title("Regulon activity",FontSize=16)
plt.legend(["GSE24080","GSE19784","EMTAB4032"])
plt.savefig(os.path.join(resultsDirectory,"ROC_regulons_all_validations.pdf"),bbox_inches="tight")

# Correlate predicted risk to observed risk

# MMRF
spearman_r_MMRF, spearman_p_MMRF = stats.spearmanr(list(guanSurvivalDfMMRF.loc[pats_MMRF_full,"GuanScore"]),
                clf.predict(np.array(diff_matrix_MMRF.loc[:,pats_MMRF_full]).T))

predicted_risk_MMRF = pd.DataFrame(
        clf.predict(np.array(diff_matrix_MMRF.loc[:,pats_MMRF_full]).T))
predicted_risk_MMRF.index = pats_MMRF_full
predicted_risk_MMRF.columns = ["risk"]
predicted_risk_MMRF = predicted_risk_MMRF.loc[guanSurvivalDfMMRF.index,:]
coxhr_predicted_MMRF = miner.survivalMedianAnalysisDirect(predicted_risk_MMRF,guanSurvivalDfMMRF)
coxhr_MMRF, coxp_MMRF = coxhr_predicted_MMRF['risk']

# GSE24080
spearman_r_GSE24080, spearman_p_GSE24080 = stats.spearmanr(list(guanSurvivalDfGSE24080UAMS.loc[pats_GSE24080UAMS,"GuanScore"]),
            clf.predict(np.array(diff_matrix_GSE24080UAMS.loc[:,pats_GSE24080UAMS]).T))

predicted_risk_GSE24080 = pd.DataFrame(
        clf.predict(np.array(dfrGSE24080UAMS.loc[:,pats_GSE24080UAMS]).T))
predicted_risk_GSE24080.index = pats_GSE24080UAMS
predicted_risk_GSE24080.columns = ["risk"]
predicted_risk_GSE24080 = predicted_risk_GSE24080.loc[guanSurvivalDfGSE24080UAMS.index,:]
coxhr_predicted_GSE24080 = miner.survivalMedianAnalysisDirect(predicted_risk_GSE24080,guanSurvivalDfGSE24080UAMS)
coxhr_GSE24080, coxp_GSE24080 = coxhr_predicted_GSE24080['risk']

# GSE19784
spearman_r_GSE19784, spearman_p_GSE19784 = stats.spearmanr(list(guanSurvivalDfGSE19784HOVON65.loc[pats_GSE19784HOVON65,"GuanScore"]),
                clf.predict(np.array(diff_matrix_GSE19784HOVON65.loc[:,pats_GSE19784HOVON65]).T))

predicted_risk_GSE19784 = pd.DataFrame(
        clf.predict(np.array(dfrGSE19784HOVON65.loc[:,pats_GSE19784HOVON65]).T))
predicted_risk_GSE19784.index = pats_GSE19784HOVON65
predicted_risk_GSE19784.columns = ["risk"]
predicted_risk_GSE19784 = predicted_risk_GSE19784.loc[guanSurvivalDfGSE19784HOVON65.index,:]
coxhr_predicted_GSE19784  = miner.survivalMedianAnalysisDirect(predicted_risk_GSE19784,guanSurvivalDfGSE19784HOVON65)
coxhr_GSE19784, coxp_GSE19784 = coxhr_predicted_GSE19784['risk']

# EMTAB4032
spearman_r_EMTAB4032, spearman_p_EMTAB4032 = stats.spearmanr(list(guanSurvivalDfEMTAB4032.loc[pats_EMTAB4032,"GuanScore"]),
                clf.predict(np.array(diff_matrix_EMTAB4032.loc[:,pats_EMTAB4032]).T))

predicted_risk_EMTAB4032 = pd.DataFrame(
        clf.predict(np.array(dfrEMTAB4032.loc[:,pats_EMTAB4032]).T))
predicted_risk_EMTAB4032.index = pats_EMTAB4032
predicted_risk_EMTAB4032.columns = ["risk"]
predicted_risk_EMTAB4032 = predicted_risk_EMTAB4032.loc[guanSurvivalDfEMTAB4032.index,:]
coxhr_predicted_EMTAB4032  = miner.survivalMedianAnalysisDirect(predicted_risk_EMTAB4032,guanSurvivalDfEMTAB4032)
coxhr_EMTAB4032, coxp_EMTAB4032 = coxhr_predicted_EMTAB4032['risk']

# Save predictor to file
pred_coefficients = pd.DataFrame(clf.coef_)
pred_coefficients.index = np.array(diff_matrix_MMRF.index).astype(str)
pred_coefficients.columns = ["Coefficient"]
pred_coefficients.to_csv(os.path.join(resultsDirectory,"RidgeCoefficientsRegulons.csv"))

# Tabulate results
Header = ["N","Median PFS","AUC","Spearman R",
          "Spearman p","Cox HR","Cox p"]

Datasets = [
        "MMRF IA12",
        "GSE24080",
        "GSE19784"
        ]

N = [
     int(guanSurvivalDfMMRF.shape[0]),
     int(guanSurvivalDfGSE24080UAMS.shape[0]),
     int(guanSurvivalDfGSE19784HOVON65.shape[0])
     ]

Median_PFS = [
        int(np.median(guanSurvivalDfMMRF.loc[:,"duration"])),
        int(np.median(guanSurvivalDfGSE24080UAMS.loc[:,"duration"])),
        int(np.median(guanSurvivalDfGSE19784HOVON65.loc[:,"duration"]))
        ]

AUC = [
       roc_MMRF,
       roc_GSE24080UAMS,
       roc_GSE19784HOVON65]

Spearman_R = [
       spearman_r_MMRF,
       spearman_r_GSE24080,
       spearman_r_GSE19784]

Spearman_p = [
       spearman_p_MMRF,
       spearman_p_GSE24080,
       spearman_p_GSE19784]

Cox_HR = [
       coxhr_MMRF,
       coxhr_GSE24080,
       coxhr_GSE19784]

Cox_p = [
       coxp_MMRF,
       coxp_GSE24080,
       coxp_GSE19784]

tabulated_results = pd.DataFrame(np.vstack([
        N,
        Median_PFS,
        AUC,
        Spearman_R,
        Spearman_p,
        Cox_HR,
        Cox_p]).T)

tabulated_results.columns = Header
tabulated_results.index = Datasets
tabulated_results.to_csv(os.path.join(resultsDirectory,"RiskPredictionResults.csv"))
