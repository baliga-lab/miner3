#!/usr/bin/env python3
import sys
import os
import pytest

import pandas as pd
from miner import miner
import json
import pickle

from .regulon_quality import confirm_regulon_coregulation

def test_cluster_quality_tfbsdb1():
    """
    Test cluster quality by comparing to coregulation using
    TFBSDB
    """
    database_path = 'miner/data/network_dictionaries/tfbsdb_tf_to_genes.pkl'
    with open(database_path, 'rb') as infile:
        tf2genes =  pickle.load(infile)

    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    #with open("testdata/init_clusters-001.json") as infile:
    #    ref_init_clusters = json.load(infile)
    init_clusters = miner.cluster(exp,
                                  minNumberGenes=6,
                                  minNumberOverExpSamples=4,
                                  maxSamplesExcluded=0.5,
                                  random_state=12,
                                  overExpressionThreshold=80)


    revised_clusters = miner.reviseInitialClusters(init_clusters, exp)
    axes = miner.principalDf(revised_clusters, exp,
                             subkey=None, minNumberGenes=1)

    mechanistic_output = miner.mechanisticInference(axes, revised_clusters,
                                                    exp,
                                                    correlationThreshold=0.2,
                                                    numCores=4,
                                                    database_path=database_path)
    coregulation_modules = miner.getCoregulationModules(mechanistic_output)
    regulons = miner.getRegulons(coregulation_modules,
                                 minNumberGenes=5,
                                 freqThreshold=0.333)
    regulon_modules, regulon_df = miner.regulonDictionary(regulons)

    overlap_rate = confirm_regulon_coregulation(tf2genes, regulon_df)
    #print("OVERLAP: ", overlap_rate)
    assert(abs(overlap_rate - 1.0) < 0.0001)


def test_cluster_quality_tfbsdb2():
    """
    Test cluster quality by comparing to coregulation using
    TFBSDB
    """
    database_path = 'miner/data/network_dictionaries/tfbsdb2_tf_to_genes.pkl'
    with open(database_path, 'rb') as infile:
        tf2genes =  pickle.load(infile)

    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    #with open("testdata/init_clusters-001.json") as infile:
    #    ref_init_clusters = json.load(infile)
    init_clusters = miner.cluster(exp,
                                  minNumberGenes=6,
                                  minNumberOverExpSamples=4,
                                  maxSamplesExcluded=0.5,
                                  random_state=12,
                                  overExpressionThreshold=80)


    revised_clusters = miner.reviseInitialClusters(init_clusters, exp)
    axes = miner.principalDf(revised_clusters, exp,
                             subkey=None, minNumberGenes=1)

    mechanistic_output = miner.mechanisticInference(axes, revised_clusters,
                                                    exp,
                                                    correlationThreshold=0.2,
                                                    numCores=4,
                                                    database_path=database_path)
    coregulation_modules = miner.getCoregulationModules(mechanistic_output)
    regulons = miner.getRegulons(coregulation_modules,
                                 minNumberGenes=5,
                                 freqThreshold=0.333)
    regulon_modules, regulon_df = miner.regulonDictionary(regulons)

    overlap_rate = confirm_regulon_coregulation(tf2genes, regulon_df)
    #print("OVERLAP: ", overlap_rate)
    assert(abs(overlap_rate - 1.0) < 0.0001)

