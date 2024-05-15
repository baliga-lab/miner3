#!/usr/bin/env python3
import sys
import os
import pytest

import pandas as pd
from miner import miner
import json
import pickle

from .regulon_quality import confirm_regulon_coregulation



def test_cluster():
    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    with open("testdata/init_clusters-002.json") as infile:
        ref_init_clusters = json.load(infile)
    init_clusters = miner.cluster(exp,
                                  minNumberGenes=6,
                                  minNumberOverExpSamples=4,
                                  maxSamplesExcluded=0.5,
                                  random_state=12,
                                  overExpressionThreshold=80)
    #assert(ref_init_clusters == init_clusters)
    for cluster in init_clusters:
        assert(len(cluster) >= 6)


def test_revise_initial_clusters():
    with open("testdata/init_clusters-002.json") as infile:
        init_clusters = json.load(infile)

    with open("testdata/revised_clusters-002.json") as infile:
        ref_revised_clusters = json.load(infile)

    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    revised_clusters = miner.reviseInitialClusters(init_clusters, exp)
    assert(ref_revised_clusters, revised_clusters)


def test_mechanistic_inference_tfbsdb1():
    with open("testdata/revised_clusters-002.json") as infile:
        revised_clusters = json.load(infile)
    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    axes = miner.principalDf(revised_clusters, exp,
                             subkey=None, minNumberGenes=1)
    database_path = os.path.join('miner', 'data', 'network_dictionaries', 'tfbsdb_tf_to_genes.pkl')
    with open('testdata/mechinf-002.json') as infile:
        ref_mechanistic_output = json.load(infile)

    mechanistic_output = miner.mechanisticInference(axes,
                                                    revised_clusters,
                                                    exp,
                                                    correlationThreshold=0.2,
                                                    numCores=5,
                                                    database_path=database_path)
    assert(ref_mechanistic_output, mechanistic_output)
