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
        ref_cluster_nums = sorted(ref_revised_clusters.keys())

    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    revised_clusters = miner.reviseInitialClusters(init_clusters, exp)
    cluster_nums = sorted(revised_clusters.keys())
    #assert(ref_revised_clusters == revised_clusters)
    assert(ref_cluster_nums == cluster_nums)
    for cluster_num in cluster_nums:
        assert(sorted(ref_revised_clusters[cluster_num]) ==
               sorted(revised_clusters[cluster_num]))


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
    assert(sorted(ref_mechanistic_output.keys()) ==
           sorted(mechanistic_output.keys()))
    for regulon in mechanistic_output.keys():
        # 1. test the contents of each regulon
        ref_elem = ref_mechanistic_output[regulon]
        elem = mechanistic_output[regulon]
        assert(sorted(ref_elem.keys()) == sorted(elem.keys()))
        # 2. now test map for each TF mapping in the regulon
        for tf in elem.keys():
            pval, genes = elem[tf]
            ref_pval, ref_genes = ref_elem[tf]
            assert(pval == ref_pval)
            assert(sorted(genes) == sorted(ref_genes))




