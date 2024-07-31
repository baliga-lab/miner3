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
    with open("testdata/init_clusters-001.json") as infile:
        ref_init_clusters = json.load(infile)
    init_clusters = miner.cluster(exp,
                                  minNumberGenes=6,
                                  minNumberOverExpSamples=4,
                                  maxSamplesExcluded=0.5,
                                  random_state=12,
                                  overExpressionThreshold=80)
    #with open("init_clusters-001.json", "w") as outfile:
    #    json.dump(init_clusters, outfile)

    for cluster in init_clusters:
        assert(len(cluster) >= 6)
    assert(len(ref_init_clusters) == len(init_clusters))
    for i in range(len(init_clusters)):
        assert(sorted(ref_init_clusters[i]) == sorted(init_clusters[i]))

EPS = 0.00001

def test_get_axes():
    cluster = []
    with open("testdata/cluster1-00.txt") as infile:
        for line in infile:
            cluster.append(line.strip())
    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    with open("testdata/ref_axes-000.json") as infile:
        ref_axes = json.load(infile)

    axes = miner.get_axes({"1": cluster}, exp, random_state=12)
    json_axes = {}
    for key, arr  in axes.items():
        ref_arr = ref_axes[key]
        json_axes[key] = list(arr)
        assert(len(arr) == len(ref_arr))
        for i in range(len(arr)):
            assert(abs(arr[i] - ref_arr[i]) < EPS)


"""
def test_recursive_decomposition():
    cluster = []
    with open("testdata/cluster1-00.txt") as infile:
        for line in infile:
            cluster.append(line.strip())
    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)

    ref_recdecomp = []
    with open("testdata/ref_recdecomp.txt") as infile:
        for line in infile:
            ref_recdecomp.append(line.strip().split(" "))

    rec_decomp = miner.recursive_decomposition(cluster, exp, 6, 80)
    assert(ref_recdecomp == rec_decomp)
"""

def test_reconstruction():
    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    decomposed_list = []
    with open("testdata/ref_recdecomp.txt") as infile:
        for line in infile:
            decomposed_list.append(line.strip().split(" "))
    cluster = miner.reconstruction(decomposed_list, exp, random_state=12)
    #with open("ref_reconstruction_cluster.json", "w") as outfile:
    #    json.dump(cluster, outfile)
    with open("testdata/ref_reconstruction_cluster-001.json") as infile:
        ref_cluster = json.load(infile)
    for cluster_num, cluster_members in ref_cluster.items():
        assert(cluster_members == cluster[int(cluster_num)])


def test_recursive_alignment():
    cluster = []
    with open("testdata/cluster1-00.txt") as infile:
        for line in infile:
            cluster.append(line.strip())
    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    ref_recalign = []
    with open("testdata/ref_recalign.txt") as infile:
        for line in infile:
            ref_recalign.append(line.strip().split(" "))

    rec_align = miner.recursive_alignment(cluster, exp, 6, 80)
    #with open("ref_recalign.txt", "w") as outfile:
    #    outfile.write("%s" % ' '.join(sorted(rec_align[0])))
    assert(ref_recalign[0] == rec_align[0])


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
    assert(ref_cluster_nums == cluster_nums)
    for cluster_num in cluster_nums:
        assert(sorted(ref_revised_clusters[cluster_num]) ==
               sorted(revised_clusters[cluster_num]))


def test_mechanistic_inference_tfbsdb1():
    with open("testdata/revised_clusters-002.json") as infile:
        revised_clusters = json.load(infile)
    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    axes = miner.principal_df(revised_clusters, exp,
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

def test_background_df():
    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    ref_background = pd.read_csv("testdata/ref_background_df-001.csv", index_col=0)
    df = miner.background_df(exp)
    #df.to_csv("ref_background_df-001.csv")
    print(df)
    print(ref_background)
    assert(df.equals(ref_background))


def test_principal_df():
    with open("testdata/revised_clusters-002.json") as infile:
        revised_clusters = json.load(infile)
    tmp = {}
    # convert string keys to int
    for cluster, genes in revised_clusters.items():
        tmp[int(cluster)] = genes
    revised_clusters = tmp
    exp = pd.read_csv('testdata/exp_data_preprocessed-002.csv', header=0,
                      index_col=0)
    ref_pdf = pd.read_csv("testdata/ref_principal_df-001.csv", header=0,
                          index_col=0)

    #print("REVISED_CLUSTERS: ", revised_clusters)
    pdf = miner.principal_df(revised_clusters, exp, subkey=None)
    #pdf.to_csv("ref_principal_df-001.csv")
    assert(pdf.shape == ref_pdf.shape)
    assert(sorted(pdf.index) == sorted(ref_pdf.index))
    assert(sorted(pdf.columns) == sorted(ref_pdf.columns))
    result = pdf.compare(ref_pdf)
    #print("TESTRESULT: ", result)

