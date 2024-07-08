#!/usr/bin/env python3

import pandas as pd
import pickle

def confirm_regulon_coregulation(tf2genes, model):
    overlaps = []
    regulons = set(model["Regulon_ID"])
    for r in sorted(regulons):
        rows = model[model['Regulon_ID'] == r]
        regulator = list(set(rows["Regulator"]))[0]
        binding_profile = set(tf2genes[regulator])
        regulon_genes = set(rows["Gene"])
        overlap = len(binding_profile & regulon_genes) / len(regulon_genes)
        overlaps.append(overlap)
    return len(overlaps) / len(regulons)


if __name__ == '__main__':
    with open('tfbsdb_tf_to_genes.pkl', 'rb') as infile:
        tf2genes =  pickle.load(infile)
    df = pd.read_csv('regulonDf.csv')
    overlap_rate = confirm_regulon_coregulation(tf2genes, df)
    print(overlap_rate)
