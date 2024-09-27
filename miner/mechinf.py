"""mechinf.py - Mechanistic inference module"""

import os
import json
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import time
import numpy as np

from miner import miner, util

NUM_CORES = 5
MIN_REGULON_GENES = 5


def mechanistic_inference(exp_data, mapfile, revised_clusters, database_path,
                          outdir, mincorr, skip_tpm, firmout, genelist):
    # get first principal component axes of clusters
    t1 = time.time()

    axes = miner.principal_df(revised_clusters, exp_data,
                              subkey=None, minNumberGenes=1)

    # analyze revised clusters for enrichment in relational database
    # (default: transcription factor binding site database)
    mechanistic_output = miner.mechanisticInference(axes, revised_clusters, exp_data,
                                                    correlationThreshold=mincorr,
                                                    numCores=NUM_CORES,
                                                    database_path=database_path)

    # write mechanistic output to .json file
    with open(os.path.join(outdir, "mechanisticOutput.json"), 'w') as outfile:
        json.dump(mechanistic_output, outfile)

    # order mechanisticOutput as {tf:{coexpressionModule:genes}}
    coregulation_modules = miner.getCoregulationModules(mechanistic_output)

    # write coregulation modules to .json file
    with open(os.path.join(outdir, "coregulationModules.json"), 'w') as outfile:
        json.dump(coregulation_modules, outfile)

    # get final regulons by keeping genes that requently appear coexpressed and associated
    # to a common regulator
    regulons = miner.getRegulons(coregulation_modules,
                                 minNumberGenes=MIN_REGULON_GENES,
                                 freqThreshold=0.333)

    # reformat regulon dictionary for consistency with revisedClusters and coexpressionModules
    regulon_modules, regulon_df = miner.regulonDictionary(regulons)

    # FIRM export: note that we do not check whether we have RefSeq or Entrez, maybe this should be
    # checked in the glue
    entrez_map = miner.make_entrez_map(mapfile)
    with open(os.path.join(outdir, firmout), 'w') as outfile:
        outfile.write('Gene\tGroup\n')
        for regulon, genes in regulon_modules.items():
            for gene in genes:
                if gene in entrez_map:
                    outfile.write('%s\t%s\n' % (entrez_map[gene], regulon))

    # OpenTargets export
    with open(os.path.join(outdir, genelist), 'w') as outfile:
        all_genes = set()
        for regulon, genes in regulon_modules.items():
            all_genes.update(genes)
        for gene in sorted(all_genes):
            outfile.write('%s\n' % gene)

    # write regulons to json file
    with open(os.path.join(outdir, "regulons.json"), 'w') as outfile:
        json.dump(regulon_modules, outfile)
    regulon_df.to_csv(os.path.join(outdir, "regulonDf.csv"))

    # define coexpression modules as composite of coexpressed regulons
    coexpression_modules = miner.getCoexpressionModules(mechanistic_output)

    # write coexpression modules to .json file
    with open(os.path.join(outdir, "coexpressionModules.json"), 'w') as outfile:
        json.dump(coexpression_modules, outfile)

    # write annotated coexpression clusters to .json file
    with open(os.path.join(outdir, "coexpressionDictionary_annotated.json"), 'w') as outfile:
        json.dump(revised_clusters, outfile)

    """
    # write annotated regulons to .json file
    with open(os.path.join(outdir, "regulons_annotated.json"), 'w') as outfile:
        json.dump(regulons, outfile)

    # reconvert coexpression modules
    annotated_coexpression_modules = mechanistic_inference.convert_dictionary(coexpression_modules, conv_table)

    # write annotated coexpression modules to .json file
    with open(os.path.join(outdir, "coexpressionModules_annotated.json"), 'w') as outfile:
        json.dump(annotated_coexpression_modules, outfile)"""


    # Get eigengenes for all modules
    eigengenes = miner.getEigengenes(regulon_modules, exp_data, regulon_dict=None,
                                     saveFolder=None)
    eigen_scale = np.percentile(exp_data, 95) / np.percentile(eigengenes, 95)
    eigengenes = eigen_scale * eigengenes
    eigengenes.index = np.array(eigengenes.index).astype(str)

    # write eigengenes to .csv
    eigengenes.to_csv(os.path.join(outdir, "eigengenes.csv"))

    t2 = time.time()
    logging.info("Completed mechanistic inference in {:.2f} minutes".format((t2 - t1) / 60.))
    logging.info("Inferred network with {:d} regulons, {:d} regulators, and {:d} co-regulated genes".format(len(regulon_df.Regulon_ID.unique()), len(regulon_df.Regulator.unique()),len(regulon_df.Gene.unique())))
