import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import json
import numpy as np

from miner import miner, util


def plot_expression_stats(exp_data, outdir):
    plt.figure()
    ind_exp_data = [exp_data.iloc[:,i] for i in range(len(exp_data.columns))]
    _ = plt.boxplot(ind_exp_data)
    plt.title("Patient expression profiles", fontsize=14)
    plt.ylabel("Relative expression", fontsize=14)
    plt.xlabel("Sample ID", fontsize=14)
    plt.savefig(os.path.join(outdir, "patient_expression_profiles.pdf"),
                bbox_inches="tight")

    plt.figure()
    _ = plt.hist(exp_data.iloc[0,:],bins=100, alpha=0.75)
    plt.title("Expression of single gene", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xlabel("Relative expression", fontsize=14)
    plt.savefig(os.path.join(outdir, "expression_single_gene.pdf"),
                bbox_inches="tight")

    plt.figure()
    _ = plt.hist(exp_data.iloc[:,0],bins=200,color=[0,0.4,0.8],alpha=0.75)
    plt.ylim(0, 350)
    plt.title("Expression of single patient sample", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xlabel("Relative expression", fontsize=14)
    plt.savefig(os.path.join(outdir, "expression_single_patient.pdf"),
                bbox_inches="tight")


def coexpression(expfile, mapfile, outdir, skip_tpm,
                 mingenes, minoverexpsamp, maxexclusion,
                 overexpthresh,
                 randstate):
    if not os.path.exists(expfile):
        sys.exit("expression file not found")
    if not os.path.exists(mapfile):
        sys.exit("identifier mapping file not found")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir, 'run_info.txt'), 'w') as outfile:
        util.write_dependency_infos(outfile)

    exp_data, conv_table = miner.preprocess(expfile, mapfile,
                                            do_preprocess_tpm=(skip_tpm))
    plot_expression_stats(exp_data, outdir)

    t1 = time.time()
    init_clusters = miner.cluster(exp_data,
                                  minNumberGenes=mingenes,
                                  minNumberOverExpSamples=minoverexpsamp,
                                  maxSamplesExcluded=maxexclusion,
                                  random_state=randstate,
                                  overExpressionThreshold=overexpthresh)

    revised_clusters = miner.reviseInitialClusters(init_clusters, exp_data)
    with open(os.path.join(outdir, "coexpressionDictionary.json"), 'w') as out:
        json.dump(revised_clusters, out)

    # retrieve first three clusters for visual inspection
    if len(revised_clusters) > 3:
        first_clusters = np.hstack([revised_clusters[i] for i in np.arange(3).astype(str)])

        # visualize background expression
        plt.figure(figsize=(8,4))
        plt.imshow(exp_data.loc[np.random.choice(exp_data.index, len(first_clusters), replace=False),:],
                   aspect="auto", cmap="viridis", vmin=-1,vmax=1)
        plt.grid(False)
        plt.ylabel("Genes", fontsize=20)
        plt.xlabel("Samples", fontsize=20)
        plt.title("Random selection of genes", fontsize=20)

        plt.savefig(os.path.join(outdir, "background_expression.pdf"),
                    bbox_inches="tight")

        # visualize first 10 clusters
        plt.figure(figsize=(8,4))
        plt.imshow(exp_data.loc[first_clusters,:], aspect="auto", cmap="viridis", vmin=-1, vmax=1)
        plt.grid(False)
        plt.ylabel("Genes", fontsize=20)
        plt.xlabel("Samples", fontsize=20)
        plt.title("First 3 clusters", fontsize=20)
        plt.savefig(os.path.join(outdir, "first_clusters.pdf"),
                    bbox_inches="tight")

    # report coverage
    if len(init_clusters) > 0:
        logging.info("Number of genes clustered: {:d}".format(len(set(np.hstack(init_clusters)))))
    else:
        logging.warn('No clusters detected')
    logging.info("Number of unique clusters: {:d}".format(len(revised_clusters)))

    t2 = time.time()
    logging.info("Completed clustering module in {:.2f} minutes".format((t2-t1)/60.))
