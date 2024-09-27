"""subtypes.py - subtypes computation module"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from miner import miner

def subtypes(exp_data, regulon_modules, outdir):
    bkgd = miner.background_df(exp_data)

    overexpressed_members = miner.biclusterMembershipDictionary(regulon_modules, bkgd, label=2,
                                                                p=0.05)
    overexpressed_members_matrix = miner.membershipToIncidence(overexpressed_members,
                                                               exp_data)
    underexpressed_members = miner.biclusterMembershipDictionary(regulon_modules, bkgd, label=0,
                                                                 p=0.05)
    underexpressed_members_matrix = miner.membershipToIncidence(underexpressed_members,
                                                                exp_data)

    sample_dictionary = overexpressed_members
    sample_matrix = overexpressed_members_matrix

    # Matt's new way of subtype discovery (START)
    # TODO: fold this into the code above
    reference_matrix = overexpressed_members_matrix - underexpressed_members_matrix
    primary_matrix = overexpressed_members_matrix
    primary_dictionary = overexpressed_members
    secondary_matrix = underexpressed_members_matrix
    secondary_dictionary = underexpressed_members

    states, centroid_clusters = inferSubtypes(reference_matrix, primary_matrix,
                                              secondary_matrix,
                                              primary_dictionary,
                                              secondary_dictionary,
                                              minClusterSize=int(np.ceil(0.01*exp_data.shape[1])),restricted_index=None)
    states_dictionary = {str(i):states[i] for i in range(len(states))}
    with open(os.path.join(outdir, "transcriptional_states.json"), 'w') as outfile:
        json.dump(states_dictionary, outfile)

    eigengenes = miner.getEigengenes(regulon_modules, exp_data,
                                     regulon_dict=None,saveFolder=None)
    eigenScale = np.percentile(exp_data, 95) / np.percentile(eigengenes, 95)
    eigengenes = eigenScale * eigengenes
    eigengenes.index = np.array(eigengenes.index).astype(str)
    eigengenes.to_csv(os.path.join(outdir, "eigengenes.csv"))
    reference_df = eigengenes.copy()
    programs, _ = miner.mosaic(dfr=reference_df, clusterList=centroid_clusters,
                               minClusterSize_x=int(np.ceil(0.01*exp_data.shape[1])),
                               minClusterSize_y=5,
                               allow_singletons=False,
                               max_groups=50,
                               saveFile=os.path.join(outdir,"regulon_activity_heatmap.pdf"),
                               random_state=12)
    transcriptional_programs, program_regulons = miner.transcriptionalPrograms(programs,
                                                                               regulon_modules)
    program_list = [program_regulons[("").join(["TP",str(i)])] for i in range(len(program_regulons))]
    programs_dictionary = {str(i):program_list[i] for i in range(len(program_list))}

    with open(os.path.join(outdir, "transcriptional_programs.json"), 'w') as outfile:
        json.dump(programs_dictionary, outfile)

    mosaic_df = reference_df.loc[np.hstack(program_list), np.hstack(states)]
    mosaic_df.to_csv(os.path.join(outdir, "regulons_activity_heatmap.csv"))

    dfr = overexpressed_members_matrix - underexpressed_members_matrix
    mtrx = dfr.loc[np.hstack(program_list),np.hstack(states)]
    plt.figure(figsize=(8,8))
    plt.imshow(mtrx, cmap="bwr", vmin=-1, vmax=1, aspect=float(mtrx.shape[1]) / float(mtrx.shape[0]))
    plt.grid(False)
    plt.savefig(os.path.join(outdir, "mosaic_all.pdf"), bbox_inches="tight")

    # Determine activity of transcriptional programs in each sample
    states_df = miner.reduceModules(df=dfr.loc[np.hstack(program_list),
                                               np.hstack(states)],programs=program_list,
                                    states=states, stateThreshold=0.50,
                                    saveFile=os.path.join(outdir, "transcriptional_programs.pdf"))

    # Cluster patients into subtypes and give the activity of each program in each subtype
    programsVsStates = miner.programsVsStates(states_df, states,
                                              filename=os.path.join(outdir, "programs_vs_states.pdf"),
                                              csvpath=os.path.join(outdir, "programs_vs_states.csv"),
                                              showplot=True)


def inferSubtypes(referenceMatrix,primaryMatrix,secondaryMatrix,primaryDictionary,secondaryDictionary,minClusterSize=5,restricted_index=None):

    t1 = time.time()

    if restricted_index is not None:
        referenceMatrix = referenceMatrix.loc[restricted_index,:]
        primaryMatrix = primaryMatrix.loc[restricted_index,:]
        secondaryMatrix = secondaryMatrix.loc[restricted_index,:]

    # perform initial subtype clustering
    similarityClusters = miner.f1Decomposition(primaryDictionary,thresholdSFM=0.1)
    similarityClusters = [list(set(cluster)&set(referenceMatrix.columns)) for cluster in similarityClusters]
    initialClasses = [i for i in similarityClusters if len(i)>4]
    if len(initialClasses)==0:
        print('No subtypes were detected')

    # expand initial subtype clusters
    centroidClusters, centroidMatrix = miner.centroidExpansion(initialClasses,
                                                               primaryMatrix,
                                                               f1Threshold=0.1,
                                                               returnCentroids=True) #0.3

    subcentroidClusters = []
    for c in range(len(centroidClusters)):
        tmp_cluster = centroidClusters[c]
        if len(tmp_cluster) < 2*minClusterSize:
            if len(tmp_cluster)>0:
                subcentroidClusters.append(tmp_cluster)
            continue

        sampleDictionary = {key:list(set(tmp_cluster)&set(secondaryDictionary[key])) for key in secondaryDictionary}
        sampleMatrix = secondaryMatrix.loc[:,tmp_cluster]

        # perform initial subtype clustering
        similarityClusters = miner.f1Decomposition(sampleDictionary,
                                                   thresholdSFM=0.1)
        initialClasses = [i for i in similarityClusters if len(i)>4]
        if len(initialClasses)==0:
            subcentroidClusters.append(tmp_cluster)
            continue

        # expand initial subtype clusters
        tmp_centroidClusters, tmp_centroidMatrix = miner.centroidExpansion(
            initialClasses,sampleMatrix,f1Threshold = 0.1,
            returnCentroids=True) #0.3
        tmp_centroidClusters.sort(key=len,reverse=True)

        if len(tmp_centroidClusters) <= 1:
            subcentroidClusters.append(tmp_cluster)
            continue

        for cc in range(len(tmp_centroidClusters)):
            new_cluster = tmp_centroidClusters[cc]
            if len(new_cluster)==0:
                continue
            if len(new_cluster) < minClusterSize:
                if cc == 0:
                    other_clusters = []
                    other_clusters.append(np.hstack(tmp_centroidClusters))
                    tmp_centroidClusters = other_clusters
                    break
                other_clusters = tmp_centroidClusters[0:cc]
                new_centroids = miner.getCentroids(other_clusters,referenceMatrix)
                unlabeled = list(set(np.hstack(tmp_centroidClusters))-set(np.hstack(other_clusters)))
                for sample in unlabeled:
                    pearson = miner.pearson_array(np.array(new_centroids).T,np.array(referenceMatrix.loc[:,sample]))
                    top_hit = np.argsort(pearson)[-1]
                    other_clusters[top_hit].append(sample)
                tmp_centroidClusters = other_clusters
                break

            elif len(new_cluster) >= minClusterSize:
                continue

        for ccc in range(len(tmp_centroidClusters)):
            if len(tmp_centroidClusters[ccc]) == 0:
                continue
            subcentroidClusters.append(tmp_centroidClusters[ccc])

    t2 = time.time()
    print("completed subtype inference in {:.2f} minutes".format((t2-t1)/60.))

    return subcentroidClusters, centroidClusters
