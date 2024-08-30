"""subtypes.py - subtypes computation module"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

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

    states, centroid_clusters = miner.inferSubtypes(reference_matrix, primary_matrix,
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
