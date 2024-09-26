import pandas as pd
import numpy as np
from scipy import stats
import os
import time

def causalNetworkAnalysis(regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,resultsDirectory,minRegulons=1,significance_threshold=0.05,causalFolder="causal_results"):
    if not os.path.isdir(resultsDirectory):
        os.mkdir(resultsDirectory)
    # create results directory
    causal_path = os.path.join(resultsDirectory,causalFolder)
    if not os.path.isdir(causal_path):
        os.mkdir(causal_path)

    t1 = time.time()

    ###
    regulon_df_bcindex = regulon_matrix.copy()
    regulon_df_bcindex.index = np.array(regulon_df_bcindex["Regulon_ID"]).astype(str)

    regulon_df_gene_index = regulon_matrix.copy()
    regulon_df_gene_index.index = regulon_df_gene_index["Gene"]

    tf_name = []
    bc_name = []
    rs_1 = []
    ps_1 = []
    index_1 = []

    # Test correlation of TF and cluster eigengene
    missing_tfs = list(set(regulon_df_bcindex.loc[:,"Regulator"])-set(expression_matrix.index))
    for key in list(set(regulon_df_bcindex.index)):
        e_gene = reference_matrix.loc[str(key),:]
        tf = list(regulon_df_bcindex.loc[key,"Regulator"])[0]
        if tf not in missing_tfs:
            tf_exp = expression_matrix.loc[tf,reference_matrix.columns]
            r, p = stats.spearmanr(tf_exp, e_gene)
        else:
            r, p = (0,1)
        tf_name.append(tf)
        bc_name.append(key)
        rs_1.append(r)
        ps_1.append(p)
        index_1.append(key)

    correlation_df_bcindex = pd.DataFrame(np.vstack([tf_name,bc_name,rs_1,ps_1]).T) # Table
    correlation_df_bcindex.columns = ["Regulator","Regulon_ID","Spearman_R","Spearman_p"]
    correlation_df_bcindex.index = np.array(index_1).astype(str)

    correlation_df_regulator_index = correlation_df_bcindex.copy() # Table
    correlation_df_regulator_index.index = correlation_df_regulator_index["Regulator"]

    # loop through each mutation
    for mut_ix in range(mutation_matrix.shape[0]):
        time.sleep(0.01)
        mutation_name = mutation_matrix.index[mut_ix]

        phenotype_2 = mutation_matrix.columns[mutation_matrix.loc[mutation_name,:]==1]
        phenotype_1 = list(set(mutation_matrix.columns)-set(phenotype_2))
        phenotype_2 = list(set(phenotype_2)&set(reference_matrix.columns))
        phenotype_1 = list(set(phenotype_1)&set(reference_matrix.columns))

        # Welch's t-test of cluster eigengene versus mutation status
        regulon_ttests = pd.DataFrame(
            np.vstack(
                stats.ttest_ind(reference_matrix.loc[:,phenotype_2],reference_matrix.loc[:,phenotype_1],equal_var=False,axis=1)
            ).T
        )

        regulon_ttests.index = reference_matrix.index
        regulon_ttests.columns = ["Regulon_t-test_t","Regulon_t-test_p"] # Table1: eigengenes ttests

        result_dfs = []
        mean_ts = []
        mean_significance = []

        # Identify TFs that are regulators of clusters, but not members of any clusters
        upstream_regulators = list(set(regulon_matrix.Regulator)-set(regulon_df_gene_index.index))

        # Iterate over regulators that are also members of clusters
        for regulator_ in list(set(regulon_matrix.Regulator)&set(regulon_df_gene_index.index)):  # analyze all regulators in also members in cluster
        #for regulator_ in list(set(regulon_matrix.Regulator)|set(regulon_df_gene_index.index)): # analyze all regulators in regulon_matrix
            if regulator_ not in upstream_regulators:
                # Collect all clusters in which the regulator is a member (i.e., included in the cluster of genes)
                tmp = regulon_df_gene_index.loc[regulator_,"Regulon_ID"]
                if type(tmp) is not pd.core.series.Series:
                    regulons_ = [str(tmp)]
                elif type(tmp) is pd.core.series.Series:
                    regulons_ = list(np.array(tmp).astype(str))

                neglogps = []
                ts = []

                # Collect the significance of the differential cluster expression for all clusters to which the regulator belongs
                for regulon_ in regulons_:
                    t, p = list(regulon_ttests.loc[regulon_,:])
                    tmp_neglogp = -np.log10(p)
                    neglogps.append(tmp_neglogp)
                    ts.append(t)

                mean_ts = np.mean(ts)
                mean_significance = np.mean(neglogps)

            else:
                #If the regulator does not belong to any clusters, just use its normalized expression
                xt, xp = stats.ttest_ind(expression_matrix.loc[regulator_,phenotype_2],expression_matrix.loc[regulator_,phenotype_1],equal_var=False)
                mean_ts = xt
                mean_significance = -np.log10(xp)

            if mean_significance >= -np.log10(significance_threshold):
                downstream_tmp = correlation_df_regulator_index.loc[regulator_,"Regulon_ID"]
                if type(downstream_tmp) is not pd.core.series.Series:
                    downstream_regulons = [str(downstream_tmp)]
                elif type(downstream_tmp) is pd.core.series.Series:
                    downstream_regulons = list(np.array(downstream_tmp).astype(str))

                if len(downstream_regulons)<minRegulons:
                    continue

                d_neglogps = []
                d_ts = []
                # Check for significant difference in expression of clusters regulated BY the regulator
                for downstream_regulon_ in downstream_regulons:
                    dt, dp = list(regulon_ttests.loc[downstream_regulon_,:])
                    tmp_neglogp = -np.log10(dp)
                    d_neglogps.append(tmp_neglogp)
                    d_ts.append(dt)

                d_neglogps = np.array(d_neglogps)
                d_ts = np.array(d_ts)

                mask = np.where(d_neglogps >= -np.log10(significance_threshold))[0]
                if len(mask) == 0:
                    continue

                # Significance of differential cluster expression in target clusters of regulator
                significant_regulons = np.array(downstream_regulons)[mask]
                significant_regulon_ts = d_ts[mask]
                significant_regulon_ps = d_neglogps[mask]

                # Significance of tf-cluster correlation for downstream clusters
                significant_Rs = np.array(correlation_df_bcindex.loc[significant_regulons,"Spearman_R"]).astype(float)
                significant_ps = np.array(correlation_df_bcindex.loc[significant_regulons,"Spearman_p"]).astype(float)

                # (mean_ts = mean t-statistic of mutation-regulator edge)*(array of corr. coeff. of tf-cluster edges)*(array of t-statistics for downstream regulon diff. exp. in WT vs mutant)
                assignment_values = mean_ts*significant_Rs*significant_regulon_ts
                #assignments = assignment_values/np.abs(assignment_values)
                alignment_mask = np.where(assignment_values>0)[0]

                if len(alignment_mask) == 0:
                    continue

                mutation_list = np.array([mutation_name for i in range(len(alignment_mask))])
                regulator_list = np.array([regulator_ for i in range(len(alignment_mask))])
                bicluster_list = significant_regulons[alignment_mask]
                mutation_regulator_edge_direction = np.array([mean_ts/np.abs(mean_ts) for i in range(len(alignment_mask))])
                mutation_regulator_edge_ps = np.array([mean_significance for i in range(len(alignment_mask))])
                regulator_bicluster_rs = significant_Rs[alignment_mask]
                regulator_bicluster_ps = significant_ps[alignment_mask]
                bicluster_ts = significant_regulon_ts[alignment_mask]
                bicluster_ps = significant_regulon_ps[alignment_mask]
                fraction_aligned = np.array([len(alignment_mask)/float(len(mask)) for i in range(len(alignment_mask))])
                fraction_effected = np.array([len(alignment_mask)/float(len(d_neglogps)) for i in range(len(alignment_mask))]) #New addition
                n_downstream = np.array([len(d_neglogps) for i in range(len(alignment_mask))])
                n_diff_exp = np.array([len(mask)for i in range(len(alignment_mask))])

                results_ = pd.DataFrame(
                    np.vstack(
                        [
                            mutation_list,
                            regulator_list,
                            bicluster_list,
                            mutation_regulator_edge_direction,
                            mutation_regulator_edge_ps,
                            regulator_bicluster_rs,
                            regulator_bicluster_ps,
                            bicluster_ts,
                            bicluster_ps,
                            fraction_aligned,
                            fraction_effected,
                            n_downstream,
                            n_diff_exp
                        ]
                    ).T
                )

                results_.columns = [
                    "Mutation",
                    "Regulator",
                    "Regulon",
                    "MutationRegulatorEdge",
                    "-log10(p)_MutationRegulatorEdge",
                    "RegulatorRegulon_Spearman_R",
                    "RegulatorRegulon_Spearman_p-value",
                    "Regulon_stratification_t-statistic",
                    "-log10(p)_Regulon_stratification",
                    "Fraction_of_edges_correctly_aligned",
                    "Fraction_of_aligned_and_diff_exp_edges",
                    "number_downstream_regulons",
                    "number_differentially_expressed_regulons"#New addition
                ]

                results_.index = bicluster_list

                result_dfs.append(results_)

            elif mean_significance < -np.log10(significance_threshold):
                continue

        if len(result_dfs) == 0:
            continue
        elif len(result_dfs) == 1:
            causal_output = result_dfs[0]
        if len(result_dfs) > 1:
            causal_output = pd.concat(result_dfs,axis=0)

        output_file = ("").join([mutation_name,"_causal_results",".csv"])
        causal_output.to_csv(os.path.join(causal_path,output_file))

    t2 = time.time()
    print('completed causal analysis in {:.2f} minutes'.format((t2-t1)/60.))

def readCausalFiles(directory):
    sample_dfs = []
    for subdir in os.listdir(directory):
        for fname in os.listdir(os.path.join(directory, subdir)):
            #print('\t%s' % fname)
            extension = fname.split(".")[-1]
            if extension == 'csv':
                path = os.path.join(directory, subdir, fname)
                #print("READING CAUSAL RESULTS FROM \"%s\"" % path)
                df = pd.read_csv(path, index_col=0, header=0)
                df.index = np.array(df.index).astype(str)
                sample_dfs.append(df)

    causalData = pd.concat(sample_dfs,axis=0)
    renamed = [("-").join(["R",str(name)]) for name in causalData.Regulon]
    causalData.Regulon = renamed
    return causalData


def wiringDiagram(causal_results, regulonModules, coherent_samples_matrix,
                  include_genes=False, savefile=None):
    cytoscape_output = []
    for regulon in list(set(causal_results.index)):

        genes = regulonModules[str(regulon)]
        samples = coherent_samples_matrix.columns[coherent_samples_matrix.loc[int(regulon),:]==1]
        condensed_genes = (";").join(genes)
        condensed_samples = (";").join(samples)
        causal_info = causal_results.loc[regulon,:]
        if type(causal_info) is pd.core.frame.DataFrame:
            for i in range(causal_info.shape[0]):
                mutation = causal_info.iloc[i,0]
                reg = causal_info.iloc[i,1]
                tmp_edge1 = causal_info.iloc[i,3]
                if tmp_edge1 >0:
                    edge1 = "up-regulates"
                elif tmp_edge1 <0:
                    edge1 = "down-regulates"
                tmp_edge2 = causal_info.iloc[i,5]
                if tmp_edge2 >0:
                    edge2 = "activates"
                elif tmp_edge2 <0:
                    edge2 = "represses"

                if include_genes is True:
                    cytoscape_output.append([mutation,edge1,reg,edge2,regulon,condensed_genes,condensed_samples])
                elif include_genes is False:
                    cytoscape_output.append([mutation,edge1,reg,edge2,regulon])

        elif type(causal_info) is pd.core.series.Series:
            for i in range(causal_info.shape[0]):
                mutation = causal_info.iloc[0]
                reg = causal_info.iloc[1]
                tmp_edge1 = causal_info.iloc[3]
                if tmp_edge1 >0:
                    edge1 = "up-regulates"
                elif tmp_edge1 <0:
                    edge1 = "down-regulates"
                tmp_edge2 = causal_info.iloc[5]
                if tmp_edge2 >0:
                    edge2 = "activates"
                elif tmp_edge2 <0:
                    edge2 = "represses"

                if include_genes is True:
                    cytoscape_output.append([mutation,edge1,reg,edge2,regulon,condensed_genes,condensed_samples])
                elif include_genes is False:
                    cytoscape_output.append([mutation,edge1,reg,edge2,regulon])

    cytoscapeDf = pd.DataFrame(np.vstack(cytoscape_output))

    if include_genes is True:
        cytoscapeDf.columns = ["mutation","mutation-regulator_edge","regulator","regulator-regulon_edge","regulon","genes","samples"]
    elif include_genes is False:
        cytoscapeDf.columns = ["mutation","mutation-regulator_edge","regulator","regulator-regulon_edge","regulon"]

    sort_by_regulon = np.argsort(np.array(cytoscapeDf["regulon"]).astype(int))
    cytoscapeDf = cytoscapeDf.iloc[sort_by_regulon,:]
    cytoscapeDf.index = cytoscapeDf["regulon"]
    rename = [("-").join(["R",name]) for name in cytoscapeDf.index]
    cytoscapeDf.loc[:,"regulon"] = rename
    if savefile is not None:
        cytoscapeDf.to_csv(savefile)

    return cytoscapeDf
