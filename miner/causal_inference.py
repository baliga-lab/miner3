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



# =============================================================================
# Functions used for causal inference, commented for now
# to see if they are actually used
# =============================================================================

"""
def causalNetworkImpact(target_genes,regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,resultsDirectory,minRegulons=1,significance_threshold=0.05,causalFolder="causal_results",return_df=False,tag=None):

    # create results directory
    if not os.path.isdir(resultsDirectory):
        os.mkdir(resultsDirectory)
    causal_path = os.path.join(resultsDirectory,causalFolder)
    if not os.path.isdir(causal_path):
        os.mkdir(causal_path)

    ###
    regulon_df_bcindex = regulon_matrix.copy()
    regulon_df_bcindex.index = np.array(regulon_df_bcindex["Regulon_ID"]).astype(str)

    regulon_df_gene_index = regulon_matrix.copy()
    regulon_df_gene_index.index = regulon_df_gene_index["Gene"]

    dfs = []
    ###
    for mut_ix in range(mutation_matrix.shape[0]):
        rows = []
        mutation_name = mutation_matrix.index[mut_ix]

        phenotype_2 = mutation_matrix.columns[mutation_matrix.loc[mutation_name,:]==1]
        phenotype_1 = list(set(mutation_matrix.columns)-set(phenotype_2))
        phenotype_2 = list(set(phenotype_2)&set(reference_matrix.columns))
        phenotype_1 = list(set(phenotype_1)&set(reference_matrix.columns))

        regulon_ttests = pd.DataFrame(
            np.vstack(
                stats.ttest_ind(reference_matrix.loc[:,phenotype_2],reference_matrix.loc[:,phenotype_1],equal_var=False,axis=1)
            ).T
        )

        regulon_ttests.index = reference_matrix.index
        regulon_ttests.columns = ["Regulon_t-test_t","Regulon_t-test_p"] # Table1: eigengenes ttests

        mean_ts = []
        mean_significance = []

        target_genes = list(set(target_genes)&set(expression_matrix.index))
        target_genes_in_network = list(set(target_genes)&set(regulon_df_gene_index.index))
        for regulator_ in target_genes: # analyze all target_genes in expression_matrix

            if regulator_ in target_genes_in_network:
                tmp = regulon_df_gene_index.loc[regulator_,"Regulon_ID"]
                if type(tmp) is not pd.core.series.Series:
                    regulons_ = [str(tmp)]
                elif type(tmp) is pd.core.series.Series:
                    regulons_ = list(np.array(tmp).astype(str))

                neglogps = []
                ts = []

                for regulon_ in regulons_:
                    t, p = list(regulon_ttests.loc[regulon_,:])
                    tmp_neglogp = -np.log10(p)
                    neglogps.append(tmp_neglogp)
                    ts.append(t)

                mean_ts = np.mean(ts)
                mean_significance = np.mean(neglogps)
                pp = 10**(-1*mean_significance)

            else:
                xt, xp = stats.ttest_ind(expression_matrix.loc[regulator_,phenotype_2],expression_matrix.loc[regulator_,phenotype_1],equal_var=False)
                mean_ts = xt
                mean_significance = -np.log10(xp)
                pp = 10**(-1*mean_significance)

            if mean_significance >= -np.log10(significance_threshold):
                results = [mutation_name,regulator_,mean_ts,mean_significance,pp]
                rows.append(results)

        if len(rows) == 0:
            continue

        output = pd.DataFrame(np.vstack(rows))
        output.columns = ["Mutation","Regulator","t-statistic","-log10(p)","p"]
        sort_values = np.argsort(np.array(output["p"]).astype(float))
        output = output.iloc[sort_values,:]

        if tag is None:
            tag = "network_impact"
        filename = ("_").join([mutation_name,tag])
        output.to_csv(("").join([os.path.join(causal_path,filename),".csv"]))
        if return_df is True:
            dfs.append(output)
    if return_df is True:
        concatenate_dfs = pd.concat(dfs,axis=0)
        concatenate_dfs.index = range(concatenate_dfs.shape[0])
        return concatenate_dfs

    return

def viewSelectedCausalResults(causalDf,selected_mutation,minimum_fraction_correctly_aligned=0.5,correlation_pValue_cutoff=0.05,regulon_stratification_pValue=0.05):
    causalDf = causalDf[causalDf.Mutation==selected_mutation]
    causalDf = causalDf[causalDf["RegulatorRegulon_Spearman_p-value"]<=correlation_pValue_cutoff]
    causalDf = causalDf[causalDf["Fraction_of_edges_correctly_aligned"]>=minimum_fraction_correctly_aligned]
    if '-log10(p)_Regulon_stratification' in causalDf.columns:
        causalDf = causalDf[causalDf["-log10(p)_Regulon_stratification"]>=-np.log10(regulon_stratification_pValue)]
    elif 'Regulon_stratification_p-value' in causalDf.columns:
        causalDf = causalDf[causalDf["Regulon_stratification_p-value"]>=-np.log10(regulon_stratification_pValue)]

    return causalDf

def causalNetworkAnalysisTask(task):

    start, stop = task[0]
    regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,minRegulons,significance_threshold,causal_path = task[1]
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
    for key in list(set(regulon_df_bcindex.index)):
        e_gene = reference_matrix.loc[str(key),:]
        tf = list(regulon_df_bcindex.loc[key,"Regulator"])[0]
        tf_exp = expression_matrix.loc[tf,reference_matrix.columns]
        r, p = stats.spearmanr(tf_exp, e_gene)
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

    ###
    for mut_ix in range(start,stop):

        mutation_name = mutation_matrix.index[mut_ix]

        phenotype_2 = mutation_matrix.columns[mutation_matrix.loc[mutation_name,:]==1]
        phenotype_1 = list(set(mutation_matrix.columns)-set(phenotype_2))

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

        upstream_regulators = list(set(regulon_matrix.Regulator)-set(regulon_df_gene_index.index))
        for regulator_ in list(set(regulon_matrix.Regulator)&set(regulon_df_gene_index.index)): # analyze all regulators in regulon_matrix

            if regulator_ not in upstream_regulators:
                tmp = regulon_df_gene_index.loc[regulator_,"Regulon_ID"]
                if type(tmp) is not pd.core.series.Series:
                    regulons_ = [str(tmp)]
                elif type(tmp) is pd.core.series.Series:
                    regulons_ = list(np.array(tmp).astype(str))

                neglogps = []
                ts = []

                for regulon_ in regulons_:
                    t, p = list(regulon_ttests.loc[regulon_,:])
                    tmp_neglogp = -np.log10(p)
                    neglogps.append(tmp_neglogp)
                    ts.append(t)

                mean_ts = np.mean(ts)
                mean_significance = np.mean(neglogps)

            else:
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

                significant_regulons = np.array(downstream_regulons)[mask]
                significant_regulon_ts = d_ts[mask]
                significant_regulon_ps = d_neglogps[mask]

                significant_Rs = np.array(correlation_df_bcindex.loc[significant_regulons,"Spearman_R"]).astype(float)
                significant_ps = np.array(correlation_df_bcindex.loc[significant_regulons,"Spearman_p"]).astype(float)

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
                            fraction_aligned
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
                    "Regulon_stratification_p-value",
                    "Fraction_of_edges_correctly_aligned"
                ]

                results_.index = bicluster_list

                result_dfs.append(results_)

            elif mean_significance < -np.log10(significance_threshold):
                continue

        causal_output = pd.concat(result_dfs,axis=0)
        output_file = ("").join([mutation_name,"_causal_results",".csv"])
        causal_output.to_csv(os.path.join(causal_path,output_file))

    return

def parallelCausalNetworkAnalysis(regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,causal_path,numCores,minRegulons=1,significance_threshold=0.05):

    # create results directory
    if not os.path.isdir(causal_path):
        os.mkdir(causal_path)

    t1 = time.time()

    taskSplit = splitForMultiprocessing(mutation_matrix.index,numCores)
    taskData = (regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,minRegulons,significance_threshold,causal_path)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    multiprocess(causalNetworkAnalysisTask,tasks)

    t2 = time.time()
    print('completed causal analysis in {:.2f} minutes'.format((t2-t1)/60.))


def biclusterTfIncidence(mechanisticOutput,regulons=None):

    if regulons is not None:
        allTfs = regulons.keys()
        tfCount = []
        ct=0
        for tf in list(regulons.keys()):
            tfCount.append([])
            for key in list(regulons[tf].keys()):
                tfCount[-1].append(str(ct))
                ct+=1

        allBcs = np.hstack(tfCount)
        bcTfIncidence = pd.DataFrame(np.zeros((len(allBcs),len(allTfs))))
        bcTfIncidence.index = allBcs
        bcTfIncidence.columns = allTfs

        for i in range(len(allTfs)):
            tf = allTfs[i]
            bcs = tfCount[i]
            bcTfIncidence.loc[bcs,tf] = 1

        index = np.sort(np.array(bcTfIncidence.index).astype(int))
        if type(bcTfIncidence.index[0]) is str:
            bcTfIncidence = bcTfIncidence.loc[index.astype(str),:]
        else:
            bcTfIncidence = bcTfIncidence.loc[index,:]

        return bcTfIncidence

    allBcs = list(mechanisticOutput.keys())
    allTfs = list(set(np.hstack([list(mechanisticOutput[i].keys()) for i in list(mechanisticOutput.keys())])))

    bcTfIncidence = pd.DataFrame(np.zeros((len(allBcs),len(allTfs))))
    bcTfIncidence.index = allBcs
    bcTfIncidence.columns = allTfs

    for bc in list(mechanisticOutput.keys()):
        bcTfs = mechanisticOutput[bc].keys()
        bcTfIncidence.loc[bc,bcTfs] = 1

    index = np.sort(np.array(bcTfIncidence.index).astype(int))
    if type(bcTfIncidence.index[0]) is str:
        bcTfIncidence = bcTfIncidence.loc[index.astype(str),:]
    else:
        bcTfIncidence = bcTfIncidence.loc[index,:]

    return bcTfIncidence

def tfExpression(expressionData,motifPath=os.path.join("..","data","all_tfs_to_motifs.pkl")):

    allTfsToMotifs = read_pkl(motifPath)
    tfs = list(set(allTfsToMotifs.keys())&set(expressionData.index))
    tfExp = expressionData.loc[tfs,:]
    return tfExp

def filterMutations(mutationPath,mutationFile,minNumMutations=None):

    filePath = os.path.join(mutationPath,mutationFile)
    mutations = pd.read_csv(filePath,index_col=0,header=0)
    if minNumMutations is None:
        minNumMutations = min(np.ceil(mutations.shape[1]*0.01),4)
    freqMuts = list(mutations.index[np.where(np.sum(mutations,axis=1)>=minNumMutations)[0]])
    filteredMutations = mutations.loc[freqMuts,:]

    return filteredMutations

def mutationMatrix(mutationPath,mutationFiles,minNumMutations=None):

    if type(mutationFiles) is str:
        mutationFiles = [mutationFiles]
    matrices = []
    for mutationFile in mutationFiles:
        matrix = filterMutations(mutationPath=mutationPath,mutationFile=mutationFile,minNumMutations=minNumMutations)
        matrices.append(matrix)
    filteredMutations = pd.concat(matrices,axis=0)

    return filteredMutations


def mutationRegulatorStratification(mutationDf,tfDf,threshold=0.05,dictionary_=False):

    incidence = pd.DataFrame(np.zeros((tfDf.shape[0],mutationDf.shape[0])))
    incidence.index = tfDf.index
    incidence.columns = mutationDf.index

    stratification = {}
    tfCols = set(tfDf.columns)
    mutCols = set(mutationDf.columns)
    for mutation in mutationDf.index:
        mut = getMutations(mutation,mutationDf)
        wt = list(mutCols-set(mut))
        mut = list(set(mut)&tfCols)
        wt = list(set(wt)&tfCols)
        tmpMut = tfDf.loc[:,mut]
        tmpWt = tfDf.loc[:,wt]
        ttest = stats.ttest_ind(tmpMut,tmpWt,axis=1,equal_var=False)
        significant = np.where(ttest[1]<=threshold)[0]
        hits = list(tfDf.index[significant])
        if len(hits) > 0:
            incidence.loc[hits,mutation] = 1
            if dictionary_ is not False:
                stratification[mutation] = {}
                for i in range(len(hits)):
                    stratification[mutation][hits[i]] = [ttest[0][significant[i]],ttest[1][significant[i]]]

    if dictionary_ is not False:
        return incidence, stratification
    return incidence

def generateEpigeneticMatrix(epigeneticFilename,expressionData,cutoff_pecentile=80,saveFile="epigeneticMatrix.csv"):
    epigenetic_regulators = pd.read_csv(os.path.join(os.path.split(os.getcwd())[0],"data",epigeneticFilename),sep="\t",header=None)
    epigenetic_regulators_list = list(epigenetic_regulators.iloc[:,0])
    epigenetic = list(set(epigenetic_regulators_list)&set(expressionData.index))
    epigenetic_expression = expressionData.loc[epigenetic,:]
    percentiles80 = np.percentile(epigenetic_expression,cutoff_pecentile,axis=1)
    epigenetic_cutoffs = [max(percentiles80[i],0) for i in range(len(percentiles80))]

    epigenetic_matrix = pd.DataFrame(np.zeros((len(epigenetic),expressionData.shape[1])))
    epigenetic_matrix.columns = expressionData.columns
    epigenetic_matrix.index = epigenetic

    for i in range(epigenetic_matrix.shape[0]):
        epi = epigenetic_matrix.index[i]
        hits = epigenetic_matrix.columns[np.where(expressionData.loc[epi,:]>=epigenetic_cutoffs[i])[0]]
        epigenetic_matrix.loc[epi,hits] = 1

    if saveFile is not None:
        epigenetic_matrix.to_csv(os.path.join(os.path.split(os.getcwd())[0],"data",saveFile))

    return epigenetic_matrix

def generateCausalInputs(expressionData,mechanisticOutput,coexpressionModules,saveFolder,mutationFile="filteredMutationsIA12.csv",regulon_dict=None):
    if not os.path.isdir(saveFolder):
        os.mkdir(saveFolder)

    # set working directory to results folder
    os.chdir(saveFolder)
    # identify the data folder
    os.chdir(os.path.join("..","data"))
    dataFolder = os.getcwd()
    # write csv files for input into causal inference module
    os.chdir(os.path.join("..","src"))

    #bcTfIncidence
    bcTfIncidence = biclusterTfIncidence(mechanisticOutput,regulons=regulon_dict)
    bcTfIncidence.to_csv(os.path.join(saveFolder,"bcTfIncidence.csv"))

    #eigengenes
    eigengenes = principal_df(coexpressionModules,expressionData,subkey=None,regulons=regulon_dict,minNumberGenes=1)
    eigengenes = eigengenes.T
    index = np.sort(np.array(eigengenes.index).astype(int))
    eigengenes = eigengenes.loc[index.astype(str),:]
    eigengenes.to_csv(os.path.join(saveFolder,"eigengenes.csv"))

    #tfExpression
    tfExp = tfExpression(expressionData)
    tfExp.to_csv(os.path.join(saveFolder,"tfExpression.csv"))

    #filteredMutations:
    filteredMutations = filterMutations(dataFolder,mutationFile)
    filteredMutations.to_csv(os.path.join(saveFolder,"filteredMutations.csv"))

    #regStratAll
    tfStratMutations = mutationRegulatorStratification(filteredMutations,tfDf=tfExp,threshold=0.01)
    keepers = list(set(np.arange(tfStratMutations.shape[1]))-set(np.where(np.sum(tfStratMutations,axis=0)==0)[0]))
    tfStratMutations = tfStratMutations.iloc[:,keepers]
    tfStratMutations.to_csv(os.path.join(saveFolder,"regStratAll.csv"))

def processCausalResults(causalPath=os.path.join("..","results","causal"),causalDictionary=False):

    causalFiles = []
    for root, dirs, files in os.walk(causalPath, topdown=True):
       for name in files:
          if name.split(".")[-1] == 'DS_Store':
              continue
          causalFiles.append(os.path.join(root, name))

    if causalDictionary is False:
        causalDictionary = {}
    for csv in causalFiles:
        tmpcsv = pd.read_csv(csv,index_col=False,header=None)
        for i in range(1,tmpcsv.shape[0]):
            score = float(tmpcsv.iloc[i,-2])
            if score <1:
                break
            bicluster = int(tmpcsv.iloc[i,-3].split(":")[-1].split("_")[-1])
            if bicluster not in list(causalDictionary.keys()):
                causalDictionary[bicluster] = {}
            regulator = tmpcsv.iloc[i,-5].split(":")[-1]
            if regulator not in list(causalDictionary[bicluster].keys()):
                causalDictionary[bicluster][regulator] = []
            mutation = tmpcsv.iloc[i,1].split(":")[-1]
            if mutation not in causalDictionary[bicluster][regulator]:
                causalDictionary[bicluster][regulator].append(mutation)

    return causalDictionary

def analyzeCausalResults(task):

    start, stop = task[0]
    preProcessedCausalResults,mechanisticOutput,filteredMutations,tfExp,eigengenes = task[1]
    postProcessed = {}
    if mechanisticOutput is not None:
        mechOutKeyType = type(mechanisticOutput.keys()[0])
    allPatients = set(filteredMutations.columns)
    keys = preProcessedCausalResults.keys()[start:stop]
    ct=-1
    for bc in keys:
        ct+=1
        if ct%10 == 0:
            print(ct)
        postProcessed[bc] = {}
        for tf in list(preProcessedCausalResults[bc].keys()):
            for mutation in preProcessedCausalResults[bc][tf]:
                mut = getMutations(mutation,filteredMutations)
                wt = list(allPatients-set(mut))
                mutTfs = tfExp.loc[tf,mut][tfExp.loc[tf,mut]>-4.01]
                if len(mutTfs) <=1:
                    mutRegT = 0
                    mutRegP = 1
                elif len(mutTfs) >1:
                    wtTfs = tfExp.loc[tf,wt][tfExp.loc[tf,wt]>-4.01]
                    mutRegT, mutRegP = stats.ttest_ind(list(mutTfs),list(wtTfs),equal_var=False)
                mutBc = eigengenes.loc[bc,mut][eigengenes.loc[bc,mut]>-4.01]
                if len(mutBc) <=1:
                    mutBcT = 0
                    mutBcP = 1
                    mutCorrR = 0
                    mutCorrP = 1
                elif len(mutBc) >1:
                    wtBc = eigengenes.loc[bc,wt][eigengenes.loc[bc,wt]>-4.01]
                    mutBcT, mutBcP = stats.ttest_ind(list(mutBc),list(wtBc),equal_var=False)
                    if len(mutTfs) <=2:
                        mutCorrR = 0
                        mutCorrP = 1
                    elif len(mutTfs) >2:
                        nonzeroPatients = list(set(np.array(mut)[tfExp.loc[tf,mut]>-4.01])&set(np.array(mut)[eigengenes.loc[bc,mut]>-4.01]))
                        mutCorrR, mutCorrP = stats.pearsonr(list(tfExp.loc[tf,nonzeroPatients]),list(eigengenes.loc[bc,nonzeroPatients]))
                signMutTf = 1
                if mutRegT < 0:
                    signMutTf = -1
                elif mutRegT == 0:
                    signMutTf = 0
                signTfBc = 1
                if mutCorrR < 0:
                    signTfBc = -1
                elif mutCorrR == 0:
                    signTfBc = 0
                if mechanisticOutput is not None:
                    if mechOutKeyType is int:
                        phyper = mechanisticOutput[bc][tf][0]
                    elif mechOutKeyType is not int:
                        phyper = mechanisticOutput[str(bc)][tf][0]
                elif mechanisticOutput is None:
                    phyper = 1e-10
                pMutRegBc = 10**-((-np.log10(mutRegP)-np.log10(mutBcP)-np.log10(mutCorrP)-np.log10(phyper))/4.)
                pWeightedTfBc = 10**-((-np.log10(mutCorrP)-np.log10(phyper))/2.)
                mutFrequency = len(mut)/float(filteredMutations.shape[1])
                postProcessed[bc][tf] = {}
                postProcessed[bc][tf]["regBcWeightedPValue"] = pWeightedTfBc
                postProcessed[bc][tf]["edgeRegBc"] = signTfBc
                postProcessed[bc][tf]["regBcHyperPValue"] = phyper
                if "mutations" not in list(postProcessed[bc][tf].keys()):
                    postProcessed[bc][tf]["mutations"] = {}
                postProcessed[bc][tf]["mutations"][mutation] = {}
                postProcessed[bc][tf]["mutations"][mutation]["mutationFrequency"] = mutFrequency
                postProcessed[bc][tf]["mutations"][mutation]["mutRegBcWeightedPValue"] = pMutRegBc
                postProcessed[bc][tf]["mutations"][mutation]["edgeMutReg"] = signMutTf
                postProcessed[bc][tf]["mutations"][mutation]["mutRegPValue"] = mutRegP
                postProcessed[bc][tf]["mutations"][mutation]["mutBcPValue"] = mutBcP
                postProcessed[bc][tf]["mutations"][mutation]["regBcCorrPValue"] = mutCorrP
                postProcessed[bc][tf]["mutations"][mutation]["regBcCorrR"] = mutCorrR

    return postProcessed

def postProcessCausalResults(preProcessedCausalResults,filteredMutations,tfExp,eigengenes,mechanisticOutput=None,numCores=5):

    taskSplit = splitForMultiprocessing(preProcessedCausalResults.keys(),numCores)
    taskData = (preProcessedCausalResults,mechanisticOutput,filteredMutations,tfExp,eigengenes)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    Output = multiprocess(analyzeCausalResults,tasks)
    postProcessedAnalysis = condenseOutput(Output)

    return postProcessedAnalysis

def causalMechanisticNetworkDictionary(postProcessedCausalAnalysis,biclusterRegulatorPvalue=0.05,regulatorMutationPvalue=0.05,mutationFrequency = 0.025,requireCausal=False):

    tabulatedResults = []
    ct=-1
    for key in list(postProcessedCausalAnalysis.keys()):
        ct+=1
        if ct%10==0:
            print(ct)
        lines = []
        regs = postProcessedCausalAnalysis[key].keys()
        for reg in regs:
            bcid = key
            regid = reg
            bcRegEdgeType = int(postProcessedCausalAnalysis[key][reg]['edgeRegBc'])
            bcRegEdgePValue = postProcessedCausalAnalysis[key][reg]['regBcWeightedPValue']
            bcTargetEnrichmentPValue = postProcessedCausalAnalysis[key][reg]['regBcHyperPValue']
            if bcRegEdgePValue <= biclusterRegulatorPvalue:
                if len(postProcessedCausalAnalysis[key][reg]['mutations'])>0:
                    for mut in list(postProcessedCausalAnalysis[key][reg]['mutations'].keys()):
                        mutFrequency = postProcessedCausalAnalysis[key][reg]['mutations'][mut]['mutationFrequency']
                        mutRegPValue = postProcessedCausalAnalysis[key][reg]['mutations'][mut]['mutRegPValue']
                        if mutFrequency >= mutationFrequency:
                            if mutRegPValue <= regulatorMutationPvalue:
                                mutid = mut
                                mutRegEdgeType = int(postProcessedCausalAnalysis[key][reg]['mutations'][mut]['edgeMutReg'])
                            elif mutRegPValue > regulatorMutationPvalue:
                                mutid = np.nan #"NA"
                                mutRegEdgeType = np.nan #"NA"
                                mutRegPValue = np.nan #"NA"
                                mutFrequency = np.nan #"NA"
                        elif mutFrequency < mutationFrequency:
                            mutid = np.nan #"NA"
                            mutRegEdgeType = np.nan #"NA"
                            mutRegPValue = np.nan #"NA"
                            mutFrequency = np.nan #"NA"
                elif len(postProcessedCausalAnalysis[key][reg]['mutations'])==0:
                    mutid = np.nan #"NA"
                    mutRegEdgeType = np.nan #"NA"
                    mutRegPValue = np.nan #"NA"
                    mutFrequency = np.nan #"NA"
            elif bcRegEdgePValue > biclusterRegulatorPvalue:
                continue
            line = [bcid,regid,bcRegEdgeType,bcRegEdgePValue,bcTargetEnrichmentPValue,mutid,mutRegEdgeType,mutRegPValue,mutFrequency]
            lines.append(line)
        if len(lines) == 0:
            continue
        stack = np.vstack(lines)
        df = pd.DataFrame(stack)
        df.columns = ["Cluster","Regulator","RegulatorToClusterEdge","RegulatorToClusterPValue","RegulatorBindingSiteEnrichment","Mutation","MutationToRegulatorEdge","MutationToRegulatorPValue","FrequencyOfMutation"]
        tabulatedResults.append(df)

    resultsDf = pd.concat(tabulatedResults,axis=0)
    resultsDf = resultsDf[resultsDf["RegulatorToClusterEdge"]!='0']
    resultsDf.index = np.arange(resultsDf.shape[0])

    if requireCausal is True:
        resultsDf = resultsDf[resultsDf["Mutation"]!="nan"]

    return resultsDf

def clusterInformation(causalMechanisticNetwork,key):
    return causalMechanisticNetwork[causalMechanisticNetwork["Cluster"]==key]

def showCluster(expressionData,coexpressionModules,key):
    plt.figure(figsize=(10,10))
    plt.imshow(expressionData.loc[coexpressionModules[key],:],vmin=-1,vmax=1)
    plt.title("Cluster Expression",fontsize=16)
    plt.xlabel("Patients",fontsize=14)
    plt.ylabel("Genes",fontsize=14)
"""
