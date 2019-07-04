# miner
mechanistic inference of node-edge relationships

# tutorial information
A template with instructions and code for performing a miner analysis is provided in the miner/src directory. 

The template can be opened using Anaconda by selecting jupyter notebook from the user interface, or by simply typing "jupyter notebook" into the command line. 

# what you need to get started
Before the miner analysis can be performed, the gene expression data to be analyzed must be added to the miner/data directory. Ensure that your expression data is of the form log2(TPM+1) or log2(FPKM+1).

If survival analysis is desired, a survival file must added to the miner/data directory

If causal analysis is desired, a mutation file must added to the miner/data directory

# where to put your data
miner will search for specific filenames in the miner/data folder. Be sure to update the lines that read your files with the appropriate paths and filenames. Consider using the following names for consistency:
    1. Name your expression data "expressionData.csv"
    2. Name your mutation data "mutations.csv" (only for causal analysis)
    3. Name your survival data "survival.csv" (only for survival analysis)
   
Note that the gene names will be converted to Ensembl Gene ID format

# common mistakes to avoid
1. miner does not yet support expression data in counts format. Ensure that data is in log2(TPM+1) or log2(FPKM+1) format.
2. mechanistic inference includes a step that enforces a minimum correlation coefficient. If your results seem too sparse, try decreasing the minCorrelation parameter.
