The miner2-riskpredict tool
===========================

This utility calculates risk prediction data from its input

You can see the tool's available options when you enter ``miner2-riskpredict -h``
at the command prompt:

.. highlight:: none

::

  usage: miner2-riskpredict [-h] [--method METHOD] input outdir

  miner-riskpredict - MINER compute risk prediction.
  MINER Version development (Git SHA 563821013b1f4189d012b54416a7989396d0811d)

  positional arguments:
    input            input specification file
    outdir           output directory

  optional arguments:
    -h, --help       show this help message and exit
     usage: miner2-riskclassifier [-h] input outdir


Parameters in detail
--------------------

``miner2-riskpredict`` expects at least these 2 arguments:

  * **input:** an input specification in JSON format
  * **outdir:** The directory where the result files will be placed in.

An example input file
---------------------

.. highlight:: none

::

  {
    "exp": "MATTDATA/expression/IA12Zscore.csv",
    "idmap": "MATTDATA/identifier_mappings.txt",
    "translocations": "MATTDATA/mutations/translocationsIA12.csv",
    "coexpression_dictionary": "MATTRESULT/coexpressionDictionary.json",
    "coexpression_modules": "MATTRESULT/coexpressionModules.json",
    "regulon_modules": "MATTRESULT/regulons.json",
    "mechanistic_output": "MATTRESULT/mechanisticOutput.json",
    "regulon_df": "MATTRESULT/regulonDf.csv",
    "overexpressed_members": "MATTRESULT/overExpressedMembers.csv",
    "underexpressed_members": "MATTRESULT/underExpressedMembers.csv",
    "eigengenes": "MATTRESULT/eigengenes.csv",
    "filtered_causal_results": "MATTRESULT/filteredCausalResults.csv",
    "transcriptional_programs": "MATTRESULT/transcriptional_programs.json",
    "transcriptional_states": "MATTRESULT/transcriptional_states.json",
    "primary_survival_data": "MATTDATA/survival/survivalIA12.csv"
  }


Output in detail
----------------

After successful completion there will be the following files in the output directory

  * ``CoxProportionalHazardsRegulons.csv`` cox hazard information by regulon
  * ``CoxProportionalHazardsPrograms.csv`` cox hazard information by program
  * plots and curves
