The miner3-mechinf tool
=======================

This utility computes the mechanistic inference.

You can see the tool's available options when you enter ``miner2-mechinf -h``
at the command prompt:

.. highlight:: none

::

    usage: miner3-mechinf [-h] [-mc MINCORR]
                          expfile mapfile coexprdict outdir

    miner3-mechinf - MINER compute mechanistic inference

    positional arguments:
      expfile               input matrix
      mapfile               identifier mapping file
      coexprdict            coexpressionDictionary.json file from miner-coexpr
      outdir                output directory

    optional arguments:
      -h, --help            show this help message and exit
      --skip_tpm            skip TPM preprocessing for single cell data
      --tfs2genes FILE      override TF to genes mapping (pickle or JSON file)
      -mc MINCORR, --mincorr MINCORR
                            minimum correlation


Parameters in detail
--------------------

``miner3-mechinf`` expects at least these 5 arguments:

  * **expfile:** The gene expression file a matrix in csv format.
  * **mapfile:** The gene identifier map file.
  * **coexprdict:** The path coexpressionDictionary.json file from the miner-coexpr tool
  * **outdir:** The path to the output directory

In addition, you can specify the following optional arguments:

  * ``--skip_tpm``: to skip the TPM preprocessing step for single-cell data
  * ``--tfs2genes``: provide alternative TF to genes mapping for computation
  * ``--mincorr`` or ``--mc``: the minimum correlation value.

Output in detail
----------------

After successful completion there will be the following files in the output directory


  * ``regulons.json`` - use this file in subsequent tools
  * ``coexpressionDictionary_annotated.json``
  * ``mechanisticOutput.json``
  * ``coexpressionModules_annotated.json``
  * ``regulons_annotated.csv``
  * ``coexpressionModules.json``
  * ``regulons_annotated.json``
  * ``coregulationModules.json``
