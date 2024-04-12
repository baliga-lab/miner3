# MINER

![Unit tests](https://github.com/baliga-lab/miner3/actions/workflows/python-test.yml/badge.svg)

**M**echanistic **I**nference of **N**ode-edge **R**elationships

## Usage

### Docker image

you can access a Docker image with a full MINER installation at  Docker Hub with the name ```weiju/isb-miner3```

```
docker pull weiju/isb-miner3
```

### Through pip

```
pip install isb-miner3
```

## Command Line Tools documentation

Please see command line documentation at https://baliga-lab.github.io/miner3/

## Tutorial information

Find a walkthrough analysis in the form of a [Jupyter Notebook here](Example%20MINER%20Analysis.ipynb)

## What you need to get started

Before the miner analysis can be performed, the gene expression data to be analyzed must be added to the miner/data directory. Ensure that your expression data is of the form log2(TPM+1) or log2(FPKM+1).

  * If survival analysis is desired, a survival file must added to the miner/data directory
  * If causal analysis is desired, a mutation file must added to the miner/data directory

## Where to put your data

miner will search for specific filenames in the miner/data folder. Be sure to update the lines that read your files with the appropriate paths and filenames. Consider using the following names for consistency:

  1. Name your expression data "expressionData.csv"
  2. Name your mutation data "mutations.csv" (only for causal analysis)
  3. Name your survival data "survival.csv" (only for survival analysis)

Note that the gene names will be converted to Ensembl Gene ID format

## Common mistakes to avoid

  1. miner does not yet support expression data in counts format. Ensure that data is in log2(TPM+1) or log2(FPKM+1) format.
  2. mechanistic inference includes a step that enforces a minimum correlation coefficient. If your results seem too sparse, try decreasing the minCorrelation parameter.


## For maintainers

### Documentation

This project's documentation is provided as github pages. It is generated from ReStructured Text files
and generated using the tool sphinx.
In order to update it, edit the `.rst` documents in docs and generate HTML files by

```
make html
```

Provided there is a functioning sphinx system installed, the documentation will be in the build html
directory and should be copied to the `gh-pages` branch of this repository.

