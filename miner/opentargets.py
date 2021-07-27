#!/usr/bin/env python3

"""
An OpenTargets Client based on the V4 GraphQL API
"""
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from collections import defaultdict

import os
import argparse
import json
from chembl_webresource_client.new_client import new_client


ENDPOINT_URL = "https://api.platform.opentargets.org/api/v4/graphql"


DRUG_QUERY = """query mydrugs {
  target(ensemblId: "%s") {
    id
    associatedDiseases {
      rows {
      	disease {
          name
          knownDrugs {
            rows {
              approvedName
              prefName
              phase
              status
              targetClass
              mechanismOfAction
              drugType
              drugId
            }
          }
        }
      }
    }
  }
}"""



def uniqify(gene_drugs):
    out = {}
    for gene, infos in gene_drugs.items():
        out_infos = {}
        if len(infos) > 0:
            print("%s" % gene)
            for info in infos:
                out_infos[info['drug']['molecule_name']] = {
                    'drug': info['drug']['molecule_name'],
                    'target_class': info['target_class'],
                    'mechanism_of_action': info['mechanism_of_action'],
                    'trial_phase': info['trial_phase']
                }
                print("  %s" % info['drug']['molecule_name'])
                print("  %s" % str(info['target_class']))
                print("  %s" % str(info['mechanism_of_action']))
                print("  %s" % str(info['trial_phase']))
            out[gene] = list(out_infos.values())
        else:
            out[gene] = []
    return out


def item_meets_criteria(item, diseases, trial_phase):
    if len(diseases) == 0:
        return 'drug' in item and (trial_phase is None or get_drug_trial_phase(item) == trial_phase)

    return ('drug' in item and item['disease']['efo_info']['label'] in diseases and
            (trial_phase is None or get_drug_trial_phase(item) == trial_phase))


def compute_backgrounds(gene_opentargets, outdir):
    num_genes = len(gene_opentargets)
    num_drugs = defaultdict(int)
    num_target_classes = defaultdict(int)
    num_mechanism_of_action = defaultdict(int)

    for gene, infos in gene_opentargets.items():
        gene_drugs = set()
        for info in infos:
            drug = info['drug']
            # only process once
            if drug not in gene_drugs:
                gene_drugs.add(drug)
                num_drugs[drug] += 1

                target_classes = info['target_class']
                mechanism_of_action = info['mechanism_of_action']
                for target_class in target_classes:
                    num_target_classes[target_class] += 1
                num_mechanism_of_action[mechanism_of_action] += 1

    with open(os.path.join(outdir, 'drugs_background.json'), 'w') as outfile:
        json.dump(num_drugs, outfile)
    with open(os.path.join(outdir, 'target_classes_background.json'), 'w') as outfile:
        json.dump(num_target_classes, outfile)
    with open(os.path.join(outdir, 'mechanism_of_action_background.json'), 'w') as outfile:
        json.dump(num_mechanism_of_action, outfile)


def get_drugs(client, gene, all_diseases, diseases, trial_phase=None):
    query = gql(DRUG_QUERY % gene)
    result = client.execute(query)
    diseases = result['target']["associatedDiseases"]["rows"]
    results = []
    for item in diseases:
        disease = item['disease']

        if disease["knownDrugs"] is None:
            drugs = []
        else:
            drugs = disease["knownDrugs"]["rows"]
        if len(drugs) > 0:
            all_diseases[disease["name"]] += 1

        for drug in drugs:
            trial_phase = drug["phase"]
            out_item = {
                "drug": {
                    'id': drug['drugId'],
                    'molecule_name': drug['prefName'],
                    'molecule_type': drug['drugType']
                },
                "trial_phase": trial_phase,
                "mechanism_of_action": drug['mechanismOfAction'],
                "target_class": drug['targetClass']
            }
            results.append(out_item)
    return results



CHEMBL_LOOKUP_ID_URL = "https://www.ebi.ac.uk/chembl/api/data/docs#collapse_GET_chembl_id_lookup_api_get_search"

DRUG_TARGETS_QUERY = """
query drug_targets {
  drug(chemblId: "%s") {
    id
    name
    linkedTargets {
      rows {
        id
      }
    }
  }
}
"""

def drug_info_for_drugs(args, result_thresh=60.0):
    """retrieve all opentargets drug information for a list of drugs"""
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    with open(args.drugs) as infile:
        drugs = [s.strip() for s in infile]

    # Step 1: collect the CHEMBL IDs
    drug_ids = {}
    molecule = new_client.molecule

    for i, drug in enumerate(drugs):
        print('%s - %d of %d' % (drug, i + 1, len(drugs)))
        try:
            res = molecule.search(drug)
            entry = res[0]
            drug_ids[drug] = entry['molecule_chembl_id']
        except:
            print("FAILURE: could not retrieve info for molecule '%s' - skipping" % drug)

    print(drug_ids)
    # Step 2: collect targets for the drug ids we obtained in step 1
    tp = RequestsHTTPTransport(url=ENDPOINT_URL,
                               verify=True, retries=3)
    client = Client(transport=tp, fetch_schema_from_transport=True)

    target_ids = set([])
    for drug in drug_ids.values():
        query = gql(DRUG_TARGETS_QUERY % drug)
        result = client.execute(query)
        targets = result["drug"]["linkedTargets"]["rows"]
        for target in targets:
            ensid = target["id"]
            target_ids.add(ensid)
    # call our central gene retrieval function
    _drug_info_for_genes(list(target_ids), args)


def drug_info_for_genes(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    with open(args.genes) as infile:
        genes = [s.strip() for s in infile]

    _drug_info_for_genes(genes, args)


def _drug_info_for_genes(genes, args):
    """Central workhorse function"""
    all_results = {}
    all_diseases = defaultdict(int)
    num_genes = len(genes)
    diseases = set(args.disease)
    if args.disease_file is not None:
        with open(args.disease_file) as infile:
            for d in infile:
                diseases.add(d.strip())
    print("Diseases: " + str(diseases))

    tp = RequestsHTTPTransport(url=ENDPOINT_URL,
                               verify=True, retries=3)

    client = Client(transport=tp, fetch_schema_from_transport=True)
    diseases = set([])
    all_diseases = defaultdict(int)
    trial_phase = None

    for i, gene in enumerate(genes):
        print('%s - %d of %d' % (gene, i + 1, num_genes))
        try:
            result = get_drugs(client, gene, all_diseases, diseases, trial_phase)
            all_results[gene] = result
        except:
            print("FAILURE: could not retrieve info for gene '%s' - skipping" % gene)


    with open(os.path.join(args.outdir, 'diseases.tsv'), 'w') as outfile:
        for disease in sorted(all_diseases.keys()):
            outfile.write('%s\t%d\n' % (disease, all_diseases[disease]))

    with open(os.path.join(args.outdir, 'gene_opentargets.json'), 'w') as outfile:
        json.dump(all_results, outfile)

    # write unique results
    unique_results = uniqify(all_results)
    with open(os.path.join(os.path.join(args.outdir, 'gene_opentargets_unique.json')), 'w') as outfile:
        json.dump(unique_results, outfile)

    # write backgrounds
    compute_backgrounds(unique_results, args.outdir)


DESCRIPTION = "get_opentargets - find opentargets data for genes and disease"
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=DESCRIPTION)
    parser.add_argument('genes', help="list of genes separated by new lines in EnsEMBL format")
    parser.add_argument('--disease', action="append", help="disease to filter by", default=[])
    parser.add_argument('--disease_file', help="disease to filter by", default=None)
    parser.add_argument('--trial_phase', type=int, help="trial phase for drugs", default=None)
    parser.add_argument('outdir', help="output directory")
    args = parser.parse_args()

    drug_info_for_genes(args)
