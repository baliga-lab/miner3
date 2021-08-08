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

DRUG_INFO_QUERY = """
query drug_targets {
  drug(chemblId: "%s") {
    id
    name
    drugType
    isApproved
    approvedIndications
    drugWarnings {
      meddraSocCode
      toxicityClass
    }
    literatureOcurrences {
      rows {
        pmid
      }
    }
    linkedTargets {
      rows {
        id
        approvedName
      }
    }
    indications {
      rows {
        disease {
          id
          name
        }
        maxPhaseForIndication
      }
    }
    maximumClinicalTrialPhase
    mechanismsOfAction {
      rows {
        mechanismOfAction
        actionType
        targetName
        targets {
          id
          approvedName
          bioType
        }
      }
    }
    knownDrugs {
      rows {
        targetId
        targetClass
        approvedName
        approvedSymbol
        diseaseId
        urls {
          name
          url
        }
      }
    }
  }
}
"""

# CSV field delimiter
CSV_DELIM = ','


def drug_info_for_drugs(args, result_thresh=60.0):
    """retrieve all opentargets drug information for a list of drugs"""
    errors = []
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    with open(args.drugs) as infile:
        drugs = [s.strip() for s in infile]

    diseases = set(args.disease)
    if args.disease_file is not None:
        with open(args.disease_file) as infile:
            for d in infile:
                diseases.add(d.strip())


    # Step 1: collect the CHEMBL IDs
    drug_ids = {}
    molecule = new_client.molecule
    chembl2drug = {}

    for i, drug in enumerate(drugs):
        print('%s - %d of %d' % (drug, i + 1, len(drugs)))
        try:
            res = molecule.search(drug)
            for r in res:
                synonyms = r['molecule_synonyms']
                for s in synonyms:
                    if s['synonyms'] == drug:
                        entry = r
                        break

            chembl_id = entry['molecule_chembl_id']
            #print(entry['molecule_synonyms'])
            drug_ids[drug] = chembl_id
            chembl2drug[chembl_id] = drug
            #with open(os.path.join(args.outdir,'%s.json' % chembl_id), 'w') as outfile:
            #    outfile.write(str(res))
        except:
            print("FAILURE: could not retrieve info for molecule '%s' - skipping" % drug)
            errors.append("FAILURE: could not retrieve info for molecule '%s' - skipping" % drug)

    with open(os.path.join(args.outdir, 'chembl2drug.csv'), 'w') as outfile:
        for chembl_id, drug in chembl2drug.items():
            outfile.write('%s\t%s\n' % (chembl_id, drug))

    #print(drug_ids)
    # Step 2: collect targets for the drug ids we obtained in step 1
    tp = RequestsHTTPTransport(url=ENDPOINT_URL,
                               verify=True, retries=3)
    client = Client(transport=tp, fetch_schema_from_transport=True)

    target_ids = set([])
    results = set()
    all_results = {}
    for i, chembl_id in enumerate(drug_ids.values()):
        print('%s - %d of %d' % (chembl_id, i, len(drug_ids)))
        try:
            query = gql(DRUG_INFO_QUERY % chembl_id)
            result = client.execute(query)
            drug = result["drug"]
            trial_urls = {}
            out_item = {}
            try:
                out_item['molecule_name'] = drug['name']
            except:
                out_item['molecule_name'] = ''
                errors.append('WARNING - no molecule name for "%s" (%s)' % (chembl_id, chembl2drug[chembl_id]))
                # This would be weird
                #continue
            try:
                out_item['drug_type'] = drug['drugType']
            except:
                out_item['drug_type'] = ''
                errors.append('WARNING - no drug type for "%s" (%s)' % (chembl_id, chembl2drug[chembl_id]))
            try:
                for d in diseases:
                    out_item['max_%s_phase' % d] = 0
                indications = drug['indications']['rows']
                out_item['indication_ids'] = set()
                out_item['indication_names'] = set()
                for indication in indications:
                    for d in diseases:
                        try:
                            indication['disease']['name'].index(d)
                            if indication['maxPhaseForIndication'] > out_item['max_%s_phase' % d]:
                                out_item['max_%s_phase' % d] = indication['maxPhaseForIndication']
                        except:
                            pass
                    out_item['indication_ids'].add(indication['disease']['id'])
                    out_item['indication_names'].add(indication['disease']['name'])
                out_item['indication_ids'] = list(out_item['indication_ids'])
                out_item['indication_names'] = list(out_item['indication_names'])
            except:
                out_item['indication_ids'] = []
                out_item['indication_names'] = []
            try:
                out_item['trial_phase'] = drug['maximumClinicalTrialPhase']
            except:
                out_item['trial_phase'] = ''
            try:
                out_item['chembl_uri'] = 'https://www.ebi.ac.uk/chembl/compound_report_card/%s/' % chembl_id
            except:
                out_item['chembl_uri'] = ''

            try:
                mech_action = drug['mechanismsOfAction']['rows'][0]
                out_item['mechanism_of_action'] = mech_action['mechanismOfAction']
                out_item['action_type'] = mech_action['actionType']
            except:
                out_item['mechanism_of_action'] = ''
                out_item['action_type'] = ''

            # error handling
            try:
                targets = drug['knownDrugs']['rows']
            except:
                targets = []

            out_item['targets'] = {}
            out_item['targetstr'] = []
            out_item['target_id'] = ''
            out_item['target_class'] = ''
            out_item['approved_name'] = ''
            for target in targets:
                try:
                    out_item['target_id'] = target['targetId']
                    if len(target['targetClass']) > 0:
                        out_item['targets'][target['targetId']] = target['targetClass'][0]
                        out_item['target_class'] = target['targetClass'][0]
                    out_item['approved_name'] = target['approvedSymbol']
                    for url in target['urls']:
                        trial_urls[url['name']] = url['url']
                except:
                    raise
                    pass

            for tid, tclass in out_item['targets'].items():
                out_item['targetstr'].append('%s:%s' % (tid, tclass))

            # Clinical Trials URL if available
            try:
                out_item['trial_url'] = trial_urls['ClinicalTrials']
            except:
                out_item['trial_url'] = ''

            all_results[chembl_id] = out_item
            out_item['literature_occ'] = []
            try:
                for row in drug['literatureOcurrences']['rows']:
                    out_item['literature_occ'].append(row['pmid'])
            except:
                pass

            out_item['toxicity_class'] = []
            out_item['meddra_soc_code'] = []
            try:
                for warning in drug['drugWarnings']:
                    out_item['toxicity_class'].append(str(warning['toxicityClass']))
                    out_item['meddra_soc_code'].append(str(warning['meddraSocCode']))
            except:
                pass
        except:
            print("FAILURE - could not retrieve targets for drug '%s' - skipping" % chembl_id)
            raise

    with open(os.path.join(args.outdir, 'errors.txt'), 'w') as outfile:
        for error in errors:
            outfile.write(error)
            outfile.write('\n')

    # output as JSON
    with open(os.path.join(args.outdir, 'drug_opentargets.json'), 'w') as outfile:
        json.dump(all_results, outfile)

    # output as CSV
    with open(os.path.join(args.outdir, 'drug_opentargets.csv'), 'w') as outfile:
        header = ['CHEMBL_ID', 'molecule_name', 'molecule_type',
                  'indication_ids', 'indication_names', 'max_trial_phase']
        for d in diseases:
            header.append('max_%s_phase' % d)
        header.extend(['chembl_uri',
                       'mechanism_of_action', 'action_type', 'target_id', 'target_class',
                       'approved_name', 'literature_occ', 'trial_url',
                       'toxicity_class', 'meddra_soc_code'])
        outfile.write(CSV_DELIM.join(header))
        outfile.write('\n')

        for chembl_id, info in all_results.items():
            out_row = [chembl_id,
                       info['molecule_name'],
                       info['drug_type'],
                       ':'.join(list(info['indication_ids'])),
                       '"' + ':'.join(list(info['indication_names'])) + '"',
                       str(info['trial_phase'])]
            for d in diseases:
                out_row.append(str(info['max_%s_phase' % d]))
            out_row.extend([info['chembl_uri'],
                            '"' + info['mechanism_of_action'] + '"',
                            info['action_type'],
                            info['target_id'], info['target_class'],
                            info['approved_name'], ':'.join(info['literature_occ']),
                            info['trial_url']])
            out_row.append(':'.join(info['toxicity_class']))
            out_row.append(':'.join(info['meddra_soc_code']))
            outfile.write(CSV_DELIM.join(out_row))
            outfile.write('\n')


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
