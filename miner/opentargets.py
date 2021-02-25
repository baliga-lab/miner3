#!/usr/bin/env python3
from opentargets import OpenTargetsClient
import json
import argparse
from collections import defaultdict
import os


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



def get_drug_trial_phase(item):
    trial_phase = None
    if "evidence" in item:
        evidence = item["evidence"]
        if "drug2clinic" in evidence:
            drug2clinic = evidence["drug2clinic"]
            if "clinical_trial_phase" in drug2clinic:
                trial_phase = drug2clinic["clinical_trial_phase"]["numeric_index"]
    return trial_phase


def get_drugs(client, gene, all_diseases, diseases, trial_phase=None):
    #response = client.get_evidence_for_target(gene, fields=['drug.*', 'target.target_class',
    #                                                        'disease.efo_info.label',
    #                                                        'evidence.target2drug.mechanism_of_action'])
    # we just get everything
    response = client.get_evidence_for_target(gene)
    result = []

    for item in response:
        # only count occurring diseases if they have a drug

        if 'drug' in item:
            all_diseases[item['disease']['efo_info']['label']] += 1

        if item_meets_criteria(item, diseases, trial_phase):
            item_trial_phase = get_drug_trial_phase(item)
            if item_trial_phase is None:
                item_trial_phase = 'NA'
            out_item = {"drug": item['drug'], 'trial_phase': item_trial_phase}
            if 'target' in item and 'target_class' in item['target']:
                out_item['target_class'] = item['target']['target_class']
            if ('evidence' in item and 'target2drug' in item['evidence']
                and 'mechanism_of_action' in item['evidence']['target2drug']):
                out_item['mechanism_of_action'] = item['evidence']['target2drug']['mechanism_of_action']
            result.append(out_item)
    return result


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


def drug_info_for_drugs(args, result_thresh=60.0):
    """retrieve all opentargets drug information for a list of drugs"""
    client = OpenTargetsClient()
    all_results = {}
    with open(args.drugs) as infile:
        for line in infile:
            drug = line.strip()
            print(drug)
            # evaluate the json result
            try:
                results = client.search(drug)
                all_results[drug] = []
                for result in results:
                    print(list(result.keys()))
                    res_score = result['score']
                    res_data = result['data']

                    # break if score too low
                    if res_score < result_thresh:
                        break

                    disease = res_data['name']
                    print(list(result['data'].keys()))
                    drugs = result['data']['drugs']  # just a list of drugs
                    print("EVIDENCE_DATA: ", drugs['evidence_data'])
                    for key, entry in drugs.items():
                        print("KEY: ", key)
                        print("ENTRY: ", entry)
                    all_results[drug].append({'disease': disease, 'score': res_score, 'drugs': drugs})
            except Exception as e:
                print("WARNING: problems retrieving results for drug '%s' - skipping" % drug)
            break

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    with open(os.path.join(args.outdir, 'drugs_opentargets.json'), 'w') as outfile:
        json.dump(all_results, outfile)


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

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    with open(args.genes) as infile:
        genes = [s.strip() for s in infile]

    client = OpenTargetsClient()

    all_results = {}
    all_diseases = defaultdict(int)
    num_genes = len(genes)
    diseases = set(args.disease)
    if args.disease_file is not None:
        with open(args.disease_file) as infile:
            for d in infile:
                diseases.add(d.strip())
    print("Diseases: " + str(diseases))

    for i, g in enumerate(genes):
        print('%s - %d of %d' % (g, i + 1, num_genes))
        result = get_drugs(client, g, all_diseases, diseases, args.trial_phase)
        all_results[g] = result

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
