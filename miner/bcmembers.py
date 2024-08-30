"""bcmembers.py - Module to compute bicluster memberships"""

import os
import json
from miner import miner

def bcmembers(exp_data, regulon_modules, outdir):
    bkgd = miner.background_df(exp_data)
    overexpressed_members = miner.biclusterMembershipDictionary(regulon_modules,
                                                                bkgd, label=2, p=0.05)
    underexpressed_members = miner.biclusterMembershipDictionary(regulon_modules,
                                                                 bkgd, label=0, p=0.05)
    dysregulated_members = miner.biclusterMembershipDictionary(regulon_modules,
                                                               bkgd, label="excluded")
    coherent_members = miner.biclusterMembershipDictionary(regulon_modules,
                                                           bkgd, label="included")

    # write the overexpressed/underexpressed members as JSON, tools later in the pipeline can
    # easier access them
    with open(os.path.join(outdir, 'overExpressedMembers.json'), 'w') as out:
        json.dump(overexpressed_members, out)
    with open(os.path.join(outdir, 'underExpressedMembers.json'), 'w') as out:
        json.dump(underexpressed_members, out)

    overexpressed_members_matrix = miner.membershipToIncidence(overexpressed_members,
                                                               exp_data)
    overexpressed_members_matrix.to_csv(os.path.join(outdir,
                                                     "overExpressedMembers.csv"))

    underexpressed_members_matrix = miner.membershipToIncidence(underexpressed_members,
                                                                exp_data)
    underexpressed_members_matrix.to_csv(os.path.join(outdir,
                                                      "underExpressedMembers.csv"))

    dysregulated_members_matrix = miner.membershipToIncidence(dysregulated_members,
                                                              exp_data)
    dysregulated_members_matrix.to_csv(os.path.join(outdir, "dysregulatedMembers.csv"))

    coherent_members_matrix = miner.membershipToIncidence(coherent_members,
                                                          exp_data)
    coherent_members_matrix.to_csv(os.path.join(outdir,
                                                "coherentMembers.csv"))
