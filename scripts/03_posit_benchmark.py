from pathlib import Path

import pandas as pd

from kinoml.core.components import BaseProtein
from kinoml.core.ligands import RDKitLigand
from kinoml.core.systems import ProteinLigandComplex
from kinoml.features.complexes import OEPositDockingFeaturizer


if __name__ == '__main__':

    print("Reading benchmark dataframe ...")
    structures = pd.read_csv("../data/docking_benchmark_dataset.csv")

    print("Getting list of files ...")
    path_list = [path.stem for path in Path("../data/posit").glob("*.sdf")]

    print("Generating systems ...")
    systems = []
    kinase_conformation_representatives = structures.groupby(
        ["kinase.klifs_id", "structure.dfg", "structure.ac_helix"]
    ).head(1)
    for _, representative in kinase_conformation_representatives.iterrows():
        klifs_kinase_id = representative["kinase.klifs_id"]
        dfg = representative["structure.dfg"]
        ac_helix = representative["structure.ac_helix"]
        kinase_conformation_group = structures[
            (structures["kinase.klifs_id"] == klifs_kinase_id) &
            (structures["structure.dfg"] == dfg) &
            (structures["structure.ac_helix"] == ac_helix)
            ]
        for _, ligand_details in kinase_conformation_group.iterrows():
            ligand_name = f"{ligand_details['structure.pdb_id']}-{ligand_details['ligand.expo_id']}"
            ligand = RDKitLigand.from_smiles(
                smiles=ligand_details["smiles"],
                name=ligand_name
            )
            for _, protein_details in kinase_conformation_group.iterrows():
                # do not re-dock ligand into derived kinase structure
                if ligand_details["structure.pdb_id"] != protein_details["structure.pdb_id"]:
                    # check if sdf file already exists
                    path_list_tmp = [path for path in path_list if path.split("_")[9] == ligand_name]
                    path_list_tmp = [path for path in path_list_tmp if path.split("_")[5] == protein_details["structure.pdb_id"]]
                    if len(path_list_tmp) > 0:
                        continue
                    protein = BaseProtein(
                        name=f"{protein_details['species.klifs']}_{protein_details['kinase.klifs_name']}"
                    )
                    protein.pdb_id = protein_details["structure.pdb_id"]
                    protein.expo_id = protein_details["ligand.expo_id"]
                    protein.chain_id = protein_details["structure.chain"]
                    # omit alternate location handling because of erroneous annotation in KLIFS
                    systems.append(ProteinLigandComplex(components=[protein, ligand]))

    print(f"Generated {len(systems)} systems!")

    print("Intitializing featurizer ...")
    featurizer = OEPositDockingFeaturizer(
        cache_dir="../data/.cache",
        output_dir="../data/posit",
        n_processes=67
    )

    print("Featurizing systems ...")
    systems = featurizer.featurize(systems)
