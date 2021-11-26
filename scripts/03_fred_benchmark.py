from pathlib import Path

from opencadd.databases.klifs import setup_remote
import pandas as pd

from kinoml.core.components import BaseProtein
from kinoml.core.ligands import RDKitLigand
from kinoml.core.systems import ProteinLigandComplex
from kinoml.features.complexes import OEFredDockingFeaturizer


if __name__ == '__main__':
    print("Connecting to KLIFS ...")
    remote = setup_remote()

    print("Reading benchmark dataframe ...")
    structures = pd.read_csv("../data/docking_benchmark_dataset.csv")

    print("Getting list of files ...")
    path_list = [path.stem for path in Path("../data/fred").glob("*.sdf")]

    print("Generating systems ...")
    pocket_resids_dict = {}
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
                    path_list_tmp = [path for path in path_list if path.split("_")[8] == ligand_name]
                    path_list_tmp = [path for path in path_list_tmp if path.split("_")[5] == protein_details["structure.pdb_id"]]
                    if len(path_list_tmp) > 0:
                        continue
                    protein = BaseProtein(
                        name=f"{protein_details['species.klifs']}_{protein_details['kinase.klifs_name']}"
                    )
                    protein.pdb_id = protein_details["structure.pdb_id"]
                    protein.expo_id = protein_details["ligand.expo_id"]
                    protein.chain_id = protein_details["structure.chain"]
                    # get KLIFS pocket residues
                    if protein_details["structure.pdb_id"] not in pocket_resids_dict.keys():
                        pocket = remote.pockets.by_structure_klifs_id(protein_details["structure.klifs_id"])
                        if any(pd.isnull(pocket["residue.id"])):
                            raise ValueError(
                                f"Pocket not availabe for KLIFS strcuture ID {protein_details['structure.klifs_id']}!"
                            )
                        else:
                            pocket_resids = [
                                int(residue_id) for residue_id in pocket["residue.id"] if residue_id != "_"
                            ]
                            pocket_resids_dict[protein_details["structure.pdb_id"]] = pocket_resids
                    else:
                        pocket_resids = pocket_resids_dict[protein_details["structure.pdb_id"]]
                    protein.pocket_resids = pocket_resids
                    # omit alternate location handling because of erroneous annotation in KLIFS
                    systems.append(ProteinLigandComplex(components=[protein, ligand]))

    print(f"Generated {len(systems)} systems!")

    print("Intitializing featurizer ...")
    featurizer = OEFredDockingFeaturizer(
        cache_dir="../data/.cache",
        output_dir="../data/fred",
        n_processes=67
    )

    print("Featurizing systems ...")
    systems = featurizer.featurize(systems)
