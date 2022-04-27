"""
This script will generate a data set that will be used to explore the impact of mutations on the
docking performance. Similar filters are applied as in 01_generate_benchmark_dataset.py except
structures are filtered to contain mutations. Furthermore, structures are filtered for kinases
that were already present in the docking_benchmark_dataset.csv.
"""
from pathlib import Path

from kinoml.databases.pdb import smiles_from_pdb
from opencadd.databases.klifs import setup_remote
import pandas as pd


def is_valid_smiles(smiles):
    """Check if given smiles can be handled by OpenEye and RDKit."""
    from kinoml.modeling.OEModeling import read_smiles
    from rdkit import Chem, RDLogger

    # turn of noisy RDKit logging
    RDLogger.DisableLog("rdApp.*")

    # test openeye
    try:
        read_smiles(smiles)
    except ValueError:
        return False

    # test rdkit
    if not Chem.MolFromSmiles(smiles):
        return False

    return True


def has_reasonable_mw(smiles, min_mw=150, max_mw=1000):
    """Check if given smiles falls within molecular weight range."""
    from kinoml.modeling.OEModeling import read_smiles
    from openeye import oechem

    mol = read_smiles(smiles)

    if min_mw < oechem.OECalculateMolecularWeight(mol) < max_mw:
        return True

    return False


def count_ligands(pdb_id, chain_id, expo_id):
    """Count the ligands in the given PDB entry."""
    from openeye import oechem

    from kinoml.modeling.OEModeling import read_molecules, select_chain, remove_non_protein
    from kinoml.databases.pdb import download_pdb_structure

    structure_path = download_pdb_structure(pdb_id)
    if pdb_id:
        structure = read_molecules(structure_path)[0]
    else:
        raise ValueError(f"Could not download PDB entry {pdb_id}!")

    structure = select_chain(structure, chain_id)
    structure = remove_non_protein(structure, exceptions=[expo_id])

    hierview = oechem.OEHierView(structure)
    count = sum([1 for residue in hierview.GetResidues() if residue.GetResidueName() == expo_id])

    return count


def has_covalent_ligand(pdb_id, chain_id, expo_id):
    """Check if the given PDB entry has a covalent ligand bound."""
    from openeye import oechem

    from kinoml.modeling.OEModeling import read_molecules, select_chain, remove_non_protein
    from kinoml.databases.pdb import download_pdb_structure

    structure_path = download_pdb_structure(pdb_id)
    if pdb_id:
        structure = read_molecules(structure_path)[0]
    else:
        raise ValueError(f"Could not download PDB entry {pdb_id}!")

    structure = select_chain(structure, chain_id)
    structure = remove_non_protein(structure, exceptions=[expo_id])

    split_options = oechem.OESplitMolComplexOptions()
    ligand = list(oechem.OEGetMolComplexComponents(
        structure, split_options, split_options.GetLigandFilter())
    )
    if len(ligand) > 0:
        return False

    split_options.SetSplitCovalent(True)
    ligand = list(oechem.OEGetMolComplexComponents(
        structure, split_options, split_options.GetLigandFilter())
    )
    if len(ligand) > 0:
        return True
    else:
        raise ValueError("Could not detect any ligand!")


path_log = Path("../data/docking_benchmark_dataset_mutation.log")
if path_log.is_file():
    path_log.unlink()

with open(path_log, "w") as log_file:
    log_file.write("Loading benchmark dataset groups ...")
    group_selection = pd.read_csv(
        "../data/docking_benchmark_dataset_groups.csv", index_col=False
    )
    log_file.write("\n\n\tspecies\tname\tklifs\tgroup\tdfg\tac\tcount\n")

group_selection.to_csv(path_log, sep="\t", mode="a", header=False)

with open(path_log, "a") as log_file:

    log_file.write("\n\nGetting all available structures ...")
    remote = setup_remote()
    klifs_structures = remote.structures.all_structures()
    log_file.write(f"\nNumber of kinases: {len(klifs_structures['kinase.klifs_id'].unique())}")
    log_file.write(f"\nNumber of structures: {len(klifs_structures)}")
    log_file.write(
        f"\nNumber of unique ligands: {len(klifs_structures['ligand.expo_id'].unique())}"
    )

    log_file.write("\n\nApplying group selection to all available structures ...")
    selected_structure_ids = []
    for index, group in group_selection.iterrows():
        selected_structure_ids += klifs_structures[
            (klifs_structures["kinase.klifs_id"] == group["kinase.klifs_id"]) &
            (klifs_structures["structure.dfg"] == group["structure.dfg"]) &
            (klifs_structures["structure.ac_helix"] == group["structure.ac_helix"])
        ]["structure.klifs_id"].to_list()
    structures = klifs_structures[
        klifs_structures["structure.klifs_id"].isin(selected_structure_ids)
    ]
    log_file.write(f"\nNumber of kinases: {len(structures['kinase.klifs_id'].unique())}")
    log_file.write(f"\nNumber of structures: {len(structures)}")
    log_file.write(f"\nNumber of ligands: {len(structures['ligand.expo_id'].unique())}")

    log_file.write("\n\nFiltering for single orthosteric ligand ...")
    # keep structures with orthosteric ligands needed for hybrid and posit
    structures = structures[
        structures["ligand.expo_id"] != "-"
        ]
    # remove structures with multiple orthosteric ligands
    structures = structures.groupby("structure.pdb_id").filter(
        lambda x: len(set(x["ligand.expo_id"])) == 1
    )
    log_file.write(f"\nNumber of kinases: {len(structures['kinase.klifs_id'].unique())}")
    log_file.write(f"\nNumber of structures: {len(structures)}")
    log_file.write(f"\nNumber of unique ligands: {len(structures['ligand.expo_id'].unique())}")

    log_file.write("\n\nFiltering for highest quality chain per PDB entry ...")
    structures = structures.sort_values(
        by=[
            "structure.qualityscore", "structure.resolution", "structure.chain",
            "structure.alternate_model"
        ],
        ascending=[False, True, True, True]
    )
    structures = structures.groupby("structure.pdb_id").head(1)
    log_file.write(f"\nNumber of kinases: {len(structures['kinase.klifs_id'].unique())}")
    log_file.write(f"\nNumber of structures: {len(structures)}")
    log_file.write(f"\nNumber of unique ligands: {len(structures['ligand.expo_id'].unique())}")

    log_file.write("\n\nRemoving structures with ligands not handled by OESpruce ...")
    excluded_ligands = ["A"]
    log_file.write("\n")
    log_file.write(
        ", ".join(structures[
            structures["ligand.expo_id"].isin(excluded_ligands)
        ]["structure.pdb_id"].to_list())
    )
    structures = structures[~structures["ligand.expo_id"].isin(excluded_ligands)]
    log_file.write(f"\nNumber of kinases: {len(structures['kinase.klifs_id'].unique())}")
    log_file.write(f"\nNumber of structures: {len(structures)}")
    log_file.write(f"\nNumber of unique ligands: {len(structures['ligand.expo_id'].unique())}")

    log_file.write(
        "\n\nRemoving structures without mutations or missing residues in the KLIFS binding "
        "pocket ..."
    )
    # Retrieve wild type KLIFS pocket sequence
    wt_pockets = {}
    for kinase_klifs_id in structures["kinase.klifs_id"].unique():
        wt_pockets[kinase_klifs_id] = remote.kinases.by_kinase_klifs_id(
            kinase_klifs_id
        )["kinase.pocket"].iloc[0]
    # Identify structures without mutations or missing residues in the KLIFS pocket sequence
    wt_structure_indices = []
    missing_residues_indices = []
    for index, row in structures.iterrows():
        wt_sequence = wt_pockets[row["kinase.klifs_id"]]
        structure_sequence = row["structure.pocket"]
        if wt_sequence == structure_sequence:  # wt sequences
            wt_structure_indices.append(index)
        elif "_" in structure_sequence:  # missing residues
            missing_residues_indices.append(index)
    # Remove structures without mutations or missing residues in the KLIFS binding pocket
    structures = structures.drop(wt_structure_indices)
    structures = structures.drop(missing_residues_indices)
    log_file.write(f"\nNumber of kinases: {len(structures['kinase.klifs_id'].unique())}")
    log_file.write(f"\nNumber of structures: {len(structures)}")
    log_file.write(f"\nNumber of unique ligands: {len(structures['ligand.expo_id'].unique())}")

    log_file.write(
        "\n\nRemoving structures with SMILES not interpretable by RDKit or OpenEye, or having an "
        "unreasonable molecular weight ..."
    )
    # add smiles column
    smiles_dict = smiles_from_pdb(structures["ligand.expo_id"].to_list())
    smiles_list = []
    for index, row in structures.iterrows():
        try:
            smiles_list.append(smiles_dict[row["ligand.expo_id"]])
        except KeyError:
            smiles_list.append(None)
    structures.insert(len(structures.columns), "smiles", smiles_list)
    # check for missing smiles
    log_file.write("\n")
    log_file.write(
        ", ".join(structures[structures['smiles'].isnull()]["structure.pdb_id"].to_list())
    )
    log_file.write(
        f"\nRemoved {len(structures[structures['smiles'].isnull()])} "
        "structures with missing smiles."
    )
    structures = structures[structures["smiles"].notnull()]
    # test smiles with OpenEye and RDKit
    invalid_smiles_indices = []
    for index, structure in structures.iterrows():
        if not is_valid_smiles(structure["smiles"]):
            invalid_smiles_indices.append(index)
    log_file.write("\n")
    log_file.write(
        ", ".join(
            structures[structures.index.isin(invalid_smiles_indices)]["structure.pdb_id"].to_list()
        )
    )
    log_file.write(f"\nRemoved {len(invalid_smiles_indices)} structures with invalid smiles.")
    structures = structures[~structures.index.isin(invalid_smiles_indices)]
    # filter for reasonable molecular weight, i.e. 150 - 1000 Da
    wrong_mw_smiles_indices = []
    for index, structure in structures.iterrows():
        if not has_reasonable_mw(structure["smiles"]):
            wrong_mw_smiles_indices.append(index)
    log_file.write("\n")
    log_file.write(
        ", ".join(
            structures[structures.index.isin(wrong_mw_smiles_indices)]["structure.pdb_id"].to_list()
        )
    )
    log_file.write(
        f"\nRemoved {len(wrong_mw_smiles_indices)} structures with ligands of unreasonable "
        "molecular weight."
    )
    structures = structures[~structures.index.isin(wrong_mw_smiles_indices)]
    log_file.write(f"\nNumber of kinases: {len(structures['kinase.klifs_id'].unique())}")
    log_file.write(f"\nNumber of structures: {len(structures)}")
    log_file.write(f"\nNumber of unique ligands: {len(structures['ligand.expo_id'].unique())}")

    log_file.write(
        "\n\nRemoving structures with multiple instances of the same ligand bound to the same "
        "chain ..."
    )
    multiple_ligands_indices = []
    for index, structure in structures.iterrows():
        if count_ligands(structure["structure.pdb_id"], structure["structure.chain"],
                structure["ligand.expo_id"]) > 1:
            multiple_ligands_indices.append(index)
    log_file.write("\n")
    log_file.write(
        ", ".join(
            structures[structures.index.isin(
                multiple_ligands_indices
            )]["structure.pdb_id"].to_list()
        )
    )
    log_file.write(
        f"\nRemoved {len(multiple_ligands_indices)} structures with multiple instances of the same "
        "ligand bound to the same chain."
    )
    structures = structures[~structures.index.isin(multiple_ligands_indices)]
    log_file.write(f"\nNumber of kinases: {len(structures['kinase.klifs_id'].unique())}")
    log_file.write(f"\nNumber of structures: {len(structures)}")
    log_file.write(f"\nNumber of unique ligands: {len(structures['ligand.expo_id'].unique())}")

    log_file.write(
        "\n\nGetting kinases with at least 10 available structures per kinase and conformation ..."
    )
    # add kinase group information, will be needed one step later
    kinase_information = remote.kinases.by_kinase_klifs_id(
        list(structures["kinase.klifs_id"].unique())
    )
    for klifs_kinase_id in kinase_information["kinase.klifs_id"]:
        structures.loc[structures["kinase.klifs_id"] == klifs_kinase_id, "kinase.group"] = \
            kinase_information[
                kinase_information["kinase.klifs_id"] == klifs_kinase_id
            ]["kinase.group"].iloc[0]
    # group
    kinase_conformation_groups = structures[[
        "species.klifs", "kinase.klifs_name", "kinase.klifs_id", "kinase.group", "structure.dfg",
        "structure.ac_helix", "structure.klifs_id"
    ]].groupby([
        "species.klifs", "kinase.klifs_name", "kinase.klifs_id", "kinase.group", "structure.dfg",
        "structure.ac_helix"
    ]).count().sort_values("structure.klifs_id", ascending=False)
    kinase_conformation_groups.rename({"structure.klifs_id": "count"}, axis=1, inplace=True)
    # pick groups with at least 10 representatives
    kinase_conformation_groups = kinase_conformation_groups[
        kinase_conformation_groups["count"] >= 10
    ]

    log_file.write("\n\nPicking most populated kinases from diverse groups and conformations ...")
    group_selection = kinase_conformation_groups.groupby(
        ["kinase.group", "structure.dfg", "structure.ac_helix"]
    ).head(1)

    log_file.write("\n\nApplying group selection to filtered structures ...")
    selected_structure_ids = []
    for index, group in group_selection.iterrows():
        selected_structure_ids += structures[
            (structures["kinase.klifs_id"] == index[2]) &
            (structures["structure.dfg"] == index[4]) &
            (structures["structure.ac_helix"] == index[5])
        ]["structure.klifs_id"].to_list()
    selected_structures = structures[structures["structure.klifs_id"].isin(selected_structure_ids)]

    log_file.write("\n\nAdding information about covalent ligands ...")
    selected_structures = selected_structures.copy()
    selected_structures.loc[:, "covalent_ligand"] = False
    covalent_indices = []
    for index, structure in selected_structures.iterrows():
        if has_covalent_ligand(
            structure["structure.pdb_id"], structure["structure.chain"], structure["ligand.expo_id"]
        ):
            # exclude erroneous classified covalent ligands
            if structure["structure.pdb_id"] not in ['1bx6']:
                covalent_indices.append(index)
    selected_structures.loc[covalent_indices, "covalent_ligand"] = True
    log_file.write(f"\nFound {len(covalent_indices)} structures with covalent ligands.")
    log_file.write("\n")
    log_file.write(
        ", ".join(selected_structures[
            selected_structures["covalent_ligand"] == True
        ]["structure.pdb_id"].to_list())
    )

    log_file.write("\nSaving benchmark data set ...")
    selected_structures.to_csv("../data/docking_benchmark_dataset_mutation.csv")
    group_selection.to_csv("../data/docking_benchmark_dataset_mutation_groups.csv")

    log_file.write("\n\nDocking benchmark data set with mutations overview:")
    log_file.write(f"\nNumber of kinases: {len(selected_structures['kinase.klifs_id'].unique())}")
    log_file.write(
        f"\nNumber of kinase groups: {len(selected_structures['kinase.group'].unique())}"
    )
    log_file.write(f"\nNumber of structures: {len(selected_structures)}")
    log_file.write(f"\nNumber of ligands: {len(selected_structures['ligand.expo_id'].unique())}")
    log_file.write(
        "\nNumber of conformations: "
        f"{len(selected_structures.groupby(['structure.dfg', 'structure.ac_helix']))}"
    )
    log_file.write(
        "\nTotal number of docking runs to perform: "
        f"{sum([x * (x - 1) for x in group_selection['count'].to_list()])}"
    )
    log_file.write("\n\nspecies\tname\tklifs\tgroup\tdfg\tac\tcount\n")

group_selection.to_csv(path_log, sep="\t", mode="a", header=False)

print("Finished!")
