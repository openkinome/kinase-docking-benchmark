"""
This script reads the docking_benchmark_dataset.csv file generated by
01_generate_benchmark_dataset.py and calculates a similarity matrix over all structures using the
RMSD of the KLIFS binding pocket.
"""
from pathlib import Path
from typing import Iterable

from openeye import oechem
import pandas as pd


CACHE_DIR = Path("../data/.cache")
klifs_residues_dict = {}
structure_dict = {}


def get_klifs_residues(structure, structure_klifs_id, klifs_residues_dict=klifs_residues_dict):
    """Get a list of all KLIFS residues in the format 'ALA123'."""
    from opencadd.databases.klifs import setup_remote
    from kinoml.modeling.OEModeling import residue_ids_to_residue_names, remove_non_protein

    if structure_klifs_id in klifs_residues_dict.keys():
        return klifs_residues_dict[structure_klifs_id]

    remote = setup_remote()

    protein = remove_non_protein(structure, remove_water=True)
    klifs_residue_ids = [
        residue_id for residue_id in
        remote.pockets.by_structure_klifs_id(structure_klifs_id)["residue.id"]
        if residue_id != "_"
    ]

    klifs_residue_names = residue_ids_to_residue_names(protein, klifs_residue_ids)

    klifs_residues = [
        residue_name + residue_id for residue_name, residue_id in
        zip(klifs_residue_names, klifs_residue_ids)
    ]

    klifs_residues_dict[structure_klifs_id] = klifs_residues

    return klifs_residues


def load_pdb_entry(pdb_id, chain_id, alternate_location, directory=CACHE_DIR):
    """Load a PDB entry as OpenEye molecule."""
    from kinoml.databases.pdb import download_pdb_structure
    from kinoml.modeling.OEModeling import read_molecules, select_chain, select_altloc

    structure_path = download_pdb_structure(pdb_id, directory)
    structure = read_molecules(structure_path)[0]
    structure = select_chain(structure, chain_id)
    if alternate_location != "-":
        try:
            structure = select_altloc(structure, alternate_location)
        except ValueError:  # KLIFS contains erroneously annotated altloc information
            print(
                f"PDB entry {pdb_id} does not contain "
                f"alternate location {alternate_location}. "
                "Continuing without selecting alternate location."
            )
    return structure


def superpose_proteins(
        reference_protein: oechem.OEMolBase,
        fit_protein: oechem.OEMolBase,
        residues: Iterable = tuple(),
        chain_id: str = " ",
        insertion_code: str = " "
) -> oechem.OEMolBase:
    """
    Superpose a protein structure onto a reference protein. The superposition
    can be customized to consider only the specified residues.

    Parameters
    ----------
    reference_protein: oechem.OEMolBase
        An OpenEye molecule holding a protein structure which will be used as reference during superposition.
    fit_protein: oechem.OEMolBase
        An OpenEye molecule holding a protein structure which will be superposed onto the reference protein.
    residues: Iterable of str
        Residues that should be used during superposition in format "GLY123".
    chain_id: str
        Chain identifier for residues that should be used during superposition.
    insertion_code: str
        Insertion code for residues that should be used during superposition.

    Returns
    -------
    superposed_protein: oechem.OEMolBase
        An OpenEye molecule holding the superposed protein structure.
    """
    from openeye import oespruce

    # do not modify input
    superposed_protein = fit_protein.CreateCopy()

    # set superposition method
    options = oespruce.OESuperpositionOptions()
    if len(residues) == 0:
        options.SetSuperpositionType(oespruce.OESuperpositionType_Global)
    else:
        options.SetSuperpositionType(oespruce.OESuperpositionType_Site)
        for residue in residues:
            options.AddSiteResidue(f"{residue[:3]}:{residue[3:]}:{insertion_code}:{chain_id}")

    # perform superposition
    superposition = oespruce.OEStructuralSuperposition(
        reference_protein, superposed_protein, options
    )
    superposition.Transform(superposed_protein)
    rmsd = superposition.GetRMSD()

    return rmsd, superposed_protein


docking_benchmark_dataset = pd.read_csv("../data/docking_benchmark_dataset.csv", index_col=0)
print(f"Number of entries: {len(docking_benchmark_dataset)}")

rmsd_matrix = []
for i, (_, entry1) in enumerate(docking_benchmark_dataset.iterrows()):
    print(f"Processing entry {i} ...")
    rmsd_row = []
    if entry1["structure.klifs_id"] in structure_dict.keys():
        structure1 = structure_dict[entry1["structure.klifs_id"]]
    else:
        structure1 = load_pdb_entry(
            entry1["structure.pdb_id"],
            entry1["structure.chain"],
            entry1["structure.alternate_model"]
        )
        structure_dict[entry1["structure.klifs_id"]] = structure1
    for j in range(10):
        try:
            klifs_residues1 = get_klifs_residues(
                structure1, entry1["structure.klifs_id"]
            )
        except:
            if j < 10:
                print(f"ConnectionError trial {i + 1}")
                time.sleep(1)
                continue
        break
    for _, entry2 in docking_benchmark_dataset.iterrows():
        if entry1["kinase.klifs_id"] != entry2["kinase.klifs_id"]:
            rmsd_row.append("NA")
            continue
        if entry2["structure.klifs_id"] in structure_dict.keys():
            structure2 = structure_dict[entry2["structure.klifs_id"]]
        else:
            structure2 = load_pdb_entry(
                entry2["structure.pdb_id"],
                entry2["structure.chain"],
                entry2["structure.alternate_model"]
            )
            structure_dict[entry2["structure.klifs_id"]] = structure2
        rmsd, structure2_superposed = superpose_proteins(
            structure1,
            structure2,
            klifs_residues1,
            entry1["structure.chain"]
        )
        rmsd_row.append(rmsd)
    rmsd_matrix.append(rmsd_row)

print("Saving results ...")
structure_klifs_ids = docking_benchmark_dataset["structure.klifs_id"]
rmsd_matrix = pd.DataFrame(rmsd_matrix, index=structure_klifs_ids, columns=structure_klifs_ids)
rmsd_matrix.index.names = [""]
rmsd_matrix.to_csv("../data/pocket_rmsd_matrix.csv")

print("Finished!")
