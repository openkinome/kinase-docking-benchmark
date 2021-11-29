import argparse
from pathlib import Path
import time

from opencadd.databases.klifs import setup_remote
from openeye import oechem
import pandas as pd

from kinoml.databases.pdb import smiles_from_pdb, download_pdb_structure
from kinoml.modeling.OEModeling import (
    read_molecules, select_chain, select_altloc, superpose_proteins, are_identical_molecules
)


CACHE_DIR = Path("../data/.cache")


def get_klifs_residues(structure, structure_klifs_id):
    """Get a list of all KLIFS residues in the format 'ALA123'."""
    from opencadd.databases.klifs import setup_remote
    from kinoml.modeling.OEModeling import residue_ids_to_residue_names, remove_non_protein

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

    return klifs_residues


def get_rmsd(oemol1, oemol2):
    """
    Calculate the heavy atom RMSD between two molecules without prior alignment
    using SpyRMSD. This tool does include valence information in RMSD calculation.
    All hydrogens will be removed to account for different tautomeric and
    protonation states.
    """
    from tempfile import NamedTemporaryFile
    from spyrmsd import io, rmsd
    from kinoml.modeling.OEModeling import write_molecules

    oemol1 = oemol1.CreateCopy()
    oemol2 = oemol2.CreateCopy()

    coords = []
    anums = []
    adj = []
    for i, oemol in enumerate([oemol1, oemol2]):
        oechem.OESuppressHydrogens(oemol)
        with NamedTemporaryFile(mode="w", suffix=".pdb") as tmp:
            write_molecules([oemol], tmp.name)
            mol = io.loadmol(tmp.name)
            coords.append(mol.coordinates)
            anums.append(mol.atomicnums)
            adj.append(mol.adjacency_matrix)

    RMSD = rmsd.symmrmsd(
        coords[0],
        coords[1:],
        anums[0],
        anums[1],
        adj[0],
        adj[1],
    )

    return RMSD[0]


def load_klifs_ligand(klifs_structure_id):
    """
    Download and load an orthosteric ligand of a kinase structure defined by its KLIFS structure ID.
    """
    from opencadd.databases.klifs import setup_remote
    from kinoml.modeling.OEModeling import read_molecules
    from kinoml.utils import LocalFileStorage

    remote = setup_remote()

    file_path = LocalFileStorage.klifs_ligand_mol2(klifs_structure_id)
    if not file_path.is_file():
        mol2_text = remote.coordinates.to_text(
            klifs_structure_id, entity="ligand", extension="mol2"
        )
        with open(file_path, "w") as wf:
            wf.write(mol2_text)
    klifs_ligand = read_molecules(file_path)[0]

    return klifs_ligand


def get_fingerprint_similarity(smiles1, smiles2):
    """
    Calculate the similarity between two molecules using Morgan feature fingerprints and the Dice
    similarity measure.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    rdkit_mol1 = Chem.MolFromSmiles(smiles1)
    rdkit_mol2 = Chem.MolFromSmiles(smiles2)

    fingerprint1 = AllChem.GetMorganFingerprint(rdkit_mol1, 2, useFeatures=True)
    fingerprint2 = AllChem.GetMorganFingerprint(rdkit_mol2, 2, useFeatures=True)

    return DataStructs.DiceSimilarity(fingerprint1, fingerprint2)


def get_shape_similarity(oemol, smiles):
    """
    Calculate the shape similarity between a molecule in 3D with known conformation and a molecule
    in 2D with unknown conformation.
    """
    from kinoml.modeling.OEModeling import (
        generate_reasonable_conformations, overlay_molecules, read_smiles
    )

    conformations_ensemble = generate_reasonable_conformations(
        read_smiles(smiles)
    )
    shape_similarity = max([
        overlay_molecules(oemol, conformations)[0] for conformations in conformations_ensemble
    ])

    return shape_similarity


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str)
    args = parser.parse_args()

    # get sdf files
    directory = Path("../data/" + args.method)
    sdf_files = list(directory.glob("*ligand.sdf"))

    # get structures from KLIFS
    remote = setup_remote()
    structures = remote.structures.all_structures()

    # analyze docking poses
    try:  # look for existing results
        benchmark_results = pd.read_csv(f"../data/{args.method}_results.csv", index_col=0)
        benchmark_results = benchmark_results.to_dict("index")
        already_analyzed = set([file[:-2] for file in benchmark_results.keys()])
    except FileNotFoundError:
        benchmark_results = {}
        already_analyzed = set()
    for i, sdf_file in enumerate(sdf_files):
        print(i, sdf_file)
        if sdf_file.name in already_analyzed:
            continue  # skip already analyzed docking poses
        # load structures and select chain and alternate location
        protein_pdb_id = sdf_file.name.split("_")[5]
        protein_chain_id = sdf_file.name.split("_")[6][5]
        if args.method == "posit":
            ligand_pdb_id, ligand_expo_id = sdf_file.name.split("_")[9].split("-")
        else:
            ligand_pdb_id, ligand_expo_id = sdf_file.name.split("_")[8].split("-")
        protein_structure_details = structures[
            (structures["structure.pdb_id"] == protein_pdb_id) &
            (structures["structure.chain"] == protein_chain_id) &
            (structures["ligand.expo_id"] != "-")
        ].iloc[0]
        ligand_structure_details = structures[
            (structures["structure.pdb_id"] == ligand_pdb_id) &
            (structures["ligand.expo_id"] == ligand_expo_id)
        ].iloc[0]
        protein_structure_path = download_pdb_structure(
            protein_structure_details["structure.pdb_id"], CACHE_DIR
        )
        ligand_structure_path = download_pdb_structure(
            ligand_structure_details["structure.pdb_id"], CACHE_DIR
        )
        protein_structure = read_molecules(protein_structure_path)[0]
        protein_structure = select_chain(protein_structure, protein_chain_id)
        if protein_structure_details["structure.alternate_model"] != "-":
            try:
                protein_structure = select_altloc(
                    protein_structure, protein_structure_details["structure.alternate_model"]
                )
            except ValueError:  # KLIFS contains erroneously annotated altloc information
                print(
                    f"PDB entry {protein_structure_details['structure.pdb_id']} does not contain "
                    f"alternate location {protein_structure_details['structure.alternate_model']}. "
                    "Continuing without selecting alternate location."
                )
        ligand_structure = read_molecules(ligand_structure_path)[0]
        ligand_structure = select_chain(
            ligand_structure, ligand_structure_details["structure.chain"]
        )
        if ligand_structure_details["structure.alternate_model"] != "-":
            try:
                ligand_structure = select_altloc(
                    ligand_structure, ligand_structure_details["structure.alternate_model"]
                )
            except ValueError:  # KLIFS contains erroneously annotated altloc information
                print(
                    f"PDB entry {ligand_structure_details['structure.pdb_id']} does not contain "
                    f"alternate location {ligand_structure_details['structure.alternate_model']}. "
                    "Continuing without selecting alternate location."
                )
        # superposition
        for j in range(10):
            try:
                klifs_residues = get_klifs_residues(
                    protein_structure, protein_structure_details["structure.klifs_id"]
                )
            except:
                if j < 10:
                    print(f"ConnectionError trial {j + 1}")
                    time.sleep(1)
                    continue
            break
        ligand_structure = superpose_proteins(
            protein_structure,
            ligand_structure,
            klifs_residues,
            protein_structure_details["structure.chain"]
        )
        # extract co-crystallized ligand
        oechem.OEPlaceHydrogens(ligand_structure)
        for atom in ligand_structure.GetAtoms():
            if oechem.OEAtomGetResidue(atom).GetName() != ligand_expo_id:
                ligand_structure.DeleteAtom(atom)
        docking_pose = read_molecules(sdf_file)[0]
        # ligand template similarity
        ligand_template_molecule = load_klifs_ligand(
            protein_structure_details["structure.klifs_id"]
        )
        ligand_smiles = smiles_from_pdb([ligand_expo_id])[ligand_expo_id]
        try:
            fingerprint_similarity = get_fingerprint_similarity(
                ligand_smiles, oechem.OEMolToSmiles(ligand_template_molecule)
            )
        except Exception as e:
            print(e)
            print(f"Could not calculate fingerprint similarity for {sdf_file.name}.")
            fingerprint_similarity = None
        shape_similarity = get_shape_similarity(ligand_template_molecule, ligand_smiles)
        # molecular identitiy
        different_molecule = False  # e.g. in case of missinterpretation or covalent molecules
        if not are_identical_molecules(ligand_structure, docking_pose):
            different_molecule = True
        # get potential poses
        pose_files = list(directory.glob(sdf_file.stem[:-6] + "pose*"))
        if len(pose_files) == 0:  # i.e. posit run, since it only gives a single pose
            pose_files = [sdf_file]
        for k, pose_file in enumerate(pose_files, start=1):
            docking_pose = read_molecules(pose_file)[0]
            # docking score
            docking_score = float(oechem.OEGetSDData(docking_pose, "Chemgauss4"))
            # posit probability
            try:
                posit_probability = float(oechem.OEGetSDData(docking_pose,"POSIT::Probability"))
            except ValueError:
                posit_probability = None
            # rmsd
            try:
                rmsd = get_rmsd(ligand_structure, docking_pose)
            except (AssertionError, ValueError) as e:
                rmsd = None
                print(e)
            # aggregate results
            results = {
                "protein_pdb_id": protein_pdb_id,
                "ligand_pdb_id": ligand_pdb_id,
                "ligand_expo_id": ligand_expo_id,
                "pose": k,
                "rmsd": rmsd,
                "docking_score": docking_score,
                "posit_probability": posit_probability,
                "different_molecule": different_molecule,
                "fingerprint_similarity": fingerprint_similarity,
                "shape_similarity": shape_similarity,
            }
            print(results)
            benchmark_results[sdf_file.name + f"_{k}"] = results
        if i % 10 == 0:
            benchmark_results_df = pd.DataFrame.from_dict(benchmark_results, orient="index")
            benchmark_results_df.to_csv(f"../data/{args.method}_results.csv")

    benchmark_results_df = pd.DataFrame.from_dict(benchmark_results, orient="index")
    benchmark_results_df.to_csv(f"../data/{args.method}_results.csv")
    print("Finished!")
