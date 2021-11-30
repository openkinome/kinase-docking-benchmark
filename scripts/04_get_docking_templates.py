from pathlib import Path
from multiprocessing import Pool

from opencadd.databases.klifs import setup_remote
import pandas as pd


CACHE_DIR = Path("../data/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def read_klifs_ligand(structure_id: int, directory):
    """Retrieve and read an orthosteric kinase ligand from KLIFS."""
    from pathlib import Path

    from kinoml.modeling.OEModeling import read_molecules
    from kinoml.utils import LocalFileStorage

    file_path = LocalFileStorage.klifs_ligand_mol2(structure_id, Path(directory))

    if not file_path.is_file():
        from opencadd.databases.klifs import setup_remote

        remote = setup_remote()
        try:
            mol2_text = remote.coordinates.to_text(structure_id, entity="ligand", extension="mol2")
        except ValueError:
            print(f"Unable to fetch ligand coordinates of structure with KLIFS ID {structure_id}.")
            return None
        with open(file_path, "w") as wf:
            wf.write(mol2_text)

    molecule = read_molecules(file_path)[0]

    return molecule


def count_ligands(pdb_id, chain_id, expo_id, directory):
    """Count the ligands in the given PDB entry."""
    from openeye import oechem

    from kinoml.modeling.OEModeling import read_molecules, select_chain, remove_non_protein
    from kinoml.databases.pdb import download_pdb_structure

    structure_path = download_pdb_structure(pdb_id, directory)
    if pdb_id:
        structure = read_molecules(structure_path)[0]
    else:
        raise ValueError(f"Could not download PDB entry {pdb_id}!")

    structure = select_chain(structure, chain_id)
    structure = remove_non_protein(structure, exceptions=[expo_id])

    hierview = oechem.OEHierView(structure)
    count = sum([1 for residue in hierview.GetResidues() if residue.GetResidueName() == expo_id])

    return count


def get_docking_template(structure, docking_templates, cache_dir):
    """Get most similar docking template in the same conformation excluding itself."""
    from kinoml.modeling.OEModeling import (
        read_smiles,
        generate_reasonable_conformations,
        overlay_molecules
    )

    # filter for conformation
    docking_templates = docking_templates[
        docking_templates["structure.dfg"] == structure["structure.dfg"]
    ]
    docking_templates = docking_templates[
        docking_templates["structure.ac_helix"] == structure["structure.ac_helix"]
    ]

    # remove itself
    docking_templates = docking_templates[
        docking_templates["structure.pdb_id"] != structure["structure.pdb_id"]
    ]

    # find entry with most similar ligand
    complex_ligands = [
        read_klifs_ligand(structure_id, cache_dir) for structure_id
        in docking_templates["structure.klifs_id"]
    ]
    complex_ligands = [complex_ligand for complex_ligand in complex_ligands if complex_ligand]
    conformations_ensemble = generate_reasonable_conformations(read_smiles(structure["smiles"]))
    overlay_scores = []
    for conformations in conformations_ensemble:
        overlay_scores += [
            [i, overlay_molecules(complex_ligand, conformations)[0]]
            for i, complex_ligand in enumerate(complex_ligands)
        ]
    docking_template_index, docking_template_similarity = sorted(
        overlay_scores, key=lambda x: x[1], reverse=True
    )[0]
    docking_template = docking_templates.iloc[docking_template_index]

    # return dictionary with key pdb_id_expo_id and dictionary as value with docking template
    # pdb_id, expo_id and chain_id
    docking_template_dict = {
        f"{structure['structure.pdb_id']}_{structure['ligand.expo_id']}" : {
            "docking_template_pdb_id": docking_template["structure.pdb_id"],
            "docking_template_chain_id": docking_template["structure.chain"],
            "docking_template_expo_id": docking_template["ligand.expo_id"],
            "docking_template_similarity": docking_template_similarity,
        }
    }

    return docking_template_dict

if __name__ == "__main__":

    print("Reading benchmark dataframe ...")
    structures = pd.read_csv("../data/docking_benchmark_dataset.csv")

    print("Retrieving available docking templates ...")
    remote = setup_remote()
    docking_templates = remote.structures.all_structures()

    print("Filtering available docking templates")
    # orthosteric ligand
    docking_templates = docking_templates[
        docking_templates["ligand.expo_id"] != "-"
    ]
    # single orthosteric ligand
    docking_templates = docking_templates.groupby("structure.pdb_id").filter(
        lambda x: len(set(x["ligand.expo_id"])) == 1
    )
    # remove structures with ligands not handled by oespruce
    docking_templates = docking_templates[docking_templates["ligand.expo_id"] != "A"]
    # sort by quality
    docking_templates = docking_templates.sort_values(
        by=[
            "structure.qualityscore", "structure.resolution", "structure.chain",
            "structure.alternate_model"
        ],
        ascending=[False, True, True, True]
    )
    # keep highest quality structure per PDB ID
    docking_templates = docking_templates.groupby("structure.pdb_id").head(1)
    # remove structues with multiple instances of the same ligand
    multiple_ligands_indices = []
    erroneous_indices = []
    for index, structure in docking_templates.iterrows():
        try:
            if count_ligands(structure["structure.pdb_id"], structure["structure.chain"],
                    structure["ligand.expo_id"], CACHE_DIR) > 1:
                multiple_ligands_indices.append(index)
        except ValueError:
            print("Error counting ligands:")
            print(
                structure["structure.pdb_id"],
                structure["structure.chain"],
                structure["ligand.expo_id"]
            )
            erroneous_indices.append(index)
    docking_templates = docking_templates[~docking_templates.index.isin(multiple_ligands_indices)]
    docking_templates = docking_templates[~docking_templates.index.isin(erroneous_indices)]

    print("Downloading ligand structures of filtered docking templates ...")
    unavailable_ligand_indices = []
    for index, docking_template in docking_templates.iterrows():
        if not read_klifs_ligand(docking_template["structure.klifs_id"], CACHE_DIR):
            unavailable_ligand_indices.append(index)
    docking_templates = docking_templates[~docking_templates.index.isin(unavailable_ligand_indices)]

    print("Getting docking templates ...")
    with Pool(processes=50) as pool:
        results = pool.starmap(
            get_docking_template,
            [(structure, docking_templates, CACHE_DIR) for i, structure in structures.iterrows()]
        )

    print("Merging and saving results ...")
    results_merged = {}
    for result in results:
        results_merged.update(result)
    results_merged = pd.DataFrame.from_dict(results_merged, orient="index")
    results_merged.to_csv("../data/docking_templates.csv")

    print("Finished!")
