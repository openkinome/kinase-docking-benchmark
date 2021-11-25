from pathlib import Path

from dask_jobqueue import LSFCluster
from dask.distributed import Client
import pandas as pd

from kinoml.core.components import BaseProtein
from kinoml.core.ligands import RDKitLigand
from kinoml.core.systems import ProteinLigandComplex
from kinoml.features.complexes import OEHybridDockingFeaturizer


if __name__ == '__main__':

    print("Reading benchmark dataframe ...")
    structures = pd.read_csv("../data/docking_benchmark_dataset.csv")

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
                    protein = BaseProtein(
                        name=f"{protein_details['species.klifs']}_{protein_details['kinase.klifs_name']}"
                    )
                    protein.pdb_id = protein_details["structure.pdb_id"]
                    protein.expo_id = protein_details["ligand.expo_id"]
                    protein.chain_id = protein_details["structure.chain"]
                    # omit alternate location handling because of erroneous annotation in KLIFS
                    systems.append(ProteinLigandComplex(components=[protein, ligand]))

    print(f"Generated {len(systems)} systems!")

    print("Setting up cluster ...")
    cluster = LSFCluster(
        cores=1,
        memory="16 GB",
        queue="cpuqueue",
        walltime="48:00",
    )
    cluster.scale(100)
    client = Client(cluster)

    print("Intitializing featurizer ...")
    featurizer = OEHybridDockingFeaturizer(
        cache_dir="../data/.cache",
        output_dir="../data/hybrid",
        dask_client=client
    )

    print("Featurizing systems ...")
    systems = featurizer.featurize(systems)
