import os
import numpy as np
import pandas as pd
from typing import List
from deepchem.molnet import *
from rdkit import Chem


MOLNET_DIRECTORY = {
    "bbbp": {"dataset_type": "classification", "load_fn": load_bbbp, "split": "scaffold",},
    "clearance": {"dataset_type": "regression", "load_fn": load_clearance, "split": "scaffold",},
    "clintox": {
        "dataset_type": "classification",
        "load_fn": load_clintox,
        "split": "scaffold",
        "tasks_wanted": ["CT_TOX"],
    },
    "delaney": {"dataset_type": "regression", "load_fn": load_delaney, "split": "scaffold",},
    "hiv": {"dataset_type": "classification", "load_fn": load_hiv, "split": "scaffold",},
    # pcba is very large and breaks the dataloader
    #     "pcba": {
    #         "dataset_type": "classification",
    #         "load_fn": load_pcba,
    #         "split": "scaffold",
    #     },
    "qm7": {"dataset_type": "regression", "load_fn": load_qm7, "split": "random",},
    "qm8": {"dataset_type": "regression", "load_fn": load_qm8, "split": "random",},
    "qm9": {"dataset_type": "regression", "load_fn": load_qm9, "split": "random",},
    "sider": {"dataset_type": "classification", "load_fn": load_sider, "split": "scaffold",},
    "tox21": {
        "dataset_type": "classification",
        "load_fn": load_tox21,
        "split": "scaffold",
        # "tasks_wanted": ["SR-p53"],
    },
    "bace": {
        "dataset_type": "classification",
        "load_fn": load_bace_classification,
        "split": "scaffold",
    },
    "muv": {"dataset_type": "classification", "load_fn": load_muv, "split": "scaffold"},
    "toxcast":{"dataset_type": "classification", "load_fn": load_toxcast, "split": "scaffold"}
}


def get_task_type(task):
    return MOLNET_DIRECTORY[task]["dataset_type"]


def load_molnet_dataset(name: str, split: str = None, tasks_wanted: List = None):
    """Loads a MolNet dataset into a DataFrame ready for either chemberta or chemprop.
    Args:
        name: Name of MolNet dataset (e.g., "bbbp", "tox21").
        split: Split name. Defaults to the split specified in MOLNET_DIRECTORY.
        tasks_wanted: List of tasks from dataset. Defaults to `tasks_wanted` in MOLNET_DIRECTORY, if specified, or else all available tasks.
        df_format: `chemberta` or `chemprop`
    Returns:
        tasks_wanted, (train_df, valid_df, test_df), transformers
    """
    # if name.startswith('ogbg'):
    #     from molecule.ogbdata import DatasetWrapper
    #     dataset = DatasetWrapper(name)
    #     return dataset.get_dataset()
    # if name.startswith("kfold"):
    #     from molecule.kfold import KfoldDataset
    #     dataset = KfoldDataset(name)
    #     return dataset.get_dataset()

    load_fn = MOLNET_DIRECTORY[name]["load_fn"]
    tasks, splits, transformers = load_fn(
        featurizer="Raw", splitter=split or MOLNET_DIRECTORY[name]["split"]
    )

    # Default to all available tasks
    if tasks_wanted is None:
        tasks_wanted = tasks
    print(f"Using tasks {tasks_wanted} from available tasks for {name}: {tasks}")

    return (
        tasks_wanted,
        [
            make_dataframe(s, MOLNET_DIRECTORY[name]["dataset_type"], tasks, tasks_wanted,)
            for s in splits
        ],
        transformers,
    )


def make_dataframe(dataset, dataset_type, tasks, tasks_wanted):
    df = dataset.to_dataframe()
    if len(tasks) == 1:
        mapper = {"y": tasks[0]}
    else:
        mapper = {f"y{y_i + 1}": task for y_i, task in enumerate(tasks)}
    df.rename(mapper, axis="columns", inplace=True)

    # Canonicalize SMILES
    # smiles_list = [Chem.MolToSmiles(s, isomericSmiles=True) for s in df["X"]]
    smiles_list = [s for s in df["ids"]]
    # Convert labels to integer for classification
    labels = df[tasks_wanted]
    if dataset_type == "classification":
        labels = labels.astype(int)
    labels = labels.values.tolist()
    return pd.DataFrame({"text": smiles_list, "labels": labels})
