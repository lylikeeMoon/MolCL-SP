import os
import os.path as osp
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from openbabel import pybel
import re
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import warnings
warnings.filterwarnings('error')
from ogb.utils.features import  (atom_to_feature_vector, bond_to_feature_vector)
import tarfile
import codecs
from subword_nmt.apply_bpe import BPE
from rdkit.Chem.rdmolfiles import SDMolSupplier
from multiprocessing import Pool
from rdkit import Chem
import torch_geometric
from rdkit.Chem import Draw

from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from functools import lru_cache
from utils import mask_tokens_batch,mask_graph_batch


def parse_atomic_symbols(token):
    """
    解析 token 中的原子符号，保证匹配 SMILES 中合法的元素符号。
    """
    periodic_table = set([
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr"
    ])

    matches = re.findall(r'[A-Z][a-z]?', token)
    return [m for m in matches if m in periodic_table]

def espf_tokenize(smile, mol, vocab_path="ESPF/drug_codes_chembl_freq_1500.txt", subword_map_path="ESPF/subword_units_map_chembl_freq_1500.csv"):
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    sub_csv = pd.read_csv(subword_map_path)
    idx2word_d = sub_csv['index'].values  # 所有子结构（token）
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    tokenized_smiles = dbpe.process_line(smile).split()
    match_atoms = []
    match_atoms_cnt = []
    for token in tokenized_smiles:
        token_atoms=parse_atomic_symbols(token)
        match_atoms.append(token_atoms)
        match_atoms_cnt.append(len(token_atoms))
    current_match_atom_cnt = [0]*len(match_atoms_cnt)

    try:
        token_ids = np.asarray([words2idx_d[token] for token in tokenized_smiles])
    except KeyError:
        token_ids = np.array([0])

    mol_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_count = len(mol_atoms)
    atom_substructure_mapping = [-1] * atom_count

    atom_idx = 0
    temp_token_pos = 0

    for atom_idx in range(0, atom_count):
        flag = False
        current_token_pos = temp_token_pos
        for token in tokenized_smiles[current_token_pos:]:

            if current_match_atom_cnt[temp_token_pos] >= match_atoms_cnt[temp_token_pos]:
                temp_token_pos += 1
            else:
                token_atoms = match_atoms[temp_token_pos]
                if mol_atoms[atom_idx] in token_atoms:
                    atom_substructure_mapping[atom_idx] = temp_token_pos
                    current_match_atom_cnt[temp_token_pos] += 1
                    match_atoms[temp_token_pos].remove(mol_atoms[atom_idx])
                    flag = True
                    break
                temp_token_pos += 1
        if not flag:
            temp_token_pos = current_token_pos



    max_length = 50
    seq_length = len(token_ids)

    if seq_length < max_length:
        padded_tokens = np.pad(token_ids, (0, max_length - seq_length), 'constant', constant_values=0)
        attention_mask = [1] * seq_length + [0] * (max_length - seq_length)
    else:
        padded_tokens = token_ids[:max_length]
        attention_mask = [1] * max_length

    return tokenized_smiles, padded_tokens, np.asarray(attention_mask), seq_length, atom_substructure_mapping




class PCQM4Mv2Dataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset/",
            transform=None,
            pre_transform=None,
            xyzdir='dataset/pcqm4m-v2/pcqm4m-v2_xyz',
            mask_ratio=0.5
    ):
        self.original_root = root
        self.mask_ratio = mask_ratio
        self.folder = osp.join(root, "pcqm4m-v2")
        self.version = 1

        self.url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"

        if osp.isdir(self.folder) and (
                not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("PCQM4Mv2 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        self.xyzdir = xyzdir

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        split_dict = self.get_idx_split()
        train_idxs = split_dict["train"].tolist()
        print("Converting SMILES strings into graphs...")
        data_list = []

        for i in tqdm(range(len(smiles_list))):
            # data = DGData()
            data = Data()
            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]

            if i in train_idxs:
                prefix = i // 10000
                prefix = "{0:04d}0000_{0:04d}9999".format(prefix)
                xyzfn = osp.join(self.xyzdir, prefix, f"{i}.sdf")
                mol_from_sdf = next(SDMolSupplier(xyzfn))
                pos_from_sdf = mol_from_sdf.GetConformer(0).GetPositions()
                smiles = Chem.MolToSmiles(mol_from_sdf, isomericSmiles=True)
                mol = Chem.MolFromSmiles(smiles)
                atom_mapping = mol.GetSubstructMatch(mol_from_sdf)
                pos = np.zeros_like(pos_from_sdf)
                for sdf_idx, smiles_idx in enumerate(atom_mapping):
                    pos[smiles_idx] = pos_from_sdf[sdf_idx]
                num_atoms = mol.GetNumAtoms()
            else:
                mol = Chem.MolFromSmiles(smiles)
                num_atoms = mol.GetNumAtoms()
                pos = np.zeros((num_atoms, 3), dtype=float)
            # atoms
            atom_features_list = []
            for atom in mol.GetAtoms():

                atom_features_list.append(atom_to_feature_vector(atom))
            x = np.array(atom_features_list, dtype=np.int64)
            # bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            edge_index = np.array(edges_list, dtype=np.int64).T
            edge_attr = np.array(edge_features_list, dtype=np.int64)

            data.x = torch.from_numpy(x).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
            data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
            data.pos = torch.from_numpy(pos).to(torch.float)
            espf_smiles, data.tokens, data.attention_mask, substructure_num, data.atom2substructure = espf_tokenize(smiles,mol)
            data.ori_smiles = smiles

            if data.pos.size()[0] == 0 or data.pos.size()[1] == 0:
                print("zero!")
                print(data.pos.size())
                continue
            data.num_nodes = num_atoms

            data_list.append(data)

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, "split_dict.pt"))
        )
        return split_dict



if __name__ == "__main__":
    dataset = PCQM4Mv2Dataset()
    print('dataset load finish')
    split_idx = dataset.get_idx_split()
    print('split idx load finish')

    train_loader = torch_geometric.loader.DataListLoader(
        dataset[split_idx['train']], batch_size=256, drop_last=True, shuffle=True
    )

    # data_test(train_loader)
    valid_loader = torch_geometric.loader.DataListLoader(
        dataset[split_idx['valid']], batch_size=256, drop_last=True, shuffle=True
    )

    train_data_len = len(train_loader)
    val_data_len = len(valid_loader)
    print('train dataset length: ', train_data_len)
    print('val dataset length: ', val_data_len)