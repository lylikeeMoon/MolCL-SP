from torch_geometric.data import InMemoryDataset
import shutil, os
import os.path as osp
import torch
import re
from torch_sparse import SparseTensor

import numpy as np
from tqdm import tqdm
from ...utils.graph import smiles2graphwithface
from rdkit import Chem
from copy import deepcopy
from .deepchem_dataloader import (
    load_molnet_dataset,
    get_task_type,
)
from copy import deepcopy
import codecs
from subword_nmt.apply_bpe import BPE
import pandas as pd
from ....MPP.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data


class DGData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face)", key)):
            return -1
        elif bool(re.search("(nf_node|nf_ring|nei_tgt_mask)", key)):
            return -1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if bool(re.search("(ring_index|nf_ring)", key)):
            return int(self.num_rings.item())
        elif bool(re.search("(index|face|nf_node)", key)):
            return self.num_nodes
        else:
            return 0

def drug2emb_encoder(smile):
    vocab_path = "ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("ESPF/subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


def eespf_tokenize(smile,vocab_path = "ESPF/drug_codes_chembl_freq_1500.txt",subword_map_path = "ESPF/subword_units_map_chembl_freq_1500.csv"):

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    # 读取子结构索引映射表
    sub_csv = pd.read_csv(subword_map_path)
    idx2word_d = sub_csv['index'].values  # 所有子结构（token）
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))  # 词到索引的映射

    tokenized_smiles = dbpe.process_line(smile).split()  # # BPE 处理后拆分成 token 列表

    try:
        token_ids = np.asarray([words2idx_d[token] for token in tokenized_smiles])  #
    except KeyError:
        token_ids = np.array([0])

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smile}")
    # 生成分子结构图并保存
    #output_path = "molecule1.png"
    #Draw.MolToFile(mol, output_path, size=(300, 300))

    #print(f"分子图已保存到 {output_path}")

    # 计算化学式
    #formula = CalcMolFormula(mol)

    #print(f"化学式: {formula}")

    atom_count = mol.GetNumAtoms()  # 获取分子中的原子数
    atom_substructure_mapping = [-1] * atom_count

    # 5. 通过字符匹配建立原子到子结构的映射
    atom_idx = 0
    substructure_idx = 0
    smile_cursor = 0  # 遍历 SMILES 的指针

    for token in tokenized_smiles:
        # 在 SMILES 中找到 token 的位置
        token_pos = smile.find(token, smile_cursor)
        if token_pos == -1:
            substructure_idx += 1
            continue  # 如果找不到，跳过

        # 计算 token 中可能包含的原子数
        sub_atom_count = sum(1 for c in token if c.isalpha())  # 估算原子数
        for i in range(sub_atom_count):
            if atom_idx < atom_count:
                atom_substructure_mapping[atom_idx] = substructure_idx
                atom_idx += 1  # 处理下一个原子

        # 移动 SMILES 的指针
        smile_cursor = token_pos + len(token)
        substructure_idx += 1  # 记录下一个子结构索引

    max_length = 50
    seq_length = len(token_ids)

    if seq_length < max_length:
        padded_tokens = np.pad(token_ids, (0, max_length - seq_length), 'constant', constant_values=0)
        attention_mask = [1] * seq_length + [0] * (max_length - seq_length)
    else:
        padded_tokens = token_ids[:max_length]
        attention_mask = [1] * max_length

    return tokenized_smiles,padded_tokens, np.asarray(attention_mask), seq_length, atom_substructure_mapping


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

    # 使用正则表达式匹配化学元素符号
    matches = re.findall(r'[A-Z][a-z]?', token)
    return [m for m in matches if m in periodic_table]

def espf_tokenize(smile, mol, vocab_path="ESPF/drug_codes_chembl_freq_1500.txt", subword_map_path="ESPF/subword_units_map_chembl_freq_1500.csv"):
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    # 读取子结构索引映射表
    sub_csv = pd.read_csv(subword_map_path)
    idx2word_d = sub_csv['index'].values  # 所有子结构（token）
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))  # 词到索引的映射

    tokenized_smiles = dbpe.process_line(smile).split()  # # BPE 处理后拆分成 token 列表
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

    # 获取分子中的所有原子符号
    mol_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_count = len(mol_atoms)
    atom_substructure_mapping = [-1] * atom_count  # 初始化映射

    # 通过字符匹配建立原子到子结构的映射
    atom_idx = 0  # 指向 mol 的当前原子
    temp_token_pos = 0

    # 遍历 SMILES 中的每个 token
    for atom_idx in range(0, atom_count):
        flag = False  # 标志位，表示当前原子是否成功匹配
        current_token_pos = temp_token_pos
        for token in tokenized_smiles[current_token_pos:]:  # 从当前 token 开始查找

            if current_match_atom_cnt[temp_token_pos] >= match_atoms_cnt[temp_token_pos]:
                temp_token_pos += 1
            else:
                # 解析 token 中的原子符号
                token_atoms = match_atoms[temp_token_pos]
                if mol_atoms[atom_idx] in token_atoms:
                    atom_substructure_mapping[atom_idx] = temp_token_pos
                    current_match_atom_cnt[temp_token_pos] += 1
                    match_atoms[temp_token_pos].remove(mol_atoms[atom_idx])
                    flag = True  # 标记为已匹配
                    break  # 当前原子找到匹配，跳出内层循环
                temp_token_pos += 1  # 记录新的子结构编号
        if not flag:
            # 如果当前原子未匹配，则下一个原子回退到当前 token 开始重新查找
            temp_token_pos = current_token_pos  # 回退到 SMILES 的开始位置



    max_length = 50
    seq_length = len(token_ids)

    if seq_length < max_length:
        padded_tokens = np.pad(token_ids, (0, max_length - seq_length), 'constant', constant_values=0)
        attention_mask = [1] * seq_length + [0] * (max_length - seq_length)
    else:
        padded_tokens = token_ids[:max_length]
        attention_mask = [1] * max_length

    return tokenized_smiles, padded_tokens, np.asarray(attention_mask), seq_length, atom_substructure_mapping


class DCGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root="dataset", transform=None, pre_transform=None):
        assert name.startswith("dc-")
        name = name[len("dc-") :]
        self.name = name
        self.dirname = f"{name}"
        self.original_root = root
        self.root = osp.join(root, self.dirname)
        print(self.root)
        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices, self._num_tasks = torch.load(self.processed_paths[0])

    def get_idx_split(self):
        path = os.path.join(self.root, "split", "split_dict.pt")
        return torch.load(path)

    @property
    def task_type(self):
        return get_task_type(self.name)

    @property
    def eval_metric(self):
        return "rocauc" if "classification" in self.task_type else "mae"

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def raw_file_names(self):
        return ["data.npz"]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        pass

    def process(self):
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        _, dfs, _ = load_molnet_dataset(self.name)

        num_tasks = len(dfs[0]["labels"].values[0])

        for insert_idx, df in zip([train_idx, valid_idx, test_idx], dfs):
            smiles_list = df["text"].values.tolist()
            labels_list = df["labels"].values.tolist()
            assert len(smiles_list) == len(labels_list)

            for smiles, labels in zip(smiles_list, labels_list):
                data = DGData()
                mol = Chem.MolFromSmiles(smiles)
                graph = smiles2graphwithface(mol)

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_nodes__ = int(graph["num_nodes"])

                if "classification" in self.task_type:
                    data.y = torch.as_tensor(labels).view(1, -1).to(torch.long)
                else:
                    data.y = torch.as_tensor(labels).view(1, -1).to(torch.float32)
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
                data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
                data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
                
                data.smiles_ori = smiles
                espf_smiles, data.tokens, data.attention_mask, substructure_num, data.atom2substructure = espf_tokenize(smiles,mol)



                data_list.append(data)
                insert_idx.append(len(data_list))
                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices, num_tasks), self.processed_paths[0])

        os.makedirs(osp.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.as_tensor(train_idx, dtype=torch.long),
                "valid": torch.as_tensor(valid_idx, dtype=torch.long),
                "test": torch.as_tensor(test_idx, dtype=torch.long),
            },
            osp.join(self.root, "split", "split_dict.pt"),
        )


if __name__ == "__main__":
    dataset = DCGraphPropPredDataset("dc-bbbp")
    split_index = dataset.get_idx_split()
    print(split_index)

