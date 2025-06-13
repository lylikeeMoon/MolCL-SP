import torch
import numpy as np
import random
from torch_geometric.utils import to_dense_batch
from rdkit import Chem


MASK_TOKEN = 2586
MASK_ATOM_TYPE = 119
MASK_ATOM_DEGREE = 0
MASK_TOKEN_RATIO = 0.2
MASK_GRAPH_RATIO = 0.25
NOISE_STD = 0.1
NOISE_RATIO = 0.35


def mask_tokens_batch(tokens, attention_mask, mask_ratio=MASK_TOKEN_RATIO, mask_token=MASK_TOKEN):

    batch_size, seq_len = tokens.shape


    masked_tokens = tokens.clone()
    all_mask_indices = torch.zeros_like(tokens, dtype=torch.bool)  # 用布尔值表示 MASK 位置

    for i in range(batch_size):
        valid_indices = torch.nonzero(attention_mask[i], as_tuple=True)[0]

        num_to_mask = int(len(valid_indices) * mask_ratio)
        if num_to_mask > 0:
            mask_indices = valid_indices[torch.randperm(len(valid_indices))[:num_to_mask]]
            masked_tokens[i, mask_indices] = mask_token
            all_mask_indices[i, mask_indices] = 1

    return masked_tokens, all_mask_indices


def mask_graph_batch(graph_batch, atom_to_sub_batch, token_mask_batch, mask_ratio=MASK_GRAPH_RATIO):
    masked_graph_batch = []
    batch_size = len(graph_batch)

    all_mask_indices = torch.zeros_like(token_mask_batch, dtype=torch.bool)
    all_masked_atom_indices = []

    num_atoms_per_graph = [graph_batch[i].x.size(0) for i in range(batch_size)]
    cumulative_num_atoms = torch.cumsum(torch.tensor(num_atoms_per_graph), dim=0)


    for i in range(batch_size):
        graph = graph_batch[i]
        atom_to_sub = torch.tensor(atom_to_sub_batch[i], dtype=torch.long)  # 确保是 Tensor
        token_mask = token_mask_batch[i]

        substructure_indices = torch.unique(atom_to_sub[atom_to_sub != -1])
        available_subs = [sub.item() for sub in substructure_indices if not token_mask[sub]]
        num_mask = int(len(available_subs) * mask_ratio)
        if num_mask == 0:
            masked_graph_batch.append(graph)  # 没有可 MASK 的子结构时，返回原图
            continue

        masked_subs = random.sample(available_subs, min(num_mask, len(available_subs)))

        masked_graph = graph.clone()

        masked_atom_indices = []

        for atom_idx, sub_idx in enumerate(atom_to_sub.tolist()):
            if sub_idx in masked_subs:
                masked_graph.x[atom_idx][0] = torch.tensor(MASK_ATOM_TYPE, dtype=torch.float)
                masked_graph.x[atom_idx][1] = torch.tensor(MASK_ATOM_DEGREE, dtype=torch.float)
                all_mask_indices[i, sub_idx] = 1
                global_idx = atom_idx if i == 0 else cumulative_num_atoms[i - 1] + atom_idx
                masked_atom_indices.append(global_idx)

        masked_graph_batch.append(masked_graph)
        all_masked_atom_indices.extend(masked_atom_indices)

    all_masked_atom_indices = torch.tensor(all_masked_atom_indices, dtype=torch.long)

    return masked_graph_batch, all_mask_indices, all_masked_atom_indices



def add_noise_to_3d_structure_batch(atom_to_sub_batch, atom_positions_batch, batch, masked_subs_1d_batch,
                                    masked_subs_2d_batch, noise_ratio=NOISE_RATIO, noise_std=NOISE_STD):

    batch_size = masked_subs_1d_batch.shape[0]
    noisy_positions_batch = atom_positions_batch.clone()

    all_mask_indices = torch.zeros_like(masked_subs_1d_batch, dtype=torch.bool)
    all_masked_atom_indices = []


    for b in range(batch_size):
        atom_indices = torch.nonzero(batch == b, as_tuple=True)[0].long()

        if len(atom_indices) == 0:
            continue

        atom_to_sub_mapping = torch.tensor(atom_to_sub_batch[b], dtype=torch.long).to(batch)  # 确保是 Tensor

        unique_subs = torch.unique(atom_to_sub_mapping[atom_to_sub_mapping!=-1])

        masked_subs_1d = masked_subs_1d_batch[b].bool()
        masked_subs_2d = masked_subs_2d_batch[b].bool()
        masked_subs = masked_subs_1d | masked_subs_2d

        available_subs = unique_subs[~masked_subs[unique_subs]]
        num_noise_subs = int(len(available_subs) * noise_ratio)

        if num_noise_subs == 0:
            continue

        selected_subs = available_subs[torch.randperm(len(available_subs))[:num_noise_subs]]

        all_mask_indices[b, selected_subs] = 1
        mask = torch.isin(atom_to_sub_mapping, selected_subs)

        selected_atoms = atom_indices[torch.isin(atom_to_sub_mapping, selected_subs)]
        noise = torch.randn_like(atom_positions_batch[selected_atoms]) * noise_std
        noisy_positions_batch[selected_atoms] += noise
        all_masked_atom_indices.extend(selected_atoms)
    all_masked_atom_indices = torch.tensor(all_masked_atom_indices, dtype=torch.long)

    return noisy_positions_batch, all_mask_indices,all_masked_atom_indices




def to_dense_with_fixed_padding(node_embeddings, batch, padding_length):
    dense_embeddings, mask = to_dense_batch(node_embeddings, batch)  # (batch_size, max_nodes, emd_dim)

    batch_size, max_nodes, emd_dim = dense_embeddings.shape

    if max_nodes < padding_length:
        pad_size = padding_length - max_nodes
        pad_tensor = torch.zeros(batch_size, pad_size, emd_dim, device=dense_embeddings.device)
        mask_pad = torch.zeros(batch_size, pad_size, dtype=torch.bool, device=dense_embeddings.device)
        dense_embeddings = torch.cat([dense_embeddings, pad_tensor], dim=1)
        mask = torch.cat([mask, mask_pad], dim=1)

    elif max_nodes > padding_length:
        dense_embeddings = dense_embeddings[:, :padding_length, :]
        mask = mask[:, :padding_length]

    return dense_embeddings, mask  # (batch_size, padding_length, emd_dim)






if __name__ == "__main__":
    # 示例：批处理数据
    batch_size = 3
    atom_to_sub_mapping_batch = [
        [0,1,2,0,0,0],
        [0,1,2,0,0,0],
        [0,1,2,0,0,0]
    ]
    atom_positions_batch = [
        [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]],
        [[2.0, 3.0, 4.0], [2.5, 3.5, 4.5], [3.0, 4.0, 5.0]],
        [[3.0, 4.0, 5.0], [3.5, 4.5, 5.5], [4.0, 5.0, 6.0]]
    ]

    masked_subs_1d_batch = torch.tensor([
        [True, False, False, False, False, False],
        [False, True, False, False, False, False],
        [True, False, False, False, False, False]
    ], dtype=torch.bool)
    masked_subs_2d_batch = torch.tensor([
        [False, True, False, False, False, False],
        [True, False, False, False, False, False],
        [False, False, True, False, False, False]
    ], dtype=torch.bool)

    noisy_positions_batch = add_noise_to_3d_structure_batch(
        atom_to_sub_mapping_batch,
        atom_positions_batch,
        masked_subs_1d_batch,
        masked_subs_2d_batch,
        noise_ratio=NOISE_RATIO,
        noise_std=NOISE_STD
    )

    print( noisy_positions_batch)
