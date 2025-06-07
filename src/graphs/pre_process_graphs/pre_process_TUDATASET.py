import os.path as osp
from typing import Dict, List, Optional, Tuple
import os
import torch
from torch import Tensor
import numpy as np
import argparse
import pickle
from torch_geometric.data import Data
from torch_geometric.io import fs, read_txt_array
from torch_geometric.utils import coalesce, cumsum, one_hot, remove_self_loops
import networkx as nx
names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def read_tu_data(
    folder: str,
    prefix: str,
) -> Tuple[Data, Dict[str, Tensor], Dict[str, int], torch.Tensor]:
    files = fs.glob(osp.join(folder, f'{prefix}_*.txt'))
    names = [osp.basename(f)[len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attribute = torch.empty((batch.size(0), 0))
    if 'node_attributes' in names:
        node_attribute = read_file(folder, prefix, 'node_attributes')
        if node_attribute.dim() == 1:
            node_attribute = node_attribute.unsqueeze(-1)

     
    edge_attribute = torch.empty((edge_index.size(1), 0))
    if 'edge_attributes' in names:
        edge_attribute = read_file(folder, prefix, 'edge_attributes')
        if edge_attribute.dim() == 1:
            edge_attribute = edge_attribute.unsqueeze(-1)

    edge_label = torch.empty((edge_index.size(1), 0))
    if 'edge_labels' in names:
        edge_label = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_label.dim() == 1:
            edge_label = edge_label.unsqueeze(-1)
        edge_label = edge_label - edge_label.min(dim=0)[0]
        edge_labels = list(edge_label.unbind(dim=-1))
        edge_labels = [one_hot(e) for e in edge_labels]
        if len(edge_labels) == 1:
            edge_label = edge_labels[0]
        else:
            edge_label = torch.cat(edge_labels, dim=-1)

    x = cat([node_attribute])
    edge_attr = cat([edge_attribute, edge_label])

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = int(edge_index.max()) + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    sizes = {
        'num_node_attributes': node_attribute.size(-1), 
        'num_edge_attributes': edge_attribute.size(-1),
        'num_edge_labels': edge_label.size(-1),
    }

    ################################################################
    # Build adjacency matrix for the FIRST graph in the dataset:
    #
    # The 'slices' dictionary tells us how many edges/nodes are in
    # each graph. For the first graph, the edges go from index
    # 0 up to slices['edge_index'][1]. Similarly, the nodes go
    # from 0 up to slices['x'][1] (if x exists).
    ################################################################

    adjs = []
    for i in range(len(slices['edge_index']) - 1):
        # Get the end index for the current graph's edges
        graph_edge_end = slices['edge_index'][i + 1].item()
        graph_edge_index = data.edge_index[:, slices['edge_index'][i]:graph_edge_end]

        if data.x is not None:
            graph_num_nodes = slices['x'][i + 1].item()
        else:
            # If data.x is None, 'data._num_nodes' was set (imitates "collate").
            # This is a list of node counts for each graph.
            graph_num_nodes = data._num_nodes[i]
        # Initialize an adjacency matrix of the right size:
        graph_adj = torch.zeros((graph_num_nodes, graph_num_nodes), dtype=torch.long)
        # Because TUDataset graphs are undirected, fill both (src, dst) and (dst, src):
        for (src, dst) in graph_edge_index.t():
            graph_adj[src, dst] = 1
            graph_adj[dst, src] = 1
        adjs.append(graph_adj)

    return adjs

def read_file(
    folder: str,
    prefix: str,
    name: str,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq: List[Optional[Tensor]]) -> Optional[Tensor]:
    values = [v for v in seq if v is not None]
    values = [v for v in values if v.numel() > 0]
    values = [v.unsqueeze(-1) if v.dim() == 1 else v for v in values]
    return torch.cat(values, dim=-1) if len(values) > 0 else None


def split(data: Data, batch: Tensor) -> Tuple[Data, Dict[str, Tensor]]:
    node_slice = cumsum(torch.bincount(batch))

    assert data.edge_index is not None
    row, _ = data.edge_index
    edge_slice = cumsum(torch.bincount(batch[row]))

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        assert isinstance(data.y, Tensor)
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, int(batch[-1]) + 2, dtype=torch.long)

    return data, slices


def save_graphs_as_gpickle(
    adjs: List[torch.Tensor],
    out_dir: str, 
    ):
    """
    Given a list of adjacency matrices (as PyTorch tensors),
    save each as a NetworkX .gpickle file in out_dir.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, adj in enumerate(adjs):
        # Convert to NetworkX graph:
        # (We convert the PyTorch tensor to a NumPy array first.)
        G = nx.from_numpy_array(adj.numpy())
 
        filename = osp.join(out_dir, f"graph_{i}.gpickle")

        with open(filename, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL) 
        print(f"Saved graph {i} -> {filename}")


def main():
    # Set up the argument parser.
    parser = argparse.ArgumentParser(
        description="Process TU dataset and compute node statistics."
    )
    parser.add_argument(
        "--TU_data_folder",
        type=str,
        default="/storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/data/real_world/TUDataset/bioinformatics",
        help="Path to the folder containing the TU data."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dblp_ct1",
        help="Name of the dataset to process (e.g., COLLAB)."
    )
    args = parser.parse_args()

    # Compute directories based on the arguments.
    TU_data_folder = args.TU_data_folder
    dataset_name = args.dataset_name
    TU_data_folder_for_this_instance = osp.join(TU_data_folder, dataset_name)
    
    # Use the provided output directory or construct one.
    out_dir = osp.join(TU_data_folder_for_this_instance, 'gpickle_files')
    
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # 1. Read TU data -> list of adjacency matrices
    adjs = read_tu_data(TU_data_folder_for_this_instance, dataset_name)

    # 2. Save each adjacency matrix as a .gpickle file
    save_graphs_as_gpickle(adjs, out_dir)

    # 3. Compute node statistics for each graph
    num_nodes_list = np.array([adj.size(0) for adj in adjs])
    stats = (
        f"Graph node statistics for dataset {dataset_name}:\n"
        f"Mean: {np.mean(num_nodes_list):.2f}\n"
        f"Std: {np.std(num_nodes_list):.2f}\n"
        f"Min: {np.min(num_nodes_list)}\n"
        f"Max: {np.max(num_nodes_list)}\n"
        f"Median: {np.median(num_nodes_list)}\n"
    )
    print(stats)

    # 4. Save the statistics to a text file under TU_data_folder_for_this_instance
    stats_file_path = osp.join(TU_data_folder_for_this_instance, "node_statistics.txt")
    with open(stats_file_path, "w") as f:
        f.write(stats)
    print(f"Node statistics saved to: {stats_file_path}")


if __name__ == "__main__":
    main()
    '''
    python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name dblp_ct1
    REDDIT-BINARY
    REDDIT-MULTI-5K
    python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name IMDB-BINARY; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name IMDB-MULTI; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name tumblr_ct1; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name tumblr_ct2; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name highschool_ct1; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name highschool_ct2; 
    
    python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name SYNTHETIC; 
    python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name SYNTHETICnew
    ; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name REDDIT-BINARY
    ; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name REDDIT-MULTI-5K
    ; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name COLORS-3; 
    

    python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name Synthie
    ; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name infectious_ct1; python /storage/home/hcoda1/3/adeza3/p-phentenryck3-1/adeza3/foundationalCO/src/processors/pre_process_TUDATASET.py --dataset_name infectious_ct2
    '''

     