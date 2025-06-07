import pickle
import networkx as nx
import torch
from typing import Tuple

def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a tensor to the range [0, 1] using min-max normalization.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor in [0, 1].
    """
    min_val = x.min()
    max_val = x.max()
    if max_val - min_val > 0:
        return (x - min_val) / (max_val - min_val)
    else:
        # If all values are the same, return zeros.
        return torch.zeros_like(x)

def compute_triangle_count(graph: nx.Graph, normalize: bool = True) -> Tuple[torch.Tensor, str]:
    """
    Compute the triangle count for each node and normalize the counts to [0, 1].

    Parameters:
        graph (nx.Graph): An undirected graph.
        normalize (bool): Whether to apply min-max normalization. Default is True.

    Returns:
        Tuple[torch.Tensor, str]:
            - A column tensor (shape: [n_nodes, 1]) of triangle counts (normalized if specified).
            - A string indicating the base level ('node').
    """
    
    triangles = nx.triangles(graph)
    feat = torch.tensor(list(triangles.values()), dtype=torch.float32).unsqueeze(1)
    if normalize:
        feat = min_max_normalize(feat)
    return feat

def compute_clustering_coefficient(graph: nx.Graph, normalize: bool = True) -> Tuple[torch.Tensor, str]:
    """
    Compute the clustering coefficient for each node and normalize the coefficients to [0, 1].

    Parameters:
        graph (nx.Graph): An undirected graph.
        normalize (bool): Whether to apply min-max normalization. Default is True.
            (Note: Clustering coefficients are typically in [0, 1], but normalization ensures consistency.)

    Returns:
        Tuple[torch.Tensor, str]:
            - A column tensor (shape: [n_nodes, 1]) of clustering coefficients.
            - A string indicating the base level ('node').
    """
    
    clustering = nx.clustering(graph)
    feat = torch.tensor(list(clustering.values()), dtype=torch.float32).unsqueeze(1)
    if normalize:
        feat = min_max_normalize(feat)
    return feat

def compute_eccentricity(graph: nx.Graph, normalize: bool = True) -> Tuple[torch.Tensor, str]:
    """
    Compute the eccentricity for each node and normalize the values to [0, 1].
    For disconnected graphs, eccentricity is computed per connected component.

    Parameters:
        graph (nx.Graph): An undirected graph.
        normalize (bool): Whether to apply min-max normalization. Default is True.

    Returns:
        Tuple[torch.Tensor, str]:
            - A column tensor (shape: [n_nodes, 1]) of eccentricity values (normalized if specified).
            - A string indicating the base level ('node').
    """
    
    if nx.is_connected(graph):
        ecc_dict = nx.eccentricity(graph)
    else:
        # For each connected component, compute eccentricity separately.
        ecc_dict = {}
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            ecc_sub = nx.eccentricity(subgraph)
            ecc_dict.update(ecc_sub)
    # Ensure the order follows graph.nodes()
    ecc_list = [ecc_dict[node] for node in graph.nodes()]
    feat = torch.tensor(ecc_list, dtype=torch.float32).unsqueeze(1)
    if normalize:
        feat = min_max_normalize(feat)
    return feat

def compute_degrees(graph: nx.Graph, log_transform: bool = True, normalize: bool = True) -> Tuple[torch.Tensor, str]:
    """
    Compute the degree for each node, optionally apply a logarithmic transformation, 
    and normalize the resulting degrees to the range [0, 1].

    Parameters:
        graph (nx.Graph): An undirected graph.
        log_transform (bool): Whether to apply log(degree + 1) to dampen high variance. Default is True.
        normalize (bool): Whether to apply min-max normalization. Default is True.

    Returns:
        Tuple[torch.Tensor, str]:
            - A column tensor (shape: [n_nodes, 1]) of node degrees (or log-transformed degrees), normalized if specified.
            - A string indicating the base level ('node').
    """
    
    degrees = [d for _, d in graph.degree()]
    feat = torch.tensor(degrees, dtype=torch.float32).unsqueeze(1)
    if log_transform:
        feat = torch.log(feat + 1)  # Avoid log(0)
    if normalize:
        feat = min_max_normalize(feat)
    return feat

def extract_node_features(graph: nx.Graph, log_degree: bool = True, normalize: bool = True) -> torch.Tensor:
    """
    Extract and concatenate normalized node features into a single feature matrix.
    The features are:
      - Degree (optionally log-transformed)
      - Clustering Coefficient
      - Triangle Count
      - Eccentricity

    Parameters:
        graph (nx.Graph): An undirected graph.
        log_degree (bool): Whether to apply a logarithmic transformation to the degree feature.
        normalize (bool): Whether to normalize each feature to the range [0, 1].

    Returns:
        torch.Tensor: A node feature matrix of shape [n_nodes, n_features] with values in [0, 1].
    """
    degree_feat = compute_degrees(graph, log_transform=log_degree, normalize=normalize)
    clustering_feat = compute_clustering_coefficient(graph, normalize=normalize)
    triangle_feat = compute_triangle_count(graph, normalize=normalize)
    eccentricity_feat = compute_eccentricity(graph, normalize=normalize)
    
    # Concatenate features along the feature dimension (columns)
    features = torch.cat([degree_feat, clustering_feat, triangle_feat, eccentricity_feat], dim=1)
    return features

# Example usage:
if __name__ == "__main__":
    p = '/storage/home/hcoda1/3/adeza3/scratch/foundationalCO/data/mis/synthetic/satlib/CBS_k3_n100_m403_b10_0.gpickle'
    with open(p, 'rb') as f:
        G = pickle.load(f)
    
    node_features = extract_node_features(G)
    print("Normalized node feature matrix shape:", node_features.shape)
