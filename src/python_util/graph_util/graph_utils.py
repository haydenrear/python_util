from random import random
import random as rn
from typing import Tuple

import networkx as nx
import numpy as np
import torch


def create_random_edgelist(n_nodes: int, probability: float) -> list[list[int]]:
    out_edgelist: list[list[int]] = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if rn.random() > probability:
                out_edgelist.append([i, j])

    return out_edgelist


def transform_from_adjacency_matrix(adj_matrix: torch.Tensor) -> torch.Tensor:
    edge_list = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:
                edge_list.append([i, j])
    return torch.tensor(edge_list).T


def extract_edgelist_from_graph(starting_graph: nx.Graph, nodes: list):
    edges = []
    graph_nodes = [n for n in starting_graph.nodes]
    for from_t, to_ in nx.edges(starting_graph, graph_nodes):
        if from_t in nodes and to_ in nodes:
            edges.append([nodes.index(from_t),
                          nodes.index(to_)])

    return torch.tensor(edges).T


def extract_edgelist_subgraph(starting_edgelist: torch.Tensor, nodes: set[int]):
    out_edges = []
    assert starting_edgelist.shape[0] == 2
    assert len(starting_edgelist.shape) == 2

    for edges in starting_edgelist.T:
        from_ = int(edges[0])
        to_ = int(edges[1])
        if from_ in nodes and to_ in nodes:
            out_edges.append([from_, to_])

    return out_edges


def transform_to_adjacency_matrix(edgelist: torch.Tensor, starting: torch.Tensor) -> torch.Tensor:
    for i in range(edgelist.shape[0]):
        for j in range(edgelist.shape[1]):
            starting[edgelist[i], edgelist[j]] = 1
            starting[edgelist[j], edgelist[i]] = 1
    return starting


def get_node_cxns_out(graph: nx.DiGraph, node_match, node_only: bool = True):
    return [i[1] if node_only else i for i in graph.out_edges(node_match)]


def get_node_cxns_in(graph: nx.DiGraph, node_match, node_only: bool = True):
    return [i[0] if node_only else i for i in graph.in_edges(node_match)]


def get_node_cxns(graph: nx.DiGraph, node_match, node_only: bool = True,
                  filter_duplicates: bool = False):

    seen = set([])
    for n in get_node_cxns_in(graph, node_match, node_only):
        if (filter_duplicates and n not in seen) or not filter_duplicates:
            seen.add(n)
            yield n
    for n in get_node_cxns_out(graph, node_match, node_only):
        if (filter_duplicates and n not in seen) or not filter_duplicates:
            seen.add(n)
            yield n
    for from_edge, to_edge in graph.edges(node_match):
        if node_only:
            n = from_edge if from_edge != node_match else to_edge
            if (filter_duplicates and n not in seen) or not filter_duplicates:
                seen.add(n)
                yield n
        else:
            n = from_edge, to_edge
            if (filter_duplicates and n not in seen) or not filter_duplicates:
                seen.add(n)
                yield n


def get_sorted_by_key(tensor: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    sorted_keys = sorted(tensor.keys())
    return [tensor[key] for key in sorted_keys]


def modularity_of_partitions(G, partition, M):
    """
    G: The subgraph
    partition: dictionary (node: community)
    M: Matrix representing the "average" from the main graph.
    """
    A = nx.to_numpy_array(G)
    nodes = G.nodes()
    degrees = np.array([G.degree(n) for n in nodes])
    m = G.number_of_edges()

    return do_calculate_modularity(A, M, m, nodes, partition)


def do_calculate_modularity(A, M, m, nodes, partition):
    Q = 0
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if partition[node_i] == partition[node_j]:
                Q += A[i, j] - M[i, j]
    return Q / (2 * m)


def modularity_from_subgraph(main_graph, sub_graph, M):
    """
    main_graph: The main graph
    sub_graph: The subgraph whose modularity you want to calculate
    M: Matrix representing the "average" from the main graph.
    """
    A = nx.to_numpy_array(main_graph)
    nodes = main_graph.nodes()
    sub_nodes = sub_graph.nodes()
    m = main_graph.number_of_edges()

    # Create partition based on sub_graph
    partition = {}
    for node in nodes:
        if node in sub_nodes:
            partition[node] = 1  # Sub-graph community
        else:
            partition[node] = 0  # Rest of the main graph community

    return do_calculate_modularity(A, M, m, nodes, partition)


def create_edge_list_from_edges(edges, nodes=None):
    edge_list = []
    for from_, to_ in edges:
        if nodes is None:
            edge_list.append([from_, to_])
        else:
            assert isinstance(nodes, list)
            if from_ in nodes and to_ in nodes:
                edge_list.append([nodes.index(from_), nodes.index(to_)])

    return torch.tensor(edge_list).T


def concatenate_graphs(graphs: list[torch.Tensor],
                       edgelists: list[torch.Tensor],
                       add_edge_probability: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param graphs:
    :param edgelists:
    :param add_edge_probability:
    :return:  Tuple where first element is graph and second element is edgelist.
    """
    concatenated_edgelist = torch.empty(0, 2)  # Empty tensor to store concatenated edgelist
    # Concatenate edgelists while adjusting node indices
    node_offset = 0
    for edgelist in edgelists:
        # Increment node indices of the current graph
        edgelist_incremented = edgelist + node_offset

        # Concatenate the adjusted edgelists along the rows
        concatenated_edgelist = torch.cat((concatenated_edgelist, edgelist_incremented), dim=0)
        if node_offset > 0 and add_edge_probability != 0.0:
            for node1 in range(node_offset, edgelist_incremented.max().item() + 1):
                for node2 in range(concatenated_edgelist.max().item() + 1 - node_offset):
                    if rn.random() < add_edge_probability:
                        new_edge = torch.tensor([[node1, node2 + node_offset]])
                        concatenated_edgelist = torch.cat((concatenated_edgelist, new_edge), dim=0)


    # Update node offset for the next graph
        node_offset += edgelist.max().item() + 1

    return torch.concat(graphs), concatenated_edgelist