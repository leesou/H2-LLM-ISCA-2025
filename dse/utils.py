import random
from typing import List, Tuple, Set

import networkx as nx


def are_nodes_not_successors(group1: List[int], group2: List[int], operator_graph: nx.DiGraph) -> bool:
    for group2_node in group2:
        successors_of_group2 = nx.descendants(operator_graph, group2_node)
        if any(group1_node in successors_of_group2 for group1_node in group1):
            return False
    return True


def check_memory_access_group_legality(node_groups: List[List[int]], operator_graph: nx.DiGraph):
    assert nx.is_directed_acyclic_graph(operator_graph), "Operator graph must be a directed acyclic graph."
    graph_nodes_set = set(operator_graph.nodes)
    union_of_groups = set()
    for group in node_groups:
        union_of_groups.update(group)
    if union_of_groups != graph_nodes_set:
        return False

    for i in range(len(node_groups)):
        for j in range(i+1, len(node_groups)):
            if not set(node_groups[i]).isdisjoint(set(node_groups[j])):
                return False

            if not are_nodes_not_successors(node_groups[i], node_groups[j], operator_graph):
                return False

    return True


def random_generate_memory_access_groups(operator_graph: nx.DiGraph) -> Tuple[List[List[int]], List[nx.DiGraph]]:
    assert nx.is_directed_acyclic_graph(operator_graph), "Operator graph must be a directed acyclic graph."

    node_groups = []
    subgraphs = []
    graph_to_remove: nx.DiGraph = operator_graph.copy()
    while graph_to_remove.nodes:
        group_size = random.randint(1, len(graph_to_remove.nodes))

        if group_size == len(graph_to_remove.nodes):
            tmp_group = [node for node in graph_to_remove.nodes]
            tmp_subgraph = operator_graph.subgraph(tmp_group).copy()
            node_groups.append(tmp_group)
            subgraphs.append(tmp_subgraph)
            break

        tmp_group = []
        for _ in range(group_size):
            zero_indegree_nodes = [node for node in graph_to_remove.nodes if graph_to_remove.in_degree(node) == 0]
            random.shuffle(zero_indegree_nodes)
            tmp_group.append(zero_indegree_nodes[0])
            graph_to_remove.remove_node(zero_indegree_nodes[0])
        tmp_subgraph = operator_graph.subgraph(tmp_group).copy()
        node_groups.append(tmp_group)
        subgraphs.append(tmp_subgraph)

    assert check_memory_access_group_legality(node_groups, operator_graph), "Illegal memory access group splitting occurs"
    return node_groups, subgraphs


def check_memory_partition_group_legality(node_groups: List[List[int]], operator_graph: nx.DiGraph):
    assert nx.is_directed_acyclic_graph(operator_graph), "Operator graph must be a directed acyclic graph."
    graph_nodes_set = set(operator_graph.nodes)
    union_of_groups = set()
    for group in node_groups:
        union_of_groups.update(group)
    if union_of_groups != graph_nodes_set:
        return False

    for i in range(len(node_groups)):
        if not nx.is_weakly_connected(operator_graph.subgraph(node_groups[i])):
            return False
        
        for j in range(i+1, len(node_groups)):
            if not set(node_groups[i]).isdisjoint(set(node_groups[j])):
                return False

            for node_i in node_groups[i]:
                for node_j in node_groups[j]:
                    if operator_graph.has_edge(node_i, node_j) or operator_graph.has_edge(node_j, node_i):
                        return False
    return True


def generate_memory_partition_groups(operator_graph: nx.DiGraph) -> Tuple[List[List[int]], List[nx.DiGraph]]:
    assert nx.is_directed_acyclic_graph(operator_graph), "Operator graph must be a directed acyclic graph."
    weakly_connected_components = list(nx.weakly_connected_components(operator_graph))
    node_groups = [list(component) for component in weakly_connected_components]
    subgraphs = [operator_graph.subgraph(component).copy() for component in weakly_connected_components]
    assert check_memory_partition_group_legality(node_groups, operator_graph), "Illegal memory parition group splitting occurs"
    return node_groups, subgraphs


def check_stratification_legality(node_groups: List[List[int]], operator_graph: nx.DiGraph):
    assert nx.is_directed_acyclic_graph(operator_graph), "Operator graph must be a directed acyclic graph."
    graph_nodes_set = set(operator_graph.nodes)
    union_of_groups = set()
    for group in node_groups:
        union_of_groups.update(group)
    if union_of_groups != graph_nodes_set:
        return False

    for i in range(len(node_groups)):
        for node_idx1 in range(len(node_groups[i])):
            for node_idx2 in range(len(node_groups[i])):
                if operator_graph.has_edge(node_groups[i][node_idx1], node_groups[i][node_idx2]) or \
                    operator_graph.has_edge(node_groups[i][node_idx2], node_groups[i][node_idx1]):
                    return False
        
        for j in range(i+1, len(node_groups)):
            if not set(node_groups[i]).isdisjoint(set(node_groups[j])):
                return False

            if not are_nodes_not_successors(node_groups[i], node_groups[j], operator_graph):
                return False
    return True


def random_generate_stratifications(operator_graph: nx.DiGraph, max_node_num_per_srtatification: int = -1) -> Tuple[List[List[int]], List[nx.DiGraph]]:
    assert nx.is_directed_acyclic_graph(operator_graph), "Operator graph must be a directed acyclic graph."

    in_degree = {node: 0 for node in operator_graph}
    for u, v in operator_graph.edges:
        in_degree[v] += 1
    zero_in_degree = [node for node in operator_graph if in_degree[node] == 0]

    node_groups = []
    subgraphs = []
    while zero_in_degree:
        random.shuffle(zero_in_degree)
        group_size = random.randint(1, len(zero_in_degree))
        if max_node_num_per_srtatification > 0:
            group_size = min(max_node_num_per_srtatification, group_size)
        group = zero_in_degree[:group_size]
        node_groups.append(group)
        subgraph = operator_graph.subgraph(group).copy()
        subgraphs.append(subgraph)

        zero_in_degree = zero_in_degree[group_size:]
        for node in group:
            for neighbor in operator_graph.successors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)

    assert check_stratification_legality(node_groups, operator_graph), "Illegal stratification splitting occurs"
    return node_groups, subgraphs


def are_sets_disjoint(set_list: List[Set]):
    for i in range(len(set_list)):
        for j in range(i + 1, len(set_list)):
            if not set_list[i].isdisjoint(set_list[j]):
                return False
    return True


def split_list_random_lengths(lst: List, k: int, allow_empty_sublist: bool) -> List[List[int]]:
    shuffle_lst = lst.copy()
    random.shuffle(shuffle_lst)

    if len(lst) == 0:
        assert allow_empty_sublist, "Given an empty list but allow_empty_sublist is false."
        return [[] for _ in range(k)]

    if allow_empty_sublist:
        split_points = sorted(random.choices(list(range(len(shuffle_lst))), k=k-1))
    else:
        if len(shuffle_lst) < k:
            raise ValueError("List length must be greater than or equal to k")
        split_points = sorted(random.sample(list(range(1, len(shuffle_lst))), k=k-1))
    
    sublists = [shuffle_lst[i:j] for i, j in zip([0] + split_points, split_points + [len(shuffle_lst)])]
    return sublists


def print_graph_attributes(graph: nx.DiGraph):
    for node_id in graph.nodes:
        print(f"{node_id} : {graph.nodes[node_id]}")
