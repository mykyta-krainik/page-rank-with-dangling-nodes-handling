import numpy as np
from collections import defaultdict
import pandas as pd


def parse_graph(file_path):
    graph = defaultdict(list)

    with open(file_path, 'r') as file:
        for line in file:
            nodes = line.strip().split()

            if nodes:
                graph[nodes[0]] = nodes[1:]

    return graph


def remove_dangling_nodes(graph):
    while True:
        dangling_nodes = [node for node in graph if len(graph[node]) == 0]

        if not dangling_nodes:
            break

        for node in dangling_nodes:
            del graph[node]

        for node in graph:
            graph[node] = [neighbor for neighbor in graph[node] if neighbor in graph]

    return graph


def pagerank(graph, tol=1e-6, max_iter=100):
    nodes = list(graph.keys())
    num_nodes = len(nodes)
    node_indices = {node: i for i, node in enumerate(nodes)}

    rank = np.ones(num_nodes) / num_nodes

    transition_matrix = np.zeros((num_nodes, num_nodes))

    for node, neighbors in graph.items():
        if neighbors:
            for neighbor in neighbors:
                transition_matrix[node_indices[neighbor], node_indices[node]] = 1 / len(neighbors)

    for _ in range(max_iter):
        new_rank = transition_matrix @ rank

        if np.linalg.norm(new_rank - rank, ord=1) < tol:
            break

        rank = new_rank

    return {node: rank[node_indices[node]] for node in nodes}, transition_matrix


def calculate_dangling_node_rank(graph, removed_nodes, ranks):
    processed_nodes = set(ranks.keys())
    remaining_nodes = set(removed_nodes)

    while remaining_nodes:
        for node in list(remaining_nodes):
            predecessors = [src for src, targets in graph.items() if node in targets]

            if all(pred in processed_nodes for pred in predecessors):
                rank = 0
                for pred in predecessors:
                    pred_out_degree = len(graph[pred])
                    rank += ranks[pred] / pred_out_degree

                ranks[node] = rank
                processed_nodes.add(node)
                remaining_nodes.remove(node)

    return ranks


def main(file_path):
    graph = parse_graph(file_path)

    print("Original graph:")
    print(pd.DataFrame(graph.items(), columns=["Node", "Neighbors"]))

    original_graph = dict(graph)

    graph = remove_dangling_nodes(graph)
    removed_nodes = set(original_graph.keys()) - set(graph.keys())

    ranks, transition_matrix = pagerank(graph)

    print("Transition matrix:")
    print(pd.DataFrame(transition_matrix, index=graph.keys(), columns=graph.keys()))

    ranks = calculate_dangling_node_rank(original_graph, removed_nodes, ranks)

    return ranks


if __name__ == "__main__":
    file_path = "./graph.txt"

    ranks = main(file_path)

    print("PageRank scores:")

    for node, rank in sorted(ranks.items(), key=lambda x: x[0]):
        print(f"{node}: {rank:.6f}")
