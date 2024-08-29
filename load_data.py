#run by $python load_data.py
import pickle
import os
from utils.dag_graph import DAGraph
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import math
import networkx as nx
import numpy as np

# Specify the path to the pickle file
pickle_file_path = "/data/shared/huiyuan/dag50_addedge/val.pkl"

# Open the file in read-binary mode and load the graphs
with open(pickle_file_path, 'rb') as f:
    loaded_graphs = pickle.load(f)
print(len(loaded_graphs))

# def arrays_to_digraph(nodes, dep_mat):
#     # Create an empty directed graph
#     G = nx.DiGraph()

#     # Add nodes to the graph
#     for i in range(len(nodes)):
#         G.add_node(i, features=nodes[i].tolist())

#     # Add edges to the graph based on the dependency matrix
#     for i in range(len(dep_mat)):
#         for j in range(len(dep_mat[i])):
#             if np.isclose(dep_mat[i][j], 1.0):
#                 G.add_edge(i, j)

#     return G

# def transitive_closure(dep_mat):
#         '''
#         return the transitive closure that has e_ij = 1 if i is an ancestor of j
#         '''
#         n_nodes = dep_mat.shape[0]
#         dep_mat = dep_mat.astype(bool)
#         identity_like_adj_mat = np.identity(n_nodes, dtype=bool)
#         closure = identity_like_adj_mat | dep_mat
#         # Floyd-Warshall Algorithm
#         for k in range(n_nodes):
#             for i in range(n_nodes):
#                 for j in range(n_nodes):
#                     closure[j, i] = closure[j, i] or (closure[k, i] and closure[j, k])
#         closure = ~identity_like_adj_mat & closure
#         return closure

# def update_transitive_closure(trans_closure, i_node, j_node):
#         n_nodes = trans_closure.shape[0]
#         trans_closure[i_node, j_node] = True
#         trans_closure[:, j_node] |= trans_closure[:, i_node]
#         for k in range(n_nodes):
#             if trans_closure[j_node, k]:
#                 trans_closure[:, k] |= trans_closure[:, i_node]
#                 trans_closure[i_node, k] = True

# def batch(iterable, n=1):
#     l = len(iterable)
#     for ndx in range(0, l, n):
#         yield iterable[ndx:min(ndx + n, l)]   

# # Specify the path to the pickle file
# pickle_file_path = "/data/shared/huiyuan/dag34_new/train.pkl"

# # Open the file in read-binary mode and load the graphs
# with open(pickle_file_path, 'rb') as f:
#     loaded_graphs = pickle.load(f)
# print(len(loaded_graphs))

# new_data = []
# valid_data = []
# resource_dim = 1
# raw_node_feature_dim = 1 + resource_dim  # (duration, resources)
# evaluator = DAGraph(resource_dim=resource_dim, feature_dim=raw_node_feature_dim)

# num_new_graphs = 10

# for batched_graphs in tqdm(batch(loaded_graphs, num_new_graphs)):
#     graph_original = batched_graphs[0]
#     valid_data.append(graph_original)
#     n_nodes = len(graph_original.nodes)
#     new_graphs = []
#     better_graph_count = 0
#     num_sample_count = 0
#     while len(new_graphs) < num_new_graphs:
#         max_add = 5
#         add_edges = []
#         graph = graph_original.copy()
#         edge_mat = nx.to_numpy_array(graph)
#         identity_mat = np.identity(n_nodes, dtype=bool)
#         trans_closure = transitive_closure(edge_mat)
#         redundant_mask = ~trans_closure
#         cyclic_mask = ~(trans_closure.T | identity_mat)
#         mask = redundant_mask & cyclic_mask
        
#         while max_add > 0:
#             i = random.randint(0, n_nodes-1)
#             j = random.randint(0, n_nodes-1)
#             if mask[i, j]:
#                 edge_mat[i, j] = 1
#                 update_transitive_closure(trans_closure, i, j)
#                 redundant_mask = ~trans_closure
#                 cyclic_mask = ~(trans_closure.T | identity_mat)
#                 mask = redundant_mask & cyclic_mask
#                 max_add -= 1
#                 add_edges.append((i, j))
#         for i, j in add_edges:
#             graph.add_edge(i, j)
#         time_cost, order = evaluator.critical_path_scheduling(graph)
#         real_cost = evaluator.scheduling_evaluator(graph_original, order)
        
#         if len(new_graphs) <= math.floor(num_new_graphs*0.7) or better_graph_count >= math.floor(num_new_graphs*0.3):
#             if real_cost < graph.graph['makespan']:
#                 better_graph_count += 1
#             print(f"makespan before: {graph.graph['makespan']}, after: {time_cost}, real_cost: {real_cost}")
#             add_graph = graph_original.copy()
#             add_graph.graph['makespan'] = real_cost
#             add_graph.graph['order'] = order
#             add_graph.graph['added_edge'] = add_edges
#             add_graph.graph['scheduler'] = 'critical_path'
        
#             new_graphs.append(add_graph)
#         else:
#             num_sample_count += 1
#             if num_sample_count <= 200:
#                 if real_cost < graph.graph['makespan']:
#                     print(f"makespan before: {graph.graph['makespan']}, after: {time_cost}, real_cost: {real_cost}")
#                     add_graph = graph_original.copy()
#                     add_graph.graph['makespan'] = real_cost
#                     add_graph.graph['order'] = order
#                     add_graph.graph['added_edge'] = add_edges
#                     add_graph.graph['scheduler'] = 'critical_path'
#                     new_graphs.append(add_graph)
#             else:
#                 print("hard to find better graph, add anyway")
#                 print(f"makespan before: {graph.graph['makespan']}, after: {time_cost}, real_cost: {real_cost}")
#                 add_graph = graph_original.copy()
#                 add_graph.graph['makespan'] = real_cost
#                 add_graph.graph['order'] = order
#                 add_graph.graph['added_edge'] = add_edges
#                 add_graph.graph['scheduler'] = 'critical_path'
#                 new_graphs.append(add_graph)

#     new_data.extend(new_graphs)
#     print(len(new_data))

# train_save_path = "/data/shared/huiyuan/dag34_addedge/train.pkl"
# valid_save_path = "/data/shared/huiyuan/dag34_addedge/valid.pkl"
# with open(train_save_path, 'wb') as f:
#     pickle.dump(new_data, f)

# with open(valid_save_path, 'wb') as f:
#     print(len(valid_data))
#     pickle.dump(valid_data, f)



# ms = []
# for graph in loaded_graphs:
#     ms.append(graph.graph['makespan'])
# plt.hist(ms, bins=10) 
# plt.savefig('heuristic_makespans.png')
# plt.show()
