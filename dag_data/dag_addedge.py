import networkx as nx
import pickle
import numpy as np
import random
import os
from tqdm import tqdm
import math
from multiprocessing import Pool
from functools import partial

def arrays_to_digraph(nodes, dep_mat):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for i in range(len(nodes)):
        G.add_node(i, features=nodes[i].tolist())

    # Add edges to the graph based on the dependency matrix
    for i in range(len(dep_mat)):
        for j in range(len(dep_mat[i])):
            if np.isclose(dep_mat[i][j], 1.0):
                G.add_edge(i, j)

    return G

def transitive_closure(dep_mat):
    '''
    return the transitive closure that has e_ij = 1 if i is an ancestor of j
    '''
    n_nodes = dep_mat.shape[0]
    dep_mat = dep_mat.astype(bool)
    identity_like_adj_mat = np.identity(n_nodes, dtype=bool)
    closure = identity_like_adj_mat | dep_mat
    # Floyd-Warshall Algorithm
    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                closure[j, i] = closure[j, i] or (closure[k, i] and closure[j, k])
    closure = ~identity_like_adj_mat & closure
    return closure

def update_transitive_closure(trans_closure, i_node, j_node):
    n_nodes = trans_closure.shape[0]
    trans_closure[i_node, j_node] = True
    trans_closure[:, j_node] |= trans_closure[:, i_node]
    for k in range(n_nodes):
        if trans_closure[j_node, k]:
            trans_closure[:, k] |= trans_closure[:, i_node]
            trans_closure[i_node, k] = True

def chunked_graphs(data, chunk_size):
    """Yield successive chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size] 

def add_edge(graph_batch_with_seed, evaluator, num_new_graphs, total_adds, test_mode):
    all_graphs = []
    imprv_ratio = []
    all_max_imprv = []
    seed, graph_batch = graph_batch_with_seed
    random.seed(seed)
    print(seed)
    print(graph_batch[0].number_of_nodes())

    for graph_original in tqdm(graph_batch):
        #valid_data.append(graph_original)
        n_nodes = len(graph_original.nodes)
        new_graphs = []
        #improvement_ratio
        better_graph_count = 0
        better_makespan = []
        num_sample_count = 0
        while len(new_graphs) < num_new_graphs:
            max_add = total_adds
            add_edges = []
            graph = graph_original.copy()
            edge_mat = nx.to_numpy_array(graph)
            identity_mat = np.identity(n_nodes, dtype=bool)
            trans_closure = transitive_closure(edge_mat)
            redundant_mask = ~trans_closure
            cyclic_mask = ~(trans_closure.T | identity_mat)
            mask = redundant_mask & cyclic_mask
                
            while max_add > 0:
                i = random.randint(0, n_nodes-1)
                j = random.randint(0, n_nodes-1)
                if mask[i, j]:
                    edge_mat[i, j] = 1
                    update_transitive_closure(trans_closure, i, j)
                    redundant_mask = ~trans_closure
                    cyclic_mask = ~(trans_closure.T | identity_mat)
                    mask = redundant_mask & cyclic_mask
                    max_add -= 1
                    add_edges.append((i, j))
            for i, j in add_edges:
                graph.add_edge(i, j)
            time_cost, order = evaluator.critical_path_scheduling(graph)
            real_cost = evaluator.scheduling_evaluator(graph_original, order)
                
            if test_mode or len(new_graphs) <= math.floor(num_new_graphs*0.1) or better_graph_count >= math.floor(num_new_graphs*0.9):
                relative_makespan = real_cost/graph.graph['makespan']
                if real_cost < graph.graph['makespan']:
                    better_graph_count += 1
                    better_makespan.append(1-relative_makespan)
                #print(f"makespan before: {graph.graph['makespan']}, after: {time_cost}, real_cost: {real_cost}")
                add_graph = graph_original.copy()
                add_graph.graph['makespan'] = relative_makespan
                add_graph.graph['order'] = order
                add_graph.graph['added_edge'] = add_edges
                add_graph.graph['scheduler'] = 'critical_path'
                new_graphs.append(add_graph)
                print(relative_makespan)
            else:
                num_sample_count += 1
                if num_sample_count <= 200:
                    if real_cost < graph.graph['makespan']:
                        relative_makespan = real_cost/graph.graph['makespan']
                        better_makespan.append(1-relative_makespan)
                        #print(f"makespan before: {graph.graph['makespan']}, after: {time_cost}, real_cost: {real_cost}")
                        add_graph = graph_original.copy()
                        add_graph.graph['makespan'] = relative_makespan
                        print(relative_makespan)
                        add_graph.graph['order'] = order
                        add_graph.graph['added_edge'] = add_edges
                        add_graph.graph['scheduler'] = 'critical_path'
                        new_graphs.append(add_graph)
                else:
                    print("hard to find better graph, add anyway")
                    #print(f"makespan before: {graph.graph['makespan']}, after: {time_cost}, real_cost: {real_cost}")
                    relative_makespan = real_cost/graph.graph['makespan']
                    add_graph = graph_original.copy()
                    add_graph.graph['makespan'] = relative_makespan
                    print(relative_makespan)
                    add_graph.graph['order'] = order
                    add_graph.graph['added_edge'] = add_edges
                    add_graph.graph['scheduler'] = 'critical_path'
                    new_graphs.append(add_graph)
        
        if len(better_makespan):
            ratio = sum(better_makespan)/len(better_makespan)
            imprv_ratio.append(ratio)
            max_imprv = max(better_makespan)
            all_max_imprv.append(max_imprv)
        else:
            ratio = 0.0
            imprv_ratio.append(ratio)
            max_imprv = 0.0
            all_max_imprv.append(max_imprv)
        all_graphs.extend(new_graphs)
    
    #avg_max_imprv = sum(all_max_imprv)/len(all_max_imprv)
    return all_graphs, imprv_ratio, all_max_imprv
                    
        

def new_graphs_addedge(loaded_graphs, evaluator, samples_per_graph, total_adds, save_dir, num_workers, random_seed, test_mode):
    new_data = []
    #valid_data = []
    improvement_ratio = []
    max_imprv_list = []

    with Pool() as p:
        ss = np.random.SeedSequence(random_seed)
        seeds = ss.spawn(num_workers)  
        
        add_edge_processor = partial(add_edge, evaluator=evaluator, num_new_graphs=samples_per_graph, total_adds=total_adds, test_mode=test_mode) 
        batch_size = len(loaded_graphs) // num_workers 
        graph_batches_with_seeds = zip(seeds, chunked_graphs(loaded_graphs, batch_size))   
        new_graphs = p.map(add_edge_processor, graph_batches_with_seeds)
    
    for new_batch, imprv_ratio, max_imprv in new_graphs:
        new_data.extend(new_batch)
        improvement_ratio.extend(imprv_ratio)
        max_imprv_list.extend(max_imprv)
    
    print(f'total trials: {len(max_imprv_list)}, {len(improvement_ratio)}')
    print(f'In-average improvement in data is: {sum(improvement_ratio)/len(improvement_ratio)*100}%')
    print(f'Max improvement in data is: {sum(max_imprv_list)/len(max_imprv_list)*100}%')
    
    if not test_mode:
        train_save_path = os.path.join(save_dir, 'train.pkl')       
        with open(train_save_path, 'wb') as f:
            pickle.dump(new_data, f)
