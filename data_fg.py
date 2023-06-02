import networkx as nx
import numpy as np
import os
import pdb
import random
import torch

import time
from torch_geometric.data import Data
from torch_scatter import scatter_add
import core


from batch import DataLoader, Dataset_mine
import csv
from utils import *
import copy
import scipy

def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output


def get_adj_matrix(graph):
    N = graph.number_of_nodes()

    adj_matrix = np.zeros((N,N,2)) #Two types of nodes

    for i in range(N):
        adj_matrix[i,i] = [1,1]
    
    for edge in graph.edges:
        adj_matrix[edge[0],edge[1]] = graph.edges[edge[0],edge[1]]['features']
        adj_matrix[edge[1],edge[0]] = graph.edges[edge[0],edge[1]]['features']

    return adj_matrix

def normalize_adj(A, method='sym', *, axis1=-2, axis2=-1, 
                  assume_symmetric_input=False,
                  check_symmetry=False, eps=1e-10,
                  array_mode=None,
                  array_default_mode='numpy',
                  array_homo_mode=None):
    """Normalize adjacency matrix defined by axis1 and axis2 in an array
    """
    dtype = A.dtype if np.issubdtype(A.dtype, np.floating) else np.float

    if method in ['row', 'col', 'column']:
        axis_to_sum = axis2 if method == 'row' else axis1
        norm = np.sum(A, axis_to_sum, dtype=dtype, keepdims=True)
        norm[norm==0] = eps
        norm = 1.0 / norm
        return A * norm
    elif method in ['ds', 'dsm', 'doubly_stochastic']:
        # step 1: row normalize
        norm = np.sum(A, axis2, dtype=dtype, keepdims=True)
        norm[norm==0] = eps
        norm = 1.0 / norm
        P = A * norm

        # step 2: P @ P^T / column_norm
        P = _ops.swapaxes(P, axis2, -1)
        P = _ops.swapaxes(P, axis1, -2)
        norm = np.sum(P, axis=-2, dtype=dtype, keepdims=True)
        norm[norm==0] = eps
        norm = 1.0 / norm
        PT = _ops.swapaxes(P, -1, -2)
        P = np.multiply(P, norm)
        T = np.matmul(P, PT)
        T = _ops.swapaxes(T, axis1, -2)
        T = _ops.swapaxes(T, axis2, -1)
        return T
    else:
        assert method in ['sym', 'symmetric']
        treat_A_as_sym = False
        if assume_symmetric_input:
            if check_symmetry:
                _utils.assert_is_symmetric(A, axis1, axis2)
            treat_A_as_sym = True
        else:
            if check_symmetry:
                treat_A_as_sym = _utils.is_symmetric(A, axis1, axis2)

        norm1 = np.sqrt(np.sum(A, axis2, dtype=dtype, keepdims=True))
        norm1[norm1==0] = 1e-10
        norm1 = 1.0 / norm1
        if treat_A_as_sym:
            norm2 = _ops.swapaxes(norm1, axis1, axis2)
        else:
            norm2 = np.sqrt(np.sum(A, axis1, dtype=dtype, keepdims=True))
            norm2[norm2==0] = 1e-10
            norm2 = 1.0 / norm2
        return A * norm1 * norm2

def load_graphs_fg(data_dir, stats_dir, community_list, core_flag):
    # load stats
    with open(stats_dir+'fg_stats.csv') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        stats = []
        for stat in data:
            stats.append(stat)

    # load graphs
    graphs = []
    nodes_communitys = []
    nodes_par1s = []
    nodes_par2s = []
    core_indexes = []
    filenames_order = os.listdir(data_dir)
    # filenames_order = sorted(filenames_order, key=lambda x: os.stat(os.path.join(data_dir,x)).st_size)
    filenames_order = sorted(filenames_order)
    for idx, filename in enumerate(filenames_order):
        if 'fg_edge' in filename:
            with open(data_dir + filename, 'rb') as fh:
                graph = nx.read_edgelist(fh)
            filename = filename[:-13] # remove postfix
            # find partite split
            for stat in stats:
                if filename == stat[0][:-4]:
                    n = graph.number_of_nodes()
                    n_var = int(stat[1])
                    n_clause = int(stat[2])
                    if graph.number_of_nodes() != n_var+n_clause:
                        print(graph.number_of_nodes())
                        print(n_var)
                        print('Stats not match!')
                        print(stat[0], filename, graph.number_of_nodes(), graph.number_of_edges(), n_var, n_clause)
                    else:
                        # relabel nodes
                        keys = [str(i + 1) for i in range(n)]
                        vals = range(n)
                        mapping = dict(zip(keys, vals))
                        nx.relabel_nodes(graph, mapping, copy=False)
                        
                        # split nodes partite
                        nodes_par1 = list(range(n_var))
                        nodes_par2 = list(range(n_var, n_var + n_clause))
                        nodes_par1s.append(nodes_par1)
                        nodes_par2s.append(nodes_par2)

                        # community nodes
                        community_info = community_list[idx].copy()
                        community_num = len(community_info)
                        nodes_community = list(range(n_var + n_clause, n_var + n_clause + community_num))
                        for i, set_i in enumerate(community_info):
                            graph.add_edges_from([(i+n_var+n_clause, j) for j in set_i],features=[0,0,1])
                        nodes_communitys.append(nodes_community)
                        
                        if core_flag:
                            core_dir = data_dir[:-3] + 'core/'
                            unsat_core_file = core_dir + f'{filename}_core'
                            core_index = core.mark_unsat_core(graph, unsat_core_file, n_var, n_clause, is_LCG=False)
                            core_indexes.append(core_index)

                        graphs.append(graph)
                    break

    return graphs, nodes_par1s, nodes_par2s, nodes_communitys, core_indexes, filenames_order

def load_graphs_vig(data_dir, stats_dir):
    # load stats
    with open(stats_dir+'lcg_stats.csv') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        stats = []
        for stat in data:
            stats.append(stat)

    # load graphs
    graphs = []
    filenames_order = os.listdir(data_dir)
    # filenames_order = sorted(filenames_order, key=lambda x: os.stat(os.path.join(data_dir,x)).st_size)
    filenames_order = sorted(filenames_order)
    for filename in filenames_order:
        if 'vig_edge' in filename:
            with open(data_dir + filename, 'rb') as fh:
                graph = nx.read_edgelist(fh)
            filename = filename[:-14] # remove postfix
            # find partite split
            for stat in stats:
                if filename == stat[0][:-4]:
                    n = graph.number_of_nodes()
                    n_var = int(stat[1])
                    n_clause = int(stat[2])
                    if graph.number_of_nodes() != n_var:
                        print('Stats not match!')
                        print(stat[0], filename, graph.number_of_nodes(), graph.number_of_edges(), n_var, n_clause)
                    else:
                        # relabel nodes
                        keys = [str(i + 1) for i in range(n)]
                        vals = range(n)
                        mapping = dict(zip(keys, vals))
                        nx.relabel_nodes(graph, mapping, copy=False)
                        graphs.append(graph)
                    break
    return graphs

class Dataset_sat_intercmt(torch.utils.data.Dataset):
    def __init__(self, graph_list, nodes_par1_list, nodes_par2_list, nodes_communitys_list, community_list, core_flag,
                 core_indexes, epoch_len, yield_prob=1, speedup=False, hop=4, simple_sample=False):
        super(Dataset_sat_intercmt, self).__init__()
        self.graph_list = graph_list
        self.nodes_par1_list = nodes_par1_list
        self.nodes_par2_list = nodes_par2_list
        self.nodes_communitys_list = nodes_communitys_list
        self.community_list = community_list
        self.epoch_len = epoch_len
        self.yield_prob = yield_prob
        self.speedup = speedup
        self.hop = hop
        self.simple_sample = simple_sample
        self.core_flag = core_flag
        self.core_indexes = core_indexes

        self.data_generator = self.get_data()

    def __getitem__(self, index):
        return next(self.data_generator)

    def __len__(self):
        # return len(self.data)
        return self.epoch_len

    @property
    def num_features(self):
        return 3

    @property
    def num_classes(self):
        return 2

    def get_graph_incmt(self):
        graph_templates = []
        nodes_par1s = []
        nodes_par2s = []
        nodes_communitys = self.nodes_communitys_list
        for i in range(len(self.graph_list)):
            graph = self.graph_list[i].copy()
            nodes_par1 = self.nodes_par1_list[i].copy()
            nodes_par2 = self.nodes_par2_list[i].copy()
            communty_info = self.community_list[i].copy()
            c_labels = torch.zeros((len(nodes_par1), )).type(torch.int)
            for j in range(len(communty_info)):
                for x in communty_info[j]:
                    c_labels[x] = j

            degree_info = list(graph.degree(nodes_par2))
            # random.shuffle(degree_info)
            degree_info.sort(key = lambda a: a[-1], reverse=True) # sort by degree
            idx = 0
            num_clauses = len(nodes_par2)
            while True:
                if idx == num_clauses:
                    print('done',node_degree)
                    graph_templates.append(graph)
                    nodes_par1s.append(nodes_par1)
                    nodes_par2s.append(nodes_par2)
                    break
                node, node_degree = degree_info[idx]
                if self.core_flag and node in self.core_indexes[i]['clause']:
                    idx += 1
                    continue
                node_nbrs = list(set(graph[node]) - {node})
                node_cmts = c_labels[node_nbrs]
                if len(set(node_cmts.tolist())) == 1:
                    idx += 1
                    continue

                node_nbr = random.choice(node_nbrs)
                node_nbr_cmt = c_labels[node_nbr].item()
                idxs = torch.nonzero(node_cmts == node_nbr_cmt).squeeze().tolist()
                node_unlinks = np.array(node_nbrs)[idxs]

                node_new = graph.number_of_nodes() # new node in nodes_par2s
                nodes_par2.append(node_new)
                if node_unlinks.size == 1:
                    edge_type = graph.edges[node, node_unlinks]['features']
                    graph.remove_edge(node, node_unlinks)
                    graph.add_edge(node_nbr, node_new, features=edge_type)
                else:
                    for node_unlink in node_unlinks:
                        edge_type = graph.edges[node, node_unlink]['features']
                        graph.remove_edge(node, node_unlink)
                        graph.add_edge(node_nbr, node_new, features=edge_type)
                        
        return graph_templates, nodes_par1s, nodes_par2s, nodes_communitys

    def get_data(self):
        # assume we hold nodes_par1, while split node in nodes_par2
        # output node pair (node, node_new) and corresponding edge_list

        # 1 pick max degree node in nodes_par2
        while True:
            id = np.random.randint(len(self.graph_list))
            graph = self.graph_list[id].copy()
            nodes_par1 = self.nodes_par1_list[id].copy()
            nodes_par2 = self.nodes_par2_list[id].copy()
            nodes_community = self.nodes_communitys_list[id].copy()
            communty_info = self.community_list[id].copy()
            c_labels = torch.zeros((len(nodes_par1), )).type(torch.int)
            for j in range(len(communty_info)):
                for x in communty_info[j]:
                    c_labels[x] = j

            degree_info = list(graph.degree(nodes_par2))
            # random.shuffle(degree_info)
            degree_info.sort(key = lambda a: a[-1], reverse=True)
            idx = 0
            num_clauses = len(nodes_par2)
            while True:
                if idx == num_clauses:
                    break
                node, node_degree = degree_info[idx]
                if self.core_flag and node in self.core_indexes[id]['clause']:
                    idx += 1
                    continue
                node_nbrs = list(set(graph[node]) - {node})
                node_cmts = c_labels[node_nbrs]
                if len(set(node_cmts.tolist())) == 1:
                    idx += 1
                    continue

                node_nbr = random.choice(node_nbrs)
                node_nbr_cmt = c_labels[node_nbr].item()
                idxs = torch.nonzero(node_cmts == node_nbr_cmt).squeeze().tolist()
                node_unlinks = np.array(node_nbrs)[idxs]

                node_new = graph.number_of_nodes() # new node in nodes_par2s
                nodes_par2.append(node_new)
                if node_unlinks.size == 1:
                    edge_type = graph.edges[node, node_unlinks]['features']
                    graph.remove_edge(node, node_unlinks)
                    graph.add_edge(node_nbr, node_new, features=edge_type)
                else:
                    for node_unlink in node_unlinks:
                        edge_type = graph.edges[node, node_unlink]['features']
                        graph.remove_edge(node, node_unlink)
                        graph.add_edge(node_nbr, node_new, features=edge_type)

                if np.random.rand()<self.yield_prob:
                    # generate output data
                    if self.speedup and not self.core_flag:
                        # construct nodes_candidates
                        nodes_par1 = list(set(graph[node_new]) - {node_new})
                        nodes_candidates = set(nodes_par2) - {node_new}
                        for node_par1 in nodes_par1:
                            node_par1_nbrs = set(graph[node_par1]) - {node_par1}
                            node_par1_cmt = list(node_par1_nbrs.intersection(nodes_community))
                            assert(len(node_par1_cmt) == 1)
                            node_par1_cmt = node_par1_cmt[0]
                            nodes_same_cmt = set(get_neigbors(graph, node_par1_cmt, depth=2)[2])
                            nodes_candidates = nodes_candidates - node_par1_nbrs
                            nodes_candidates_intercmt = nodes_candidates - nodes_same_cmt
                            if len(nodes_candidates_intercmt) != 0: 
                                nodes_candidates = nodes_candidates_intercmt
                        
                        node_sample = random.sample(nodes_candidates, k=1)[0] # sample negative examples

                        nodes_sub1 = set(dict(nx.single_source_shortest_path_length(graph, node, cutoff=self.hop)).keys())
                        nodes_sub2 = set(dict(nx.single_source_shortest_path_length(graph, node_new, cutoff=self.hop)).keys())
                        nodes_sub3 = set(dict(nx.single_source_shortest_path_length(graph, node_sample, cutoff=self.hop)).keys())
                        graph_sub = graph.subgraph(nodes_sub1.union(nodes_sub2,nodes_sub3))
                        keys = list(graph_sub.nodes)
                        vals = range(len(keys))
                        mapping = dict(zip(keys, vals))
                        graph_sub = nx.relabel_nodes(graph_sub, mapping, copy=True)
                        x = torch.zeros((len(keys), 3))
                        nodes_par2_mapped = []
                        for i,key in enumerate(keys):
                            if key<len(nodes_par1):
                                x[i,0]=1
                            elif key in nodes_community:
                                x[i,2]=1
                            else:
                                x[i,1]=1
                                nodes_par2_mapped.append(i)

                        for i in graph_sub.nodes:
                            graph_sub.add_edge(i,i, features=[1,1,0])

                        edge_index = np.array(list(graph_sub.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)

                        edge_features = []
                        for edge in edge_index:
                            edge_features.append(graph_sub.edges[edge[0], edge[1]]['features'])

                        edge_features = np.array(edge_features)  
                        edge_features = torch.from_numpy(edge_features)

                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

                        # # compute GCN norm
                        # row, col = edge_index
                        # deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
                        # deg_inv_sqrt = deg.pow(-0.5)
                        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

                        #adj_matrix = get_adj_matrix(graph_sub)

                        node_index_positive = torch.from_numpy(np.array([[mapping[node]], [mapping[node_new]]])).long()
                        node_index_negative = torch.from_numpy(np.array([[mapping[node]], [mapping[node_sample]]])).long()
                    else:
                        x = torch.zeros((graph.number_of_nodes(), 3))  # 3 types of nodes
                        for i in range(graph.number_of_nodes()):
                            if i<len(nodes_par1):
                                x[i,0]=1
                            elif i in nodes_community:
                                x[i,2]=1
                            else:
                                x[i,1]=1

                        for i in graph.nodes:
                            graph.add_edge(i,i, features=[1,1,0])

                        edge_index = np.array(list(graph.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)

                        edge_features = []
                        for edge in edge_index:
                            edge_features.append(graph.edges[edge[0], edge[1]]['features'])

                        edge_features = np.array(edge_features)  
                        edge_features = torch.from_numpy(edge_features)

                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)
                        node_index_positive = torch.from_numpy(np.array([[node], [node_new]])).long()
                        # sample negative examples
                        if self.simple_sample:
                            node_neg_pair = random.sample(nodes_par2,2)
                            node_index_negative = torch.from_numpy(np.array([[node_neg_pair[0]], [node_neg_pair[1]]])).long()
                        else:
                            # construct nodes_candidates
                            nodes_par1 = list(set(graph[node_new]) - {node_new})
                            nodes_candidates = set(nodes_par2) - {node_new}
                            for node_par1 in nodes_par1:
                                node_par1_nbrs = set(graph[node_par1]) - {node_par1}
                                node_par1_cmt = list(node_par1_nbrs.intersection(nodes_community))
                                assert(len(node_par1_cmt) == 1)
                                node_par1_cmt = node_par1_cmt[0]
                                nodes_same_cmt = set(get_neigbors(graph, node_par1_cmt, depth=2)[2])
                                nodes_candidates = nodes_candidates - node_par1_nbrs
                                nodes_candidates_intercmt = nodes_candidates - nodes_same_cmt
                                if len(nodes_candidates_intercmt) != 0: 
                                    nodes_candidates = nodes_candidates_intercmt
                            
                            node_sample = random.sample(nodes_candidates, k=1)[0] # sample negative examples
                            node_index_negative = torch.from_numpy(np.array([[node_new], [node_sample]])).long()

                        #adj_matrix = get_adj_matrix(graph)
                        

                    # adj_matrix = normalize_adj(adj_matrix, axis1=0, axis2=1)
                    # adj_matrix = torch.from_numpy(adj_matrix)
                    
                    # print(edge_index.shape)
                    # print(edge_features.shape)
                    
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_features.float())
                    #data.adj_matrix=adj_matrix
                    data.node_index_positive = node_index_positive
                    data.node_index_negative = node_index_negative
                    
                    yield data
                else:
                    continue

class Dataset_sat_incmt(torch.utils.data.Dataset):
    def __init__(self, graph_list, nodes_par1_list, nodes_par2_list, nodes_communitys_list, community_list, core_flag,
                 core_indexes, epoch_len, yield_prob=1, speedup=False, hop=4, simple_sample=False):
        super(Dataset_sat_incmt, self).__init__()
        self.graph_list = graph_list
        self.nodes_par1_list = nodes_par1_list
        self.nodes_par2_list = nodes_par2_list
        self.nodes_communitys_list = nodes_communitys_list
        self.community_list = community_list
        self.epoch_len = epoch_len
        self.yield_prob = yield_prob
        self.speedup = speedup
        self.hop = hop
        self.simple_sample = simple_sample
        self.core_flag = core_flag
        self.core_indexes = core_indexes

        self.data_generator = self.get_data()

    def __getitem__(self, index):
        return next(self.data_generator)

    def __len__(self):
        # return len(self.data)
        return self.epoch_len

    @property
    def num_features(self):
        return 3

    @property
    def num_classes(self):
        return 2

    def get_template(self):
        graph_templates = []
        nodes_par1s = []
        nodes_par2s = []
        nodes_communitys = self.nodes_communitys_list
        for i in range(len(self.graph_list)):
            graph = self.graph_list[i].copy()
            nodes_par1 = self.nodes_par1_list[i].copy()
            nodes_par2 = self.nodes_par2_list[i].copy()
            if self.core_flag:
                # different ending flag to divide core and non-core nodes
                remain_clause = nodes_par2.copy()
                while True:
                    if len(remain_clause) == 0:
                        print('done incmt with core')
                        graph_templates.append(graph)
                        nodes_par1s.append(nodes_par1)
                        nodes_par2s.append(nodes_par2)
                        break
                    degree_info = list(graph.degree(remain_clause))
                    node, node_degree = max(degree_info, key=lambda item:item[1]) # (node, degree)
                    if node in self.core_indexes[i]['clause'] or node_degree == 1:
                        remain_clause.remove(node)
                        continue
                    node_nbrs = list(set(graph[node]) - {node})
                    node_nbr = random.choice(node_nbrs)
                    edge_type = graph.edges[node, node_nbr]['features']
                    graph.remove_edge(node, node_nbr)
                    node_new = graph.number_of_nodes()  # new node in nodes_par2
                    nodes_par2.append(node_new)
            else:
                while True:
                    degree_info = list(graph.degree(nodes_par2))
                    node, node_degree = max(degree_info, key=lambda item: item[1])  # (node, degree)
                    if node_degree == 1:
                        print('done',node_degree)
                        graph_templates.append(graph)
                        nodes_par1s.append(nodes_par1)
                        nodes_par2s.append(nodes_par2)
                        break
                    node_nbrs = list(set(graph[node]) - {node})
                    node_nbr = random.choice(node_nbrs)
                    edge_type = graph.edges[node, node_nbr]['features']
                    graph.remove_edge(node, node_nbr)
                    node_new = graph.number_of_nodes()  # new node in nodes_par2
                    nodes_par2.append(node_new)
                graph.add_edge(node_nbr, node_new, features=edge_type)
        return graph_templates, nodes_par1s, nodes_par2s, nodes_communitys

    def get_data(self):
        # assume we hold nodes_par1, while split node in nodes_par2
        # output node pair (node, node_new) and corresponding edge_list

        # 1 pick max degree node in nodes_par2
        while True:
            id = np.random.randint(len(self.graph_list))
            graph = self.graph_list[id].copy()
            nodes_par1 = self.nodes_par1_list[id].copy()
            nodes_par2 = self.nodes_par2_list[id].copy()
            nodes_community = self.nodes_communitys_list[id].copy()
            communty_info = self.community_list[id].copy()
            c_labels = torch.zeros((len(nodes_par1), )).type(torch.int)
            for j in range(len(communty_info)):
                for x in communty_info[j]:
                    c_labels[x] = j

            if self.core_flag:
                # different ending flag to divide core and non-core nodes
                remain_clause = nodes_par2.copy()
            while True:
                if self.core_flag:
                    if len(remain_clause) == 0:
                        break
                    degree_info = list(graph.degree(remain_clause))
                    node, node_degree = max(degree_info, key=lambda item:item[1]) # (node, degree)
                    if node in self.core_indexes[i]['clause'] or node_degree == 1:
                        remain_clause.remove(node)
                        continue
                else:
                    degree_info = list(graph.degree(nodes_par2))
                    random.shuffle(degree_info)
                    node, node_degree = max(degree_info, key=lambda item:item[1]) # (node, degree)
                    if node_degree==1:
                        break
                node_nbrs = list(set(graph[node]) - {node})
                node_nbr = random.choice(node_nbrs)
                edge_type = graph.edges[node,node_nbr]['features']
                graph.remove_edge(node, node_nbr)
                node_new = graph.number_of_nodes() # new node in nodes_par2
                nodes_par2.append(node_new)
                graph.add_edge(node_nbr, node_new, features=edge_type)

                if np.random.rand()<self.yield_prob:
                    # generate output data
                    if self.speedup and not self.core_flag:
                        # sample negative examples
                        node_par1 = list(set(graph[node_new]) - {node_new})[0]
                        node_par1_nbrs = set(graph[node_par1]) - {node_par1}
                        node_par1_cmt = list(node_par1_nbrs.intersection(nodes_community))
                        assert(len(node_par1_cmt) == 1)
                        node_par1_cmt = node_par1_cmt[0]
                        nodes_same_cmt = set(get_neigbors(graph, node_par1_cmt, depth=2)[2])
                        nodes_candidates = set(nodes_par2) - node_par1_nbrs - {node_new}
                        nodes_candidates = nodes_candidates.intersection(nodes_same_cmt)
                        node_sample = random.sample(nodes_candidates, k=1)[0]

                        nodes_sub1 = set(dict(nx.single_source_shortest_path_length(graph, node, cutoff=self.hop)).keys())
                        nodes_sub2 = set(dict(nx.single_source_shortest_path_length(graph, node_new, cutoff=self.hop)).keys())
                        nodes_sub3 = set(dict(nx.single_source_shortest_path_length(graph, node_sample, cutoff=self.hop)).keys())
                        graph_sub = graph.subgraph(nodes_sub1.union(nodes_sub2,nodes_sub3))
                        keys = list(graph_sub.nodes)
                        vals = range(len(keys))
                        mapping = dict(zip(keys, vals))
                        graph_sub = nx.relabel_nodes(graph_sub, mapping, copy=True)
                        x = torch.zeros((len(keys), 3))
                        nodes_par2_mapped = []
                        for i,key in enumerate(keys):
                            if key<len(nodes_par1):
                                x[i,0]=1
                            elif key in nodes_community:
                                x[i,2]=1
                            else:
                                x[i,1]=1
                                nodes_par2_mapped.append(i)

                        for i in graph_sub.nodes:
                            graph_sub.add_edge(i,i, features=[1,1,0])

                        edge_index = np.array(list(graph_sub.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)

                        edge_features = []
                        for edge in edge_index:
                            edge_features.append(graph_sub.edges[edge[0], edge[1]]['features'])

                        edge_features = np.array(edge_features)  
                        edge_features = torch.from_numpy(edge_features)

                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

                        # # compute GCN norm
                        # row, col = edge_index
                        # deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
                        # deg_inv_sqrt = deg.pow(-0.5)
                        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

                        #adj_matrix = get_adj_matrix(graph_sub)

                        node_index_positive = torch.from_numpy(np.array([[mapping[node]], [mapping[node_new]]])).long()
                        node_index_negative = torch.from_numpy(np.array([[mapping[node]], [mapping[node_sample]]])).long()
                    else:
                        x = torch.zeros((graph.number_of_nodes(), 3))  # 3 types of nodes
                        for i in range(graph.number_of_nodes()):
                            if i<len(nodes_par1):
                                x[i,0]=1
                            elif i in nodes_community:
                                x[i,2]=1
                            else:
                                x[i,1]=1

                        for i in graph.nodes:
                            graph.add_edge(i,i, features=[1,1,0])

                        edge_index = np.array(list(graph.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)

                        edge_features = []
                        for edge in edge_index:
                            edge_features.append(graph.edges[edge[0], edge[1]]['features'])

                        edge_features = np.array(edge_features)  
                        edge_features = torch.from_numpy(edge_features)

                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)
                        node_index_positive = torch.from_numpy(np.array([[node], [node_new]])).long()
                        # sample negative examples
                        if self.simple_sample:
                            node_neg_pair = random.sample(nodes_par2,2)
                            node_index_negative = torch.from_numpy(np.array([[node_neg_pair[0]], [node_neg_pair[1]]])).long()
                        else:
                            # sample additional node
                            node_par1 = list(set(graph[node_new]) - {node_new})[0]
                            node_par1_nbrs = set(graph[node_par1]) - {node_par1}
                            node_par1_cmt = list(node_par1_nbrs.intersection(nodes_community))
                            assert(len(node_par1_cmt) == 1)
                            node_par1_cmt = node_par1_cmt[0]
                            nodes_same_cmt = set(get_neigbors(graph, node_par1_cmt, depth=2)[2])
                            nodes_candidates = set(nodes_par2) - node_par1_nbrs - {node_new}
                            nodes_candidates = nodes_candidates.intersection(nodes_same_cmt)
                            node_sample = random.sample(nodes_candidates, k=1)[0]
                            node_index_negative = torch.from_numpy(np.array([[node_new], [node_sample]])).long()
                        #adj_matrix = get_adj_matrix(graph)
                        

                    # adj_matrix = normalize_adj(adj_matrix, axis1=0, axis2=1)
                    # adj_matrix = torch.from_numpy(adj_matrix)
                    
                    # print(edge_index.shape)
                    # print(edge_features.shape)
                    
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_features.float())
                    #data.adj_matrix=adj_matrix
                    data.node_index_positive = node_index_positive
                    data.node_index_negative = node_index_negative
                    
                    yield data
                else:
                    continue

class graph_generator():
    def __init__(self, graph, nodes_par1, nodes_par2, nodes_community, community_info, core_flag, core_indexes, sample_size = 100, device='cpu'): # clause_num=None
        self.graph_raw = graph
        self.graph = self.graph_raw.copy()
        self.nodes_par1 = nodes_par1
        self.nodes_par2_raw = nodes_par2
        self.nodes_par2 = nodes_par2.copy()
        self.nodes_community = nodes_community
        self.core_flag = core_flag
        self.core_indexes = core_indexes
        self.sample_size = sample_size
        self.device = device
        # init once
        self.n = self.graph.number_of_nodes()
        self.data = Data()
        x = torch.zeros((graph.number_of_nodes(), 3))  # 3 types of nodes
        for i in range(graph.number_of_nodes()):
            if i<len(nodes_par1):
                x[i,0]=1
            elif i in nodes_community:
                x[i,2]=1
            else:
                x[i,1]=1
        self.data.x = x
        self.data.node_index = torch.zeros((2,self.sample_size),dtype=torch.long)

        self.community_info = community_info
        c_labels = torch.zeros((len(self.nodes_par1), )).type(torch.int)
        for j in range(len(self.community_info)):
            for x in self.community_info[j]:
                c_labels[x] = j

        self.reset()

        self.data.to(device)


    def reset(self):
        # reset graph to init state
        self.graph = self.graph_raw.copy()
        self.node_par2s = self.nodes_par2_raw.copy()
        self.data.edge_index = np.array(list(self.graph.edges))
        self.data.edge_index = np.concatenate((self.data.edge_index, self.data.edge_index[:, ::-1]), axis=0)

        edge_features = []
        for edge in self.data.edge_index:
            edge_features.append(self.graph.edges[edge[0], edge[1]]['features'])

        edge_features = np.array(edge_features)
        edge_features = torch.from_numpy(edge_features).to(self.device)

        self.data.edge_index = torch.from_numpy(self.data.edge_index).long().permute(1, 0).to(self.device)
        self.data.edge_attr=edge_features.float()
        self.resample(1, False)

    # picked version
    def resample(self, clause_num, flag_intercmt=True):
        # select new node to merge
        degree_info = list(self.graph.degree(self.node_par2s))
        random.shuffle(degree_info)
        node_new, node_degree = min(degree_info, key=lambda item: item[1])
        #print(node_degree)
        # if node_degree>1:
        # print(len(self.node_par2s), self.clause_num)
        if len(self.node_par2s) == clause_num:
            return True  # exit_flag = True
        node_par1s = list(set(self.graph[node_new]) - {node_new})
        # sample additional node
        if flag_intercmt:
            nodes_candidates = set(self.node_par2s) - {node_new}
            for node_par1 in list(node_par1s):
                node_par1_nbrs = set(self.graph[node_par1]) - {node_par1}
                nodes_candidates = nodes_candidates - node_par1_nbrs
        else:
            nodes_candidates = set(self.node_par2s) - {node_new}
            for node_par1 in list(node_par1s):
                node_par1_nbrs = set(self.graph[node_par1]) - {node_par1}
                node_par1_cmt = list(node_par1_nbrs.intersection(self.nodes_community))
                assert(len(node_par1_cmt) == 1)
                node_par1_cmt = node_par1_cmt[0]
                nodes_same_cmt = set(get_neigbors(self.graph, node_par1_cmt, depth=2)[2])
                nodes_candidates = nodes_candidates - node_par1_nbrs
                nodes_candidates = nodes_candidates.intersection(nodes_same_cmt)

        sample_size = min(self.sample_size,len(nodes_candidates))
        nodes_sample = torch.tensor(random.sample(nodes_candidates, k=sample_size),dtype=torch.long)
        # generate queries
        self.data.node_index = torch.zeros((2,sample_size),dtype=torch.long,device=self.device)
        self.data.node_index[0, :] = node_new
        self.data.node_index[1, :] = nodes_sample
        # pdb.set_trace()
        return False

    def merge(self, node_pair):
        # node_pair: node_new, node
        # merge node
        self.data.edge_index[self.data.edge_index==node_pair[0]] = node_pair[1]
        node_pair = node_pair.cpu().numpy()
        if node_pair[0] < node_pair[1]:
            node_pair[0], node_pair[1] = node_pair[1], node_pair[0]
        self.graph = nx.contracted_nodes(self.graph, node_pair[1], node_pair[0])
        self.node_par2s.remove(node_pair[0])

    def update(self, node_pair, clause_num, flag_intercmt=True):
        # node_pair: node_new, node
        if node_pair[0] in self.core_indexes['clause'] or node_pair[0] in self.core_indexes['clause']:
            return False
        self.merge(node_pair)
        return self.resample(clause_num, flag_intercmt)


    def get_graph(self):
        # graph = nx.Graph()

        # for i in range(edge_index.shape[1]):
        #     graph.add_edge(edge_index[:,i].tolist(), features=self.edge_attr[i])

        graph = self.graph.copy()
        graph.remove_nodes_from(self.nodes_community)    

        return graph
