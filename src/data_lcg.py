import networkx as nx
import numpy as np
import os
import random
import torch

from torch_geometric.data import Data
import core
import eval.conversion_lcg as conversion

import csv
from utils import *

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

def load_graphs_lcg(data_dir, stats_dir, community_list, core_flag):
    # load stats
    with open(stats_dir+'lcg_stats.csv') as csvfile:
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
    filenames_order = sorted(filenames_order)
    filenames = []
    for idx, filename in enumerate(filenames_order):
        if '.cnf' in filename:
            graph = conversion.sat_to_LCG(data_dir + filename)
            filename = filename[:-4]
            # find partite split
            for stat in stats:
                if filename == stat[0].split('/')[-1]:
                    n = graph.number_of_nodes()
                    n_var = int(stat[1])
                    n_clause = int(stat[2])
                    if graph.number_of_nodes() != n_var*2+n_clause:
                        print('Stats not match!')
                        print(stat[0], filename, graph.number_of_nodes(), graph.number_of_edges(), n_var, n_clause)
                    else:
                        print('Stats match.')
                        print(stat[0], filename, graph.number_of_nodes(), graph.number_of_edges(), n_var, n_clause)
                        # relabel nodes
                        keys = [(i + 1) for i in range(n)]
                        vals = range(n)
                        mapping = dict(zip(keys, vals))
                        nx.relabel_nodes(graph, mapping, copy=False)
                        # add links between v and -v
                        graph.add_edges_from([(i, i + n_var) for i in range(n_var)])
                        # split nodes partite
                        nodes_par1 = list(range(n_var * 2))
                        nodes_par2 = list(range(n_var * 2, n_var * 2 + n_clause))
                        nodes_par1s.append(nodes_par1)
                        nodes_par2s.append(nodes_par2)
                        # community nodes
                        community_info = community_list[idx].copy()
                        community_num = len(community_info)
                        nodes_community = list(range(n_var * 2 + n_clause, n_var * 2 + n_clause + community_num))
                        for i, set_i in enumerate(community_info):
                            graph.add_edges_from([(i+n_var*2+n_clause, j) for j in set_i])
                            graph.add_edges_from([(i+n_var*2+n_clause, j+n_var) for j in set_i])
                        nodes_communitys.append(nodes_community)

                        if core_flag:
                            core_dir = data_dir[:-1] + '_core/'
                            unsat_core_file = core_dir + f'{filename}_core'
                            core_index = core.mark_unsat_core(graph, unsat_core_file, n_var, n_clause)
                            core_indexes.append(core_index)
                        graphs.append(graph)
                        filenames.append(filename)
                    break

    return graphs, nodes_par1s, nodes_par2s, nodes_communitys, core_indexes, filenames

def load_graphs_lcg_raw(data_dir, stats_dir):
    # load stats
    with open(stats_dir+'lcg_stats.csv') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        stats = []
        for stat in data:
            stats.append(stat)

    # load graphs
    graphs = []
    nodes_par1s = []
    nodes_par2s = []
    filenames_order = os.listdir(data_dir)
    # filenames_order = sorted(filenames_order, key=lambda x: os.stat(os.path.join(data_dir,x)).st_size)
    filenames_order = sorted(filenames_order)
    for idx, filename in enumerate(filenames_order):
        if 'lcg_edge' in filename:
            with open(data_dir + filename, 'rb') as fh:
                graph = nx.read_edgelist(fh)
            filename = filename[:-14] # remove postfix
            # find partite split
            for stat in stats:
                if filename == stat[0][:-4]:
                    n = graph.number_of_nodes()
                    n_var = int(stat[1])
                    n_clause = int(stat[2])
                    if graph.number_of_nodes() != n_var*2+n_clause:
                        print('Stats not match!')
                        print(stat[0], filename, graph.number_of_nodes(), graph.number_of_edges(), n_var, n_clause)
                    else:
                        # relabel nodes
                        keys = [str(i + 1) for i in range(n)]
                        vals = range(n)
                        mapping = dict(zip(keys, vals))
                        nx.relabel_nodes(graph, mapping, copy=False)
                        # add links between v and -v
                        graph.add_edges_from([(i, i + n_var) for i in range(n_var)])
                        # split nodes partite
                        nodes_par1 = list(range(n_var * 2))
                        nodes_par2 = list(range(n_var * 2, n_var * 2 + n_clause))
                        nodes_par1s.append(nodes_par1)
                        nodes_par2s.append(nodes_par2)
                        graphs.append(graph)
                    break

    return graphs, nodes_par1s, nodes_par2s

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
    filenames_order = sorted(filenames_order)
    for filename in filenames_order:
        if '.cnf' in filename:
            graph = conversion.sat_to_VIG(data_dir + filename)
            filename = filename[:-4]
            # find partite split
            for stat in stats:
                if filename == stat[0].split('/')[-1]:
                    n = graph.number_of_nodes()
                    n_var = int(stat[1])
                    n_clause = int(stat[2])
                    if graph.number_of_nodes() != n_var:
                        print('Stats not match!')
                        print(stat[0], filename, graph.number_of_nodes(), graph.number_of_edges(), n_var, n_clause)
                    else:
                        print('Stats match.')
                        print(stat[0], filename, graph.number_of_nodes(), graph.number_of_edges(), n_var, n_clause)
                        # relabel nodes
                        keys = [i + 1 for i in range(n)]
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
        return self.epoch_len

    @property
    def num_features(self):
        return 4

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
                    if x >= len(c_labels) or x+len(nodes_par1)//2 >= len(c_labels):
                        continue
                    c_labels[x] = j
                    c_labels[x+len(nodes_par1)//2] = j

            degree_info = list(graph.degree(nodes_par2))
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
                node_nbrs = list(graph[node])
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
                    graph.remove_edge(node, node_unlinks)
                    graph.add_edge(node_unlinks, node_new)
                else:
                    for node_unlink in node_unlinks:
                        graph.remove_edge(node, node_unlink)
                        graph.add_edge(node_unlink, node_new)

        return graph_templates, nodes_par1s, nodes_par2s, nodes_communitys

    def get_data(self):
        # assume we hold nodes_par1, while split node in nodes_par2
        # output node pair (node, node_new) and corresponding edge_list

        # pick max degree node in nodes_par2
        while True:
            id = np.random.randint(len(self.graph_list))
            graph = self.graph_list[id].copy()
            nodes_par1 = self.nodes_par1_list[id].copy()
            nodes_par2 = self.nodes_par2_list[id].copy()
            nodes_community = self.nodes_communitys_list[id].copy()
            communty_info = self.community_list[id].copy()
            core_indexes = self.core_indexes[id].copy()
            c_labels = torch.zeros((len(nodes_par1), )).type(torch.int)
            for i in range(len(communty_info)):
                for x in communty_info[i]:
                    if x >= len(c_labels) or x+len(nodes_par1)//2 >= len(c_labels):
                        continue
                    c_labels[x] = i
                    c_labels[x+len(nodes_par1)//2] = i

            degree_info = list(graph.degree(nodes_par2))
            degree_info.sort(key = lambda a: a[-1], reverse=True)
            idx = 0
            num_clauses = len(nodes_par2)
            while True:
                if idx == num_clauses:
                    break
                node, node_degree = degree_info[idx]
                if self.core_flag and node in core_indexes['clause']:
                    idx += 1
                    continue
                node_nbrs = list(graph[node])
                node_cmts = c_labels[node_nbrs]
                if len(set(node_cmts.tolist())) == 1:
                    idx += 1
                    continue
                
                assert(node not in core_indexes['clause'])
                
                node_nbr = random.choice(node_nbrs)
                node_nbr_cmt = c_labels[node_nbr].item()
                idxs = torch.nonzero(node_cmts == node_nbr_cmt).squeeze().tolist()
                node_unlinks = np.array(node_nbrs)[idxs]

                node_new = graph.number_of_nodes() # new node in nodes_par2s
                assert(node_new not in core_indexes['clause'])
                nodes_par2.append(node_new)
                if node_unlinks.size == 1:
                    graph.remove_edge(node, node_unlinks)
                    graph.add_edge(node_unlinks, node_new)
                else:
                    for node_unlink in node_unlinks:
                        graph.remove_edge(node, node_unlink)
                        graph.add_edge(node_unlink, node_new)

                if np.random.rand()<self.yield_prob:
                    # generate output data
                    if self.speedup and not self.core_flag:
                        # construct nodes_candidates
                        nodes_par1 = list(graph[node_new])
                        nodes_candidates = set(nodes_par2) - {node_new}
                        for node_par1 in nodes_par1:
                            if node_par1 < len(nodes_par1) // 2:
                                node_par1_not = node_par1 + len(nodes_par1) // 2
                            else:
                                node_par1_not = node_par1 - len(nodes_par1) // 2
                            node_par1_nbrs = set(graph[node_par1])
                            node_par1_not_nbrs = set(graph[node_par1_not])
                            node_par1_cmt = list(node_par1_nbrs.intersection(nodes_community))
                            assert(len(node_par1_cmt) == 1)
                            node_par1_cmt = node_par1_cmt[0]
                            nodes_same_cmt = set(get_neigbors(graph, node_par1_cmt, depth=2)[2])
                            nodes_candidates = nodes_candidates - node_par1_nbrs - node_par1_not_nbrs
                            nodes_candidates_intercmt = nodes_candidates - nodes_same_cmt
                            if len(nodes_candidates_intercmt) != 0: 
                                nodes_candidates = nodes_candidates_intercmt
                        
                        node_sample = random.sample(nodes_candidates, k=1)[0]
                        if self.core_flag:
                            cnt = 0
                            while node_sample in core_indexes['clause']:
                                node_sample = random.sample(nodes_candidates, k=1)[0]
                                cnt += 1
                                if cnt > 20:
                                    cnt = -1
                                    break
                            if cnt == -1:
                                continue

                        nodes_sub1 = set(dict(nx.single_source_shortest_path_length(graph, node, cutoff=self.hop)).keys())
                        nodes_sub2 = set(dict(nx.single_source_shortest_path_length(graph, node_new, cutoff=self.hop)).keys())
                        nodes_sub3 = set(dict(nx.single_source_shortest_path_length(graph, node_sample, cutoff=self.hop)).keys())
                        graph_sub = graph.subgraph(nodes_sub1.union(nodes_sub2,nodes_sub3))
                        keys = list(graph_sub.nodes)
                        vals = range(len(keys))
                        mapping = dict(zip(keys, vals))
                        graph_sub = nx.relabel_nodes(graph_sub, mapping, copy=True)
                        x = torch.zeros((len(keys), 4))
                        nodes_par2_mapped = []
                        for i,key in enumerate(keys):
                            if key<len(nodes_par1) // 2:
                                x[i,0]=1
                            elif key<len(nodes_par1):
                                x[i,1]=1
                            elif key in nodes_community:
                                x[i,3]=1
                            else:
                                x[i,2]=1
                                nodes_par2_mapped.append(i)

                        edge_index = np.array(list(graph_sub.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

                        node_index_positive = torch.from_numpy(np.array([[mapping[node]], [mapping[node_new]]])).long()
                        node_index_negative = torch.from_numpy(np.array([[mapping[node]], [mapping[node_sample]]])).long()
                    else:
                        x = torch.zeros((graph.number_of_nodes(), 4))  # 3 types of nodes
                        for i in range(graph.number_of_nodes()):
                            if i<len(nodes_par1) // 2:
                                x[i,0]=1
                            elif i<len(nodes_par1):
                                x[i,1]=1
                            elif i in nodes_community:
                                x[i,3]=1
                            else:
                                x[i,2]=1

                        edge_index = np.array(list(graph.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)
                        node_index_positive = torch.from_numpy(np.array([[node], [node_new]])).long()
                        # sample negative examples
                        if self.simple_sample:
                            node_neg_pair = random.sample(nodes_par2,2)
                            if self.core_flag:
                                cnt = 0
                                while node_neg_pair[0] in core_indexes['clause'] or node_neg_pair[1] in core_indexes['clause']:
                                    node_neg_pair = random.sample(nodes_par2, 2)
                                    cnt += 1
                                    if cnt > 20:
                                        cnt = -1
                                        break
                                if cnt == -1:
                                    continue
                            node_index_negative = torch.from_numpy(np.array([[node_neg_pair[0]], [node_neg_pair[1]]])).long()
                        else:
                            # construct nodes_candidates
                            nodes_par1 = list(graph[node_new])
                            nodes_candidates = set(nodes_par2) - {node_new}
                            for node_par1 in nodes_par1:
                                if node_par1 < len(nodes_par1) // 2:
                                    node_par1_not = node_par1 + len(nodes_par1) // 2
                                else:
                                    node_par1_not = node_par1 - len(nodes_par1) // 2
                                node_par1_nbrs = set(graph[node_par1])
                                node_par1_not_nbrs = set(graph[node_par1_not])
                                node_par1_cmt = list(node_par1_nbrs.intersection(nodes_community))
                                assert(len(node_par1_cmt) == 1)
                                node_par1_cmt = node_par1_cmt[0]
                                nodes_same_cmt = set(get_neigbors(graph, node_par1_cmt, depth=2)[2])
                                nodes_candidates = nodes_candidates - node_par1_nbrs - node_par1_not_nbrs
                                nodes_candidates_intercmt = nodes_candidates - nodes_same_cmt
                                if len(nodes_candidates_intercmt) != 0: 
                                    nodes_candidates = nodes_candidates_intercmt
                            
                            node_sample = random.sample(nodes_candidates, k=1)[0] # sample negative examples
                            if self.core_flag:
                                cnt = 0
                                while node_sample in core_indexes['clause']:
                                    node_sample = random.sample(nodes_candidates, k=1)[0]
                                    cnt += 1
                                    if cnt > 20:
                                        cnt = -1
                                        break
                                if cnt == -1:
                                    continue
                            node_index_negative = torch.from_numpy(np.array([[node_new], [node_sample]])).long()
                    data = Data(x=x, edge_index=edge_index)
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
        return self.epoch_len

    @property
    def num_features(self):
        return 4

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
                        node_nbrs = list(graph[node])
                        node_nbr = random.choice(node_nbrs)
                        graph.remove_edge(node, node_nbr)
                        node_new = graph.number_of_nodes()  # new node in nodes_par2
                        nodes_par2.append(node_new)
                        graph.add_edge(node_nbr, node_new)
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
                    node_nbrs = list(graph[node])
                    node_nbr = random.choice(node_nbrs)
                    graph.remove_edge(node, node_nbr)
                    node_new = graph.number_of_nodes()  # new node in nodes_par2
                    nodes_par2.append(node_new)
                    graph.add_edge(node_nbr, node_new)
        return graph_templates, nodes_par1s, nodes_par2s, nodes_communitys


    def get_data(self):
        # assume we hold nodes_par1, while split node in nodes_par2
        # output node pair (node, node_new) and corresponding edge_list

        # pick max degree node in nodes_par2
        while True:
            id = np.random.randint(len(self.graph_list))
            graph = self.graph_list[id].copy()
            nodes_par1 = self.nodes_par1_list[id].copy()
            nodes_par2 = self.nodes_par2_list[id].copy()
            nodes_community = self.nodes_communitys_list[id].copy()
            communty_info = self.community_list[id].copy()
            core_indexes = self.core_indexes[id].copy()
            c_labels = torch.zeros((len(nodes_par1), )).type(torch.int)
            for i in range(len(communty_info)):
                for x in communty_info[i]:
                    if x >= len(c_labels) or x+len(nodes_par1)//2 >= len(c_labels):
                        continue
                    c_labels[x] = i
                    c_labels[x+len(nodes_par1)//2] = i

            if self.core_flag:
                remain_clause = nodes_par2.copy()
            while True:
                if self.core_flag:
                    if len(remain_clause) == 0:
                        break
                    degree_info = list(graph.degree(remain_clause))
                    random.shuffle(degree_info)
                    node, node_degree = max(degree_info, key=lambda item:item[1]) # (node, degree)
                    if node in core_indexes['clause'] or node_degree == 1:
                        remain_clause.remove(node)
                        continue
                    node_nbrs = list(graph[node])
                    node_nbr = random.choice(node_nbrs)
                    graph.remove_edge(node, node_nbr)
                    node_new = graph.number_of_nodes()  # new node in nodes_par2
                    nodes_par2.append(node_new)
                    graph.add_edge(node_nbr, node_new)
                else:
                    degree_info = list(graph.degree(nodes_par2))
                    random.shuffle(degree_info)
                    node, node_degree = max(degree_info, key=lambda item:item[1]) # (node, degree)
                    if node_degree==1:
                        break
                    node_nbrs = list(graph[node])
                    node_nbr = random.choice(node_nbrs)
                    graph.remove_edge(node, node_nbr)
                    node_new = graph.number_of_nodes() # new node in nodes_par2
                    nodes_par2.append(node_new)
                    graph.add_edge(node_nbr, node_new)

                if np.random.rand()<self.yield_prob:
                    # generate output data
                    if self.speedup and not self.core_flag:
                        # sample negative examples
                        node_par1 = list(graph[node_new])[0]
                        if node_par1 < len(nodes_par1) // 2:
                            node_par1_not = node_par1 + len(nodes_par1) // 2
                        else:
                            node_par1_not = node_par1 - len(nodes_par1) // 2
                        node_par1_nbrs = set(graph[node_par1])
                        node_par1_not_nbrs = set(graph[node_par1_not])
                        node_par1_cmt = list(node_par1_nbrs.intersection(nodes_community))
                        assert(len(node_par1_cmt) == 1)
                        node_par1_cmt = node_par1_cmt[0]
                        nodes_same_cmt = set(get_neigbors(graph, node_par1_cmt, depth=2)[2])
                        nodes_candidates = set(nodes_par2) - node_par1_nbrs - node_par1_not_nbrs - {node_new}
                        nodes_candidates = nodes_candidates.intersection(nodes_same_cmt)
                        node_sample = random.sample(nodes_candidates, k=1)[0]
                        if self.core_flag:
                            while node_sample in core_indexes['clause']:
                                node_sample = random.sample(nodes_candidates, k=1)[0]

                        nodes_sub1 = set(dict(nx.single_source_shortest_path_length(graph, node, cutoff=self.hop)).keys())
                        nodes_sub2 = set(dict(nx.single_source_shortest_path_length(graph, node_new, cutoff=self.hop)).keys())
                        nodes_sub3 = set(dict(nx.single_source_shortest_path_length(graph, node_sample, cutoff=self.hop)).keys())
                        graph_sub = graph.subgraph(nodes_sub1.union(nodes_sub2,nodes_sub3))
                        keys = list(graph_sub.nodes)
                        vals = range(len(keys))
                        mapping = dict(zip(keys, vals))
                        graph_sub = nx.relabel_nodes(graph_sub, mapping, copy=True)
                        x = torch.zeros((len(keys), 4))
                        nodes_par2_mapped = []
                        for i,key in enumerate(keys):
                            if key<len(nodes_par1) // 2:
                                x[i,0]=1
                            elif key<len(nodes_par1):
                                x[i,1]=1
                            elif key in nodes_community:
                                x[i,3]=1
                            else:
                                x[i,2]=1
                                nodes_par2_mapped.append(i)

                        edge_index = np.array(list(graph_sub.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

                        node_index_positive = torch.from_numpy(np.array([[mapping[node]], [mapping[node_new]]])).long()
                        node_index_negative = torch.from_numpy(np.array([[mapping[node]], [mapping[node_sample]]])).long()
                    else:
                        x = torch.zeros((graph.number_of_nodes(), 4))  # 3 types of nodes
                        for i in range(graph.number_of_nodes()):
                            if i<len(nodes_par1) // 2:
                                x[i,0]=1
                            elif i<len(nodes_par1):
                                x[i,1]=1
                            elif i in nodes_community:
                                x[i,3]=1
                            else:
                                x[i,2]=1

                        edge_index = np.array(list(graph.edges))
                        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
                        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)
                        node_index_positive = torch.from_numpy(np.array([[node], [node_new]])).long()
                        # sample negative examples
                        if self.simple_sample:
                            node_neg_pair = random.sample(nodes_par2,2)
                            if self.core_flag:
                                while node_neg_pair[0] in core_indexes['clause'] or node_neg_pair[1] in core_indexes['clause']:
                                    node_neg_pair = random.sample(nodes_par2,2)
                            node_index_negative = torch.from_numpy(np.array([[node_neg_pair[0]], [node_neg_pair[1]]])).long()
                        else:
                            # sample additional node
                            node_par1 = list(graph[node_new])[0]
                            if node_par1 < len(nodes_par1) // 2:
                                node_par1_not = node_par1 + len(nodes_par1) // 2
                            else:
                                node_par1_not = node_par1 - len(nodes_par1) // 2
                            node_par1_nbrs = set(graph[node_par1])
                            node_par1_not_nbrs = set(graph[node_par1_not])
                            node_par1_cmt = list(node_par1_nbrs.intersection(nodes_community))
                            assert(len(node_par1_cmt) == 1)
                            node_par1_cmt = node_par1_cmt[0]
                            nodes_same_cmt = set(get_neigbors(graph, node_par1_cmt, depth=2)[2])
                            nodes_candidates = set(nodes_par2) - node_par1_nbrs - node_par1_not_nbrs - {node_new}
                            nodes_candidates = nodes_candidates.intersection(nodes_same_cmt)
                            node_sample = random.sample(nodes_candidates, k=1)[0]
                            if self.core_flag:
                                while node_sample in core_indexes['clause']:
                                    node_sample = random.sample(nodes_candidates, k=1)[0]
                            node_index_negative = torch.from_numpy(np.array([[node_new], [node_sample]])).long()
                    
                    assert(node_index_positive[0] not in core_indexes['clause'])
                    assert(node_index_positive[1] not in core_indexes['clause'])
                    assert(node_index_negative[0] not in core_indexes['clause'])
                    assert(node_index_negative[1] not in core_indexes['clause'])
                        
                    data = Data(x=x, edge_index=edge_index)
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
        self.sample_size = sample_size
        self.core_flag = core_flag
        self.core_indexes = core_indexes
        # self.clause_num = clause_num
        self.device = device
        # init once
        self.n = self.graph.number_of_nodes()
        self.data = Data()
        x = torch.zeros((graph.number_of_nodes(), 4))  # 3 types of nodes
        for i in range(graph.number_of_nodes()):
            if i<len(nodes_par1) // 2:
                x[i,0]=1
            elif i<len(nodes_par1):
                x[i,1]=1
            elif i in nodes_community:
                x[i,3]=1
            else:
                x[i,2]=1
        self.data.x = x
        self.data.node_index = torch.zeros((2,self.sample_size),dtype=torch.long)

        self.community_info = community_info
        c_labels = torch.zeros((len(self.nodes_par1), )).type(torch.int)
        for j in range(len(self.community_info)):
            for x in self.community_info[j]:
                if x >= len(c_labels) or x+len(self.nodes_par1)//2 >= len(c_labels):
                    continue
                c_labels[x] = j
                c_labels[x+len(self.nodes_par1)//2] = j

        self.reset()

        self.data.to(device)


    def reset(self):
        # reset graph to init state
        self.graph = self.graph_raw.copy()
        self.node_par2s = self.nodes_par2_raw.copy()
        self.data.edge_index = np.array(list(self.graph.edges))
        self.data.edge_index = np.concatenate((self.data.edge_index, self.data.edge_index[:, ::-1]), axis=0)
        self.data.edge_index = torch.from_numpy(self.data.edge_index).long().permute(1, 0).to(self.device)
        self.resample(1, False)

    # picked version
    def resample(self, clause_num, flag_intercmt=True):
        bad_node = []
        while True:
            # select new node to merge
            # select only non-core clause
            selected_node = set(self.node_par2s) - set(self.core_indexes['clause'])
            selected_node = list(selected_node)
            degree_info = list(self.graph.degree(selected_node))
            if len(degree_info) == 0:
                return True
            random.shuffle(degree_info)
            node_new, node_degree = min(degree_info, key=lambda item: item[1])
            if self.core_flag:
                while node_new in self.core_indexes or node_new in bad_node:
                    degree_info.remove((node_new, node_degree))
                    if len(degree_info) == 0:
                        return True
                    node_new, node_degree = min(degree_info, key=lambda item: item[1])
            if len(self.node_par2s) == clause_num:
                return True
            node_par1s = self.graph[node_new]
            if flag_intercmt:
                nodes_candidates = set(self.node_par2s) - {node_new}
                for node_par1 in list(node_par1s):
                    if node_par1 < len(self.nodes_par1) // 2:
                        node_par1_not = node_par1 + len(self.nodes_par1) // 2
                    else:
                        node_par1_not = node_par1 - len(self.nodes_par1) // 2
                    node_par1_nbrs = set(self.graph[node_par1])
                    node_par1_not_nbrs = set(self.graph[node_par1_not])
                    nodes_candidates = nodes_candidates - node_par1_nbrs - node_par1_not_nbrs
            else:
                nodes_candidates = set(self.node_par2s) - {node_new}
                for node_par1 in list(node_par1s):
                    if node_par1 < len(self.nodes_par1) // 2:
                        node_par1_not = node_par1 + len(self.nodes_par1) // 2
                    else:
                        node_par1_not = node_par1 - len(self.nodes_par1) // 2
                    node_par1_nbrs = set(self.graph[node_par1])
                    node_par1_not_nbrs = set(self.graph[node_par1_not])
                    node_par1_cmt = list(node_par1_nbrs.intersection(self.nodes_community))
                    assert(len(node_par1_cmt) == 1)
                    node_par1_cmt = node_par1_cmt[0]
                    nodes_same_cmt = set(get_neigbors(self.graph, node_par1_cmt, depth=2)[2])
                    nodes_candidates = nodes_candidates - node_par1_nbrs - node_par1_not_nbrs
                    nodes_candidates = nodes_candidates.intersection(nodes_same_cmt)
            
            if self.core_flag:
                tmp = nodes_candidates.copy()
                for tp in tmp:
                    if tp in self.core_indexes['clause']:
                        nodes_candidates.remove(tp)

            sample_size = min(self.sample_size,len(nodes_candidates))
            if sample_size == 0:
                bad_node.append(node_new)
                print(f"found bad node!: {node_new}")
                print(f"len(self.node_par2s) & clause_num = {len(self.node_par2s)}, {clause_num}")
                continue
            nodes_sample = torch.tensor(random.sample(nodes_candidates, k=sample_size),dtype=torch.long)
            # generate queries
            self.data.node_index = torch.zeros((2,sample_size),dtype=torch.long,device=self.device)
            self.data.node_index[0, :] = node_new
            self.data.node_index[1, :] = nodes_sample
            return False

    def merge(self, node_pair):
        # node_pair: node_new, node
        # merge node
        self.data.edge_index[self.data.edge_index==node_pair[0]] = node_pair[1]
        node_pair = node_pair.cpu().numpy()
        self.graph = nx.contracted_nodes(self.graph, node_pair[1], node_pair[0])
        try:
            self.node_par2s.remove(node_pair[0])
        except:
            pass

    def update(self, node_pair, clause_num, flag_intercmt=True):
        # node_pair: node_new, node
        if node_pair[0] in self.core_indexes['clause'] or node_pair[1] in self.core_indexes['clause']:
            print(f'no change on clause {node_pair[0]} and {node_pair[1]}')
            return self.resample(clause_num, flag_intercmt)
        self.merge(node_pair)
        return self.resample(clause_num, flag_intercmt)


    def get_graph(self):
        graph = self.graph.copy()
        graph.remove_nodes_from(self.nodes_community)
        return graph
