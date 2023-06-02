import os
import networkx as nx
import random
import numpy as np


# adding unsat-core info to given graph
def mark_unsat_core(graph, core_file, num_vars, num_clause, is_LCG=True):
    core_file = open(core_file)
    content = core_file.readlines()
    while content[0].split()[0] == 'c':
        content = content[1:]
    while len(content[-1].split()) <= 1:
        content = content[:-1]
    
    core_index = dict()
    core_index['clause'] = []
    core_index['vars'] = set()
    
    # parameters
    parameters = content[0].split()
    assert(parameters[0] == 'p')
    core_index['num_vars'] = int(parameters[2])
    core_index['num_clause'] = int(parameters[3])

    # mark line by line
    formula = content[1:]
    for i in range(len(formula)):
        if is_LCG:
            lits = [(int(x)-1 if int(x) > 0 else (abs(int(x)) + num_vars-1)) for x in formula[i].split()[:-1]]
            clause = set(graph[lits[0]])
            for lit in lits[1:]:
                clause = clause & set(graph[lit])
            tmp = clause.copy()
            for node in tmp:
                if node > 2*num_vars + num_clause:
                    clause.remove(node)
                elif len(graph[node]) != len(lits):
                    for nodelits in graph[node]:
                        if nodelits not in lits and nodelits < 2*num_vars + num_clause:
                            clause.remove(node)
                            break
            assert(len(list(clause)) == 1)
            clause = list(clause)[0]
            graph.nodes[clause]['core'] = True
            core_index['clause'].append(clause)
            tmp = set([abs(int(x))-1 for x in formula[i].split()[:-1]])
            core_index['vars'] = core_index['vars'] | tmp
        else:
            vars = [abs(int(x))-1 for x in formula[i].split()[:-1]]
            signs = [([1,0,0] if int(x) > 0 else [0,1,0]) for x in formula[i].split()[:-1]]
            clause = set()
            for cls in graph[vars[0]]:
                if graph.edges[vars[0], cls]['features'] == signs[0]:
                    clause.add(cls)
            for idx in range(len(vars)):
                tmp = set()
                for cls in graph[vars[idx]]:
                    if graph.edges[vars[idx], cls]['features'] == signs[idx]:
                        tmp.add(cls)
                clause = clause & tmp
            tmp = clause.copy()
            for node in tmp:
                if node > num_vars + num_clause:
                    clause.remove(node)
                elif len(graph[node]) != len(vars):
                    clause.remove(node)
            assert(len(list(clause)) == 1)
            clause = list(clause)[0]
            graph.nodes[clause]['core'] = True
            core_index['clause'].append(clause)
            core_index['vars'] = core_index['vars'] | set(vars)
        
    core_index['vars'] = list(core_index['vars'])
    return core_index


def scramble_graphs(graphs, node_pars1, core_indexes, scramble_ratio, is_LCG=True):
    for i in range(len(graphs)):
        if is_LCG: 
            graphs[i] = scrambling_core_lcg(graphs[i], len(node_pars1[i])//2, core_indexes[i], scramble_ratio)
        else:
            graphs[i] = scrambling_core_fg(graphs[i], len(node_pars1[i]), core_indexes[i], scramble_ratio)
    return graphs

# scrambling unsat-core
def scrambling_core_lcg(LCG, num_vars, core_index, scramble_ratio):
    # scrambling_ratio: a list with 3 elements
    # num of permuting variable, permuting clause and flipping variable
    
    for i in range(int(scramble_ratio[0] * len(core_index['vars']))):
        var1 = random.choice(core_index['vars'])
        var2 = var1
        while var2 == var1:
            var2 = random.choice(core_index['vars'])
        LCG = permute_variable_lcg(var1, var2, LCG, num_vars)
    
    for i in range(int(scramble_ratio[1] * len(core_index['clause']))):
        clause1 = random.choice(core_index['clause'])
        clause2 = clause1
        while clause2 == clause1:
            clause2 = random.choice(core_index['clause'])
        LCG = permute_clause(clause1, clause2, LCG, num_vars)
        
    for i in range(int(scramble_ratio[2] * len(core_index['vars']))):
        var = random.choice(core_index['vars'])
        LCG = flip_variable_lcg(var, LCG, num_vars)
        
    return LCG



# scrambling unsat-core
def scrambling_core_fg(LCG, num_vars, core_index, scramble_ratio):
    # scrambling_ratio: a list with 3 elements
    # num of permuting variable, permuting clause and flipping variable
    
    for i in range(int(scramble_ratio[0] * len(core_index['vars']))):
        var1 = random.choice(core_index['vars'])
        var2 = var1
        while var2 == var1:
            var2 = random.choice(core_index['vars'])
        LCG = permute_variable_fg(var1, var2, LCG, num_vars)
    
    for i in range(int(scramble_ratio[1] * len(core_index['clause']))):
        clause1 = random.choice(core_index['clause'])
        clause2 = clause1
        while clause2 == clause1:
            clause2 = random.choice(core_index['clause'])
        LCG = permute_clause(clause1, clause2, LCG, num_vars)
        
    for i in range(int(scramble_ratio[2] * len(core_index['vars']))):
        var = random.choice(core_index['vars'])
        LCG = flip_variable_fg(var, LCG, num_vars)
    
    return LCG
    
    
def output_core(graphs, node_pars1, core_indexes, graph_names, out_dir, is_LCG):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for idx, graph in enumerate(graphs):
        node_par1 = node_pars1[idx]
        core_dir = core_indexes[idx]
        graph_name = graph_names[idx]
        # graph_name = graph_name.strip('_lcg_edge_list')
        # graph_name = graph_name.strip('_fg_edge_list')
        
        if is_LCG:
            num_var = len(node_par1) // 2
            core_clauses = []
            for clause_node in core_dir['clause']:
                assert (clause_node >= num_var * 2)
                neighbors = list(graph.neighbors(clause_node))
                clause = ""
                assert(len(neighbors) > 0)
                for lit in neighbors:
                    if lit < num_var:
                        clause += "{} ".format(lit + 1)
                    else:
                        assert(lit < 2 * num_var)
                        clause += "{} ".format(-(lit - num_var + 1))
                clause += "0\n"
                core_clauses.append(clause)
        else:
            #TODO FG
            pass
    
        save_name = f'{out_dir}/{graph_name}_core'
        with open(save_name, 'w') as out_file:
            out_file.write("p cnf {} {} \n".format(num_var, len(core_clauses)))
            for clause in core_clauses:
                out_file.write(clause)
    
    return
    
    

def permute_variable_lcg(var1, var2, LCG, num_vars):
    neg_var1 = var1 + num_vars
    neg_var2 = var2 + num_vars
    mapping = {}
    mapping[var1] = var2
    mapping[var2] = var1
    mapping[neg_var1] = neg_var2
    mapping[neg_var2] = neg_var1
    LCG = nx.relabel_nodes(LCG, mapping)
    return LCG


def permute_variable_fg(var1, var2, LCG, num_vars):
    mapping = {}
    mapping[var1] = var2
    mapping[var2] = var1
    LCG = nx.relabel_nodes(LCG, mapping)
    return LCG


def permute_clause(clause1, clause2, LCG, num_vars):
    mapping = {}
    mapping[clause1] = clause2
    mapping[clause2] = clause1
    LCG = nx.relabel_nodes(LCG, mapping)
    return LCG
    

def flip_variable_lcg(var, LCG, num_vars):
    assert(var < num_vars)
    neg_var = var + num_vars
    mapping = {}
    mapping[var] = neg_var
    mapping[neg_var] = var
    LCG = nx.relabel_nodes(LCG, mapping)
    return LCG


def flip_variable_fg(var, LCG, num_vars):
    assert(var < num_vars)
    neighbour_clause = list(LCG[var])
    for cls in neighbour_clause:
        if LCG.edges[var, cls]['features'] == [1,0,0]:
            LCG.edges[var, cls]['features'] = [0,1,0]
        elif LCG.edges[var, cls]['features'] == [0,1,0]:
            LCG.edges[var, cls]['features'] = [1,0,0]
    return LCG