import numpy as np
import os
import csv
import torch
from networkx.algorithms.community import greedy_modularity_communities

from args import make_args
from utils import save_graph_list
from train import test
from model import GCN, SAGE
from data_lcg import Dataset_sat_intercmt, Dataset_sat_incmt, graph_generator, load_graphs_vig, load_graphs_lcg

### args
args = make_args()
print(args)
np.random.seed(123)
args.name = '{}_{}_core{}'.format(args.data_name, args.model, args.core_flag)

### set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

### create graph templates to initailize generator
if not os.path.isdir('graphs/'):
    os.mkdir('graphs/')
template_name = f'graphs/template_small_{args.data_name}.dat'
train_name = f'graphs/train_small_{args.data_name}.dat'

### load data
graphs_train_vig = load_graphs_vig(data_dir=f'dataset/{args.data_name}/', stats_dir='dataset/')
community_list = []
for G in graphs_train_vig:
    c = greedy_modularity_communities(G)
    community_list.append(c)
print([len(c) for c in community_list])

# if args.recompute_template or not os.path.isfile(template_name):
graphs_train, nodes_par1s_train, nodes_par2s_train, nodes_communitys_train, core_indexes_train, file_names = \
    load_graphs_lcg(data_dir=f'dataset/{args.data_name}/', stats_dir='dataset/', community_list=community_list, core_flag=args.core_flag)

save_graph_list(graphs_train, train_name, file_names, has_par=True, nodes_par1_list=nodes_par1s_train, nodes_par2_list=nodes_par2s_train)
print('Train graphs saved!', len(graphs_train))
node_nums = [graph.number_of_nodes() for graph in graphs_train]
edge_nums = [graph.number_of_edges() for graph in graphs_train]
print('Num {}, Node {} {} {}, Edge {} {} {}'.format(
    len(graphs_train), min(node_nums), max(node_nums), sum(node_nums) / len(node_nums), min(edge_nums),
    max(edge_nums), sum(edge_nums) / len(edge_nums)))


dataset_train_intercmt = Dataset_sat_intercmt(graphs_train, nodes_par1s_train, nodes_par2s_train, nodes_communitys_train, 
                                     community_list, args.core_flag, core_indexes_train, epoch_len=5000, yield_prob=args.yield_prob, speedup=True)
graphs_train_incmt, nodes_par1s_train_incmt, nodes_par2s_train_incmt, nodes_communitys_train_incmt = dataset_train_intercmt.get_graph_incmt()

dataset_train_incmt = Dataset_sat_incmt(graphs_train_incmt, nodes_par1s_train_incmt, nodes_par2s_train_incmt, nodes_communitys_train_incmt,
                                        community_list, args.core_flag, core_indexes_train, epoch_len=5000, yield_prob=args.yield_prob, speedup=False)

graph_templates, nodes_par1s, nodes_par2s, nodes_communitys = dataset_train_incmt.get_template()

clause_num_intercmt = [len(x) for x in nodes_par2s_train]
clause_num_incmt = [len(x) for x in nodes_par2s_train_incmt]
print(clause_num_intercmt)
print(clause_num_incmt)
print([len(x) for x in nodes_par2s])


print('Template num', len(graph_templates))


# load stats
stat_file = 'lcg_stats.csv'
with open('dataset/' + stat_file) as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    stats = []
    for stat in data:
        stats.append(stat)
generator_list = []
tmp_names = file_names.copy()
file_names = []
for i in range(len(graph_templates)):
    for j in range(args.repeat):
        generator_list.append(graph_generator(
            graph_templates[i], nodes_par1s[i], nodes_par2s[i], nodes_communitys[i], community_list[i], args.core_flag, core_indexes_train[i], sample_size=args.sample_size, device=device))
        tmp_name = tmp_names[i].strip('_lcg_edge_list')
        tmp_name = tmp_names[i].strip('_fg_edge_list')
        file_names.append(f'{tmp_names[i]}_repeat{j}')


input_dim = 4
model_intercmt = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim, output_dim=args.output_dim,
                feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
model_incmt = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim, output_dim=args.output_dim,
                feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
model_intercmt.load_state_dict(torch.load('model/'+args.name+"_"+str(args.epoch_load)+"_intercmt", map_location=device))
model_intercmt.to(device).eval()
model_incmt.load_state_dict(torch.load('model/'+args.name+"_"+str(args.epoch_load)+"_incmt", map_location=device))
model_incmt.to(device).eval()
print('Models loaded!')

clause_num_intercmt = np.array(clause_num_intercmt) * args.clause_ratio
clause_num_incmt = np.array(clause_num_incmt) * args.clause_ratio
alpha = args.alpha
clause_num_intercmt = (1 - alpha) * clause_num_incmt + alpha * clause_num_intercmt
clause_num_intercmt, clause_num_incmt = clause_num_intercmt.astype(int), clause_num_incmt.astype(int)

test(args, generator_list, model_intercmt, model_incmt, clause_num_intercmt, 
     clause_num_incmt, file_names, repeat=args.repeat)




