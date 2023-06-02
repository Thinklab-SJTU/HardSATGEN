import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from networkx.algorithms.community import greedy_modularity_communities

from args import make_args
from train import train
from model import GCN, SAGE
from data_lcg import Dataset_sat_intercmt, Dataset_sat_incmt, load_graphs_vig, load_graphs_lcg

### args
args = make_args()
print(args)
np.random.seed(123)

args.name = '{}_{}_core{}'.format(args.data_name, args.model, args.core_flag)
writer_train = SummaryWriter(comment=args.name+'train')
writer_test = SummaryWriter(comment=args.name+'test')
args.graphs_save_path = 'graphs/'

### set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda) 
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

### load data
graphs_train_vig = load_graphs_vig(data_dir=f'dataset/{args.data_name}/', stats_dir='dataset/')
community_list = []
for G in graphs_train_vig:
    c = greedy_modularity_communities(G)
    community_list.append(c)
print([len(c) for c in community_list])

graphs_train, nodes_par1s_train, nodes_par2s_train, nodes_communitys_train, core_indexes_train, file_names = \
    load_graphs_lcg(data_dir=f'dataset/{args.data_name}/', stats_dir='dataset/', community_list=community_list, core_flag=args.core_flag)
node_nums = [graph.number_of_nodes() for graph in graphs_train]
edge_nums = [graph.number_of_edges() for graph in graphs_train]
print('Num {}, Node {} {} {}, Edge {} {} {}'.format(
    len(graphs_train),min(node_nums),max(node_nums),sum(node_nums)/len(node_nums),min(edge_nums),max(edge_nums),sum(edge_nums)/len(edge_nums)))



dataset_train_intercmt = Dataset_sat_intercmt(graphs_train, nodes_par1s_train, nodes_par2s_train, nodes_communitys_train, 
                                     community_list, args.core_flag, core_indexes_train, epoch_len=5000, yield_prob=args.yield_prob, speedup=True)
dataset_test_intercmt = Dataset_sat_intercmt(graphs_train, nodes_par1s_train, nodes_par2s_train, nodes_communitys_train, 
                                     community_list, args.core_flag, core_indexes_train, epoch_len=1000, yield_prob=args.yield_prob, speedup=False)

loader_train_intercmt = DataLoader(dataset_train_intercmt, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_num)
loader_test_intercmt = DataLoader(dataset_test_intercmt, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_num)

graphs_train_incmt, nodes_par1s_train_incmt, nodes_par2s_train_incmt, nodes_communitys_train_incmt = dataset_train_intercmt.get_graph_incmt()

dataset_train_incmt = Dataset_sat_incmt(graphs_train_incmt, nodes_par1s_train_incmt, nodes_par2s_train_incmt, nodes_communitys_train_incmt,
                                        community_list, args.core_flag, core_indexes_train, epoch_len=5000, yield_prob=args.yield_prob, speedup=True)
dataset_test_incmt = Dataset_sat_incmt(graphs_train_incmt, nodes_par1s_train_incmt, nodes_par2s_train_incmt, nodes_communitys_train_incmt,
                                       community_list, args.core_flag, core_indexes_train, epoch_len=1000, yield_prob=args.yield_prob, speedup=False)

loader_train_incmt = DataLoader(dataset_train_incmt, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_num)
loader_test_incmt = DataLoader(dataset_test_incmt, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_num)

input_dim = 4
model_intercmt = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                    hidden_dim=args.hidden_dim, output_dim=args.output_dim,
                    feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
model_incmt = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim, output_dim=args.output_dim,
                feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)


optimizer_intercmt = torch.optim.Adam(model_intercmt.parameters(), lr=args.lr, weight_decay=5e-4)
optimizer_incmt = torch.optim.Adam(model_incmt.parameters(), lr=args.lr, weight_decay=5e-4)

train(args, loader_train_intercmt, loader_test_intercmt, model_intercmt, optimizer_intercmt, writer_train, writer_test, device, "intercmt")
train(args, loader_train_incmt, loader_test_incmt, model_incmt, optimizer_incmt, writer_train, writer_test, device, "incmt")




