import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.data import Data
import torch_geometric.datasets
# from torch_geometric.datasets import *
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
# from torch_geometric.nn import GCNConv
from torch.nn import init
from sklearn.metrics import roc_auc_score
import pickle
import os.path
import random
from random import shuffle

import pdb
import copy
import networkx as nx
import numpy as np
import os
from tensorboardX import SummaryWriter

from model import *
from utils import *
# from data import *
import core

import logging
import sys

from joblib import Parallel, delayed
import multiprocessing



def train(args, loader_train, loader_test, model, optimizer,
          writer_train, writer_test, device, model_label, save_dir='model/'):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    loss_func = nn.BCEWithLogitsLoss()
    out_act = nn.Sigmoid()
    for epoch in range(args.epoch_num):
        loss_train = 0
        auc_train = 0
        acc_train = 0
        grad_norm_min = 0
        grad_norm_max = 0
        grad_norm_mean = 0
        pred_np_min = 0
        pred_np_max = 0
        pred_np_mean = 0
        counter = 0
        time_model = 0
        time_total = time.time()
        for data in loader_train:
            time1 = time.time()
            model.train()
            optimizer.zero_grad()
            data.to(device)

            out = model(data)

            edge_mask = torch.cat((data.node_index_positive, data.node_index_negative), dim=-1)
            nodes_first = torch.index_select(out, 0, edge_mask[0, :].long().to(device))
            nodes_second = torch.index_select(out, 0, edge_mask[1, :].long().to(device))
            pred = torch.sum(nodes_first * nodes_second, dim=-1)
            label_positive = torch.ones([data.node_index_positive.shape[1], ], dtype=pred.dtype)
            label_negative = torch.zeros([data.node_index_negative.shape[1], ], dtype=pred.dtype)
            label = torch.cat((label_positive, label_negative)).to(device)
            loss = loss_func(pred, label)
            loss_train += loss.cpu().data.numpy()
            label_np = label.flatten().cpu().numpy()
            pred_np = out_act(pred).flatten().data.cpu().numpy()
            pred_np_min += pred_np.min()
            pred_np_max += pred_np.max()
            pred_np_mean += pred_np.mean()
            auc_train += roc_auc_score(label_np, pred_np)
            acc_train += np.mean((pred_np>0.5).astype(int)==label_np)
            # update
            loss.backward()
            optimizer.step()

            grad_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().cpu().numpy())
            grad_norm_min += min(grad_norms)
            grad_norm_max += max(grad_norms)
            grad_norm_mean += sum(grad_norms)/len(grad_norms)
            counter += 1
            time2 = time.time()
            time_model += time2-time1
        time_total = time.time() - time_total
        if epoch % args.epoch_log == 0:
            print('[Epoch {}/{}] Train time per epoch: total {:.4f}, model {:.4f}'.format(epoch, args.epoch_num, time_total,time_model))

        loss_train /= counter
        auc_train /= counter
        acc_train /= counter
        grad_norm_min /= counter
        grad_norm_max /= counter
        grad_norm_mean /= counter
        pred_np_min /= counter
        pred_np_max /= counter
        pred_np_mean /= counter
        writer_train.add_scalar('auc', auc_train, epoch)
        writer_train.add_scalar('loss', loss_train, epoch)
        writer_train.add_scalar('grad_norm_min', grad_norm_min, epoch)
        writer_train.add_scalar('grad_norm_max', grad_norm_max, epoch)
        writer_train.add_scalar('grad_norm_mean', grad_norm_max, epoch)
        writer_train.add_scalar('pred_np_min', pred_np_min, epoch)
        writer_train.add_scalar('pred_np_max', pred_np_max, epoch)
        writer_train.add_scalar('pred_np_mean', pred_np_mean, epoch)

        if epoch % args.epoch_save == 0:
            torch.save(model.state_dict(), save_dir+args.name+"_"+str(epoch)+"_"+model_label)
            print('model saved!')

        if epoch % args.epoch_log == 0 and epoch>=args.epoch_test:
            # test
            loss_test = 0
            auc_test = 0
            acc_test = 0
            counter = 0
            for data in loader_test:
                # evaluate
                data.to(device)

                out = model(data)

                edge_mask = torch.cat((data.node_index_positive, data.node_index_negative), dim=-1)
                nodes_first = torch.index_select(out, 0, edge_mask[0, :].long().to(device))
                nodes_second = torch.index_select(out, 0, edge_mask[1, :].long().to(device))
                pred = torch.sum(nodes_first * nodes_second, dim=-1)
                label_positive = torch.ones([data.node_index_positive.shape[1], ], dtype=pred.dtype)
                label_negative = torch.zeros([data.node_index_negative.shape[1], ], dtype=pred.dtype)
                label = torch.cat((label_positive, label_negative)).to(device)
                loss_test += loss_func(pred, label).cpu().data.numpy()
                label_np = label.flatten().cpu().numpy()
                pred_np = out_act(pred).flatten().data.cpu().numpy()
                auc_test += roc_auc_score(label_np, pred_np)
                acc_test += np.mean((pred_np > 0.5).astype(int) == label_np)
                counter += 1
            loss_test /= counter
            auc_test /= counter
            acc_test /= counter
            print('Model {}'.format(args.model), epoch, 'Loss {:.4f}'.format(loss_train),
                  'Train AUC: {:.4f}'.format(auc_train), 'Test AUC: {:.4f}'.format(auc_test),
                'Train ACC: {:.4f}'.format(acc_train), 'Test ACC: {:.4f}'.format(acc_test))

            writer_test.add_scalar('auc', auc_test, epoch)
            writer_test.add_scalar('loss', loss_test, epoch)

        f_out = open("progress.txt", "w")
        f_out.write(str(epoch))
        f_out.close()

    return model





def test(args, generator_list, model_intercmt, model_incmt, clause_num_intercmt, clause_num_incmt, file_names, repeat=0, outdir='graphs/'):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    clause_num_intercmt = np.repeat(clause_num_intercmt, repeat)
    clause_num_incmt = np.repeat(clause_num_incmt, repeat)
    # generate graph batch
    for i,generator in enumerate(generator_list):
        time0 = time.time()
        time_model = 0
        generator.reset()
        while True:
            time1 = time.time()
            flag_intercmt = len(generator.node_par2s) <= clause_num_incmt[i]
            # flag_intercmt = False
            # if len(generator.node_par2s) % 10 == 0: print(len(generator.node_par2s), flag_intercmt)
            out = model_intercmt(generator.data) if flag_intercmt else model_incmt(generator.data)

            nodes_first = torch.index_select(out, 0, generator.data.node_index[0, :])
            nodes_second = torch.index_select(out, 0, generator.data.node_index[1, :])
            pred = torch.sum(nodes_first * nodes_second, dim=-1)
            try:
                pred_id = torch.argmax(pred).data
            except RuntimeError:
                print(f'error on pred length: {len(pred)}')
                assert(False)
            time2 = time.time()
            time_model += time2 - time1
            exit_flag = generator.update(generator.data.node_index[:, pred_id], clause_num_intercmt[i], flag_intercmt)
            if exit_flag:
                break
        time3 = time.time()
        print('Generate time for 1 graph: total {:.4f}, model {:.4f}'.format(
            time3 - time0, time_model))
        logging.info('Generate time for 1 graph: total {:.4f}, model {:.4f}'.format(
            time3 - time0, time_model))


    graphs = [generator.get_graph() for generator in generator_list]
    nodes_par1s = [generator.nodes_par1 for generator in generator_list]
    core_indexes = [generator.core_indexes for generator in generator_list]

    ### scramble unsat-core
    if args.core_flag:
        if args.model in ["SAGE", "GCN"]:
            is_LCG = True
        else:
            is_LCG = False
        scramble_ratio = args.scramble_ratio.split('-')
        scramble_ratio = [int(r) * 0.1 for r in scramble_ratio]
        graphs = core.scramble_graphs(graphs, nodes_par1s, core_indexes, scramble_ratio, is_LCG)
        core.output_core(graphs, nodes_par1s, core_indexes, file_names, f'formulas/{args.data_name}_core/', is_LCG)


    if args.model in ["SAGE", "GCN"]: 
        save_graph_list(graphs, f'{outdir}{args.name}_{str(args.alpha)}_{str(args.clause_ratio)}_{str(args.repeat)}.dat', file_names)
    elif args.model in ["EGNN", "ECC"]:
        save_graph_list_fg([generator.graph], f'{outdir}{args.name}_{str(args.alpha)}_{str(args.clause_ratio)}_{str(args.repeat)}.dat', file_names)
    node_nums = [graph.number_of_nodes() for graph in graphs]
    edge_nums = [graph.number_of_edges() for graph in graphs]
    print('Num {}, Node {} {} {}, Edge {} {} {}'.format(
        len(graphs), min(node_nums), max(node_nums), sum(node_nums) / len(node_nums), min(edge_nums),
        max(edge_nums), sum(edge_nums) / len(edge_nums)))
    logging.info('Num {}, Node {} {} {}, Edge {} {} {}'.format(
        len(graphs), min(node_nums), max(node_nums), sum(node_nums) / len(node_nums), min(edge_nums),
        max(edge_nums), sum(edge_nums) / len(edge_nums)))

