import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
# from torch_geometric.nn import GCNConv
from torch.nn import init
from torch_scatter import scatter_mean
import pdb

####################### SAGE #############################

class SAGE(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

####################### GCN #############################

class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

####################### EGNN #############################

class EGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(EGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = SAGEConvEGNN(feature_dim, hidden_dim)
        else:
            self.conv_first = SAGEConvEGNN(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([SAGEConvEGNN(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = SAGEConvEGNN(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index, edge_attr)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index, edge_attr)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index, edge_attr)
        x = F.normalize(x, p=2, dim=-1)
        return x

####################### ECC #############################

class ECC(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, num_edge_features=3,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(ECC, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.NNConv(feature_dim, hidden_dim, nn.Sequential(nn.Linear(num_edge_features, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim*hidden_dim)))
        else:
            self.conv_first = tg.nn.NNConv(input_dim, hidden_dim, nn.Sequential(nn.Linear(num_edge_features, input_dim*hidden_dim)))
        self.conv_hidden = nn.ModuleList([tg.nn.NNConv(hidden_dim, hidden_dim, nn.Sequential(nn.Linear(num_edge_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim*hidden_dim))) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.NNConv(hidden_dim, output_dim, nn.Sequential(nn.Linear(num_edge_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim*output_dim)))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index, edge_attr)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index, edge_attr)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index, edge_attr)
        x = F.normalize(x, p=2, dim=-1)
        return x

####################### ConvNNs #############################

class SAGEConvEGNN(tg.nn.MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized. (default: :obj:`False`)
        concat (bool, optional): If set to :obj:`True`, will concatenate
            current node features with aggregated ones. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, normalize=False,
                 concat=True, bias=True, **kwargs):
        super(SAGEConvEGNN, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.concat = concat

        in_channels = 3 * in_channels if concat else in_channels
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.bias, -1, 1)
        torch.nn.init.uniform_(self.weight, -1, 1)
        # uniform(self.weight.size(0), self.weight)
        # uniform(self.weight.size(0), self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        # if not self.concat and torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
            # edge_index, edge_weight = add_remaining_self_loops(
            #     edge_index, edge_weight, 1, x.size(self.node_dim))

        return self.propagate(edge_index, size=size, x=x,
                              edge_attr=edge_attr)


    def message(self, x_j, edge_attr):
        # print("edge_attr:", edge_attr.shape)
        # print(x_j.shape)
        emb = torch.stack([x_j]*edge_attr.shape[1], dim=0)
        # print("emb:", emb.shape)
        edge_attr_aux = edge_attr.permute(1,0)
        # print(emb.shape)
        # Y = edge_attr_aux @ emb
        for i in range(emb.shape[0]):
            emb[i] = emb[i]*edge_attr[:,i].unsqueeze(1).float()

        # Multi-edge agreggation
        if self.concat:
            emb = torch.cat(torch.unbind(emb, dim=0), dim=-1)
        else:
            emb = torch.mean(emb, 0)

        return emb

    def update(self, aggr_out, x):
        aggr_out = torch.matmul(aggr_out, self.weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# class EGNNConv(torch.nn.Module):
#     def __init__(self, in_features, out_features, edge_features_dim=2, bias=True, activation=None, node_dropout=0.4, edge_dropout=0.4,
#                  multi_edge_aggregation='concat', **kwargs):
#         super(EGNNConv, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.activation = activation
#         self.bias = bias
#         self.node_dropout = node_dropout
#         self.edge_dropout = edge_dropout
#         self.multi_edge_aggregation = multi_edge_aggregation

#         self.node_dropout_layer = nn.Dropout(node_dropout)
#         self.edge_dropout_layer = nn.Dropout(edge_dropout)
#         self.embedding_layer = nn.Linear(in_features, out_features, bias=True)

#         if self.bias:
#             if multi_edge_aggregation == 'concat':
#                 self.bias = nn.Parameter(torch.Tensor(out_features*edge_features_dim))
#             else:
#                 self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.bias is not None:
#             torch.nn.init.uniform_(self.bias, -1, 1)

#     def node_agreggate(self, XDW, XDWD, E, ED, out_E):
#         if len(list(E.size())) == len(list(XDW.size())):
#             ED = torch.unsqueeze(ED, 0)
#         else:
#             ED = ED.permute(2, 1, 0)

#         EC = list(ED.size())[0]
#         XDWD = torch.stack([XDWD]*EC, dim=0)
#         Y = ED @ XDWD 

#         return Y, out_E

#     def forward(self, data):
#         X, E = data
#         IC = list(E.size())
#         out_E = E

#         # Node dropout
#         XD = self.node_dropout_layer(X)

#         # Node feature embedding
#         XDW = self.embedding_layer(XD)

#         # Embeded node dropout
#         XDWD = self.node_dropout_layer(XDW)

#         # Edge dropout
#         ED = self.edge_dropout_layer(E)

#         # Node agreggation
#         Y, out_E = self.node_agreggate(XDW, XDWD, E, ED, out_E)

#         # Multi-edge agreggation
#         if self.multi_edge_aggregation == 'concat':
#             Y = torch.cat(torch.unbind(Y, dim=0), dim=-1)
#         else:
#             Y = torch.mean(Y, 0)

#         # Add bias
#         Y = Y + self.bias

#         # Activation
#         if self.activation:
#             Y = self.activation(Y)

#         return (Y, out_E)


# class EGNN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, edge_features_dim, layer_num=2, dropout=True, **kwargs):
#         super(EGNN, self).__init__()
#         self.layer_num = layer_num
#         self.dropout = dropout

#         self.conv_first = EGNNConv(input_dim, hidden_dim, edge_features_dim=edge_features_dim, activation=torch.nn.ELU())
#         self.conv_hidden = nn.ModuleList([EGNNConv(hidden_dim*edge_features_dim, hidden_dim, edge_features_dim=edge_features_dim, activation=torch.nn.ELU()) for i in range(layer_num - 2)])
#         self.conv_out = EGNNConv(hidden_dim*edge_features_dim, output_dim, edge_features_dim=edge_features_dim, multi_edge_aggregation='mean')

#     def forward(self, data):
#         x, edges = data.x, data.adj_matrix
#         x = x.float()
#         edges = edges.float()
#         x, edges = self.conv_first((x,edges))
        
#         for i in range(self.layer_num-2):
#             x, edges = self.conv_hidden[i]((x,edges))

#         x, edges = self.conv_out((x,edges))
#         return x