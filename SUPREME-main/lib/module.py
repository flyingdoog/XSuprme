import os
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv,GINConv,GATConv
from lib.GNNLayers import GSATConv
from collections import Counter

class GCN(torch.nn.Module):
    def __init__(self, in_size=16, hid_size=8, out_size=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb

    
class MLP(torch.nn.Module):
    def __init__(self, in_size=16, hid_size=8, out_size=2):
        super(MLP, self).__init__()
        self.conv1 = torch.nn.Linear(in_size, hid_size)
        self.conv2 = torch.nn.Linear(hid_size, out_size)

    def forward(self, data):
        x_emb = self.conv1(data)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        return x

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_size=16, hid_sizes=[8]):
        super(GCNEncoder, self).__init__()
        sizes = [in_size]+hid_sizes
        self.convs = torch.nn.ModuleList()
        for i in range(1,len(sizes)):
            self.convs.append(GCNConv(sizes[i-1], sizes[i]))
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for layer in self.convs:
            x = F.dropout(x, training=self.training)
            x = layer(x, edge_index)
            x = F.relu(x)
        return x


class GATEncoder(torch.nn.Module):
    def __init__(self, in_size=16, hid_sizes=[8]):
        super(GATEncoder, self).__init__()
        sizes = [in_size]+hid_sizes
        self.convs = torch.nn.ModuleList()
        for i in range(1,len(sizes)):
            self.convs.append(GATConv(sizes[i-1], sizes[i]))
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for layer in self.convs:
            x = F.dropout(x, training=self.training)
            x = layer(x, edge_index)
            x = F.relu(x)
        return x

    
class GSATEncoder(torch.nn.Module):
    def __init__(self, in_size=16, hid_sizes=[8]):
        super(GSATEncoder, self).__init__()
        sizes = [in_size]+hid_sizes
        self.convs = torch.nn.ModuleList()
        for i in range(1,len(sizes)):
            self.convs.append(GSATConv(sizes[i-1], sizes[i]))
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for layer in self.convs:
            x = F.dropout(x, training=self.training)
            x = layer(x, edge_index,tau=1.0)
            x = F.relu(x)
        return x

class SelfAttention(torch.nn.Module):
    def __init__(self, feature_dim, heads=8):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.heads = heads
        self.scale = feature_dim ** -0.5

        self.query = torch.nn.Linear(feature_dim, feature_dim * heads)
        self.key = torch.nn.Linear(feature_dim, feature_dim * heads)
        self.value = torch.nn.Linear(feature_dim, feature_dim * heads)

        self.out = torch.nn.Linear(feature_dim * heads, feature_dim)

    def forward(self, x):
        # x has shape [batch, channel, feature_dim]
        b, n, _ = x.size()

        # Transform x using the query, key, and value linear layers
        queries = self.query(x).view(b, n, self.heads, self.feature_dim)
        keys = self.key(x).view(b, n, self.heads, self.feature_dim)
        values = self.value(x).view(b, n, self.heads, self.feature_dim)

        # Scaled dot-product attention
        attn = (queries @ keys.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ values

        # Reshape and concatenate the multi-head outputs
        out = out.view(b, n, -1)

        # Project the concatenation down to the original number of features
        return self.out(out)

    
    
class xGNN(torch.nn.Module):
    def __init__(self, in_size=16, hid_sizes=[8], out_size=2,nmb_networks=1,encoder='GAT'):
        super(xGNN, self).__init__()
        self.nmb_networks = nmb_networks
        if encoder=='GCN':
            self.gnns = torch.nn.ModuleList([GCNEncoder(in_size,hid_sizes) for _ in range(nmb_networks)])
        elif encoder=='GAT':
            self.gnns = torch.nn.ModuleList([GATEncoder(in_size,hid_sizes) for _ in range(nmb_networks)])
        elif encoder=='GSAT':
            self.gnns = torch.nn.ModuleList([GSATEncoder(in_size,hid_sizes) for _ in range(nmb_networks)])
        else:
            assert 0
            
        # for feature selection
        self.feature_selection = torch.nn.Parameter(torch.ones(nmb_networks))
        self.feature_map = torch.nn.Linear(in_size,hid_sizes[-1])
        self.attention = SelfAttention(hid_sizes[-1],heads=1)
        self.outputlayer = torch.nn.Linear(hid_sizes[-1],out_size)
        
    def forward(self, graphs):
        feature_emb = self.feature_map(graphs[0].x)
        
        embs = [feature_emb]
        for idx,(data,gnn) in enumerate(zip(graphs,self.gnns)):
            emb = gnn(data)
            embs.append(emb)
        stack_embs = torch.stack(embs)
        att_out = self.attention(stack_embs).transpose(0,1)
        read_out = torch.mean(att_out,1)
        
        prob = self.outputlayer(read_out)
        
        return prob
