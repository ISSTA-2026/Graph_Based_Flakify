import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import RelGraphConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling

class GatedRGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, activation, dropout=0.0):
        super().__init__()
        self.rgcn = RelGraphConv(
            in_feat=in_dim,
            out_feat=out_dim,
            num_rels=num_rels,
            regularizer="basis",
            num_bases=None,
            activation=None,
            self_loop=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h, edge_type):
        h_out = self.rgcn(g, h, edge_type)
        h_out = self.norm(h_out)
        return self.dropout(self.activation(h_out))

class GatedRGCNClassifier_CNN_Improved(nn.Module):
    def __init__(self, in_dim, num_rels, hidden_dim, out_dim,
                 num_layers, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(nn.Linear(784, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(dropout))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GatedRGCNLayer(hidden_dim, hidden_dim, num_rels, activation=F.gelu, dropout=dropout))

        # Enhanced Attention Pooling
        self.readout = GlobalAttentionPooling(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, g, type_vec, code_vec, edge_type):
        h = torch.cat([type_vec, code_vec], dim=-1)
        h = self.input_proj(h)
        for layer in self.layers:
            h = layer(g, h, edge_type)
        g.ndata['h'] = h
        graph_repr_batch = self.readout(g, h)  
        output = self.classifier(graph_repr_batch)
        return self.log_softmax(output)