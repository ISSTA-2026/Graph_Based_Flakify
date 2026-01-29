import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import RelGraphConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from transformers import AutoModel, AutoConfig

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

class GatedRGCNClassifier_OnlyConcat(nn.Module):
    def __init__(self, in_dim, num_rels, hidden_dim, out_dim,
                 num_layers, dropout, lm_model_name):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(nn.Linear(784, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(dropout))

        # GatedRGCN layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GatedRGCNLayer(hidden_dim, hidden_dim, num_rels, activation=F.gelu, dropout=dropout))
        
        model_name = lm_model_name
        model_config = AutoConfig.from_pretrained(model_name, return_dict=True, output_hidden_states=True)
        self.codebert = AutoModel.from_pretrained(model_name, config=model_config)

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
            nn.Dropout(dropout),
            nn.Linear(768*2, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, seq, mask, g, type_vec, code_vec, edge_type):
        h = torch.cat([type_vec, code_vec], dim=-1)
        h = self.input_proj(h)
        for layer in self.layers:
            h = layer(g, h, edge_type)
        g.ndata['h'] = h
        graph_repr_batch = self.readout(g, h)  
        codebert_output = self.codebert(input_ids=seq, attention_mask=mask)
        codebert_cls = codebert_output.last_hidden_state[:, 0, :] 
        feats = torch.cat([codebert_cls, graph_repr_batch], dim=-1)
        output = self.classifier(feats)
        return self.log_softmax(output)