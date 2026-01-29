import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import RelGraphConv, GlobalAttentionPooling
from transformers import AutoModel, AutoConfig


# -----------------------------
# R-GCN layer
# -----------------------------
class GatedRGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, activation, dropout=0.0):
        super().__init__()
        self.rgcn = RelGraphConv(
            in_feat=in_dim, out_feat=out_dim,
            num_rels=num_rels, regularizer="basis",
            num_bases=None, activation=None,
            self_loop=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h, edge_type):
        h_out = self.rgcn(g, h, edge_type)
        h_out = self.norm(h_out)
        return self.dropout(self.activation(h_out))


class MaskedGatedPooling(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, X, mask): 
        scores = self.gate(X).squeeze(-1)
        all_false = (~mask).all(dim=1, keepdim=True)
        safe_mask = mask.clone()
        if all_false.any():
            safe_mask[all_false.squeeze(1), 0] = True
        scores = scores.masked_fill(~safe_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (X * attn.unsqueeze(-1)).sum(dim=1)
        out = torch.where(all_false, torch.zeros_like(out), out)
        return out


class _BiCrossBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, ffn_mult=4):
        super().__init__()
        # T <- H
        self.t_norm1 = nn.LayerNorm(hidden_dim)
        self.t_attn  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.t_norm2 = nn.LayerNorm(hidden_dim)
        self.t_ffn   = nn.Sequential(
            nn.Linear(hidden_dim, ffn_mult * hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_mult * hidden_dim, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)
        # H <- T
        self.h_norm1 = nn.LayerNorm(hidden_dim)
        self.h_attn  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.h_norm2 = nn.LayerNorm(hidden_dim)
        self.h_ffn   = nn.Sequential(
            nn.Linear(hidden_dim, ffn_mult * hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_mult * hidden_dim, hidden_dim),
        )

    def forward(self, T, T_mask, H, H_mask):
        kpm_H = ~H_mask
        kpm_T = ~T_mask

        # Text <- Graph
        T_attn, _ = self.t_attn(query=self.t_norm1(T), key=H, value=H, key_padding_mask=kpm_H)
        T = T + self.drop(T_attn)
        T_ffn = self.t_ffn(self.t_norm2(T))
        T = T + self.drop(T_ffn)
        T = T.masked_fill(~T_mask.unsqueeze(-1), 0)

        # Graph <- Text
        H_attn, _ = self.h_attn(query=self.h_norm1(H), key=T, value=T, key_padding_mask=kpm_T)
        H = H + self.drop(H_attn)
        H_ffn = self.h_ffn(self.h_norm2(H))
        H = H + self.drop(H_ffn)
        H = H.masked_fill(~H_mask.unsqueeze(-1), 0)

        return T, H


class BiCrossEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=8, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_BiCrossBlock(hidden_dim, num_heads, dropout, ffn_mult) for _ in range(num_layers)]
        )

    def forward(self, H_batched, H_mask, T_batched, T_mask):
        T, H = T_batched, H_batched
        for blk in self.blocks:
            T, H = blk(T, T_mask, H, H_mask)
        return H, T


class GatedRGCNClassifier_CrossAttention(nn.Module):

    def __init__(self,
                 in_dim,          
                 num_rels,
                 hidden_dim,      
                 out_dim,
                 lm_model_name,
                 num_gnn_layers=4,
                 num_cross_layers=2,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # R-GCN
        self.layers = nn.ModuleList([
            GatedRGCNLayer(hidden_dim, hidden_dim, num_rels, activation=F.gelu, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])

        # raw graph readout (GAP)
        self.graph_readout = GlobalAttentionPooling(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

        cfg = AutoConfig.from_pretrained(lm_model_name, return_dict=True, output_hidden_states=True)
        self.codebert = AutoModel.from_pretrained(lm_model_name, config=cfg)
        assert cfg.hidden_size == hidden_dim, \
            f"hidden_dim({hidden_dim}) must match LM hidden size ({cfg.hidden_size})"

        # Cross-Attention encoder
        self.cross_encoder = BiCrossEncoder(
            hidden_dim, num_layers=num_cross_layers, num_heads=num_heads, dropout=dropout
        )

        self.cross_graph_pool = MaskedGatedPooling(hidden_dim, dropout=dropout)
        self.cross_text_pool  = MaskedGatedPooling(hidden_dim, dropout=dropout)

        self.norm_raw_graph  = nn.LayerNorm(hidden_dim)
        self.norm_raw_text   = nn.LayerNorm(hidden_dim)
        self.norm_cross_graph = nn.LayerNorm(hidden_dim)
        self.norm_cross_text  = nn.LayerNorm(hidden_dim)

        # Classifier（4D → out）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    @staticmethod
    def _pad_graph_nodes(g, node_h):

        graphs = dgl.unbatch(g)
        B = len(graphs)
        H = node_h.size(-1)
        counts = [gg.num_nodes() for gg in graphs]
        maxN = max(counts) if len(counts) > 0 else 1

        out = node_h.new_zeros(B, maxN, H)
        mask = torch.zeros(B, maxN, dtype=torch.bool, device=node_h.device)
        start = 0
        for i, N in enumerate(counts):
            if N > 0:
                out[i, :N] = node_h[start:start + N]
                mask[i, :N] = True
            else:
                mask[i, 0] = True 
            start += N
        return out, mask

    def forward(self, seq, mask, g, type_vec, code_vec, edge_type):
        # ---- Graph encode ----
        h_in = torch.cat([type_vec, code_vec], dim=-1)   # [sum_N, in_dim]
        h = self.input_proj(h_in)                        # -> [sum_N, D]
        for layer in self.layers:
            h = layer(g, h, edge_type)

        # raw graph
        with g.local_scope():
            g.ndata['h'] = h
            raw_graph = self.graph_readout(g, g.ndata['h'])  # [B, D]

        # ---- Text encode ----
        out = self.codebert(input_ids=seq, attention_mask=mask)
        T = out.last_hidden_state                         # [B, L, D]
        raw_text = T[:, 0, :]                             # [CLS]
        text_mask = mask.bool()                           # [B, L]

        # ---- Pack graph nodes & Cross-Attn ----
        H_batched, graph_mask = self._pad_graph_nodes(g, h)     # [B, Nmax, D], [B, Nmax]
        H_ref, T_ref = self.cross_encoder(H_batched, graph_mask, T, text_mask)

        # ---- Pool after cross ----
        cross_graph = self.cross_graph_pool(H_ref, graph_mask)  # [B, D]
        cross_text  = self.cross_text_pool(T_ref, text_mask)    # [B, D]

        # ---- Normalize & fuse ----
        raw_graph  = self.norm_raw_graph(raw_graph)
        raw_text   = self.norm_raw_text(raw_text)
        cross_graph = self.norm_cross_graph(cross_graph)
        cross_text  = self.norm_cross_text(cross_text)

        feats = torch.cat([cross_graph, cross_text, raw_graph, raw_text], dim=-1)  # [B, 4*D]
        logits = self.classifier(feats)
        return self.log_softmax(logits)
