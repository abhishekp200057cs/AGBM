

# Multi-Center Medication Recommendation via Adaptive Graph-Based Modeling (AGBM)


 


import os
import pickle
import ast
import re
import time
import json
import random
import torch
import dgl
import dgl.function as fn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fpdf import FPDF
from sklearn.metrics import f1_score, jaccard_score, average_precision_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import Counter, defaultdict
from itertools import combinations, product
#from dgl.nn import GATConv
#from dgl.nn import GATConv
from dgl.nn import GATv2Conv

from collections import defaultdict
import math



from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup


from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import textwrap
from PIL import Image, ImageDraw, ImageFont




# ---------------------------
# Config / Hyperparameters
# ---------------------------
SEEDS = [42, 43, 44, 45, 46]


HOSPITAL_IDS = [79, 141, 142, 143, 144, 146, 148, 154, 157, 165, 167, 183, 184, 197,
                198, 199, 202, 206, 208, 215, 217, 224, 226, 227, 244, 245, 248, 252,
                256, 259, 264, 268, 269, 271, 272, 275, 277, 279, 280, 281, 282, 283,
                300, 301, 307, 310, 312, 318, 328, 331, 336, 337, 338, 345, 353, 365,
                411, 413, 416, 417, 419, 443, 444, 449, 452, 458, 459, 152, 92, 220,
                188, 181, 195, 171, 110, 176, 122, 420, 243, 140]

DATA_DIR = "data/eicu/handled"
VOCAB_PKL = os.path.join(DATA_DIR, 'vocab.raw.pkl')

# Model hyperparams

NODE_EMB_DIM = 300
GAT_HIDDEN = 300
GAT_HEADS = 2
TRANS_LAYERS = 2
GAT_LAYERS = 2
DROPOUT = 0.2
NUM_EDGE_TYPES = 5
#(do not change NUM_EDGE_TYPES, its how many edge type your Patient Graph Creation create)


LR = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 10

CHECKPOINT_MODEL = "model_best.pt"

CHECKPOINT_GLOBAL = "model_global.pt"



# Recommendation threshold
PRED_THRESHOLD = 0.3








 



class Voc:
    def __init__(self, name):
        self.name = name
        self.idx2word = []
        self.word2idx = {}
        self.word_cnt = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word_cnt[word] = 1
        else:
            self.word_cnt[word] += 1
            
            


 
# ---------------------------
# Utilities
# ---------------------------
def _safe_load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# Robust parser for csv-like messy fields
def parse_field(x):
    """Return a list of tokens from x which may be list, ndarray, or string like "['a' 'b']" or "[1.0,2.0]"."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("nan", "none"):
            return []
        # Try single-quoted elements first
        quoted = re.findall(r"'([^']*)'", s)
        if quoted:
            return [q.strip() for q in quoted if q.strip() != ""]
        # Try ast literal_eval
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return list(val)
            if isinstance(val, str):
                parts = [p.strip() for p in re.split(r',\s*|\s+', val) if p.strip() != ""]
                return parts
        except Exception:
            parts = [p.strip() for p in re.split(r',\s*|\s+', s) if p.strip() != ""]
            return parts
    return [x]

def normalize_med_token(tok):
    """Normalize med token to number if possible (int or float) or string if not numeric."""
    try:
        if isinstance(tok, (int, np.integer)):
            return int(tok)
        if isinstance(tok, (float, np.floating)):
            f = float(tok)
            if f.is_integer():
                return int(f)
            return f
        s = str(tok).strip()
        if s == "":
            return None
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
            return f
        except:
            return s
    except Exception:
        return tok

# ---------------------------
# Dataset
# ---------------------------

class PatientGraphDataset(Dataset):
    """
    Student graph  : Diagnosis + Procedure
    Pure clinical graph (NO drug nodes, NO hospital node)

    Graph features:
    • Unique nodes with log-scaled frequency
    • Node importance prior (diag < proc)
    • PMI-weighted same-type edges
    • Sqrt-scaled cross-type edges
    • Star backbone for structural stability
    • Multi-relational edges
    """

    def __init__(self, dataframe, vocab, med_size, hospital_id):
        self.df = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.MED_SIZE = med_size
        self.hospital_id = str(hospital_id)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_diag = row.get('icd9_code', [])
        raw_proc = row.get('pro_code', [])
        raw_med  = row.get('drug_id', [])

        diag_tokens = parse_field(raw_diag)
        proc_tokens = parse_field(raw_proc)
        med_tokens  = parse_field(raw_med)
        med_tokens_norm = [normalize_med_token(m) for m in med_tokens if m is not None]

        # ---------------------------
        # TOKEN → ID MAPPING
        # ---------------------------
        diag_ids = [self.vocab["diag"][d] for d in diag_tokens if d in self.vocab["diag"]]
        proc_ids = [self.vocab["proc"][p] for p in proc_tokens if p in self.vocab["proc"]]

        med_ids = []
        for m in med_tokens_norm:
            if m in self.vocab['med']:
                med_ids.append(self.vocab['med'][m]); continue
            if str(m) in self.vocab['med']:
                med_ids.append(self.vocab['med'][str(m)]); continue
            try:
                if isinstance(m, int) and float(m) in self.vocab['med']:
                    med_ids.append(self.vocab['med'][float(m)])
            except:
                pass

        # ---------------------------
        # NODE FREQUENCY
        # ---------------------------
        diag_cnt = Counter(diag_ids)
        proc_cnt = Counter(proc_ids)

        uniq_diag = list(diag_cnt.keys()) or [0]
        uniq_proc = list(proc_cnt.keys())

        # ---------------------------
        # NODE INDEXING
        # ---------------------------
        node_map = {}
        node_types = []
        token_ids = []
        node_freq = []

        idx_ptr = 0

        for d in uniq_diag:
            node_map[("diag", d)] = idx_ptr
            token_ids.append(d)
            node_types.append(0)
            node_freq.append(diag_cnt[d])
            idx_ptr += 1

        for p in uniq_proc:
            node_map[("proc", p)] = idx_ptr
            token_ids.append(p)
            node_types.append(1)
            node_freq.append(proc_cnt[p])
            idx_ptr += 1

        num_nodes = max(1, idx_ptr)

        # ---------------------------
        # EDGE BUILDING
        # ---------------------------
        edge_dict = defaultdict(float)
        etype_dict = {}

        def add_edge(u, v, w, etype):
            if u == v:
                return
            edge_dict[(u, v)] += float(w)
            etype_dict[(u, v)] = etype

        def freq(t, id_):
            if t == "diag": return diag_cnt[id_]
            if t == "proc": return proc_cnt[id_]
            return 1

        # ---------------------------
        # SAME-TYPE EDGES (PMI)
        # ---------------------------
        eps = 1e-8
        for (t, uniq_list, etype) in [
            ("diag", uniq_diag, 0),
            ("proc", uniq_proc, 2),
        ]:
            for a, b in combinations(uniq_list, 2):
                u = node_map[(t, a)]
                v = node_map[(t, b)]
                fi = freq(t, a)
                fj = freq(t, b)
                pij = min(fi, fj)

                pmi = math.log((pij + eps) / (fi * fj + eps))
                w = max(pmi, 0.0)

                add_edge(u, v, w, etype)
                add_edge(v, u, w, etype)

        # ---------------------------
        # CROSS-TYPE EDGES
        # ---------------------------
        for a, b in product(uniq_diag, uniq_proc):
            u = node_map[("diag", a)]
            v = node_map[("proc", b)]
            fi = freq("diag", a)
            fj = freq("proc", b)
            w = math.sqrt(fi * fj)

            add_edge(u, v, w, 1)
            add_edge(v, u, w, 1)

        # ---------------------------
        # STAR BACKBONE
        # ---------------------------
        def add_star_edges(node_list, node_type, etype):
            if len(node_list) <= 1:
                return
            center = node_map[(node_type, node_list[0])]
            for n in node_list[1:]:
                u = center
                v = node_map[(node_type, n)]
                add_edge(u, v, 2.0, etype)
                add_edge(v, u, 2.0, etype)

        add_star_edges(uniq_diag, "diag", 3)
        add_star_edges(uniq_proc, "proc", 4)

        # ---------------------------
        # FINALIZE GRAPH
        # ---------------------------
        if edge_dict:
            src, dst, weights, etypes = [], [], [], []
            for (u, v), w in edge_dict.items():
                src.append(u)
                dst.append(v)
                weights.append(w)
                etypes.append(etype_dict[(u, v)])

            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.edata['w'] = torch.tensor(weights, dtype=torch.float32)
            g.edata['etype'] = torch.tensor(etypes, dtype=torch.long)
        else:
            g = dgl.graph(([], []), num_nodes=1)
            g.add_edges(0, 0)
            g.edata['w'] = torch.tensor([1.0])
            g.edata['etype'] = torch.tensor([0])

        g = dgl.add_self_loop(g)

        # ---------------------------
        # NODE FEATURES
        # ---------------------------
        token_ids  = torch.LongTensor(token_ids) if token_ids else torch.LongTensor([0])
        node_types = torch.LongTensor(node_types) if node_types else torch.LongTensor([0])

        node_freq = torch.log1p(torch.FloatTensor(node_freq)) \
            if node_freq else torch.FloatTensor([1.0])

        importance_map = {0: 1.0, 1: 1.2}
        node_importance = torch.FloatTensor(
            [importance_map[t.item()] for t in node_types]
        )

        # ---------------------------
        # LABEL (Drugs still predicted)
        # ---------------------------
        label = torch.zeros(self.MED_SIZE, dtype=torch.float32)
        for m in med_ids:
            if 0 <= m < self.MED_SIZE:
                label[m] = 1.0

        return g, token_ids, node_types, node_freq, node_importance, label, self.hospital_id, idx

def collate_batch(batch):
    graphs, token_ids_list, node_types_list, node_freq_list, \
    node_imp_list, labels, hids, indices = zip(*batch)

    batched_graph = dgl.batch(graphs)

    token_ids       = torch.cat(token_ids_list, dim=0)
    node_types      = torch.cat(node_types_list, dim=0)
    node_freq       = torch.cat(node_freq_list, dim=0)
    node_importance = torch.cat(node_imp_list, dim=0)
    labels          = torch.stack(labels, dim=0)

    return (
        batched_graph,
        token_ids,
        node_types,
        node_freq,
        node_importance,
        labels,
        list(hids),
        list(indices),
    )


# ----------------------------------------
# Model: GAT-v2 Encoder
# ----------------------------------------

# ----------------------------------------
# Model: Frequency-Aware Relational GAT Encoder, Custom Edge-Aware Relational GATv2 Layer
# ----------------------------------------

class EdgeRelationalGATv2(nn.Module):
    def __init__(self, in_dim, out_dim, heads, edge_feat_dim, dropout):
        super().__init__()

        self.heads = heads
        self.out_dim = out_dim
        self.scale = out_dim ** -0.5

        self.q_proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, heads * out_dim, bias=False)

        self.edge_bias = nn.Linear(edge_feat_dim, heads, bias=False)

        # 🔥 NEW: attention temperature (critical for PRAUC)
        self.attn_temp = nn.Parameter(torch.tensor(0.7))

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(heads * out_dim, heads * out_dim)

    def forward(self, g, h, edge_feat, edge_weight):
        with g.local_scope():
            N = h.size(0)

            Q = self.q_proj(h).view(N, self.heads, self.out_dim)
            K = self.k_proj(h).view(N, self.heads, self.out_dim)
            V = self.v_proj(h).view(N, self.heads, self.out_dim)

            g.ndata['Q'] = Q
            g.ndata['K'] = K
            g.ndata['V'] = V

            eb = self.edge_bias(edge_feat)
            g.edata['eb'] = eb
            g.edata['ew'] = edge_weight.unsqueeze(-1)

            def edge_attention(edges):
                score = (edges.src['Q'] * edges.dst['K']).sum(-1) * self.scale

                # 🔥 CRITICAL: sharpen attention
                score = score / torch.clamp(self.attn_temp, 0.5, 2.0)

                score = score + edges.data['eb']
                score = score * edges.data['ew']
                return {'a': score}

            g.apply_edges(edge_attention)

            g.edata['a'] = dgl.nn.functional.edge_softmax(g, g.edata['a'])
            g.edata['a'] = self.dropout(g.edata['a'])

            def message_func(edges):
                return {'m': edges.src['V'] * edges.data['a'].unsqueeze(-1)}

            def reduce_func(nodes):
                return {'h': nodes.mailbox['m'].sum(dim=1)}

            g.update_all(message_func, reduce_func)

            h_out = g.ndata['h'].reshape(N, self.heads * self.out_dim)
            return self.out_proj(h_out)




class GATEncoder(nn.Module):

    def __init__(self, diag_vocab_sz, proc_vocab_sz,
                 node_emb_dim=NODE_EMB_DIM,
                 hidden=GAT_HIDDEN,
                 heads=GAT_HEADS,
                 layers=GAT_LAYERS,
                 num_edge_types=NUM_EDGE_TYPES,
                 dropout=DROPOUT):

        super().__init__()

        self.diag_emb = nn.Embedding(diag_vocab_sz, node_emb_dim)
        self.proc_emb = nn.Embedding(proc_vocab_sz, node_emb_dim)

        self.type_emb = nn.Embedding(2, node_emb_dim)
        self.freq_proj = nn.Linear(1, node_emb_dim)
        self.importance_proj = nn.Linear(1, node_emb_dim)
        
        self.global_alpha = nn.Parameter(torch.tensor(0.3))

        self.edge_type_emb = nn.Embedding(num_edge_types, node_emb_dim)

        self.layers   = nn.ModuleList()
        self.norms    = nn.ModuleList()
        self.resproj  = nn.ModuleList()

        # 🔥 NEW: layer-wise gates (prevents oversmoothing)
        self.layer_gates = nn.ParameterList()

        in_dim = node_emb_dim

        for _ in range(layers):
            self.layers.append(
                EdgeRelationalGATv2(
                    in_dim=in_dim,
                    out_dim=hidden // heads,
                    heads=heads,
                    edge_feat_dim=node_emb_dim,
                    dropout=dropout
                )
            )

            self.norms.append(nn.LayerNorm(in_dim))

            self.layer_gates.append(nn.Parameter(torch.tensor(0.5)))

            if in_dim != hidden:
                self.resproj.append(nn.Linear(in_dim, hidden))
            else:
                self.resproj.append(nn.Identity())

            in_dim = hidden

        # 🔥 NEW: global context projection (VERY IMPORTANT)
        self.global_proj = nn.Linear(hidden, hidden)

        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout)
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden)

    def forward(self, g, token_ids, node_types, node_freq, node_importance):

        device = token_ids.device
        N = token_ids.size(0)
        emb_dim = self.diag_emb.embedding_dim

        node_emb = torch.zeros((N, emb_dim), device=device)

        mask_diag = node_types == 0
        mask_proc = node_types == 1

        if mask_diag.any():
            node_emb[mask_diag] = self.diag_emb(token_ids[mask_diag])
        if mask_proc.any():
            node_emb[mask_proc] = self.proc_emb(token_ids[mask_proc])

        node_emb = (
            node_emb
            + self.type_emb(node_types)
            + self.freq_proj(node_freq.unsqueeze(-1))
            + self.importance_proj(node_importance.unsqueeze(-1))
        )

        if 'w' in g.edata:
            w = g.edata['w']
            w_norm = w / (w.mean() + 1e-6)
        else:
            w_norm = torch.ones(g.num_edges(), device=device)

        if 'etype' in g.edata:
            edge_feat = self.edge_type_emb(g.edata['etype'])
        else:
            edge_feat = torch.zeros(g.num_edges(), emb_dim, device=device)

        h = node_emb

        for layer, norm, proj, gate in zip(self.layers, self.norms, self.resproj, self.layer_gates):
            h_res = h

            h = norm(h)
            h_new = layer(g, h, edge_feat, w_norm)

            h_new = self.act(h_new)
            h_new = self.dropout(h_new)

            # 🔥 CRITICAL: gated residual (prevents oversmoothing)
            g_val = torch.sigmoid(gate)
            h = g_val * h_new + (1 - g_val) * proj(h_res)

        # 🔥 GLOBAL CONTEXT INJECTION (THIS BOOSTS PRAUC)
        g_repr = h.mean(dim=0, keepdim=True)          # [1, D]
        g_repr = self.global_proj(g_repr)
        #h = h + 0.3 * g_repr
        h = h + torch.clamp(self.global_alpha, 0.0, 1.0) * g_repr
        
        # FFN
        h_res = h
        h = self.ffn_norm(h)
        h = self.ffn(h)
        h = h + h_res

        h = self.final_norm(h)

        return h


class AttentionPool(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1)
        )

        # 🔥 NEW: temperature for sharper pooling
        self.temp = nn.Parameter(torch.tensor(0.7))

    def forward(self, g, node_emb, node_types, node_freq, node_importance):

        ns = g.batch_num_nodes().tolist()

        splits_emb  = torch.split(node_emb, ns, dim=0)
        splits_type = torch.split(node_types, ns, dim=0)
        splits_freq = torch.split(node_freq, ns, dim=0)
        splits_imp  = torch.split(node_importance, ns, dim=0)

        pooled = []

        for nodes, types, freq, imp in zip(
            splits_emb, splits_type, splits_freq, splits_imp
        ):

            mask = (types == 0) | (types == 1)

            if mask.sum() == 0:
                pooled.append(nodes.mean(dim=0, keepdim=True))
                continue

            nodes = nodes[mask]
            freq  = freq[mask]
            imp   = imp[mask]

            scores = self.att(nodes).squeeze(-1)

            # 🔥 sharpen pooling
            scores = scores / torch.clamp(self.temp, 0.5, 2.0)

            scores = scores * (torch.log1p(freq) + imp)

            weights = torch.softmax(scores, dim=0).unsqueeze(-1)
            pooled.append((weights * nodes).sum(dim=0, keepdim=True))

        return torch.cat(pooled, dim=0)


###################################################
##################################################
############ Hospital Adaptive Decoding logic

class DrugGraphTransformer(nn.Module):
    def __init__(self, dim, heads, layers, dropout):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict({
                    "norm_q": nn.LayerNorm(dim),
                    "norm_kv": nn.LayerNorm(dim),
                    "norm_self": nn.LayerNorm(dim),
                    "norm_ff": nn.LayerNorm(dim),

                    "cross_attn": nn.MultiheadAttention(
                        embed_dim=dim,
                        num_heads=heads,
                        dropout=dropout,
                        batch_first=True
                    ),
                    "self_attn": nn.MultiheadAttention(
                        embed_dim=dim,
                        num_heads=heads,
                        dropout=dropout,
                        batch_first=True
                    ),
                    "ff": nn.Sequential(
                        nn.Linear(dim, dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim * 4, dim),
                        nn.Dropout(dropout),
                    ),
                })
            )

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, patient_token, drug_tokens):
        x = drug_tokens

        for layer in self.layers:
            q = layer["norm_q"](x)
            kv = layer["norm_kv"](patient_token)
            cross_out, _ = layer["cross_attn"](q, kv, kv)
            x = x + cross_out

            s = layer["norm_self"](x)
            self_out, _ = layer["self_attn"](s, s, s)
            x = x + self_out

            f = layer["norm_ff"](x)
            x = x + layer["ff"](f)

        return self.final_norm(x)


class UnifiedHospitalDecoder(nn.Module):
    def __init__(self, emb_dim, med_size, num_hospitals):
        super().__init__()

        self.emb_dim = emb_dim
        self.med_size = med_size

        # ==============================
        # Global Drug Tokens
        # ==============================
        self.drug_tokens = nn.Parameter(torch.randn(med_size, emb_dim) * 0.02)

        # ==============================
        # Patient-conditioned token mixer
        # ==============================
        self.token_mixer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # ==============================
        # Hospital Embedding
        # ==============================
        self.hosp_emb = nn.Embedding(num_hospitals, emb_dim)

        # ==============================
        # Pre-Norm FiLM
        # ==============================
        self.film_gamma = nn.Linear(emb_dim, emb_dim)
        self.film_beta  = nn.Linear(emb_dim, emb_dim)
        self.patient_norm = nn.LayerNorm(emb_dim)

        # ==============================
        # Transformer Backbone
        # ==============================
        self.transformer = DrugGraphTransformer(
            dim=emb_dim,
            heads=GAT_HEADS,
            layers=TRANS_LAYERS,
            dropout=DROPOUT
        )

        # ==============================
        # Shared Scoring Head
        # ==============================
        self.drug_norm = nn.LayerNorm(emb_dim)

        self.scorer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(emb_dim * 2, 1)
        )

        # ==============================
        # 🔥 Hospital Low-Rank Adapter (WEAKER)
        # ==============================
        rank = emb_dim // 8   # ↓ reduced capacity

        self.hosp_U = nn.Embedding(num_hospitals, med_size * rank)
        self.hosp_V = nn.Linear(emb_dim, rank, bias=False)

        # 🔥 Patient-aware hospital gate
        self.hosp_gate = nn.Linear(emb_dim, med_size)

        # 🔥 Residual scaling (CRITICAL)
        self.hosp_scale = nn.Parameter(torch.tensor(0.1))

        # ==============================
        # Hospital Bias
        # ==============================
        self.hosp_bias = nn.Embedding(num_hospitals, med_size)

        # ==============================
        # Global Patient Residual
        # ==============================
        self.patient_residual = nn.Linear(emb_dim, med_size)

        # ==============================
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, patient_emb, hosp_ids):
        B, D = patient_emb.shape
        M = self.med_size

        # -------------------------
        # Hospital FiLM on patient embedding
        # -------------------------
        h_emb = self.hosp_emb(hosp_ids)

        gamma = self.film_gamma(h_emb)
        beta  = self.film_beta(h_emb)

        patient_emb = self.patient_norm(patient_emb)
        patient_emb = patient_emb * (1 + gamma) + beta

        # -------------------------
        # Patient-conditioned drug tokens
        # -------------------------
        base_tokens = self.drug_tokens.unsqueeze(0).expand(B, -1, -1)
        patient_mix = self.token_mixer(patient_emb).unsqueeze(1)
        drug_tokens = base_tokens + patient_mix

        patient_token = patient_emb.unsqueeze(1)

        # -------------------------
        # Global Drug Reasoning
        # -------------------------
        drug_context = self.transformer(patient_token, drug_tokens)

        # -------------------------
        # Shared logits
        # -------------------------
        drug_context = self.drug_norm(drug_context)
        logits = self.scorer(drug_context).squeeze(-1)

        # -------------------------
        # Global patient residual
        # -------------------------
        logits = logits + self.patient_residual(patient_emb)

        # -------------------------
        # Hospital Low-Rank Residual
        # -------------------------
        U = self.hosp_U(hosp_ids).view(B, M, -1)        # (B,M,R)
        V = self.hosp_V(patient_emb).unsqueeze(-1)     # (B,R,1)
        res = torch.bmm(U, V).squeeze(-1)              # (B,M)

        # 🔥 Patient-aware gate
        gate_input = h_emb + patient_emb
        gate = torch.sigmoid(self.hosp_gate(gate_input))  # (B,M)

        # 🔥 Scaled residual (prevents overpowering global)
        logits = logits + self.hosp_scale * gate * res

        # -------------------------
        # Hospital bias
        # -------------------------
        logits = logits + self.hosp_bias(hosp_ids)

        # -------------------------
        # Temperature scaling
        # -------------------------
        logits = logits / self.temp

        probs = torch.sigmoid(logits)
        return probs, logits############ Hospital Adaptive Decoding logic
###################################################
##################################################




class MedRecommenderPerHosp(nn.Module):

    def __init__(self,
                 diag_vocab_size,
                 proc_vocab_size,
                 med_vocab_size,
                 hosp_list,
                 num_hospitals):

        super().__init__()

        self.encoder = GATEncoder(
            diag_vocab_size,
            proc_vocab_size,
            node_emb_dim=NODE_EMB_DIM,
            hidden=GAT_HIDDEN,
            heads=GAT_HEADS,
            layers=GAT_LAYERS
        )
        
        self.hosp_list = [str(h) for h in hosp_list]
        self.hosp2idx = {h: i for i, h in enumerate(self.hosp_list)}

        self.pool = AttentionPool(GAT_HIDDEN)


        self.decoder = UnifiedHospitalDecoder(
            emb_dim=GAT_HIDDEN,
            med_size=med_vocab_size,
            num_hospitals=num_hospitals
        )

        


    def forward(self, g, token_ids, node_types, node_freq, node_importance, hosp_ids):

        
        device = token_ids.device
        
    
        
        
        # Convert hospital IDs → LongTensor indices
        if isinstance(hosp_ids, list):
            hosp_ids = torch.tensor(
                [self.hosp2idx[str(h)] for h in hosp_ids],
                dtype=torch.long,
                device=device
            )
        
        node_emb = self.encoder(g, token_ids, node_types, node_freq, node_importance)
        patient_emb = self.pool(g, node_emb, node_types, node_freq, node_importance)
        probs, logits = self.decoder(patient_emb, hosp_ids)

        return probs, logits


########################################################
########################################################


# ---------------------------
# Metrics and evaluation (threshold fixed to 0.3)
# ---------------------------
#def evaluate(model, loader, device, thresholds=None):
def evaluate(model, loader, device, threshold=PRED_THRESHOLD):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for g, token_ids, node_types, node_freq, node_importance, y, hids, indices in loader:
            g = g.to(device) #  on GPU
            
            token_ids = token_ids.to(device)             
            node_types = node_types.to(device) 
            node_freq = node_freq.to(device)
            node_importance = node_importance.to(device)
            
            probs, logits = model(g, token_ids, node_types, node_freq, node_importance, hids)
            #probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    if len(all_probs) == 0:
        return {'F1': 0.0, 'Jaccard': 0.0, 'PRAUC': 0.0}
    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)

    # remove examples with no positive labels (cannot compute PRAUC)
    mask = labels.sum(axis=1) > 0
    if mask.sum() == 0:
        return {'F1': 0.0, 'Jaccard': 0.0, 'PRAUC': 0.0}
    probs = probs[mask]
    labels = labels[mask]

    ################################
    # Dynamic Top-K (based on average label count)
    #avg_k = int(labels.sum(axis=1).mean())
    #binarized = np.zeros_like(probs)
    #for i in range(probs.shape[0]):
    #   top_k_idx = np.argsort(probs[i])[-avg_k:]
    #   binarized[i, top_k_idx] = 1
    #
    ###############################
    
    ##################
    # REMOVE DYNAMIC TOP-K and using fixed PRED_THRESHOLD
    binarized = (probs > threshold).astype(int)
    #
    ####################
    
    #if thresholds is None:
    #   binarized = (probs > 0.3).astype(int)
    #else:
    #   if thresholds.ndim == 1:  # per-label thresholds
    #      binarized = (probs > thresholds.reshape(1, -1)).astype(int)
    #   else:
    #      binarized = (probs > thresholds).astype(int)

    return {
        'F1': f1_score(labels, binarized, average='micro', zero_division=0),
        'Jaccard': jaccard_score(labels, binarized, average='micro', zero_division=0),
        'PRAUC': average_precision_score(labels, probs, average='micro'),
    }


########################################################
########################################################


def approx_ndcg_loss(logits, labels, temperature=1.0):
    scores = logits / temperature
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    pos_scores = scores[pos_mask].unsqueeze(1)
    neg_scores = scores[neg_mask].unsqueeze(0)
    margin = 0.5                                      # reduced from 1.0
    loss = F.softplus(neg_scores - pos_scores + margin).mean()
    # Normalise by number of pairs to prevent magnitude explosion
    n_pairs = pos_mask.sum() * neg_mask.sum()
    return loss / (n_pairs.float().sqrt() + 1.0)      # key fix

class LabelSmoothBCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)







# ---------------------------
# TRAIN: 
# ---------------------------

def train_model(model, train_loader, 
                val_loader, device,
                epochs=EPOCHS, patience=PATIENCE,
                checkpoint_path=CHECKPOINT_MODEL,
                checkpoint_path_global=CHECKPOINT_GLOBAL):

    train_losses, val_praucs = [], []
    best_overall_score = -1.0

    #criterion = nn.BCEWithLogitsLoss()
    criterion = LabelSmoothBCE(smoothing=0.05)
    #criterion = FocalLoss(gamma=1.5)
    

    print("\n==============================")
    print("TRAINING — AGBM MODEL")
    print("==============================")

    decoder = model.decoder
    encoder = model.encoder

    # ============================================================
    # PHASE 1 — GLOBAL LEARNING
    # ============================================================
    print("\n==============================")
    print("PHASE 1 — GLOBAL LEARNING")
    print("==============================")

    # -------- Train everything (you were already doing this) --------
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_score = -1.0
    counter = 0
    epoch_idx = 0

    while counter < patience and epoch_idx < epochs:
        epoch_idx += 1
        model.train()
        total_loss, it = 0.0, 0
        start = time.time()

        for g, token_ids, node_types, node_freq, node_importance, y, hids, indices in tqdm(
            train_loader, desc=f"[AGBM-Global Learning] Epoch {epoch_idx}"
        ):
            g = g.to(device)
            token_ids = token_ids.to(device)
            node_types = node_types.to(device)
            node_freq = node_freq.to(device)
            node_importance = node_importance.to(device)
            y = y.to(device)

            hosp_idx = torch.tensor(
                [model.hosp2idx[str(h)] for h in hids],
                dtype=torch.long, device=device
            )

            _, logits_s = model(
                g, token_ids, node_types, node_freq, node_importance, hosp_idx
            )

            #loss = criterion(logits_s, y)
            
            L_bce = criterion(logits_s, y)
            # Add pairwise ranking loss per sample
            L_rank = torch.stack([
                approx_ndcg_loss(logits_s[i], y[i])
                for i in range(logits_s.size(0))
            ]).mean()
            
            loss = L_bce + 0.05 * L_rank

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            it += 1

        avg_loss = total_loss / max(it, 1)
        train_losses.append(avg_loss)
        print(f"[AGBM-Global Learning] loss={avg_loss:.4f} time={time.time()-start:.1f}s")

        val_metrics = evaluate(model, val_loader, device)
        print(f"  Val PRAUC: {val_metrics['PRAUC']:.4f}  F1: {val_metrics['F1']:.4f}  Jaccard: {val_metrics['Jaccard']:.4f}")
        #val_score = val_metrics['Jaccard']
        val_score = 0.4 * val_metrics['PRAUC'] + 0.3 * val_metrics['F1'] + 0.3 * val_metrics['Jaccard']

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), checkpoint_path_global)
            print("  ✅ Saved BEST AGBM (GLOBAL LEARNING) MODEL")
            counter = 0
        else:
            counter += 1

    # ============================================================
    # PHASE 2 — HOSPITAL ADAPTATION
    # ============================================================
    print("\n==============================")
    print("PHASE 2 — HOSPITAL ADAPTATION")
    print("==============================")

    # 🔥 CRITICAL — reload best global model
    print("Loading BEST AGBM (GLOBAL LEARNING) MODEL...")
    model.load_state_dict(torch.load(checkpoint_path_global))

    # -------- Freeze global parts --------
    for p in encoder.parameters(): p.requires_grad = False
    for p in decoder.transformer.parameters(): p.requires_grad = False
    for p in decoder.token_mixer.parameters(): p.requires_grad = False
    for p in decoder.drug_norm.parameters(): p.requires_grad = False
    for p in decoder.patient_residual.parameters(): p.requires_grad = False
    decoder.drug_tokens.requires_grad = False
    #decoder.temp.requires_grad = False
    
    decoder.temp.requires_grad = True

    # 🔥 Allow slight calibration
    for p in decoder.scorer.parameters():
        p.requires_grad = True

    # -------- Unfreeze hospital-specific parts --------
    for p in decoder.hosp_emb.parameters(): p.requires_grad = True
    for p in decoder.film_gamma.parameters(): p.requires_grad = True
    for p in decoder.film_beta.parameters():  p.requires_grad = True
    for p in decoder.patient_norm.parameters(): p.requires_grad = True
    for p in decoder.hosp_U.parameters(): p.requires_grad = True
    for p in decoder.hosp_V.parameters(): p.requires_grad = True
    for p in decoder.hosp_gate.parameters(): p.requires_grad = True
    for p in decoder.hosp_bias.parameters(): p.requires_grad = True
    decoder.hosp_scale.requires_grad = True

    # 🔥 Lower LR for adapters
    LR_LOCAL = LR * 0.2

    optimizer = torch.optim.AdamW([
        {"params": decoder.hosp_emb.parameters(), "lr": LR_LOCAL},
        {"params": decoder.film_gamma.parameters(), "lr": LR_LOCAL},
        {"params": decoder.film_beta.parameters(), "lr": LR_LOCAL},
        {"params": decoder.patient_norm.parameters(), "lr": LR_LOCAL},
        {"params": decoder.hosp_U.parameters(), "lr": LR_LOCAL},
        {"params": decoder.hosp_V.parameters(), "lr": LR_LOCAL},
        {"params": decoder.hosp_gate.parameters(), "lr": LR_LOCAL},
        {"params": decoder.hosp_bias.parameters(), "lr": LR_LOCAL},
        {"params": [decoder.hosp_scale], "lr": LR_LOCAL},
        {"params": decoder.scorer.parameters(), "lr": LR * 0.05},
        {"params": [decoder.temp], "lr": LR * 0.01},
    ], weight_decay=WEIGHT_DECAY)

    best_score = -1.0
    counter = 0
    epoch_idx = 0

    while counter < patience and epoch_idx < epochs:
        epoch_idx += 1
        model.train()
        total_loss, it = 0.0, 0
        start = time.time()

        for g, token_ids, node_types, node_freq, node_importance, y, hids, indices in tqdm(
            train_loader, desc=f"[AGBM-Hospital Adaptation] Epoch {epoch_idx}"
        ):
            g = g.to(device)
            token_ids = token_ids.to(device)
            node_types = node_types.to(device)
            node_freq = node_freq.to(device)
            node_importance = node_importance.to(device)
            y = y.to(device)

            hosp_idx = torch.tensor(
                [model.hosp2idx[str(h)] for h in hids],
                dtype=torch.long, device=device
            )

            _, logits_s = model(
                g, token_ids, node_types, node_freq, node_importance, hosp_idx
            )

            # 🔥 Primary loss
            #L_bce = criterion(logits_s, y)

            # 🔥 Residual magnitude regularization
            #L_reg = 1e-4 * (decoder.hosp_scale ** 2)

            #loss = L_bce + L_reg
            #loss = L_bce
            
            
            # Phase 2 loss — inside the fine-tuning loop
            L_bce = criterion(logits_s, y)
            # Ranking loss with SMALLER weight than Phase 1
            L_rank = torch.stack([
                approx_ndcg_loss(logits_s[i], y[i])
                for i in range(logits_s.size(0))
            ]).mean()
            # Regularisation (re-enabled)
            L_reg = (
                1e-3 * (decoder.hosp_scale ** 2) +
                1e-4 * decoder.hosp_U.weight.norm(2) +
                1e-4 * decoder.hosp_bias.weight.norm(2)
            )
            # Phase 2 weight is 0.05 — half of Phase 1's 0.1
            loss = L_bce + 0.05 * L_rank + L_reg

            optimizer.zero_grad()
            loss.backward()

            # 🔥 tighter clipping for adapters
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )

            optimizer.step()

            total_loss += loss.item()
            it += 1

        avg_loss = total_loss / max(it, 1)
        train_losses.append(avg_loss)
        print(f"[AGBM-Hospital Adaptation] loss={avg_loss:.4f} time={time.time()-start:.1f}s")

        val_metrics = evaluate(model, val_loader, device)
        print(f"  Val PRAUC: {val_metrics['PRAUC']:.4f}  F1: {val_metrics['F1']:.4f}  Jaccard: {val_metrics['Jaccard']:.4f}")
        #val_score = val_metrics['Jaccard']
        val_score = 0.4 * val_metrics['PRAUC'] + 0.3 * val_metrics['F1'] + 0.3 * val_metrics['Jaccard']
        val_praucs.append(val_score)

        if val_score > best_overall_score:
            best_overall_score = val_score
            torch.save(model.state_dict(), checkpoint_path)
            print("  ✅ Saved BEST AGBM (GLOBAL LEARNING + HOSPITAL ADAPTATION) MODEL")

        if val_score > best_score:
            best_score = val_score
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("AGBM early stopping")
                break

    return train_losses, val_praucs


 

# ---------------------------
# Data loader helper
# ---------------------------
def load_data(seed, hospital_id):
    base = os.path.join(DATA_DIR, str(seed), str(hospital_id))
    with open(os.path.join(base, 'train.pkl'), 'rb') as f:
        train_df = pickle.load(f)
    with open(os.path.join(base, 'val.pkl'), 'rb') as f:
        val_df = pickle.load(f)
    with open(os.path.join(base, 'test.pkl'), 'rb') as f:
        test_df = pickle.load(f)
    return train_df, val_df, test_df

# ---------------------------
# MAIN: Runs per-seed
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_raw = _safe_load_pickle(VOCAB_PKL)

    vocab = {
        'diag': vocab_raw['diag_voc'].word2idx,
        'proc': vocab_raw['pro_voc'].word2idx,
        'med':  vocab_raw['med_voc'].word2idx
    }

    diag_vocab_size = len(vocab['diag'])
    proc_vocab_size = len(vocab['proc'])
    med_vocab_size  = len(vocab['med'])
    
    # ============================================================
    # Hospital Index Mapping (REQUIRED for new decoder)
    # ============================================================
    #hosp_list = sorted(HOSPITAL_IDS)  # e.g. from dataset
    hosp_list = HOSPITAL_IDS  # e.g. from dataset
    hosp2idx = {str(h): i for i, h in enumerate(hosp_list)}
    idx2hosp = {i: str(h) for i, h in enumerate(hosp_list)}
    num_hospitals = len(hosp_list)
    
    

    print("Diagnosis vocab size:", diag_vocab_size)
    print("Procedure vocab size:", proc_vocab_size)
    print("Medication vocab size:", med_vocab_size)

    results_json = {}
    results_json_global = {}
    overall_metrics_storage = []
    overall_metrics_storage_global = []

    

    # =========================================================
    # LOOP OVER SEEDS
    # =========================================================
    for seed in SEEDS:

        print(f"\n==============================")
        print(f"Processing SEED {seed}")
        print(f"==============================")

        train_ds_teacher, train_ds = [], []
        val_ds, test_ds = [], []

        # -----------------------------------------
        # Build datasets for THIS seed only
        # -----------------------------------------
        for hid in HOSPITAL_IDS:
            try:
                tr, va, te = load_data(seed, hid)
            except Exception as e:
                print(f"Skipping hospital {hid} seed {seed}: {e}")
                continue



            train_ds.append(
                PatientGraphDataset(tr, vocab, med_vocab_size, hid)
            )

            val_ds.append(
                PatientGraphDataset(va, vocab, med_vocab_size, hid)
            )

            test_ds.append(
                PatientGraphDataset(te, vocab, med_vocab_size, hid)
            )

        if len(train_ds) == 0:
            print(f"No data found for seed {seed}. Skipping.")
            continue



        val_loader = DataLoader(
            ConcatDataset(val_ds),
            batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_batch
        )

        test_loader = DataLoader(
            ConcatDataset(test_ds),
            batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_batch
        )

        train_loader = DataLoader(
            ConcatDataset(train_ds),
            batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_batch
        )

        model = MedRecommenderPerHosp(
           diag_vocab_size,
           proc_vocab_size,
           med_vocab_size,
           hosp_list,
           num_hospitals 
        ).to(device)    #  on GPU

        

        print("Training AGBM Model...")
        train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=EPOCHS,
            patience=PATIENCE,
            checkpoint_path=f"model_best_seed_{seed}.pt",
            checkpoint_path_global=f"model_global_seed_{seed}.pt"
        )        

        
        
        ###########
        
        model.load_state_dict(
            torch.load(f"model_global_seed_{seed}.pt")
        )
        model.eval()

        # =====================================================
        # TEST EVALUATION (THIS SEED)
        # =====================================================
        seed_patient_results_global = {}
        all_probs_global, all_labels_global, all_hids_global = [], [], []

        with torch.no_grad():
            for g, token_ids, node_types, node_freq, node_importance, y_true, hids, indices in test_loader:
                g = g.to(device)    #  on GPU                
                token_ids = token_ids.to(device)   #  on GPU                
                node_types = node_types.to(device)    #  on GPU                
                node_freq = node_freq.to(device)    #  on GPU
                node_importance = node_importance.to(device)
                

                probs_global, logits_global = model(g, token_ids, node_types, node_freq, node_importance, hids)
                probs_global = torch.sigmoid(logits_global)
                
                all_probs_global.append(probs_global.cpu().numpy())
                all_labels_global.append(y_true.numpy())
                all_hids_global.extend(hids)

        probs_global = np.vstack(all_probs_global)
        labels_global = np.vstack(all_labels_global)        

        mask_global = labels_global.sum(axis=1) > 0
        probs_global = probs_global[mask_global]
        labels_global = labels_global[mask_global]
        filtered_hids_global = np.array(all_hids_global)[mask_global]
        
        ##################
        # Using fixed PRED_THRESHOLD
        binarized_global = (probs_global > PRED_THRESHOLD).astype(int)
        #####################

        for i in range(probs_global.shape[0]):
            patient_id = f"{i:010d}"
            seed_patient_results_global[patient_id] = {
                "model": "AGBM",
                "seed": seed,
                "hos_id": int(filtered_hids_global[i]),
                "jaccard": float(jaccard_score(labels_global[i], binarized_global[i], zero_division=0)),
                "f1": float(f1_score(labels_global[i], binarized_global[i], zero_division=0)),
                "prauc": float(average_precision_score(labels_global[i], probs_global[i]))
            }        
        
        
        seed_metrics_global = {
            'F1': f1_score(labels_global, binarized_global, average='micro', zero_division=0),
            'Jaccard': jaccard_score(labels_global, binarized_global, average='micro', zero_division=0),
            'PRAUC': average_precision_score(labels_global, probs_global, average='micro')
        }    
        
        ###########  
        
        
        
                      
        
        model.load_state_dict(
            torch.load(f"model_best_seed_{seed}.pt")
        )
        model.eval()

        # =====================================================
        # TEST EVALUATION (THIS SEED)
        # =====================================================
        seed_patient_results = {}
        all_probs, all_labels, all_hids = [], [], []

        with torch.no_grad():
            for g, token_ids, node_types, node_freq, node_importance, y_true, hids, indices in test_loader:
                g = g.to(device)    #  on GPU                
                token_ids = token_ids.to(device)   #  on GPU                
                node_types = node_types.to(device)    #  on GPU                
                node_freq = node_freq.to(device)    #  on GPU
                node_importance = node_importance.to(device)
                

                probs, logits = model(g, token_ids, node_types, node_freq, node_importance, hids)
                probs = torch.sigmoid(logits)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y_true.numpy())
                all_hids.extend(hids)

        probs = np.vstack(all_probs)
        labels = np.vstack(all_labels)        

        mask = labels.sum(axis=1) > 0
        probs = probs[mask]
        labels = labels[mask]
        filtered_hids = np.array(all_hids)[mask]
        
        ##################
        # Using fixed PRED_THRESHOLD
        binarized = (probs > PRED_THRESHOLD).astype(int)
        #####################
        
   

        for i in range(probs.shape[0]):
            patient_id = f"{i:010d}"
            seed_patient_results[patient_id] = {
                "model": "AGBM",
                "seed": seed,
                "hos_id": int(filtered_hids[i]),
                "jaccard": float(jaccard_score(labels[i], binarized[i], zero_division=0)),
                "f1": float(f1_score(labels[i], binarized[i], zero_division=0)),
                "prauc": float(average_precision_score(labels[i], probs[i]))
            }

        seed_metrics = {
            'F1': f1_score(labels, binarized, average='micro', zero_division=0),
            'Jaccard': jaccard_score(labels, binarized, average='micro', zero_division=0),
            'PRAUC': average_precision_score(labels, probs, average='micro')
        }
        
        print(f"\n===== Seed {seed} Test Results AGBM (Global Learning + Hospital Adaptation) =====")
        print(f"PRAUC   : {seed_metrics['PRAUC']:.4f}")
        print(f"F1      : {seed_metrics['F1']:.4f}")
        print(f"Jaccard : {seed_metrics['Jaccard']:.4f}")
        print("=" * 40)
        print(f"\n===== Seed {seed} Results AGBM (Global Learning) =====")
        print(f"PRAUC   : {seed_metrics_global['PRAUC']:.4f}")
        print(f"F1      : {seed_metrics_global['F1']:.4f}")
        print(f"Jaccard : {seed_metrics_global['Jaccard']:.4f}")
        print("=" * 40) 

        
        overall_metrics_storage.append(seed_metrics)
        overall_metrics_storage_global.append(seed_metrics_global)

        results_json_global[f"seed_{seed}"] = {
            "patients": seed_patient_results_global,
            "metrics": seed_metrics_global
        }        
        
        results_json[f"seed_{seed}"] = {
            "patients": seed_patient_results,
            "metrics": seed_metrics
        }

    # =========================================================
    # MEAN ± STD
    # =========================================================
    print("\n===== AGBM (Global Learning + Hospital Adaptation) Performance =====")

    metric_names = overall_metrics_storage[0].keys()
    mean_std_results = {}

    for metric in metric_names:
        values = [m[metric] for m in overall_metrics_storage]
        mean = float(np.mean(values))
        std = float(np.std(values))
        mean_std_results[f"{metric}_mean"] = mean
        mean_std_results[f"{metric}_std"] = std
        print(f"{metric}: {mean:.4f} ± {std:.4f}")
        
    print("\n===== AGBM (Global Learning) Performance =====")

    metric_names_global = overall_metrics_storage_global[0].keys()
    mean_std_results_global = {}

    for metric in metric_names_global:
        values = [m[metric] for m in overall_metrics_storage_global]
        mean = float(np.mean(values))
        std = float(np.std(values))
        mean_std_results_global[f"{metric}_mean"] = mean
        mean_std_results_global[f"{metric}_std"] = std
        print(f"{metric}: {mean:.4f} ± {std:.4f}")        


    results_json_global["overall_mean_std"] = mean_std_results_global

    with open("AGBM_global.json", "w") as f:
        json.dump(results_json_global, f, indent=2)

    print("Saved → AGBM_global.json")

    results_json["overall_mean_std"] = mean_std_results

    with open("AGBM.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("Saved → AGBM.json")

    # =========================================================
    # PDF REPORT
    # =========================================================
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Multi-Center Medication Recommendation via Adaptive ", ln=True)
    pdf.cell(0, 10, "Graph-Based Modeling (AGBM)", ln=True)
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, "AGBM (Global Learning + Hospital Adaptation) Performance:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    for metric in metric_names:
        pdf.cell(0, 10,
                 f"{metric}: {mean_std_results[f'{metric}_mean']:.4f} ± {mean_std_results[f'{metric}_std']:.4f}",
                 ln=True)
                 
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, "AGBM (Global Learning) Performance:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    for metric in metric_names_global:
        pdf.cell(0, 10,
                 f"{metric}: {mean_std_results_global[f'{metric}_mean']:.4f} ± {mean_std_results_global[f'{metric}_std']:.4f}",
                 ln=True)
                 

    pdf.output("AGBM.pdf")
    print("Saved AGBM.pdf")

if __name__ == '__main__':
    main()

