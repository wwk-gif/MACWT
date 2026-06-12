import copy
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset, uniform
from torch_scatter import scatter_add
import numpy as np
import warnings
from torch_geometric.utils import degree


# ==================== 新增：小波变换模块 ====================
class SpaWaveletTransform(nn.Module):
    """空间转录组小波变换模块 - 修复版"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 简化小波变换：使用简单的线性变换
        self.wavelet_levels = 2

        # 小波分解层
        self.decompose1 = nn.Linear(in_features, max(1, in_features // 2))
        self.decompose2 = nn.Linear(max(1, in_features // 2), max(1, in_features // 4))

        # 小波重建层
        self.reconstruct1 = nn.Linear(max(1, in_features // 4), max(1, in_features // 2))
        self.reconstruct2 = nn.Linear(max(1, in_features // 2), out_features)

        # 残差连接
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        # x: [num_nodes, features]
        original_features = x

        # 小波分解
        decomposed1 = F.relu(self.decompose1(x))
        decomposed2 = F.relu(self.decompose2(decomposed1))

        # 小波重建
        reconstructed1 = F.relu(self.reconstruct1(decomposed2))
        reconstructed = self.reconstruct2(reconstructed1)

        # 残差连接
        residual = self.residual_proj(original_features)

        return reconstructed + residual


# ==================== 新增：因果推理模块 ====================
class SpaCausalInference(nn.Module):
    """空间因果推理模块"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 空间因果注意力
        self.spatial_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 基因表达因果注意力
        self.gene_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_mask=None, gene_mask=None):
        # x: [num_nodes, hidden_dim]

        # 空间因果注意力（考虑空间邻近性）
        spatial_output, _ = self.spatial_attention(x, x, x, key_padding_mask=spatial_mask)
        x = self.norm1(x + self.dropout(spatial_output))

        # 基因表达因果注意力（考虑基因表达相关性）
        gene_output, _ = self.gene_attention(x, x, x, key_padding_mask=gene_mask)
        x = self.norm2(x + self.dropout(gene_output))

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ==================== 原有函数保持不变 ====================
def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def full_block(in_features, out_features, p_drop, act=nn.ELU()):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        act,
        nn.Dropout(p=p_drop),
    )


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
        self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features, cached=True)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x


# ==================== 修改Encoder类 ====================
class Encoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = config['feat_hidden1']
        self.feat_hidden2 = config['feat_hidden2']
        self.gcn_hidden = config['gcn_hidden']
        self.latent_dim = config['latent_dim']
        self.p_drop = config['p_drop']

        # 默认启用小波变换和因果推理模块
        self.use_wavelet = config.get('use_wavelet', False)  # 如果你想默认只开小波，可改为 True
        self.use_causal = config.get('use_causal', True)  # 默认关闭因果推理

        print(f"增强的编码器初始化:")
        print(f"  输入维度: {input_dim}")
        print(f"  使用小波变换: {self.use_wavelet}")
        print(f"  使用因果推理: {self.use_causal}")

        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        # 小波变换模块（默认启用）
        self.wavelet_module = SpaWaveletTransform(
            self.feat_hidden2,
            self.feat_hidden2
        )

        # 因果推理模块（默认启用）
        self.causal_module = SpaCausalInference(
            self.feat_hidden2,
            num_heads=4,
            dropout=self.p_drop
        )

        # GCN layers
        self.gc1 = GraphConv(self.feat_hidden2, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.gc2 = GraphConv(self.gcn_hidden, self.latent_dim, dropout=self.p_drop, act=lambda x: x)

    def forward(self, x, edge_index):
        # 基础特征提取
        base_features = self.encoder(x)

        # 应用小波变换（默认启用）
        wavelet_features = self.wavelet_module(base_features)

        # 应用因果推理（默认启用）
        causal_features = self.causal_module(wavelet_features)

        # 图卷积
        x = self.gc1(causal_features, edge_index)
        x = self.gc2(x, edge_index)
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, config, imputation=True):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        self.imputation = imputation
        if self.imputation:
            self.layer1 = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.layer1 = GraphConv(self.input_dim, self.output_dim, dropout=self.p_drop, act=nn.Identity())

    def forward(self, x, edge_index):
        if self.imputation:
            return self.layer1(x)
        return self.layer1(x, edge_index)


# ==================== 修改SpaCross_model类 ====================
class SpaCross_model(nn.Module):
    def __init__(self, input_dim, config, imputation=True):
        super().__init__()
        self.imputation = imputation
        self.dec_in_dim = config['latent_dim']

        # 默认启用小波变换和因果推理
        self.use_wavelet = False  # 默认启用小波变换
        self.use_causal = True  # 默认启用因果推理
        self.causal_reg_weight = 0.01  # 默认因果正则化权重

        print(f"\n增强的SpaCross模型初始化 (默认启用增强功能):")
        print(f"  输入维度: {input_dim}")
        print(f"  潜在维度: {self.dec_in_dim}")
        print(f"  使用小波变换: {self.use_wavelet}")
        print(f"  使用因果推理: {self.use_causal}")
        print(f"  因果正则化权重: {self.causal_reg_weight}")

        self.online_encoder = Encoder(input_dim, config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._init_target()

        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, config['project_dim'], bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.projector = GraphConv(config['project_dim'], self.dec_in_dim, dropout=config['p_drop'], act=lambda x: x)
        self.decoder = Decoder(input_dim, config, self.imputation)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self.mask_rate = config['mask_rate']
        self.t = config['t']
        self.momentum_rate = config['momentum_rate']
        self.replace_rate = 0.05
        self.mask_token_rate = 1 - self.replace_rate
        self.anchor_pair = None

        self.weight = nn.Parameter(torch.empty(self.dec_in_dim, self.dec_in_dim))
        uniform(self.dec_in_dim, self.weight)

    def _init_target(self):
        for param_teacher in self.target_encoder.parameters():
            param_teacher.detach()
            param_teacher.requires_grad = False

    def momentum_update(self):
        base_momentum = self.momentum_rate
        for param_encoder, param_teacher in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + param_encoder.data * (1. - base_momentum)

    def encoding_mask_noise(self, x, edge_index, mask_rate=0.3):
        num_nodes = x.shape[0]
        self.num_nodes = num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_edge_index = edge_index.clone()

        return out_x, use_edge_index, (mask_nodes, keep_nodes)

    def generate_neg_nodes(self, mask_nodes):
        num_mask_nodes = mask_nodes.size(0)
        neg_nodes_x = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        neg_nodes_y = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        return neg_nodes_x, neg_nodes_y

    def compute_causal_regularization(self, features, edge_index):
        """计算因果正则化损失"""
        if not self.use_causal or self.causal_reg_weight <= 0:
            return torch.tensor(0.0, device=features.device)

        # 处理不同格式的 edge_index
        if isinstance(edge_index, torch.sparse.Tensor):
            # 对于稀疏张量，转换为坐标格式
            edge_index = edge_index.coalesce()
            src, dst = edge_index.indices()
        elif isinstance(edge_index, tuple) and len(edge_index) == 2:
            # 如果是元组 (src, dst)
            src, dst = edge_index
        elif isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            # 如果是形状为 [2, num_edges] 的张量
            src, dst = edge_index[0], edge_index[1]
        elif isinstance(edge_index, list):
            # 如果是列表，假设每个元素是边列表
            if len(edge_index) == 0:
                return torch.tensor(0.0, device=features.device)
            # 取第一个元素作为边
            edge_tensor = torch.tensor(edge_index[0], device=features.device)
            src, dst = edge_tensor[:, 0], edge_tensor[:, 1]
        else:
            # 未知格式，返回0损失
            return torch.tensor(0.0, device=features.device)

        # 邻近点特征应该相似（空间平滑性）
        if len(src) > 0 and len(dst) > 0 and src.numel() == dst.numel():
            src_features = features[src]
            dst_features = features[dst]
            spatial_loss = F.mse_loss(src_features, dst_features)
        else:
            spatial_loss = torch.tensor(0.0, device=features.device)

        # 特征独立性正则化（防止过拟合）
        # 使用批处理协方差矩阵的对角线外元素作为正则化
        if features.size(0) > 1:
            # 计算特征的相关系数矩阵
            features_normalized = F.normalize(features, dim=0)
            correlation = torch.matmul(features_normalized.T, features_normalized)
            # 惩罚非对角线元素（鼓励特征独立性）
            identity = torch.eye(correlation.size(0), device=features.device)
            causal_indep_loss = torch.mean((correlation - identity).abs())
        else:
            causal_indep_loss = torch.tensor(0.0, device=features.device)

        total_loss = spatial_loss + causal_indep_loss
        return total_loss * self.causal_reg_weight

    def mask_attr_prediction(self, x, edge_index, anchor_pair):
        use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, edge_index, self.mask_rate)

        # 编码器前向传播
        enc_rep = self.online_encoder(use_x, use_adj)

        with torch.no_grad():
            x_t = x.clone()
            x_t[keep_nodes] = 0.0
            x_t[keep_nodes] += self.enc_mask_token
            rep_t = self.target_encoder(x_t, use_adj)

        # 修复：确保cl_loss始终是torch张量
        if anchor_pair is not None:
            anchor, positive, negative = anchor_pair
            summary = self.avg_readout(enc_rep, [anchor, positive])
            num_mask_nodes = mask_nodes.size(0)
            neg_nodes = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
            cl_loss = self.dgi_loss(enc_rep[mask_nodes], enc_rep[neg_nodes], summary[mask_nodes])
        else:
            cl_loss = torch.tensor(0.0, device=x.device)  # 修复：改为torch张量

        rep = enc_rep
        rep = self.encoder_to_decoder(rep)
        rep[mask_nodes] = 0.0
        rep = self.projector(rep, use_adj)

        match_loss = self.match_loss(rep, rep_t, mask_nodes)
        recon = self.decoder(rep, use_adj)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)

        # 计算因果正则化损失
        causal_loss = self.compute_causal_regularization(enc_rep, use_adj)

        return match_loss, rec_loss, cl_loss, causal_loss

    def match_loss(self, rep, rep_t, mask_nodes, t=2):
        pox_x_index, pox_y_index = mask_nodes, mask_nodes
        neg_x_index, neg_y_index = self.generate_neg_nodes(mask_nodes)
        std_emb = F.normalize(rep.clone(), p=2, dim=-1)
        tgt_emb = F.normalize(rep_t.clone(), p=2, dim=-1)

        pox_x = std_emb[pox_x_index]
        pox_y = tgt_emb[pox_y_index]
        neg_x = std_emb[neg_x_index]
        neg_y = tgt_emb[neg_y_index]

        pos_cos = (0.5 * (1 + (pox_x * pox_y).sum(dim=-1))).pow(t)
        pos_loss = -torch.log(pos_cos)
        neg_cos = (0.5 * (1 + (neg_x * neg_y).sum(dim=-1))).pow(t)
        neg_loss = -torch.log(1 - neg_cos)
        loss = 0.5 * (pos_loss.mean() + neg_loss.mean())
        return loss

    def sce_loss(self, x, y, t=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        loss = -torch.log(cos_m.pow_(t))
        return loss.mean()

    def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
        anchor_arr = emb[anchor]
        positive_arr = emb[positive]
        negative_arr = emb[negative]
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output

    def forward(self, x, edge_index, anchor_pair):
        # 总是返回4个值，如果某些功能未启用，对应损失为0
        return self.mask_attr_prediction(x, edge_index, anchor_pair)

    @torch.no_grad()
    def evaluate(self, x, edge_index):
        # 编码器评估
        enc_rep = self.online_encoder(x, edge_index)
        rep = self.encoder_to_decoder(enc_rep)
        rep = self.projector(rep, edge_index)
        recon = self.decoder(rep, edge_index)
        return enc_rep, recon

    @torch.no_grad()
    def std_tgt_embedding(self, x, edge_index):
        s_rep = self.online_encoder(x, edge_index)
        t_rep = self.target_encoder(x, edge_index)
        return s_rep, t_rep

    def avg_readout(self, rep_pos_x, edge_index):
        # 处理 edge_index 类型
        if isinstance(edge_index, torch.sparse.Tensor):
            # 对于稀疏张量，转换为稠密获取边
            adj = edge_index.to_dense()
            src, dst = torch.nonzero(adj, as_tuple=True)
        elif isinstance(edge_index, list):
            # 如果edge_index是列表，假设它是[anchor, positive]
            # 直接计算这些点的特征均值
            anchor, positive = edge_index
            src = anchor
            dst = positive
        else:
            src, dst = edge_index

        neighbor_sum = scatter_add(rep_pos_x[src], dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = scatter_add(torch.ones_like(src, dtype=torch.float), dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = neighbor_count.clamp(min=1)
        summary = neighbor_sum / neighbor_count.unsqueeze(-1)
        return torch.sigmoid(summary)

    def discriminate(self, z, summary, sigmoid=True):
        assert isinstance(summary, torch.Tensor), "Summary should be a torch.Tensor"
        value = torch.matmul(z, torch.matmul(self.weight, summary.t()))
        return torch.sigmoid(value) if sigmoid else value

    def dgi_loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

    def CL_Loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        Cos_loss = -torch.log(1 - F.cosine_similarity(pos_z, neg_z) + 1e-15).mean()
        loss = Cos_loss + pos_loss + neg_loss  # 50
        return loss