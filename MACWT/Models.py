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
    """
    空间转录组小波变换模块

    结构（对应论文 Section 2.2）:
      分解: X → decomposed1(d/2) → decomposed2(d/4)     [渐进抽象]
      重建: decomposed2 → reconstructed1(d/2) → reconstructed(d) [对称恢复]
      门控融合: α ⊙ reconstructed + (1-α) ⊙ residual_proj(X)    [自适应平衡]

    decomposed2 = 最粗尺度（全局域模式）
    residual = X - reconstructed = 高频细节
    α = 每个维度的粗/细偏好（可学习）
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.wavelet_levels = 2

        # 小波分解层
        d_h1 = max(1, in_features // 2)
        d_h2 = max(1, in_features // 4)
        self.decompose1 = nn.Linear(in_features, d_h1)
        self.decompose2 = nn.Linear(d_h1, d_h2)

        # 小波重建层
        self.reconstruct1 = nn.Linear(d_h2, d_h1)
        self.reconstruct2 = nn.Linear(d_h1, out_features)

        # 残差投影
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = nn.Identity()

        # 门控融合向量 α（论文 Eq.3）
        # 存储 logit(α)，forward 时 sigmoid 转为 (0,1)
        self.alpha_logit = nn.Parameter(torch.ones(out_features) * 0.8473)  # sigmoid(0.8473) ≈ 0.7

    def forward(self, x, return_all_scales=False):
        """
        Args:
            x: [N, in_features]
            return_all_scales: 如果 True，返回所有中间尺度（可视化用）
        Returns:
            默认: x_wavelet [N, out_features]
            return_all_scales=True: dict {
                'decomposed1': (N, d_h1),
                'decomposed2': (N, d_h2),    ← 最粗尺度
                'reconstructed1': (N, d_h1),
                'reconstructed': (N, out_features), ← 重建（平滑）
                'residual': (N, out_features),    ← 高频残差
                'x_wavelet': (N, out_features),   ← 融合输出
                'alpha': (out_features,),         ← 门控权重
            }
        """
        original_features = x

        # 小波分解 (Eq.1)
        decomposed1 = F.relu(self.decompose1(x))      # (N, d_h1)
        decomposed2 = F.relu(self.decompose2(decomposed1))  # (N, d_h2) ← 最粗

        # 小波重建 (Eq.2)
        reconstructed1 = F.relu(self.reconstruct1(decomposed2))  # (N, d_h1)
        reconstructed = self.reconstruct2(reconstructed1)         # (N, out_features)

        # 残差投影
        residual = self.residual_proj(original_features)  # (N, out_features)

        # 门控融合 (Eq.3): α ⊙ reconstructed + (1-α) ⊙ residual
        alpha = torch.sigmoid(self.alpha_logit)  # (out_features,)
        x_wavelet = alpha * reconstructed + (1. - alpha) * residual

        if return_all_scales:
            return {
                'decomposed1': decomposed1,
                'decomposed2': decomposed2,          # 最粗尺度
                'reconstructed1': reconstructed1,
                'reconstructed': reconstructed,       # 重建（平滑）
                'residual': residual - reconstructed, # 高频残差
                'x_wavelet': x_wavelet,               # 融合输出
                'alpha': alpha,                       # 门控权重
            }
        return x_wavelet


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
        )  #Q=YWQ,K=YWK,V=YWV

        # 前馈网络
        self.ffn = nn.Sequential(      #FFN(Y2)=W2⋅GELU(W1Y2+b1)+b2
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

    def forward(self, x, spatial_attn_mask=None, gene_attn_mask=None): #Mspatial[i,j]=0,Aij=1 −∞,otherwiseZspatial=softmax(QK⊤dk+Mspatial)V,#(7) Mgene[i,j]=0,Agene[ψ(i),ψ(j)]=1−∞,otherwiseZgene=softmax(QK⊤dk+Mgene)V,    (8)
        """
        Args:
            x: [num_nodes, hidden_dim]  多尺度特征矩阵 Y
            spatial_attn_mask: [num_nodes, num_nodes]  空间因果mask
                              0.0 = 允许注意力（有因果边）
                              -inf = 阻断注意力（无因果边）
            gene_attn_mask: [num_nodes, num_nodes]  基因因果mask
                            0.0 = 允许注意力（共调控）
                            -inf = 阻断注意力（非共调控）
        Returns:
            Z: [num_nodes, hidden_dim]  因果增强的特征矩阵
        """

        spatial_output, _ = self.spatial_attention(
            x, x, x, attn_mask=spatial_attn_mask
        )
        x = self.norm1(x + self.dropout(spatial_output))   #Y1=LayerNorm(Y+Dropout(Zspatial))

        # 通道2 — 基因表达因果注意力（模拟转录共调控）
        # M^G 从基因共表达网络 G^G 导出：仅共调控基因间允许信息流
        gene_output, _ = self.gene_attention(
            x, x, x, attn_mask=gene_attn_mask
        )
        x = self.norm2(x + self.dropout(gene_output))   #Y2=LayerNorm(Y1+Dropout(Zgene))

        # 前馈网络 — 非线性因果特征增强
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))  #Z'=LayerNorm(Y2+Dropout(FFN(Y2)))

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
        self.use_wavelet = config.get('use_wavelet', True)  # 如果你想默认只开小波，可改为 True
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

    def _build_spatial_causal_mask(self, edge_index, num_nodes):
        """
        从3D空间邻接图构建空间因果注意力mask M^S.

        论文公式:
            M^S_ij = 1  if e_ij ∈ G  (空间因果边存在)
            M^S_ij = 0   otherwise

        实现: 在softmax前用加法mask
            mask_ij = 0.0   if e_ij ∈ G   → softmax允许注意力
            mask_ij = -inf  otherwise      → softmax阻断注意力

        Returns:
            spatial_mask: [num_nodes, num_nodes]
        """
        mask = torch.full((num_nodes, num_nodes), float('-inf'),
                          device=edge_index.device if isinstance(edge_index, torch.Tensor) else 'cpu')

        # 解析edge_index获取源-目标对
        if isinstance(edge_index, torch.Tensor):
            if edge_index.is_sparse:
                edge_index = edge_index.coalesce()
                src, dst = edge_index.indices()
            elif edge_index.dim() == 2:
                src, dst = edge_index[0], edge_index[1]
            else:
                return torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        else:
            return torch.zeros(num_nodes, num_nodes)

        # 设置空间因果边: 0.0 表示允许注意力
        mask[src, dst] = 0.0
        mask[dst, src] = 0.0  # 无向图, 双向因果
        # 自环: 每个spot对自身有因果影响
        mask[torch.arange(num_nodes), torch.arange(num_nodes)] = 0.0

        return mask

    def _build_gene_causal_mask(self, features, topk_ratio=0.1):
        """
        从基因共表达网络构建基因表达因果注意力mask M^G.

        论文:
            基于联合表达矩阵计算成对Pearson相关, 阈值化保留显著边 → G^G
            M^G_ij = 1  if (g_i, g_j) ∈ G^G  (基因共调控)
            M^G_ij = 0   otherwise

        代码实现 (以PCA特征余弦相似度近似表达相关性):
            1. 归一化特征向量
            2. 计算余弦相似度矩阵 S [N,N]
            3. 对每个spot保留topk最相似的邻居 (模拟"共调控"阈值化)
            4. mask_ij = 0.0 (相似度高, 允许因果注意) / -inf (不相似, 阻断)

        Args:
            features: [num_nodes, hidden_dim]
            topk_ratio: 保留topk比例相似邻居 (默认10%)
        Returns:
            gene_mask: [num_nodes, num_nodes]
        """
        num_nodes = features.shape[0]
        topk = max(1, int(num_nodes * topk_ratio))

        # 余弦相似度矩阵 [N,N]
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.matmul(features_norm, features_norm.T)

        # 对每个spot保留topk最相似邻居 (模拟Pearson相关阈值化)
        _, topk_indices = torch.topk(similarity, k=min(topk, num_nodes), dim=1)

        # 构建mask: 默认阻断(-inf), topk相似邻居允许(0.0)
        gene_mask = torch.full((num_nodes, num_nodes), float('-inf'),
                               device=features.device)
        row_idx = torch.arange(num_nodes, device=features.device).unsqueeze(1).expand(-1, topk)
        gene_mask[row_idx, topk_indices] = 0.0
        # 自环始终允许
        gene_mask[torch.arange(num_nodes), torch.arange(num_nodes)] = 0.0

        return gene_mask

    def forward(self, x, edge_index, return_wavelet_features=False, return_causal_features=False):
        # 基础特征提取
        base_features = self.encoder(x)

        # 应用小波变换
        if self.use_wavelet and return_wavelet_features:
            wf = self.wavelet_module(base_features, return_all_scales=True)
            wavelet_features = wf['x_wavelet']
        elif self.use_wavelet:
            wavelet_features = self.wavelet_module(base_features, return_all_scales=False)
        else:
            wavelet_features = base_features

        # 应用因果推理 (带因果结构mask)
        if self.use_causal:
            # 构建空间因果mask M^S: 基于3D邻接图 G
            spatial_attn_mask = self._build_spatial_causal_mask(
                edge_index, wavelet_features.shape[0]
            )
            # 构建基因因果mask M^G: 基于特征共表达相似度
            gene_attn_mask = self._build_gene_causal_mask(wavelet_features)

            # 因果双通道注意力 (M^S + M^G 约束)
            causal_features = self.causal_module(
                wavelet_features,
                spatial_attn_mask=spatial_attn_mask,
                gene_attn_mask=gene_attn_mask
            )
        else:
            causal_features = wavelet_features

        # 图卷积 (在因果特征上进一步传播)
        x = self.gc1(causal_features, edge_index)
        x = self.gc2(x, edge_index)

        # 构建小波特征字典
        if self.use_wavelet and return_wavelet_features:
            wf['causal_features'] = causal_features
            wf['gcn1_features'] = x  # gc2输入
            if return_causal_features:
                return x, wf, causal_features
            return x, wf

        # 返回因果模块输出用于正则化
        if return_causal_features:
            return x, causal_features

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
        self.use_wavelet = True  # 默认启用小波变换
        self.use_causal = True  # 默认启用因果推理
        self.causal_reg_weight = config.get('causal_reg_weight', 0.01)
        self.gamma1 = config.get('gamma1', 0.5)  # γ₁: spatial smoothness
        self.gamma2 = config.get('gamma2', 0.2)  # γ₂: feature independence

        print(f"\n增强的SpaCross模型初始化 (默认启用增强功能):")
        print(f"  输入维度: {input_dim}")
        print(f"  潜在维度: {self.dec_in_dim}")
        print(f"  使用小波变换: {self.use_wavelet}")
        print(f"  使用因果推理: {self.use_causal}")
        print(f"  因果正则化权重: {self.causal_reg_weight}")
        print(f"  γ₁ (spatial): {self.gamma1:.2f}, γ₂ (independence): {self.gamma2:.2f}")

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

    def compute_causal_regularization(self, causal_features, edge_index):
        """
        因果正则化损失 L_causal = γ₁ L_spatial + γ₂ L_indep

        L_spatial (因果马尔可夫条件):
            空间因果图中相邻的spot应具有相似表征
            L_spatial = E_{(i,j)~E} ||Z_i - Z_j||²₂

        L_indep (独立因果机制):
            去相关化:不同因果因子应独立运作
            L_indep = ||Cov(Z) - I||_F

        Args:
            causal_features: 因果模块输出 Z [N, hidden_dim]
            edge_index: 3D空间邻接图的边集
        Returns:
            标量因果正则化损失
        """
        if not self.use_causal or self.causal_reg_weight <= 0:
            return torch.tensor(0.0, device=causal_features.device)

        # 处理不同格式的 edge_index
        if isinstance(edge_index, torch.sparse.Tensor):
            edge_index = edge_index.coalesce()
            src, dst = edge_index.indices()
        elif isinstance(edge_index, tuple) and len(edge_index) == 2:
            src, dst = edge_index
        elif isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            src, dst = edge_index[0], edge_index[1]
        elif isinstance(edge_index, list):
            if len(edge_index) == 0:
                return torch.tensor(0.0, device=causal_features.device)
            edge_tensor = torch.tensor(edge_index[0], device=causal_features.device)
            src, dst = edge_tensor[:, 0], edge_tensor[:, 1]
        else:
            return torch.tensor(0.0, device=causal_features.device)

        # L_spatial: 因果马尔可夫条件 — 因果图邻居的表征应相似
        if len(src) > 0 and len(dst) > 0 and src.numel() == dst.numel():
            src_features = causal_features[src]
            dst_features = causal_features[dst]
            spatial_loss = F.mse_loss(src_features, dst_features)
        else:
            spatial_loss = torch.tensor(0.0, device=causal_features.device)

        # L_indep: 独立因果机制 — 特征维度去相关
        if causal_features.size(0) > 1:
            features_normalized = F.normalize(causal_features, dim=0)
            correlation = torch.matmul(features_normalized.T, features_normalized)
            identity = torch.eye(correlation.size(0), device=causal_features.device)
            causal_indep_loss = torch.mean((correlation - identity).abs())
        else:
            causal_indep_loss = torch.tensor(0.0, device=causal_features.device)

        # L_causal = γ₁ L_spatial + γ₂ L_indep
        total_loss = self.gamma1 * spatial_loss + self.gamma2 * causal_indep_loss

        return total_loss * self.causal_reg_weight

    def mask_attr_prediction(self, x, edge_index, anchor_pair):
        use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, edge_index, self.mask_rate)

        # 编码器前向传播 — 同时获取因果模块输出用于正则化
        enc_rep, causal_features = self.online_encoder(
            use_x, use_adj, return_causal_features=True
        )

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

        # 计算因果正则化损失 — 在因果模块输出Z上施加
        causal_loss = self.compute_causal_regularization(causal_features, use_adj)

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
    def get_wavelet_features(self, x, edge_index):
        """
        【可视化专用】提取小波模块各尺度特征。

        必须满足两个条件才会返回有效的小波特征:
          1. config 中 use_wavelet = True
          2. 模型已通过 Encoder(use_wavelet=True) 初始化

        Returns:
            (latent, wavelet_dict) 其中 wavelet_dict 包含:
                - decomposed1:  中间尺度 (N, d_h1)
                - decomposed2:  最粗尺度 (N, d_h2) ← 全局域模式
                - reconstructed: 重建特征 (N, out_features)
                - residual:      高频残差 (N, out_features)
                - x_wavelet:     门控融合输出 (N, out_features)
                - alpha:         门控权重 (out_features,)
                - causal_features: 因果推理输出 (N, out_features)
        """
        self.eval()
        result = self.online_encoder(x, edge_index, return_wavelet_features=True)
        if isinstance(result, tuple):
            latent, wf = result
            return latent, wf
        else:
            # 小波未启用，返回 None
            return result, None

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