import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 通用跨模态注意力模块
# ==========================================
class CrossModalAlignment(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, key_padding_mask=None):
        # query: [B, L_q, D], key: [B, L_k, D]
        attn_out, attn_weights = self.mha(query, key, key, key_padding_mask=key_padding_mask)
        # 残差连接与归一化
        out = self.norm(query + self.dropout(attn_out))
        return out, attn_weights


# ==========================================
# 2. MSGA-RNA: 多尺度全局-局部对齐模型
# ==========================================
class MSGA_RNA(nn.Module):
    def __init__(self, args):
        super(MSGA_RNA, self).__init__()
        self.args = args
        dim_hid = args['embed_dim']
        dropout = args['dropout']
        nhead = args['nhead']

        # --- 1. 输入投影层 ---
        # RNA 分支
        self.rna_orig_proj = nn.Linear(args['rinalmo_dim'], dim_hid)
        self.rna_window_proj = nn.Linear(args['rinalmo_dim'], dim_hid)

        # 小分子分支
        self.mol_global_proj = nn.Linear(args['chemberta_dim'], dim_hid)
        self.mol_frag_proj = nn.Linear(args['melm_dim'], dim_hid)

        # --- 2. 特征增强 (Transformer) ---
        enc_layer = nn.TransformerEncoderLayer(d_model=dim_hid, nhead=nhead, batch_first=True, dropout=dropout)
        num_mol_layers = args.get('mol_encoder_layers', 2)
        self.mol_frag_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_mol_layers)

        # --- 3. Bi-level cross-modal alignment module ---
        # Coarse-grained branch: ligand-level representation + full-length RNA representation.
        self.macro_align = CrossModalAlignment(dim_hid, nhead, dropout)
        # Fine-grained branch: fragment-level ligand representation + k-mer-level RNA representation.
        self.micro_align = CrossModalAlignment(dim_hid, nhead, dropout)

        # --- 4. 辅助预测头 (Auxiliary Heads) ---
        # 强制要求每一轨特征都能独立进行初步预测，增强表征区分度
        self.aux_head_macro = nn.Linear(dim_hid, args['n_output'])
        self.aux_head_micro = nn.Linear(dim_hid, args['n_output'])

        # --- 5. 最终融合预测头 ---
        fusion_dim = args.get('fusion_hidden_dim', dim_hid // 2)
        self.fc_fusion = nn.Sequential(
            nn.Linear(dim_hid * 2, dim_hid),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hid, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, args['n_output'])
        )

    def forward(self, rna_orig, rna_window, mol_global, mol_frags,
                rna_orig_mask=None, rna_window_mask=None, frag_mask=None):
        """
        Args:
            rna_orig: RNA全长特征 [B, L1, 1280]
            rna_window: RNA窗口特征 [B, L2, 1280]
            mol_global: 分子全局特征 [B, 384]
            mol_frags: 分子碎片特征 [B, Nf, 768]
        """
        # 1. 投影与增强
        h_rna_orig = self.rna_orig_proj(rna_orig)  # [B, L1, D]
        h_rna_win = self.rna_window_proj(rna_window)  # [B, L2, D]

        v_mol_glob = self.mol_global_proj(mol_global)  # [B, D]
        h_mol_frag = self.mol_frag_proj(mol_frags)  # [B, Nf, D]
        h_mol_frag = self.mol_frag_encoder(h_mol_frag, src_key_padding_mask=frag_mask)

        # 2. Coarse-grained alignment branch
        q_macro = v_mol_glob.unsqueeze(1)
        v_macro_seq, attn_macro = self.macro_align(q_macro, h_rna_orig, key_padding_mask=rna_orig_mask)
        v_macro = v_macro_seq.squeeze(1)
        self._current_attn_macro = attn_macro

        # 3. Fine-grained alignment branch
        h_micro_seq, attn_micro = self.micro_align(h_mol_frag, h_rna_win, key_padding_mask=rna_window_mask)
        self._current_attn_micro = attn_micro

        # Mask-aware 池化获取微观向量
        if frag_mask is not None:
            mask_expanded = (~frag_mask).unsqueeze(-1).to(h_micro_seq.dtype)
            v_micro = (h_micro_seq * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        else:
            v_micro = h_micro_seq.mean(dim=1)  # [B, D]

        # 4. 辅助预测 (用于计算 Loss_aux)
        out_macro = self.aux_head_macro(v_macro)
        out_micro = self.aux_head_micro(v_micro)

        # 5. 最终特征融合预测
        combined = torch.cat([v_macro, v_micro], dim=-1)  # [B, 2*D]
        out_main = self.fc_fusion(combined)

        return {
            'out': out_main,
            'out_macro': out_macro,
            'out_micro': out_micro,
            'v_macro': v_macro,
            'v_micro': v_micro,
            'attn_macro': getattr(self, '_current_attn_macro', None),
            'attn_micro': getattr(self, '_current_attn_micro', None)
        }


# ==========================================
# 3. 适配 5080 的测试函数
# ==========================================
def test_run():
    args = {
        'embed_dim': 256,
        'rinalmo_dim': 1280,
        'melm_dim': 768,
        'chemberta_dim': 384,
        'dropout': 0.1,
        'nhead': 8,
        'n_output': 1
    }

    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 初始化模型并转为半精度 (FP16) 节省显存
    model = MSGA_RNA(args).to(device).half()
    model.eval()

    # 构造模拟输入 (Batch=4, RNA_Len=200, Frag_Len=12)
    b, r, f = 4, 200, 12
    rna_o = torch.randn(b, r, 1280, device=device).half()
    rna_w = torch.randn(b, r - 10, 1280, device=device).half()
    mol_g = torch.randn(b, 384, device=device).half()
    mol_f = torch.randn(b, f, 768, device=device).half()

    with torch.no_grad():
        res = model(rna_o, rna_w, mol_g, mol_f)

    print("-" * 30)
    print(f"✅ 模型前向传播成功！")
    print(f"主输出形状:   {res['out'].shape}")
    print(f"辅助输出(宏): {res['out_macro'].shape}")
    print(f"辅助输出(微): {res['out_micro'].shape}")
    print("-" * 30)

    # 计算总损失示例 (训练时参考)
    # loss_total = loss_fn(res['out'], target) + 0.3 * (loss_fn(res['out_macro'], target) + loss_fn(res['out_micro'], target))


if __name__ == "__main__":
    test_run()
