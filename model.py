import torch
import torch.nn as nn



class PhyNetDTA(nn.Module):
    def __init__(self, smi_feat_dim=384, prot_feat_dim=480, embed_dim=256, num_heads=4, use_cross_attention=True):
        super(PhyNetDTA, self).__init__()
        self.use_cross_attention = use_cross_attention
        
        # Project heterogeneous features to a common embedding space
        self.smi_proj = nn.Linear(smi_feat_dim, embed_dim)
        self.prot_proj = nn.Linear(prot_feat_dim, embed_dim)
        
        # Cross-Attention modules
        self.attn_l2p = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_p2l = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        fc_in_dim = embed_dim * 2
        
        # Main branch
        self.fc_main = nn.Sequential(
            nn.Linear(fc_in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Physics branches (Enthalpy and Entropy)
        self.fc_h = nn.Sequential(
            nn.Linear(fc_in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.fc_s = nn.Sequential(
            nn.Linear(fc_in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, smi_feat, prot_feat, smi_mask=None, prot_mask=None):
        # smi_feat: (batch, seq_len, smi_feat_dim)
        # prot_feat: (batch, seq_len, prot_feat_dim)
        
        smi_seq = self.smi_proj(smi_feat)
        prot_seq = self.prot_proj(prot_feat)
        
        if self.use_cross_attention:
            # Ligand queries Protein (Where does ligand look in protein?)
            # key_padding_mask ignores padded tokens
            smi_attn_out, smi_attn_weights = self.attn_l2p(
                smi_seq, prot_seq, prot_seq, key_padding_mask=prot_mask
            )
            
            # Protein queries Ligand (Where does protein look in ligand?)
            prot_attn_out, prot_attn_weights = self.attn_p2l(
                prot_seq, smi_seq, smi_seq, key_padding_mask=smi_mask
            )
            
            # Residual connections
            smi_seq = smi_seq + smi_attn_out
            prot_seq = prot_seq + prot_attn_out
        else:
            smi_attn_weights = None
            prot_attn_weights = None
            
        # Apply padding mask before pooling to prevent picking up padded 0.0s as maximums
        if smi_mask is not None:
            smi_seq = smi_seq.masked_fill(smi_mask.unsqueeze(-1), -1e9)
        if prot_mask is not None:
            prot_seq = prot_seq.masked_fill(prot_mask.unsqueeze(-1), -1e9)
            
        # Global Max Pooling expects (batch, channels, length)
        smi_x = smi_seq.permute(0, 2, 1)
        prot_x = prot_seq.permute(0, 2, 1)
        
        smi_x = self.global_pool(smi_x).squeeze(-1) # (batch, embed_dim)
        prot_x = self.global_pool(prot_x).squeeze(-1) # (batch, embed_dim)
        
        concat_features = torch.cat([smi_x, prot_x], dim=1) # (batch, embed_dim * 2)
        
        y_main = self.fc_main(concat_features)
        h_pred = self.fc_h(concat_features)
        s_pred = self.fc_s(concat_features)
        
        return y_main, h_pred, s_pred, smi_attn_weights, prot_attn_weights
