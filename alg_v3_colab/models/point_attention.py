import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pointbert_layers import PointTransformer

class PositionalEncoding(nn.Module):

    def __init__(self, enc_dim, in_dim=None, enc_type='fourier', max_freq=10, freq_scale=0.1, dropout=None, concat=True, learnable_pos_index=None):
        super(PositionalEncoding, self).__init__()
        self.enc_dim = enc_dim
        self.in_dim = enc_dim if in_dim is None else in_dim
        self.enc_type = enc_type
        self.max_freq = max_freq
        self.freq_scale = freq_scale
        self.concat = concat
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        if concat:
            self.fc = nn.Linear(self.enc_dim + self.in_dim, self.enc_dim)
        if learnable_pos_index is not None:
            if not isinstance(learnable_pos_index, torch.Tensor):
                learnable_pos_index = torch.LongTensor(learnable_pos_index)
            self.learnable_pos_index = learnable_pos_index
            self.learned_pe_res = nn.Parameter(torch.zeros(learnable_pos_index.shape[0], self.enc_dim))
        else:
            self.learnable_pos_index = None

    def original_positional_encoding(self, pos):
        pos = pos.unsqueeze(-1)
        mul_term = torch.exp(torch.arange(0, self.enc_dim, 2).to(pos.device) * (-np.log(10000.0) / self.enc_dim))
        pe = torch.stack([torch.sin(pos * mul_term), torch.cos(pos * mul_term)], dim=-1)
        pe = pe.view(-1, self.enc_dim)
        return pe

    def fourier_positional_encoding(self, pos):
        pos = pos.unsqueeze(-1)
        num_freq = self.enc_dim // 2
        mul_term = torch.exp(torch.arange(num_freq).to(pos.device) * (np.log(self.max_freq) / num_freq)) * self.freq_scale
        pe = torch.stack([torch.sin(pos * mul_term), torch.cos(pos * mul_term)], dim=-1)
        pe = pe.view(-1, self.enc_dim)
        return pe

    def generate_positional_encoding(self, pos):
        if self.enc_type == 'original':
            pe = self.original_positional_encoding(pos)
        elif self.enc_type == 'fourier':
            pe = self.fourier_positional_encoding(pos)
        else:
            raise ValueError('Unknown positional encoding type!')

        if self.learnable_pos_index is not None:
            pe[self.learnable_pos_index] += self.learned_pe_res
        return pe


    def forward(self, x=None, pos=None, seq_dim=1, x_shape=None, device=None, pos_offset=0):
        if x is not None:
            x_shape = x.shape 

        if pos is None:
            pos = torch.arange(x_shape[seq_dim], device=device if x is None else x.device)
            if pos_offset > 0:
                pos += pos_offset
        pe = self.generate_positional_encoding(pos)

        for _ in range(len(x_shape) - seq_dim - 2):
            pe = pe.unsqueeze(1)
        for _ in range(seq_dim):
            pe = pe.unsqueeze(0)
        
        if x is not None:
            if self.concat:
                pe_exp = pe.expand(x_shape[:-1] + (self.enc_dim,))
                x = torch.cat([x, pe_exp], dim=-1)
                x = self.fc(x)
            else:
                x = x + pe
        else:
            x = pe.expand(x_shape[:-1] + (self.enc_dim,))

        if self.dropout is not None:
            x = self.dropout(x)
        return x





class AttnPointEnc(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Initialize PointBert-based Encoder
        self.PointBERTEnc = PointTransformer(config.point_bert)
        # if config.point_bert.pretrained != "None":
        #     self.PointBERTEnc.load_model_from_ckpt(bert_ckpt_path = config.point_bert.pretrained)
        
        # Implement Factorized Spatio-Temporal Encoder
        self.in_chans = config.in_chans # 1280
        self.hid_chans = config.hid_chans # 1024
        self.out_chans = config.out_chans # 512

        self.fc_1 = nn.Linear(self.in_chans, self.hid_chans)
        self.fc_2 = nn.Linear(self.hid_chans, self.out_chans)

        self.temporal_pos_enc_type = config.temporal_pos_enc_type # 'original'
        self.temporal_n_head = config.temporal_n_head # 8
        self.temporal_n_layer = config.temporal_n_layer # 4
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, self.hid_chans))
        self.spatial_pos_enc_type = config.spatial_pos_enc_type # 'original'
        self.spatial_n_head = config.spatial_n_head # 8
        self.spatial_n_layer = config.spatial_n_layer # 4
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, self.out_chans))
        self.pooling_mode = config.pooling_mode
        self.drop_rate = config.drop_rate # 0.1
        
        self.temporal_pos_enc = PositionalEncoding(
            enc_dim = self.hid_chans,
            enc_type = self.temporal_pos_enc_type,
            concat = True
        )
        _temporal_transformer_layer = nn.TransformerEncoderLayer(
            d_model = self.hid_chans,
            nhead = self.temporal_n_layer,
            dropout = self.drop_rate,
            batch_first = True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer = _temporal_transformer_layer,
            num_layers = self.temporal_n_layer
        )

        self.spatial_pos_enc = PositionalEncoding(
            enc_dim = self.out_chans,
            enc_type = self.spatial_pos_enc_type,
            concat = True
        )
        _spatial_transformer_layer = nn.TransformerEncoderLayer(
            d_model = self.out_chans,
            nhead = self.spatial_n_head,
            dropout = self.drop_rate,
            batch_first = True
        )
        self.spatial_transformer = nn.TransformerEncoder(
            encoder_layer = _spatial_transformer_layer,
            num_layers = self.spatial_n_layer
        )


        # Using the weight initialization introduced in https://github.com/huggingface/transformers/blob/204c54d411c2b4c7f31405203533a51632f46ab1/src/transformers/models/bert/modeling_bert.py#L731-L734
        torch.nn.init.normal_(self.spatial_cls_token, std=0.02)
        torch.nn.init.normal_(self.temporal_cls_token, std=0.02)

    
    def spatial_to_batch(self, x):
        assert len(x.shape) == 4
        B, G, T, C = x.shape
        return x.reshape(B*G, T, C)
    
    def batch_to_spatial(self, x, batch_size):
        C = x.shape[-1]
        return x.reshape(batch_size, -1, C)

    def pool(self, x):
        ndim = len(x.shape)
        if self.pooling_mode == "cls":
            if ndim == 3:
                return x[:, 0, :]
            elif ndim == 4:
                return x[:, 0, 0, :]
        elif self.pooling_mode == "avg":
            if ndim == 3:
                return torch.mean(x, dim=1)
            elif ndim == 4:
                return torch.mean(x, dim=(1, 2)) 

    def forward(self, pts):
        # pts.shape: B, T, N, C
        # PC encoding
        T = pts.shape[1]
        _ret = []
        for t in range(T):
            _ret.append(self.PointBERTEnc(pts[:, t, :, :].contiguous()).unsqueeze(1))
        x = torch.cat(_ret, dim = 1) # B, T, G, C = 1280
        batch_size = x.shape[0]
        x = x.swapaxes(1, 2)
        # B, G, T, C = x.shape
        x = self.fc_1(x) # B, T, G, hid_chans = 1024

        # Temporal branch
        x = self.spatial_to_batch(x) # B*G, T, hid_chans
        x = torch.cat((self.temporal_cls_token.expand(x.shape[0], -1, -1), x), dim=1) # B*G, T+1,hid_chans
        x = self.temporal_pos_enc(x) # B*G, T+1, hid_chans
        x = self.temporal_transformer(x) # B*G, T+1, hid_chans
        x = self.pool(x) # B*G, hid_chans
        
        # Spatial branch
        x = self.batch_to_spatial(x, batch_size) # B, G, hid_chans
        f_p_temporal = self.fc_2(x) # B, G, out_chans = 512
        x = f_p_temporal
        x = torch.cat((self.spatial_cls_token.expand(x.shape[0], -1, -1), x), dim=1) # B, G+1, out_chans
        x = self.spatial_pos_enc(x) # B, G+1, out_chans
        x = self.spatial_transformer(x) # B, G+1, out_chans
        f_p = self.pool(x) # B, out_chans

        return f_p_temporal, f_p # B, G, out_chans / B, out_chans
