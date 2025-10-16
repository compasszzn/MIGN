''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gelu(x):
   return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class RelativePosition(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, pos_mat):
        """pos_mat: relative position matrix, n * n * pos_dim"""
        assert pos_mat.shape[0] == pos_mat.shape[1]
        # all seq share one relative positional matrix
        n_element = pos_mat.shape[0]
        pos_dim = pos_mat.shape[-1]
        positions = pos_mat.view(-1, pos_dim)
        pos_embeddings = self.linear_2(self.linear_1(positions))

        # [sz_b x len_q x len_q x d_v/d_k]
        return pos_embeddings.view(n_element, n_element, -1)  # added: batch_size dim


class RelativeScaledDotProductAttention(nn.Module):
    ''' attn: sum over element-wise product of three vectors'''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, a_k, d_k, d_v, n_head, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Transpose for attention dot product: sz_b x n_head x len_q x dv
        # Separate different heads: sz_b x len_q x n_head x dv
        r_q1, r_k1, r_v1 = q.view(sz_b, len_q, n_head, d_k).permute(0, 2, 1, 3), \
                           k.view(sz_b, len_q, n_head, d_k).permute(0, 2, 1, 3), \
                           v.view(sz_b, len_v, n_head, d_v).permute(0, 2, 1, 3)

        # r_q1: [sz_b, n_head, len_q, 1, d_k], r_k1: [sz_b, n_head, 1, len_q, d_k]
        attn1 = torch.mul(r_q1.unsqueeze(2), r_k1.unsqueeze(3))
        # attn1: [sz_b, n_head, len_q, len_q, d_k], a: [len_q, len_q, d_k]
        attn = torch.sum(torch.mul(attn1, a_k), -1)
        attn = attn / self.temperature  # [sz_b x n_head x len_q x len_k]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, r_v1)

        return output, attn

class NewRelativeMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, pos_dim, temperature, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # self.relative_position = RelativePosition(pos_dim, d_k, d_k)
        self.attention = nn.MultiheadAttention(4,1)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, pos_mat, mask=None):
        d_k, d_v, d_model, n_head = self.d_k, self.d_v, self.d_model, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        q = self.w_qs(q)#torch.Size([1, 42664, 64])
        k = self.w_ks(k)#torch.Size([1, 42664, 64])
        v = self.w_vs(v)#torch.Size([1, 42664, 64])

        # generate the spatial relative position embeddings (SRPEs)
        # a_k = self.relative_position(pos_mat)

        if mask is not None:  # used to achieve Shielded Attention
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, a_k, d_k, d_v, n_head, mask=mask)

        # Transpose to move the head dimension back: sz_b x len_q x n_head x dv
        # Combine the last two dimensions to concatenate all the heads together: sz_b x len_q x (n_head*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class NewRelativeEncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, d_pos, dropout=0.1, temperature=None):
        super(NewRelativeEncoderLayer, self).__init__()
        if temperature is None:
            temperature = d_k ** 0.5

        self.slf_attn = NewRelativeMultiHeadAttention(n_head, d_model, d_k, d_v, d_pos, temperature, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, pos_mat, attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, pos_mat, mask=attn_mask)#(1,40000,2)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class TwoLayerFCN(nn.Module):
    def __init__(self, feat_dim, n_hidden1, n_hidden2):
        super().__init__()
        self.feat_dim = feat_dim
        self.linear_1 = nn.Linear(feat_dim, n_hidden1)
        self.linear_2 = nn.Linear(n_hidden1, n_hidden2)

    def forward(self, in_vec, non_linear=False):
        """pos_vec: absolute position vector, n * feat_dim"""
        assert in_vec.shape[-1] == self.feat_dim, f"in_vec.shape: {in_vec.shape}, feat_dim:{self.feat_dim}"

        if non_linear:
            mid_emb = F.relu(self.linear_1(in_vec))
        else:
            mid_emb = self.linear_1(in_vec)

        out_emb = self.linear_2(mid_emb)
        return out_emb


class SpaFormer(nn.Module):
    def __init__(self, d_feat=1, d_pos=2, n_layers=3, n_head=2, d_k=16, d_v=16,
                 d_model=16, d_inner=256, dropout=0.1, scale_emb=False, return_attns=False, temperature=None):
        super().__init__()

        self.d_model = d_model
        self.scale_emb = scale_emb
        self.return_attns = return_attns

        self.feature_enc = TwoLayerFCN(d_feat, d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList(
            [NewRelativeEncoderLayer(d_model, d_inner, n_head, d_k, d_v, d_pos, dropout=dropout, temperature=temperature)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.decoder = TwoLayerFCN(d_model, d_model, 1)

    def forward(self, feat_seq,  masked_pos,r_pos_mat=None, attn_mask=None):
        feat_seq = feat_seq.unsqueeze(0)
        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.feature_enc(feat_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5

        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, r_pos_mat, attn_mask=attn_mask)
            enc_slf_attn_list += [enc_slf_attn] if self.return_attns else []

        masked_pos = masked_pos[:, :, None].expand(-1, -1, enc_output.size(-1))  # [batch_size, max_pred, d_model]

        # get masked position from final output of transformer.
        h_masked_1 = torch.gather(enc_output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked_2 = self.layer_norm(self.activ2(self.linear(h_masked_1)))
        dec_output = self.decoder(h_masked_2)  # [batch_size, max_pred, n_vocab]

        if self.return_attns:
            return dec_output, h_masked_1, h_masked_2, enc_slf_attn_list
        return dec_output, h_masked_1, h_masked_2

