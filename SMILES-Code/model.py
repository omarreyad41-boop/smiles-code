

import torch
import torch.nn as nn
import math


# مهم لاستقرار التدريب فكرته بدل ان تكون القيم كبيرة جدا او صغيرة جدا يجعلها متوازنة
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        # رقم صغير جدا حتى لانقسم على صفر هنا عشرة اُس سالب ستة
        self.eps = eps
        # ينشئ باراميتر قابل للتعلم كل القيم تبدأ بواحد- كلمة ان ان باراميتور هو تينسور يجب ان يتعلمه النموذج ويعدله خلال التدريب كفكرة تحسين اوبتمايزيشن
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # يحصل على المتوسط ديم سالب واحد يعني اخر دايمونشين  لانه يحسب المتوسط عبر دي مودل
        mean = x.mean(-1, keepdim=True)
        # يحصل على الانحراف المعياري
        std = x.std(-1, keepdim=True)
        # معادلة نورماليزيشن..هذا النموذج ينتج تعديل النتيجة خلال التدريب
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncodeing(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.softmax(dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ value, scores

    def forward(self, q, k, v, mask):
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        B = Q.shape[0]
        Q = Q.view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.h, self.d_k).transpose(1, 2)
        x, _ = self.attention(Q, K, V, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.w_o(x)


# ENCODE:


class EncoderBlock(nn.Module):
    def __init__(self, features, self_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residuals = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residuals[0](
            x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residuals[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# DECODER


class DecoderBlock(nn.Module):
    def __init__(self, features, self_attention, cross_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residuals = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residuals[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residuals[1](x, lambda x: self.cross_attention(
            x, encoder_output, encoder_output, src_mask))
        x = self.residuals[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


# Projection

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)


# Transformer

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self, src, src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection(x)


def bulid_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 256,
        N: int = 4,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 512,
) -> Transformer:
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncodeing(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncodeing(d_model, tgt_seq_len, dropout)

    encoder_block = nn.ModuleList([
        EncoderBlock(
            d_model,
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout,
        )
        for _ in range(N)
    ])

    decoder_block = nn.ModuleList([
        DecoderBlock(
            d_model,
            MultiHeadAttentionBlock(d_model, h, dropout),
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout,
        )
        for _ in range(N)
    ])

    transformer = Transformer(
        Encoder(d_model, encoder_block),
        Decoder(d_model, decoder_block),
        src_embed, tgt_embed, src_pos, tgt_pos,
        ProjectionLayer(d_model, tgt_vocab_size),
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
