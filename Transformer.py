#!/usr/bin/python3

"""
Created on: 03 Nov 2023 18:54:00
By: Michael Koch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import sqrt


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.Q = nn.Linear(embed_dim, head_dim)
        self.K = nn.Linear(embed_dim, head_dim)
        self.V = nn.Linear(embed_dim, head_dim)

    @staticmethod
    def scaled_dot_product_attention(Q, K, V, mask):
        # print(Q.size())
        # print(K.size())
        # print(V.size())
        dim_k = Q.size(-1)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(dim_k)
        # print(attn_scores.size())
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # print(attn_scores)
        # print(attn_scores.size())
        weights = F.softmax(attn_scores, dim=-1)
        attn = torch.bmm(weights, V)

        return attn

    def forward(self, x, mask):
        attn_output = AttentionHead.scaled_dot_product_attention(
            self.K(x), self.Q(x), self.V(x), mask
        )

        return attn_output


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(self.model_dim, self.head_dim) for _ in range(self.num_heads)]
        )
        self.out_linear = nn.Linear(model_dim, model_dim)

    def forward(self, x, mask):
        output = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        output = self.out_linear(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, dr=0.9):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dr)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dr=0.9):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(model_dim)
        self.layernorm_2 = nn.LayerNorm(model_dim)
        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.ff = FeedForward(model_dim, ff_dim, dr=0.9)

    def forward(self, x, mask):
        x = x + self.attention(self.layernorm_1(x), mask)
        x = x + self.ff(self.layernorm_2(x))

        return x


class Embeddings(nn.Module):
    def __init__(self, vocab_size, model_dim, max_seq_len, dr=0.9):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-12)
        self.dropout = nn.Dropout(dr)

    def forward(self, src):
        seq_length = src.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # print("position_ids: ", position_ids)
        word_embeddings = self.word_embedding(src)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = word_embeddings + position_embeddings
        # print("embeddings size: ", embeddings.size())
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Transformer(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, ff_dim, max_seq_len, dr=0.9):
        super().__init__()
        self.embedding = Embeddings(vocab_size, model_dim, max_seq_len, dr)
        self.layers = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ff_dim, dr) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(model_dim, vocab_size)
        self.dropout = nn.Dropout(dr)

    @staticmethod
    def create_causal_mask(sent):
        sent_mask = (sent != 0).unsqueeze(1)
        # print(sent_mask)
        # print("sent_mask size: ", sent_mask.size())
        seq_len = sent.size(1)
        causal_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        # print("causal mask: ", causal_mask)
        mask = sent_mask & causal_mask
        # print("final mask\n", mask.size())
        return mask

    def forward(self, x):
        # print(x.size())
        mask = Transformer.create_causal_mask(x)
        # print(mask.size())
        x = self.embedding(x)
        # print(x.size())
        for layer in self.layers:
            x = layer(x, mask)
        x = self.fc(x)
        preds = self.dropout(x)

        return preds

