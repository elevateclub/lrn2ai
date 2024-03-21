import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        self.encoding = self.encoding.to(x.device)
        return x + self.encoding[:, :x.size(1)]

def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        attention = F.scaled_dot_product_attention(query, key, value, mask)

        attention = attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = attention.view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)

        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, key_padding_mask=mask, need_weights=False)  # Self attention
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # Add & Norm

        ffn_output = self.ffn(out1)  # Feed Forward
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # Add & Norm

        return out2

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(d_model, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # Self attention with look ahead mask
        attn1, _= self.mha1(x, x, x, attn_mask=look_ahead_mask, need_weights=False)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        # Encoder-decoder attention
        attn2, _ = self.mha2(out1, enc_output, enc_output, key_padding_mask=padding_mask, need_weights=False)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        # Feed forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, pe_input, pe_target, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding_input = PositionalEncoding(d_model, pe_input)
        self.pos_encoding_target = PositionalEncoding(d_model, pe_target)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_decoder_layers)])

        self.dropout = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_input = self.dropout(self.pos_encoding_input(self.encoder_embedding(inp)))
        dec_input = self.dropout(self.pos_encoding_target(self.decoder_embedding(tar)))

        enc_output = enc_input
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, enc_padding_mask)

        dec_output = dec_input
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output
    
    @torch.no_grad()
    def generate(self, in_idx, out_idx, max_new_tokens, temperature=1.0, top_k=None):
        block_size = 16 # TODO
        device = next(self.parameters()).device

        enc_padding_mask = create_padding_mask(in_idx).to(device)

        for i in range(max_new_tokens):
            # idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size]
            look_ahead_mask = create_look_ahead_mask(out_idx.size(1)).to(device)
            logits = self(in_idx, out_idx, enc_padding_mask, look_ahead_mask)
            logits = logits[:, -1, :]/temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            out_idx = torch.cat((out_idx, idx_next), dim=1)

            if idx_next == special_tokens['<eos>']:
                break
        return out_idx


special_tokens = {
    '<sos>': 0,
    '<eos>': 1,
    '<pad>': 2,
    '<unk>': 3
} 

def create_padding_mask(seq):
    return (seq == special_tokens['<pad>']).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask.masked_fill(mask==1, float('-inf'))

