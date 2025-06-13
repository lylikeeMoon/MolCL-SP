import torch
import torch.nn as nn
import numpy as np
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_size, max_len, dropout):
        super(Embeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.LayerNorm = LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)

        embeddings = self.LayerNorm(token_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_scores = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_scores


class EncoderLayer(nn.Module):
    def __init__(self, emb_size, intermediate_size, num_heads, attention_dropout, hidden_dropout):
        super(EncoderLayer, self).__init__()

        self.attention = Attention(emb_size, num_heads,
                                   attention_dropout, hidden_dropout)
        self.layernorm1 = nn.LayerNorm(emb_size)

        self.ffn = nn.Sequential(
            nn.Linear(emb_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, emb_size)
        )

        self.dropout = nn.Dropout(hidden_dropout)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, mask):
        attn_output, attn_weights = self.attention(x, mask)


        ffn_output = self.ffn(x)
        x = self.layernorm(x + self.dropout(ffn_output))

        return x, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        pos: [1, x.size(), d_model]
        '''
        pos = self.pe[:x.size(0), :]
        return self.dropout(pos)

class Encoder_1d(nn.Module):
    def __init__(self, num_layers, emb_size, intermediate_size, num_heads, attention_dropout, hidden_dropout):
        super(Encoder_1d, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(emb_size, intermediate_size, num_heads, attention_dropout, hidden_dropout)
            for _ in range(num_layers)
        ])
        self.position_embeddings = PositionalEncoding(d_model=emb_size, max_len=50)

    def forward(self, x, mask):
        position_embeddings = self.position_embeddings(x)
        x=x+position_embeddings
        for layer in self.layers:
            x, attn_weights = layer(x, mask)

        return x, attn_weights


class transformer_1d(nn.Sequential):
    def __init__(self):
        super(transformer_1d, self).__init__()
        input_dim_drug = 2586 + 1 # last for mask
        transformer_emb_size_drug = 128
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.emb = Embeddings(input_dim_drug,
                         transformer_emb_size_drug,
                         50,
                         transformer_dropout_rate)

        self.encoder = Encoder_1d(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
    def forward(self, emb, mask):
        e = emb.long()
        e_mask = mask.long()
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers, attention_scores = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers