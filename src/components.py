import torch
import torch.nn as nn
from torch.nn import functional as F

from config import fetch_model_params, fetch_device
DEVICE = fetch_device()
CONTEXT_LENGTH, N_EMBED, DROPOUT_RATE, _, NUM_HEADS, NUM_BLOCKS = fetch_model_params()


class Head(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        # The head_size determines the final output of the "C" dimension after the masked self-attention is complete.
        self.key_layer = nn.Linear(N_EMBED, head_size, bias=False)
        self.query_layer = nn.Linear(N_EMBED, head_size, bias=False)
        self.value_layer = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)))
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.scale = head_size ** -0.5
        # The scale is introduced to control the variance of the weights so that softmax doesn't converge towards one-hot vectors.
        # When this happens every other token will aggregate information from a single token. (the max weight value before softmax is applied)

    def forward(self, tokens: torch.tensor):
        B, T, C = tokens.shape
        key = self.key_layer(tokens)     # shape: B, T, head_size
        query = self.query_layer(tokens)    # shape: B, T, head_size
        value = self.value_layer(tokens)    # shape: B, T, head_size

        weights = key @ query.transpose(-2, -1) * self.scale    # Transpose for matrix multiplication. (B, T, head_size @ B, head_size, T) = B, T, T
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        output = weights @ value    # shape: B, T, head_size (B, T, T @ B, T, head_size)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)     # num_heads * head_size = embedding_size
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, tokens):
        out = torch.cat([h(tokens) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),   # Projection layer
            nn.Dropout(DROPOUT_RATE)
        )

    def forward(self, tokens: torch.tensor) -> torch.tensor:
        return self.net(tokens)


class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.heads = MultiHeadAttention(num_heads, head_size)
        self.ffn = FeedForward(embedding_size=n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, tokens: torch.tensor) -> torch.tensor:
        tokens = self.ln1(tokens)
        inputs = tokens + self.heads(tokens)    # Residual connections to maintain gradient flow.
        normalised_inputs = self.ln2(inputs)
        outputs = inputs + self.ffn(normalised_inputs)      # Residual connections to enhance gradient flow.
        return outputs


class Model(nn.Module):
    def __init__(self, vocab_size: int, context_length: int,  embedding_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(context_length, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size, NUM_HEADS) for _ in range(NUM_BLOCKS)])
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape    # T represents the context_length to be passed into the position_embedding_table
        token_embed = self.token_embedding_table(idx)   # shape: (B, T, C)
        position_embed = self.position_embedding_table(torch.arange(T, device=DEVICE))  # shape: (T, C)
        inputs = position_embed + token_embed   # shape: (B, T, C)  = position_embed (T, C) + token_embed (B, T, C) broadcasting of the B dimension.
        inputs = self.blocks(inputs)
        inputs = self.layer_norm(inputs)
        logits = self.fc(inputs)    # shape B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.tensor, max_new_tokens: int) -> torch.tensor:
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -CONTEXT_LENGTH:]
            logits, _ = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

