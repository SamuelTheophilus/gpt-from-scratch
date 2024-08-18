import os.path
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import fetch_device, fetch_model_params, fetch_training_hyperparams

DEVICE = fetch_device()
CONTEXT_LENGTH, EMBEDDING_LENGTH, DROPOUT_RATE, BATCH_SIZE = fetch_model_params()
TRAIN_EPOCHS, EVAL_EPOCHS, LEARNING_RATE = fetch_training_hyperparams()
torch.manual_seed(42)

DATA_PATH = os.path.join("..", "data", "input.txt")
with open(DATA_PATH, "r") as f:
    input_data = f.read()

# Naive Tokenization
chars = sorted(list(set(input_data)))
data_vocab_size = len(chars)

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}

encode = lambda s_in: [stoi[c] for c in s_in]
decode = lambda i_in: "".join([itos[i] for i in i_in])

encoded_data = torch.tensor(encode(input_data), dtype=torch.long)

# Split data into training and validation
split = int(0.9 * len(encoded_data))
train_data, val_data = encoded_data[:split], encoded_data[split:]


def get_batch(data_type: str) -> (torch.tensor, torch.tensor):
    data = train_data if data_type == "train" else val_data
    rand_idx = torch.randint((len(data) - CONTEXT_LENGTH), (BATCH_SIZE, ))
    xb = torch.stack([data[i:i+CONTEXT_LENGTH] for i in rand_idx])
    yb = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in rand_idx])
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    return xb, yb


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int,  embedding_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(context_length, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape    # T represents the context_length to be passed into the position_embedding_table
        token_embed = self.token_embedding_table(idx)   # shape: (B, T, C)
        position_embed = self.position_embedding_table(torch.arange(T, device=DEVICE))  # shape: (T, C)
        inputs = position_embed + token_embed   # shape: (B, T, C)  = position_embed (T, C) + token_embed (B, T, C) broadcasting of the B dimension.
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



