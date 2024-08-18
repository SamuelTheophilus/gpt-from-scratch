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


@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_EPOCHS)
        for eval_step in range(EVAL_EPOCHS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[eval_step] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        logits = self.fc(logits)

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


model = BigramLanguageModel(vocab_size=data_vocab_size, embedding_size=EMBEDDING_LENGTH)
model = model.to(device=DEVICE)


# Optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)


# Training Loop
for step in range(TRAIN_EPOCHS):
    if step % 200 == 0:
        output = estimate_loss()
        print(f"Training Loss:: {output['train']:.4f}\nValidation Loss::{output['val']:.4f}\n===================\n")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

eg_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
generated_tokens = model.generate(eg_idx, max_new_tokens=100)[0].tolist()
generated_text = decode(generated_tokens)
print(f"Generated Text after training::\n{generated_text}")
