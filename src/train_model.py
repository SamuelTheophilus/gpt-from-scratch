from typing import Dict

import torch
import torch.nn as nn

from config import fetch_training_hyperparams, fetch_model_params, fetch_device
from components import Model
from main import get_batch, decode

TRAIN_EPOCHS, EVAL_EPOCHS, LEARNING_RATE = fetch_training_hyperparams()
DEVICE = fetch_device()
CONTEXT_LENGTH, N_EMBED, DROPOUT_RATE, BATCH_SIZE = fetch_model_params()
model = Model(vocab_size=65, context_length=CONTEXT_LENGTH, embedding_size=N_EMBED).to(DEVICE)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss() -> Dict:
    output_dict = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_EPOCHS)
        for eval_step in range(EVAL_EPOCHS):
            contexts, targets = get_batch(split)
            _, eval_loss = model(contexts, targets)
            losses[eval_step] = eval_loss.item()
        output_dict[split] = losses.mean()
    return output_dict


def train(training_model: nn.Module, epochs: int, eval_interval: int = 200) -> None:
    for step in range(epochs):
        if step % eval_interval == 0:
            output = estimate_loss()
            print(f"Training Loss:: {output['train']:.4f}\nValidation Loss::{output['val']:.4f}\n===================\n")
        xb, yb = get_batch("train")
        _, loss = training_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def generate_text(_model: nn.Module) -> str:
    eg_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_tokens = _model.generate(eg_idx, max_new_tokens=100)[0].tolist()
    generated_text = decode(generated_tokens)
    return f"Generated Text after training::\n{generated_text}"


train(training_model=model, epochs=TRAIN_EPOCHS)
print(generate_text(_model=model))

