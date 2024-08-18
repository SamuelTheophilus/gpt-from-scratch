from os.path import join
from typing import Tuple, Dict, List
import torch

from config import fetch_device, fetch_model_params
DEVICE = fetch_device()
CONTEXT_LENGTH, EMBEDDING_LENGTH, DROPOUT_RATE, BATCH_SIZE, _, _ = fetch_model_params()

# Read in text
torch.manual_seed(42)


def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def tokenization(data: str):
    all_chars = sorted(list(set(data)))
    return all_chars


def build_vocab(characters) -> Tuple[Dict, Dict]:
    string_to_index = {char: i for i, char in enumerate(characters)}
    index_to_string = {i: char for i, char in enumerate(characters)}

    return string_to_index, index_to_string


def encode(s_in: str, string_to_index: Dict) -> List[int]:
    return [string_to_index[char] for char in s_in]


def decode(l_in: List[int], index_to_string: Dict) -> str:
    return "".join([index_to_string[index] for index in l_in])


def split_data(split_ratio: float, data: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    split = int(split_ratio * len(data))
    train, val = data[:split], data[split:]
    return train, val


DATA_PATH = join("..", "data", "input.txt")
input_data = read_data(DATA_PATH)
chars = tokenization(input_data)
stoi, itos = build_vocab(chars)
VOCAB_SIZE = len(chars)
encoded_data = torch.tensor(encode(input_data, stoi), dtype=torch.long)


def get_batch(data_type: str) -> Tuple[torch.tensor, torch.tensor]:
    train_data, val_data = split_data(0.9, encoded_data)
    data = train_data if data_type == "train" else val_data
    rand_idx = torch.randint((len(data) - CONTEXT_LENGTH), (BATCH_SIZE, ))
    xb = torch.stack([data[i:i+CONTEXT_LENGTH] for i in rand_idx])
    yb = torch.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in rand_idx])
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    return xb, yb
