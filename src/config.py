import os
import torch
from dotenv import load_dotenv
load_dotenv()


def fetch_training_hyperparams():
    """

    :return:
        training_epochs: number of training epochs
        evaluation_epochs: number of evaluation epochs
        learning_rate: learning rate
    """
    training_epochs = os.getenv("TRAINING_EPOCHS")
    evaluation_epochs = os.getenv("EVALUATION_EPOCHS")
    learning_rate = os.getenv("LEARNING_RATE")

    return int(training_epochs), int(evaluation_epochs), float(learning_rate)


def fetch_device():
    """

    :return:
        device: (mps, cuda, cpu)
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return device


def fetch_model_params():
    """
    Returns the model params specified in the .env folder

    :return:
        context_length
        embeddings_length
        dropout rate
        batch_size
    """
    context_length = os.getenv("CONTEXT_LENGTH")
    embeddings_length = os.getenv("N_EMBEDDINGS")

    dropout_rate = os.getenv("DROPOUT_RATE")
    batch_size = os.getenv("BATCH_SIZE")
    return int(context_length), int(embeddings_length), float(dropout_rate), int(batch_size)

