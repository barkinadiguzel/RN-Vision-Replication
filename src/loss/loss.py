import torch.nn as nn

def build_loss(task="classification"):
    if task == "classification":
        return nn.CrossEntropyLoss()
    elif task == "regression":
        return nn.MSELoss()
    else:
        raise ValueError("Unknown task type")
