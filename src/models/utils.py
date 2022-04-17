import torch
import dill
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

use_cuda = torch.cuda.is_available()


def get_variable(x):
    # Converts tensors to cuda, if available.
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    # Get numpy array for both cuda and not.
    if use_cuda:
        """return x.cpu().item()
        return x.item()"""
        return x.cpu().data.numpy()
    return x.data.numpy()


@torch.no_grad()
def pred(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(get_variable(x)))
        else:
            py.append(torch.softmax(model(get_variable(x)), dim=-1))

    return torch.cat(py).cpu()


def save_laplace(la, filepath):
    with open(filepath, "wb") as outpt:
        dill.dump(la, outpt)


def load_laplace(filepath):
    with open(filepath, "rb") as input:
        la = pickle.load(input)
    return la
