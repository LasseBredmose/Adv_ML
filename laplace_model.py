import warnings

import dill

import torch
import torchvision.transforms as transforms
from laplace import Laplace
from torch.utils.data import DataLoader

from src.data.dataloader import MURADataset
from src.models.models import CNN
from netcal.metrics import ECE

batch_size = 32
shuffle = True
num_workers = 1

warnings.filterwarnings("ignore")


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x))
        else:
            py.append(torch.softmax(model(x), dim=-1))

    return torch.cat(py).cpu()


def save_laplace(la, filepath):
    with open(filepath, "wb") as outpt:
        dill.dump(la, outpt)


if __name__ == "__main__":
    # Cuda Stuff
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = torch.cuda.is_available()

    # Transforming the data, such that they all follow the same path
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = MURADataset(
        "data",
        "MURA-v1.1/train_labeled_studies.csv",
        "MURA-v1.1/train_image_paths.csv",
        transform=transform,
    )

    train_set, validation_set = torch.utils.data.random_split(
        dataset, [25765, len(dataset) - 25765]
    )

    test_set = MURADataset(
        "data",
        "MURA-v1.1/valid_labeled_studies.csv",
        "MURA-v1.1/valid_image_paths.csv",
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_set,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        dataset=validation_set,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_set,
        shuffle=False,  # very important!
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = CNN(input_channels=3, input_height=256, input_width=256, num_classes=7).to(
        device
    )

    model.eval()

    model.load_state_dict(
        torch.load(
            "./models/STATEtrained_model_epocs70_24-03-2022_22.pt",
            map_location=torch.device(device),
        )
    )

    print("Model loaded")

    # Get targets
    targets = torch.cat([y for x, y in test_loader], dim=0)

    # User-specified LA flavor
    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="diag",
    )
    la.fit(train_loader)
    print("Finished fitting")

    # la.optimize_prior_precision(method="CV", val_loader=validation_loader)

    la.optimize_prior_precision(method="marglik")

    print("Hyperparameters optimized")

    probs_map = predict(test_loader, model, laplace=False)
    acc_map = (probs_map.argmax(-1) == targets).float().sum() / len(targets)
    ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}")

    probs_laplace = predict(test_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().sum() / len(targets)
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    print(f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%} ")

    # Store the probabilities returned by Laplace
    torch.save(probs_laplace, "./reports/probs_laplace.pt")

    save_laplace(la, "./models/laplace.pkl")
