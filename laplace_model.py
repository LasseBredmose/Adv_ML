import warnings

import torch
import torchvision.transforms as transforms
from laplace import Laplace
from torch.utils.data import DataLoader

from src.data.dataloader import MURADataset
from src.models.models import CNN

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
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = CNN(input_channels=3, input_height=256, input_width=256, num_classes=7).to(
        device
    )

    model.eval()

    model.load_state_dict(
        torch.load(
            "./models/STATEtrained_model_epocs2_24-03-2022_14.pt",
            # map_location=torch.device("cpu"),
        )
    )

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
    # la.optimize_prior_precision(method="CV", val_loader=validation_loader)
    la.optimize_prior_precision(method="marglik")

    # User-specified predictive approx.
    # pred = la(test_loader, link_approx="probit")

    probs_laplace = predict(test_loader, la, laplace=True)

    torch.save(probs_laplace, "./reports/probs_laplace.pt")
    print(f"Shape: {probs_laplace.shape}")
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().sum() / 3197

    print(f"[Laplace] Acc.: {acc_laplace:.1%}")
