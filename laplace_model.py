import warnings

import torch
import torchvision.transforms as transforms
from laplace import Laplace
from torch.utils.data import DataLoader

from src.data.dataloader import MURADataset
from src.models.models import CNN

batch_size = 32
shuffle = True

warnings.filterwarnings("ignore")

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
    model.load_state_dict(
        torch.load("./models/STATEtrained_model_epocs1_17-03-2022_14.pt")
    )

    # User-specified LA flavor
    la = Laplace(
        model, "classification", subset_of_weights="all", hessian_structure="diag"
    )
    la.fit(train_loader)
    la.optimize_prior_precision(method="CV", val_loader=validation_loader)

    # User-specified predictive approx.
    pred = la(x, link_approx="probit")
