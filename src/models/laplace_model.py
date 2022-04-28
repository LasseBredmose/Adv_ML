import time
import warnings
from datetime import datetime

import torch
import torchvision.transforms as transforms
from laplace import Laplace
from netcal.metrics import ECE
from torch.utils.data import DataLoader

from src.data.dataloader import MURADataset
from src.models.models import CNN, CNN_nomax
from src.models.utils import load_laplace, pred, save_laplace

batch_size = 32
shuffle = True
num_workers = 1

warnings.filterwarnings("ignore")


def laplace(model_path, hessian, mp):
    # Cuda Stuff
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforming the data, such that they all follow the same path
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    test_loader = DataLoader(
        dataset=test_set,
        shuffle=False,  # very important!
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if mp == 0:
        model = CNN_nomax(
            input_channels=3, input_height=256, input_width=256, num_classes=7
        ).to(device)
    else:
        model = CNN(
            input_channels=3, input_height=256, input_width=256, num_classes=7
        ).to(device)

    model.eval()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device(device),
        )
    )
    print("model loaded")
    # Get targets
    targets = torch.cat([y for x, y in test_loader], dim=0)

    # User-specified LA flavor
    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure=hessian,
    )
    t0 = time.time()
    la.fit(train_loader)
    t1 = time.time()
    print(f"Time elapsed (fit): {t1-t0} seconds")

    # la.optimize_prior_precision(method="CV", val_loader=validation_loader)

    la.optimize_prior_precision(method="marglik")

    probs_laplace = pred(test_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().sum() / len(targets)
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    print(f"[Laplace] Acc.: {acc_laplace:.2%}; ECE: {ece_laplace:.2%} ")

    # Store the probabilities returned by Laplace
    '''date_time = datetime.now().strftime("%d-%m-%Y_%H")

    model_name = model_path.split('/')[-1].split('.')[0]
    torch.save(probs_laplace, f"./reports/probs_laplace_{hessian}_{date_time}_{model_name}.pt")

    save_laplace(la, f"./models/laplace_{hessian}_{date_time}_{model_name}.pkl")'''


def laplace_eval(la_path):
    # Transforming the data, such that they all follow the same path
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_set = MURADataset(
        "data",
        "MURA-v1.1/valid_labeled_studies.csv",
        "MURA-v1.1/valid_image_paths.csv",
        transform=transform,
    )

    test_loader = DataLoader(
        dataset=test_set,
        shuffle=False,  # very important!
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Get targets
    targets = torch.cat([y for x, y in test_loader], dim=0)

    la = load_laplace(la_path)
    print("Laplace loaded!")

    probs_laplace = pred(test_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().sum() / len(targets)
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    print(f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%} ")


def laplace_sample(la_path, N, method):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    la = load_laplace(la_path)

    la._device = torch.device(device)

    la_sample = la.sample(N)

    if method == "average":
        samples = la_sample.mean(axis=0)
    if method == "intersect":
        samples = la_sample.min(axis=0).values
    if method == "union":
        samples = la_sample.max(axis=0).values

    model = CNN(input_channels=3, input_height=256, input_width=256, num_classes=7).to(
        device
    )

    model.eval()

    model.load_state_dict(
        torch.load(
            "models/STATEtrained_model_epocs100_21_04_21_trans_1_layers_5_arr_0.pt",
            map_location=torch.device(device),
        )
    )

    model.l_out.weight.data = torch.reshape(
        torch.reshape(samples, (257, 7))[:-1], (7, 256)
    )
    model.l_out.bias.data = torch.reshape(samples, (257, 7))[-1]

    hess = la_path.split('_')[1]

    date_time = datetime.now().strftime("%d-%m-%Y_%H")
    torch.save(
        model.state_dict(),
        f"./models/BNN_{hess}_{method}_{date_time}.pt",
    )
    