import warnings
import time

from datetime import datetime
import torch
import torchvision.transforms as transforms
from laplace import Laplace
from torch.utils.data import DataLoader

from src.data.dataloader import MURADataset
from src.models.models import CNN, CNN_3
from netcal.metrics import ECE
from src.models.utils import pred, save_laplace, load_laplace

batch_size = 32
shuffle = True
num_workers = 1

warnings.filterwarnings("ignore")


def laplace(model_path, hessian):
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
            model_path,
            map_location=torch.device(device),
        )
    )

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
    print(f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%} ")

    # Store the probabilities returned by Laplace
    date_time = datetime.now().strftime("%d-%m-%Y_%H")
    torch.save(probs_laplace, f"./reports/probs_laplace_{hessian}_{date_time}.pt")

    save_laplace(la, f"./models/laplace_{hessian}_{date_time}.pkl")


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
    la = load_laplace(la_path)
    if method == 'average':
        samples = la.sample(N).mean(axis=0)
    if method == 'intersect':
        samples = la.sample(N).min(axis=0)
    if method == 'union':
        samples = la.sample(N).max(axis=0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN_3(input_channels=3, input_height=256, input_width=256, num_classes=7).to(
        device
    )

    model.eval()

    model.load_state_dict(
        torch.load(
            'models/STATEtrained_model_epocs70_24-03-2022_22.pt',
            map_location=torch.device(device),
        )
    )

    # Change the parameters
    model.l_out.weight.data = torch.reshape(samples.T, (7,100))
    
    date_time = datetime.now().strftime("%d-%m-%Y_%H")
    torch.save(
        model.state_dict(),
        f"./models/BNN_{date_time}.pt",
    )
