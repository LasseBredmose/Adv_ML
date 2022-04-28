import warnings

import torch
import torchvision.transforms as transforms
from netcal.metrics import ECE
from torch.utils.data import DataLoader

from src.data.dataloader import MURADataset
from src.models.models import CNN, CNN_nomax
from src.models.utils import pred

batch_size = 32
shuffle = True
num_workers = 1

warnings.filterwarnings("ignore")


def predict(model_path, mp):
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

    print("Model loaded")

    # Get targets
    targets = torch.cat([y for x, y in test_loader], dim=0)

    probs_cnn = pred(test_loader, model, laplace=False)
    acc_cnn = (probs_cnn.argmax(-1) == targets).float().sum() / len(targets)
    ece_cnn = ECE(bins=15).measure(probs_cnn.numpy(), targets.numpy())

    model_name = model_path.split("/")[-1].split(".")[0]
    torch.save(probs_cnn, f"./reports/cnn_probs/probs_{model_name}")

    print(f"Model path: {model_path}")
    print(f"[CNN] Acc.: {acc_cnn:.2%}; ECE: {ece_cnn:.2%}")
