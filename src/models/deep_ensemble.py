from os import listdir

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from netcal.metrics import ECE
from torch.utils.data import DataLoader

from src.data.dataloader import MURADataset
from src.models.cam import cam


def deep_ensemble(num_models):
    folder_path = "reports/cnn_probs/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        shuffle=False,
    )

    paths = sorted(listdir(folder_path))[:num_models]

    def prob_loader(path):
        return torch.load(
            f"reports/cnn_probs/{path}",
            map_location=torch.device(device),
        )

    probs = [prob_loader(p) for p in paths]
    de_prob = torch.stack(probs).mean(axis=0)

    targets = torch.tensor(test_loader.dataset.annotations["label"].values)

    acc = (de_prob.argmax(-1) == targets).float().sum() / len(targets)

    ece = ECE(bins=15).measure(de_prob.numpy(), targets.numpy())

    print(f"[CNN] Acc.: {acc:.2%}; ECE: {ece:.2%}")


def ensemble_cam(num_models, image, mp):
    model_folder = "models/deep_ensemble"
    models = sorted(listdir(model_folder))[:num_models]

    CAMs = [cam(image, f"models/deep_ensemble/{m}", mp)[0] for m in models]
    de_cam = np.stack(CAMs).mean(axis=0)
    img = cv2.imread(image)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(
        cv2.resize(de_cam, (width, height)).astype(np.uint8), cv2.COLORMAP_JET
    )
    result = heatmap * 0.3 + img * 0.5

    cv2.imwrite(f"deepensemble_{num_models}_{'_'.join(image.split('/')[3:])}", result)
