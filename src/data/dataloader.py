from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch


class MURADataset(Dataset):
    def __init__(self, data_dir, annotation_file, paths_file, transform=None):
        anno_df = pd.read_csv(f"{data_dir}/{annotation_file}", names=["file", "label"])
        anno_df["label"] = anno_df.file.str.split("/").str[2].str[3:]

        files_df = pd.read_csv(f"{data_dir}/{paths_file}", names=["img_file"])

        files_df["file"] = files_df["img_file"].str.split("image").str[0]
        anno_df = files_df.merge(anno_df, on="file")

        self.data_dir = data_dir
        self.annotations = anno_df[["img_file", "label"]]
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index]["img_file"]
        img = Image.open(f"{self.data_dir}/{img_path}").convert("RGB")
        label = torch.tensor(float(self.annotations.iloc[index]["label"]))

        if self.transform is not None:
            img = self.transform(img)

        return (img, torch.FloatTensor(label))

    def __len__(self):
        return len(self.annotations)


dataset = MURADataset(
    "data", "MURA-v1.1/train_labeled_studies.csv", "MURA-v1.1/train_image_paths.csv"
)

img, label = dataset[0]
print(img)
