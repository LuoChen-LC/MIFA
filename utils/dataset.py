import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

class ImagePairDataset(Dataset):
    def __init__(self, cfg, transform=None):
        self.cfg = cfg
        dataset_name = self.cfg.data.dataset
        
        if dataset_name == "LFW":
            self.image_dir = "Dataset/LFW/lfw_funneled"
            self.csv_path = "Dataset/LFW/lfw.csv"
        elif dataset_name == "CelebA-HQ":
            self.image_dir = "Dataset/CelebA-HQ/img_align_celeba"
            self.csv_path = "Dataset/CelebA-HQ/celeba_hq1.csv"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        
        if transform is not None:
            self.transform_fn = transform
        else:
            self.transform_fn = transforms.Compose([
                transforms.Resize(
                    (cfg.model.input_size, cfg.model.input_size), 
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True 
                ),
                transforms.CenterCrop(cfg.model.input_size), 
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Lambda(lambda img: to_tensor(img)),

            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        attacker_path = os.path.join(
            self.image_dir, str(self.df.iloc[idx]["attacker_img_name"])
        )
        victim_path = os.path.join(
            self.image_dir, str(self.df.iloc[idx]["victim_img_name"])
        )
        
        attacker_img = Image.open(attacker_path)
        victim_img = Image.open(victim_path)

        if self.transform_fn is not None:
            attacker_tensor = self.transform_fn(attacker_img)
            victim_tensor = self.transform_fn(victim_img)

        return {
            "attacker": attacker_tensor, 
            "victim": victim_tensor,
            "attacker_name": self.df.iloc[idx]["attacker_img_name"], 
            "victim_name": self.df.iloc[idx]["victim_img_name"]
        }
    
def get_eval_config(dataset_name):
    if dataset_name == "LFW":
        return "Dataset/LFW/lfw.csv", "Dataset/LFW/lfw_funneled"
    elif dataset_name == "CelebA-HQ":
        return "Dataset/CelebA-HQ/celeba_hq.csv", "Dataset/CelebA-HQ/img_align_celeba"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")