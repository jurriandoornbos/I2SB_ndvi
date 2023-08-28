import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImagePairDataset(Dataset):
    def __init__(self, opt, transform=None):
        self.root_a = os.path.join(opt.dataset_dir, "A")
        self.root_b = os.path.join(opt.dataset_dir,  "B")
        self.transform = transform
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for img_name in os.listdir(self.root_a):
            if img_name.lower().endswith(".png"):
                img_path_a = os.path.join(self.root_a, img_name)
                img_path_b = os.path.join(self.root_b, img_name)
                samples.append((img_path_a, img_path_b))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path_a, img_path_b = self.samples[index]
        img_a = Image.open(img_path_a).convert('RGB')
        img_b = Image.open(img_path_b).convert('RGB')
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b

def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        #can do like flips and stuff, but as we deal with NDVI sat-imagery, does not matter that much
        #I assume.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def build_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def build_dataset(opt,log,train):
    transform = build_transform(image_size=opt.image_size)
    dataset = ImagePairDataset(opt, transform)
    return dataset