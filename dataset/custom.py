import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImagePairDataset(Dataset):
    def __init__(self, root_a, root_b, transforma=None,transformb=None):
        self.root_a = root_a
        self.root_b = root_b
        self.transforma = transforma
        self.transformb = transformb
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for img_name in os.listdir(self.root_a):
            if img_name.lower().endswith((".png", ".jpg")):
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
        if self.transforma:
            img_a = self.transforma(img_a)
            img_b = self.transformb(img_b)
        return img_b, img_a , img_b #clean, corrupt, y

def build_transform_3ch(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        #can do like flips and stuff, but as we deal with NDVI sat-imagery, does not matter that much
        #I assume.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
def build_transform_1ch(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        #can do like flips and stuff, but as we deal with NDVI sat-imagery, does not matter that much
        #I assume.
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def build_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def build_dataset(opt,log,train):
    if train:
        transforma = build_transform_3ch(image_size=opt.image_size)
        transformb = build_transform_1ch(image_size=opt.image_size)
        root_a = os.path.join(opt.dataset_dir, "A", "train")
        root_b = os.path.join(opt.dataset_dir, "B", "train")
        dataset = ImagePairDataset(root_a, root_b, transforma, transformb)
    else:
        transforma = build_transform_3ch(image_size=opt.image_size)
        transformb = build_transform_1ch(image_size=opt.image_size)
        root_a = os.path.join(opt.dataset_dir, "A", "val")
        root_b = os.path.join(opt.dataset_dir, "B", "val")
        dataset = ImagePairDataset(root_a, root_b, transforma, transformb)
    return dataset

def build_test_dataset(opt,log,test):
    if test:
        transforma = build_transform_3ch(image_size=opt.image_size)
        transformb = build_transform_1ch(image_size=opt.image_size)
        root_a = os.path.join(opt.dataset_dir, "A", "test")
        root_b = os.path.join(opt.dataset_dir, "B", "test")
        dataset = ImagePairDataset(root_a, root_b, transforma, transformb)
    else:
        transforma = build_transform_3ch(image_size=opt.image_size)
        transformb = build_transform_1ch(image_size=opt.image_size)
        root_a = os.path.join(opt.dataset_dir, "A", "val")
        root_b = os.path.join(opt.dataset_dir, "B", "val")
        dataset = ImagePairDataset(root_a, root_b, transforma, transformb)
    return dataset