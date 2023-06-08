import os

import torch
from torch.utils.data.datapipes.utils.decoder import imagehandler
from torchvision import transforms

from torchdata.datapipes.iter import IterableWrapper


def apply_transform(data):
    cal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data["image"] = cal_transform(data["image"])
    return data


def decode(item):
    key, value = item
    if key.endswith(".jpg"):
        decoder = imagehandler("pil")
        image = decoder("jpg", value.read())
        cls = int(os.path.split(os.path.dirname(key))[1].split(".")[0])
        id = os.path.basename(key).split(".")[0]
    else:
        raise ValueError("Unknown file type: " + key)

    return {"id": id, "image": image, "cls": cls}


def collate_fn(batch):
    if isinstance(batch[0]['image'], torch.Tensor):
        id = [sample['id'] for sample in batch]
        images = torch.stack([sample['image'] for sample in batch]).to(torch.float16)
        classes = torch.Tensor([sample['cls'] for sample in batch]).to(torch.float16)

    return {"id": id, "image": images, "cls": classes}


def Caltech256(_path):
    BATCH_SIZE = 120
    dp = IterableWrapper([_path])
    dp = dp.list_files(recursive=True).sharding_filter().open_files(mode="b")
    dp = dp.map(decode).map(apply_transform)
    dp = dp.batch(BATCH_SIZE)

    return dp
