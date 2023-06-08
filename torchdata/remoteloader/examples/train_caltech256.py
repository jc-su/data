import os.path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.datapipes.utils.decoder import imagehandler
from torchvision import models, transforms

from torchdata.dataloader2 import MultiProcessingReadingService, DataLoader2
from torchdata.datapipes.iter import IterableWrapper
from torchdata.remoteloader import RemoteDataloader, Worker


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
        id = os.path.basename(key).split(".")
    else:
        raise ValueError("Unknown file type: " + key)

    return {"id": id, "image": image, "cls": cls}


def collate_fn(batch):
    if isinstance(batch[0]['image'], torch.Tensor):
        id = [sample['id'] for sample in batch]
        images = torch.stack([sample['image'] for sample in batch]).to(torch.bfloat16)
        classes = torch.Tensor([sample['cls'] for sample in batch]).to(torch.long)

    return {"id": id, "image": images, "cls": classes}


def Caltech256(_path):
    BATCH_SIZE = 288
    dp = IterableWrapper([_path])
    dp = dp.list_files(recursive=True).shuffle().sharding_filter().open_files(mode="b")
    dp = dp.map(decode).map(apply_transform)
    dp = dp.batch(BATCH_SIZE, drop_last=True).collate(collate_fn)

    return dp

def ImportaceSampler(rank_info, num_samples, replacement=False):
    # rank_info: list of rank tensor
    import numpy as np
    rank_info = np.concatenate(rank_info)



def cal_loss_rank(loss_tensor):
    total_sum, sublist_sums = loss_tensor.sum(), loss_tensor.sum(dim=1)
    rank_tensor = (loss_tensor.view(-1) * total_sum) / sublist_sums.repeat(loss_tensor.size(1))
    return rank_tensor.argsort().detach().cpu().numpy()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = "/home/jcsu/Dev/motivation/dataset/256_ObjectCategories"
    rs = MultiProcessingReadingService(num_workers=8)
    dl = DataLoader2(
        Caltech256(data_path),
        reading_service=rs,
        )

    model = models.resnet18(weights=None).to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(100):
        # if epoch > 1:
        #     dp = ImportaceSampler(data_path)
        loss_list = []
        id_list = []
        train_loss_list = []
        compute_time_list = []
        for idx, mini_batch in enumerate(dl):
            collated_batch = mini_batch
            images = collated_batch["image"].to(device)
            labels = collated_batch["cls"].to(device)
            data_id = collated_batch["id"]
            optimizer.zero_grad()
            id_list.append(data_id)
            start_time = time.time()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss)
            compute_time_list.append(time.time() - start_time)
            print(f"Batch {idx}")
                # print(f"Batch {idx} loss: {loss}")
            optimizer.zero_grad()
            train_loss = loss.mean()
            train_loss_list.append(train_loss)
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)

            scaler.step(optimizer)
            scaler.update()
        loss_tensor = torch.stack(loss_list)
        rank_info = cal_loss_rank(loss_tensor)
        # zip(id_list, rank_tensor)
        import dill
        dill.dump(id_list, open(f"loss_time_archive/id_list_{epoch}.pkl", "wb"))
        dill.dump(loss_tensor.detach().cpu().numpy(), open(f"loss_time_archive/loss_tensor_{epoch}.pkl", "wb"))
        dill.dump(compute_time_list, open(f"loss_time_archive/compute_time_list_{epoch}.pkl", "wb"))
        dill.dump(rank_info, open(f"loss_time_archive/rank_info_{epoch}.pkl", "wb"))