import os
from functools import partial

import torch
from torch.utils.data.datapipes.iter import Collator, ShardingFilter, Shuffler
from torch.utils.data.datapipes.utils.decoder import imagehandler
from torch.utils.data.graph import traverse_dps
from torchvision import transforms

from torchdata.dataloader2.graph import find_dps, replace_dp
from torchdata.datapipes.iter import FileLister, IterableWrapper
from torchdata.remoteloader.substitute import SubstituteIterDataPipe


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
        data_id = os.path.basename(key).split(".")[0]
    else:
        raise ValueError("Unknown file type: " + key)

    return {"data_id": data_id, "image": image, "cls": cls}


def collate_fn(batch):
    if isinstance(batch[0]['image'], torch.Tensor):
        data_id = [sample['data_id'] for sample in batch]
        images = torch.stack([sample['image'] for sample in batch])
        classes = torch.Tensor([sample['cls'] for sample in batch])

    return {"data_id": data_id, "image": images, "cls": classes}


def Caltech256(_path):
    BATCH_SIZE = 120
    dp = FileLister(_path, recursive=True).shuffle().sharding_filter().open_files(mode="b").load_from_tar().shuffle()
    dp = dp.map(decode).map(apply_transform)
    dp = dp.batch(BATCH_SIZE).collate(collate_fn)

    return dp


def find_sub_dp(_datapipe, dp_type):
    graph = traverse_dps(_datapipe)

    def recursive_find(graph):
        for _, value in graph.items():
            if isinstance(value[0], dp_type):
                return value[0]
            else:
                return recursive_find(value[1])
    return recursive_find(graph)


if __name__ == "__main__":
    import time

    # start = time.time()
    dp = Caltech256("/home/jcsu/Dev/motivation/dataset/256_ObjectCategories.tar")
    # fn = dp.collate.args[-1].fn
    # dp2 = Caltech256("/home/jcsu/Dev/motivation/dataset/256_ObjectCategories")
    # fn = find_sub_dp(dp, FileLister)

    # print(fn)
    # a = find_dps(traverse_dps(dp), Collator)[0].fn
    replace_dp(
        traverse_dps(dp),
        find_dps(traverse_dps(dp), Shuffler)[-1],
        SubstituteIterDataPipe(find_dps(traverse_dps(dp), Shuffler)[-1])
    )
    print("-------------------")
    print(traverse_dps(dp))
    # start_time = time.time()
    # for i, data in enumerate(dp):
    #     print(i, data)
    # print("Time: ", time.time() - start_time)
