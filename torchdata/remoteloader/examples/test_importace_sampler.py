import hashlib
import os
from typing import (Any, AnyStr, Callable, Dict, Iterator, Optional, Tuple,
                    Union)

import dill
import numpy as np
from scipy.stats import norm

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

U = Union[bytes, bytearray, str]





def order_by(info, key):
    sorted_result = sorted(info.items(), key=lambda item: item[1][key])
    sorted_ids = [item[0] for item in sorted_result]

    yield from sorted_ids


def importance_sampling(data, num_samples=100, replace=True):
    # Set seed for reproducibility
    np.random.seed(0)

    # Extract keys and corresponding 'loss_rank' values
    keys = []
    loss_info = []
    num_samples = max(num_samples, len(data))
    for key, value in data.items():
        keys.append(key)
        loss_info.append(value['loss'])
    loss_info = np.array(loss_info)

    # Compute mean and standard deviation of the data
    mean = np.max(loss_info)
    std_dev = np.std(loss_info)

    # Compute probabilities from the normal distribution
    probabilities = norm.pdf(loss_info, mean, std_dev)

    # Normalize the probabilities so they sum to 1
    probabilities = probabilities / np.sum(probabilities)

    # Use np.random.choice to generate samples
    samples = np.random.choice(keys, size=num_samples, replace=replace, p=probabilities)

    # shuffle samples
    np.random.shuffle(samples)

    return samples


if __name__ == "__main__":
    loss_path = "loss_time_archive/loss_tensor_{}.pkl"
    compute_time_path = "loss_time_archive/compute_time_list_{}.pkl"
    id_path = "loss_time_archive/id_list_{}.pkl"
    rank_path = "loss_time_archive/rank_info_{}.pkl"
    for i in range(1):
        # loss_info = dill.load(open(loss_path.format(i), "rb"))
        # compute_time_info = dill.load(open(compute_time_path.format(i), "rb"))
        # id_list = dill.load(open(id_path.format(i), "rb"))
        # rank_info = dill.load(open(rank_path.format(i), "rb"))
        # id_list = np.array(id_list)
        # compute_time_info = np.array(compute_time_info)
        # # print(compute_time_info.max(), compute_time_info.min(), compute_time_info.std())
        # result = {
        #     str(data_id): {"loss_info": float(loss), "compute_time": float(compute_time)}
        #     for i, compute_time in enumerate(compute_time_info)
        #     for data_id, loss in zip(id_list[i], loss_info[i])
        # }
        # # for i in result:
        # #     print(i, result[i]["loss_info"])
        # # print(len(set(sample_data(result))), len(result.keys()))

        # # print(result["9d8586390d0f8bb8578c5ff8943ffc02"])
        # print(rank_info)

        # # print(_order_by_loss(result))
        # sample_list = sample_data(result, len(result.keys()))
        # print(sample_list)
        # # print(sample_list)
        # _path = "/home/jcsu/Dev/motivation/dataset/256_ObjectCategories"
        # dp = IterableWrapper([_path]).list_files(recursive=True).shuffle().sharding_filter().substitute()
        # dp2 = IterableWrapper([_path]).list_files(recursive=True).shuffle().sharding_filter().substitute(sample_list)
        # import time
        # start_time = time.time()
        # for i in dp2:
        #     print(i)
        # t1 = time.time()-start_time

        # start_time = time.time()
        # for i in dp:
        #     print(i)
        # print(time.time()-start_time, t1)
        info = dill.load(open(f"importance_info_{i}.pkl", "rb"))

        print(info)
        samples = importance_sampling(info)
        print(samples)
        print(len(samples))
        print(len(set(samples)))
