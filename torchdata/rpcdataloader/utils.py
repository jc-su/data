import random

import numpy as np
import torch


def unpickle_tensor(buffer, dtype, shape):
    return torch.frombuffer(buffer, dtype=dtype).view(shape)


def pickle_tensor(t):
    return unpickle_tensor, (t.ravel().numpy().view("b"), t.dtype, t.shape)


pkl_dispatch_table = {torch.Tensor: pickle_tensor}


def set_random_seeds(base_seed, worker_id):
    """Set the seed of default random generators from python, torch and numpy.

    This should be called once on each worker.
    Note that workers may run tasks out of order, so this does not ensure
    reproducibility, only non-redundancy between workers.

    Example:

    >>> base_seed = torch.randint(0, 2**32-1, [1]).item()
    >>> for i, (host, port) in enumerate(workers):
    ...     rpc_async(host, port, set_random_seeds, args=[base_seed, i])
    """

    seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)
    np_seed = torch.utils.data._utils.worker._generate_state(base_seed, worker_id)
    np.random.seed(np_seed)
