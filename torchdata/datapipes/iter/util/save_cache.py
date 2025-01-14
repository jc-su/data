# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import mmap
import os

from typing import Any, Callable, Iterator, Optional, Tuple, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import dill
U = Union[bytes, bytearray, str]


@functional_datapipe("hierarchy_cache")
class HierarchyCacheIterDataPipe(IterDataPipe[str]):
    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[Any, U]],
        priority: Optional[dict] = None,
    ):
        self.source_datapipe: IterDataPipe[Tuple[Any, U]] = source_datapipe
        self.memory_cache_path = "/dev/shm"
        self.nvme_cache_path = "/mnt/nvme"
        self.priority = priority

    def __iter__(self) -> Iterator[str]:
        for data_id, data in self.source_datapipe:
            if data_id in self.priority["memory"]:
                cache_path = os.path.join(self.memory_cache_path, data_id)
            elif data_id in self.priority["nvme"]:
                cache_path = os.path.join(self.nvme_cache_path, data_id)
            else:
                raise Exception("Unknown data id")
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            # with open(os.path.join(cache_path, "data"), "wb") as f:
            #     f.write(dill.dumps(data))
            # mmmap write
            with open(os.path.join(cache_path, "data"), "wb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as m:
                    m.write(dill.dumps(data))

            yield cache_path

    def __len__(self) -> int:
        return len(self.source_datapipe)
