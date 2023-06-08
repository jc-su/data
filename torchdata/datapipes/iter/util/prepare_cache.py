import os
import mmap

from typing import Iterator

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import dill


@functional_datapipe("hierarchy_fetch")
class HierarchyFetchIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe[str], priority):
        self.source_datapipe = source_datapipe
        self.priority = priority
        self.memory_cache_path = "/dev/shm"
        self.nvme_cache_path = "/mnt/nvme"

    def __iter__(self) -> Iterator:
        for data_id in self.source_datapipe:
            if data_id in self.priority["memory"]:
                cache_path = os.path.join(self.memory_cache_path, data_id)
            elif data_id in self.priority["nvme"]:
                cache_path = os.path.join(self.nvme_cache_path, data_id)
            else:
                continue
            data_path = os.path.join(cache_path, f"{data_id}")
            with open(data_path, "rb") as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as m:
                    data = dill.loads(m.read())
            yield data

    def __len__(self) -> int:
        return len(self.source_datapipe)
