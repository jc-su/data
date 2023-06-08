import os
import mmap

from typing import Iterator

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("break_pipe")
class BreakIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe[str]):
        self.source_datapipe = source_datapipe

    def __iter__(self) -> Iterator:
        # TODO: break the datapipe
        for data in self.source_datapipe:
            break

    def __len__(self) -> int:
        return len(self.source_datapipe)
