from typing import Any, Callable, Iterator, Optional, Tuple, Union, Dict

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import dill
U = Union[bytes, bytearray, str]


@functional_datapipe("importance_sampler")
class ImportanceSamplerIterDataPipe(IterDataPipe[str]):
    def __init__(
        self,
        source_datapipe: IterDataPipe[Tuple[Any, U]],
        rank_info: Dict
    ):
        super().__init__()
        self.source_datapipe = source_datapipe
        self.rank_info = rank_info


    def __iter__(self) -> Iterator[str]:
        for file in self.source_datapipe:
            print(file)

            yield file

    def __len__(self) -> int:
        return len(self.source_datapipe)


if __name__ == "__main__":
    loss_info = dill.load(open("loss_time_archive/loss_tensor_0.pkl", "rb"))
    compute_time_info = dill.load(open("loss_time_archive/compute_time_list_0.pkl", "rb"))
    id_list = dill.load(open("loss_time_archive/id_list_0.pkl", "rb"))
    rank_info = dill.load(open("loss_time_archive/rank_info_0.pkl", "rb"))


    print(rank_info)