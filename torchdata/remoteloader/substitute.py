import hashlib
from typing import Iterator, Optional, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

U = Union[bytes, bytearray, str]

@functional_datapipe("substitute")
class SubstituteIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe[str], substitute_hash_list: Optional[list] = None):
        self.source_datapipe = source_datapipe
        self.substitute_hash_list = substitute_hash_list

    def __iter__(self) -> Iterator:
        if self.substitute_hash_list is not None:
            # Create a dictionary mapping hash ids to file names
            hash_to_file = {hashlib.md5(file_name.encode()).hexdigest(): file_name for file_name in self.source_datapipe}

            # For each hash id in substitute_hash_list, yield the corresponding file name
            for hash_id in self.substitute_hash_list:
                if hash_id in hash_to_file:
                    yield hash_to_file[hash_id]
        else:
            yield from self.source_datapipe

    def __len__(self) -> int:
        return len(self.source_datapipe)