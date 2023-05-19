import torch.distributed.rpc as rpc, RRef

from torchdata import datapipes
from torchdata.dataloader2.graph._serialization import serialize_datapipe, deserialize_datapipe

from .utils import composite_trainer_status


class RemoteDataloader:
    r"""
    - controller IP: 
    - serialized_datapipe: bytes
    - local IP:
    """

    def __init__(self, _datapipe, node_type, controller_name):
        self.status = composite_trainer_status()
        self.controller_name = controller_name
        if node_type == "COMPUTE":
            self.datapipe = _datapipe
            self.payload = {"status": self.status, "serialized_datapipe": serialize_datapipe(self.datapipe)}
        elif node_type == "PREPROCESSING":
            self.datapipe = None
            self.payload = {"status": self.status}

    def __iter__(self):
        r"""
        Request next RPC batch from controller
        - serialized_datapipe: bytes
        """
        if self.datapipe is not None:
            for data in self.datapipe:
                yield data
        else:
            # Assuming that the controller has a function called `get_next_batch`
            while True:
                serialized_datapipe = rpc.rpc_sync(self.controller_name, "get_next_batch", args=())
                datapipe = deserialize_datapipe(serialized_datapipe)
                for data in datapipe:
                    yield data
