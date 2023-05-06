import torch.distributed.rpc as rpc

from torchdata import datapipes
from torchdata.dataloader2.graph._serialization import serialize_datapipe

from .utils import composite_trainer_status


class RemoteDataloader:
    r"""
    - controller IP: 
    - serialized_datapipe: bytes
    - local IP:
    """

    def __init__(self, _datapipe) -> None:
        self.status = composite_trainer_status()
        self.datapipe = _datapipe
        self.payload = {"status": self.status, "serialized_datapipe": serialize_datapipe(self.datapipe)}

    def __iter__(self):
        r"""
        Request next RPC batch from controller
        - serialized_datapipe: bytes
        """
        
        response = rpc.rpc_sync(self.status["controller_ip_port"], get_next_batch, args=(self.payload,))
        yield response
