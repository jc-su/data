import os
import threading
from typing import (Any, Dict, Generic, Iterable, Iterator, Optional, TypeVar,
                    Union)

import hashlib
import zmq
from utils import get_local_ip

from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.adapter import Adapter
from torchdata.dataloader2.graph._serialization import (DataPipe, MapDataPipe,
                                                        clone,
                                                        deserialize_datapipe,
                                                        serialize_datapipe)
from torchdata.dataloader2.reading_service import ReadingServiceInterface
from agent import GPUAgent, SystemAgent
from torchdata.datapipes.iter import IterableWrapper, Mapper
import time


class RemoteDataloader(DataLoader2):
    r"""
    RemoteDataloader is a remote dataloader that can be used to load data from a remote datapipe.

    Args:
        node_type: The type of the node, can be "TRAINER" or "PREPROCESSOR"
        *args: Arguments for :class:`torchdata.dataloader2.DataLoader2`
        **kwargs: Keyword arguments for :class:`torchdata.dataloader2.DataLoader2`
    """

    def __init__(
        self,
        datapipe: Optional[DataPipe],
        datapipe_adapter_fn: Optional[Union[Iterable[Adapter], Adapter]] = None,
        reading_service: Optional[ReadingServiceInterface] = None,
    ):
        super().__init__(datapipe, datapipe_adapter_fn, reading_service)

        self.status = []
        self.loss = []
        # serialize_datapipe(self.datapipe) as id
        self.id = hashlib.sha1(serialize_datapipe(self.datapipe)).hexdigest()

        self.ip = get_local_ip()
        self.port = os.environ.get("PORT")
        self.controller_ip = os.environ.get("CONTROLLER_IP")
        self.controller_port = os.environ.get("CONTROLLER_PORT")

        self.node_type = os.environ.get("NODE_TYPE")
        self.iter_num = 1
        self.agent = None

        self.ctx = zmq.Context()
        self.controller_skt = self.ctx.socket(zmq.REQ)
        self.controller_skt.connect(f"tcp://{self.controller_ip}:{self.controller_port}")

        self.recored_time = None

        if self.node_type == "TRAINER":
            # self.agent = GPUAgent()
            self.payload = {"type": "init", "id": self.id, "datapipe": serialize_datapipe(self.datapipe)}
            self.controller_skt.send_pyobj(self.payload)

            self.minibatch_recv = self.ctx.socket(zmq.PULL)
            self.minibatch_recv.bind("tcp://*:5557")

        elif self.node_type == "PREPROCESSOR":
            self.agent = SystemAgent()
            # try:
            #     self.datapipe = deserialize_datapipe(self.payload["serialized_datapipe"])
            #     self.trainer_ip = os.environ.get("trainer_ip")
            #     self.trainer_port = os.environ.get("trainer_port")
            #     self.minibatch_send = self.ctx.socket(zmq.PUSH)
            #     self.minibatch_send.connect(f"tcp://{self.trainer_ip}:{self.trainer_port}")

            # except Exception as e:
            #     print(e)
            #     raise Exception("Please initialize the trainer first")

    def __iter__(self):
        self.iter_num += 1
        if self.iter_num == 1:
            self.monitor_thread = threading.Thread(target=self.agent.record, args=(self.status,), daemon=True)
            self.monitor_thread.start()
            self.recored_time = time.time()
        else:
            self.recored_time = time.time() - self.recored_time

        if self.iter_num % 100 == 0:
            self.controller_skt.send_pyobj({"type": "status_record", "id": self.iter_num, "status": self.status, "loss": self.loss})
            self.status.clear()

        if self.node_type == "PREPROCESSOR":
            return super().__iter__()
        elif self.node_type == "TRAINER":
            return self.minibatch_recv.recv_pyobj()
        else:
            raise Exception("Please specify the node type")

    def _record_loss(self, loss):
        self.controller_skt.send_pyobj({"type": "loss_record", "id": self.iter_num, "loss": loss})

    def _tweak_batch_size(self, _batch_size):
        self.datapipe.batch.args[2].batch_size = _batch_size

    def _teak_num_workers(self, _num_workers):
        self.reading_service.num_workers = _num_workers

    def shutdown(self) -> None:
        r"""
        Shuts down ``ReadingService`` and clean up iterator.
        """
        try:
            if not self._terminated:
                self._terminated = True
                if self.reading_service is not None:
                    self.reading_service.finalize_iteration()
                    self.reading_service.finalize()
            if not self._reset_iter:
                self._reset_iter = True
                self._datapipe_iter = None
            self.monitor_thread.join()
        # Ignore AttributeError in case any attribute has been removed before `__del__`
        except AttributeError:
            pass


if __name__ == "__main__":
    os.environ["PORT"] = "5555"
    os.environ["CONTROLLER_IP"] = "localhost"
    os.environ["CONTROLLER_PORT"] = "5556"
    for i in range(1):
        dp = IterableWrapper(range((i + 1) * 10)).batch(i + 1)
        rdl = RemoteDataloader(dp)
        print(rdl)
