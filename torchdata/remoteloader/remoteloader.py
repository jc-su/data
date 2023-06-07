from functools import reduce
import hashlib
import operator
import os
import threading
import time
from typing import Iterable, Optional, Union

import dill
import numpy as np
import zmq
from agent import GPUAgent
from utils import get_local_ip

from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.adapter import Adapter
from torchdata.dataloader2.reading_service import ReadingServiceInterface
from torchdata.datapipes.iter import IterableWrapper
from torchdata.dataloader2.graph import DataPipe, traverse_dps
dill.settings['recurse'] = True


class Worker:
    pass


class Trainer:
    pass


Service = Union[Worker, Trainer]


class RemoteDataloader(DataLoader2):
    r"""
    RemoteDataloader is a remote dataloader that can be used to load data from a remote datapipe.

    Args:
        datapipe: The datapipe to load data from.
        datapipe_adapter_fn: The adapter function to apply to the datapipe.
        reading_service: The reading service to use to load data from the datapipe.
        node_type: The type of the node, can be "TRAINER" or "PREPROCESSING_SERVICE"
    """

    def __init__(
        self,
        datapipe: Optional[DataPipe],
        datapipe_adapter_fn: Optional[Union[Iterable[Adapter], Adapter]] = None,
        reading_service: Optional[ReadingServiceInterface] = None,
        node_type: Optional[Union[Worker, Trainer]] = Trainer,
    ):
        super().__init__(datapipe, datapipe_adapter_fn, reading_service)

        # serialize_datapipe(self.datapipe) as id
        self.id = hashlib.sha1(dill.dumps(traverse_dps(self.datapipe))).hexdigest()

        self.local_ip = get_local_ip()
        self.local_port = os.environ.get("PORT")
        self.controller_ip = os.environ.get("CONTROLLER_IP")
        self.controller_port = os.environ.get("CONTROLLER_PORT")

        self.node_type = node_type

        self.ctx = zmq.Context()
        self.controller_skt = self.ctx.socket(zmq.REQ)
        self.controller_skt.connect(f"tcp://{self.controller_ip}:{self.controller_port}")

        self.worker_len = 1
        self.epoch = 0
        self.batch_size = 1

        if self.node_type == Trainer:
            # self.agent = GPUAgent()
            self.payload = {"type": "init", "id": self.id, "datapipe": dill.dumps(self.datapipe), "list_datapipe": dill.dumps(list_datapipe)}
            if self.reading_service is not None:
                self.payload["num_workers"] = self.reading_service.num_workers

            self._send_to_controller(self.payload)

            self.ctx = zmq.Context()
            self.pull_skt = self.ctx.socket(zmq.PULL)
            # self.pull_skt.bind("tcp://*:3000")
            self.pull_skt.connect(f"tcp://localhost:3000")
            self.poller = zmq.Poller()
            self.poller.register(self.pull_skt, zmq.POLLIN)
            print("Waiting for workers to connect")

        elif self.node_type == Worker:
            self.ctx = zmq.Context()
            self.push_skt = self.ctx.socket(zmq.PUSH)
            self.push_skt.connect("tcp://localhost:2000")

            import uuid
            import signal
            import sys
            self.uuid = uuid.uuid1().bytes

            def termination_handler(sig=None, frame=None):
                if os.getpid() == os.getpgid(0):
                    print("Terminating worker")
                    self._send_to_trainer("END")
                    sys.exit(1)
                sys.exit(0)
            signal.signal(signal.SIGTERM, termination_handler)
            signal.signal(signal.SIGINT, termination_handler)

    def __iter__(self):
        if self.node_type == Worker:
            yield from super().__iter__()
        elif self.node_type == Trainer:
            id_set = set()
            while True:
                events = dict(self.poller.poll())
                if self.worker_len == 0:
                    break
                if self.pull_skt in events and events[self.pull_skt] == zmq.POLLIN:
                    try:
                        id, messages = self.pull_skt.recv_multipart()
                        print(f"Received {int.from_bytes(id, 'big')}")
                        data = dill.loads(messages)
                        id_set.add(id)
                        if data == "END":
                            print(f"{int.from_bytes(id, 'big')} END")
                            id_set.discard(id)  # doesn't raise an error if the id doesn't exist
                            if not id_set:
                                break
                            continue

                        minibatch = [data for _ in id_set]
                        self.batch_size = len(minibatch)
                        # flatten the minibatch received from multiple workers
                        # functools.reduce has better performance than extending from a empty list
                        minibatch = reduce(operator.iconcat, minibatch, [])
                        yield minibatch
                    except Exception as e:
                        print(e)
                        continue
        else:
            raise Exception("Please specify the node type")

    def _send_to_controller(self, message):
        self.controller_skt.send_pyobj(message, flags=zmq.NOBLOCK)
        if self.controller_skt.recv_pyobj()["status"] != "OK":
            raise Exception("Failed to send message to controller")

    def _recv_from_controller(self):
        return self.controller_skt.recv_pyobj()

    def _send_to_trainer(self, message):
        self.push_skt.send_multipart([self.uuid, dill.dumps(message)])

    def _tweak_batch_size(self, _batch_size):
        self.datapipe.batch.args[2].batch_size = _batch_size

    def _tweak_num_workers(self, _num_workers):
        self.reading_service.num_workers = _num_workers

    def cal_loss_rank(self, loss_tensor):
        pass


if __name__ == "__main__":
    os.environ["PORT"] = "5555"
    os.environ["CONTROLLER_IP"] = "localhost"
    os.environ["CONTROLLER_PORT"] = "5556"

    def _map_fn(x):
        return np.array(x)

    def _map_fn2(x):
        return x + 1
    for i in range(1):
        list_datapipe = IterableWrapper(range((i + 1) * 10)).shuffle()
        dp = list_datapipe.sharding_filter().map(_map_fn2).map(lambda x: np.array([x, x + 1])).map(_map_fn).batch(124)
        # from torchdata.dataloader2.graph import DataPipe, traverse_dps
        # import torch
        # graph = traverse_dps(dp)

        # print(graph)
        # print(torch.utils.data.graph_settings.get_all_graph_pipes(graph))
        # for d in dp:
        #     print(d)
    rdl = RemoteDataloader(dp)
    # dl = DataLoader2(list_datapipe)
    for epoch in range(2):
        for idx, i in enumerate(rdl):
            print(f"batch_size: {len(i)}")
            print("-----")
        print("epoch", epoch)
        time.sleep(1)
        # per two iter
        # set.add(i[0])
        # if idx % 2 == 0:

        # time.sleep(1)
