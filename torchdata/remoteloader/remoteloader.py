import hashlib
import operator
import os
import threading
import time
from contextlib import contextmanager
from functools import reduce
from typing import Iterable, Optional, Union

import dill
import numpy as np
import zmq

from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.adapter import Adapter
from torchdata.dataloader2.graph import DataPipe, traverse_dps
from torchdata.dataloader2.reading_service import ReadingServiceInterface

from .agent import GPUAgent

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
        self.dp_id = hashlib.sha1(dill.dumps(traverse_dps(self.datapipe))).hexdigest()

        self.controller_ip = os.environ.get("CONTROLLER_IP")
        self.controller_port = os.environ.get("CONTROLLER_PORT")

        self.node_type = node_type

        self.ctx = zmq.Context()
        self.controller_skt = self.ctx.socket(zmq.REQ)
        self.controller_skt.connect(f"tcp://{self.controller_ip}:{self.controller_port}")

        self.worker_len = 1
        self.epoch = 0
        self.batch_size = 1
        self.compute_time_info = {}

        if self.node_type == Trainer:
            # self.agent = GPUAgent()
            # threading.Thread(target=self.agent.run).start()
            self.payload = {"type": "init", "id": self.dp_id, "datapipe": dill.dumps(self.datapipe)}
            if self.reading_service is not None:
                self.payload["num_workers"] = self.reading_service.num_workers

            self._send_to_controller(self.payload)

            self.ctx = zmq.Context()
            self.pull_skt = self.ctx.socket(zmq.PULL)
            # self.pull_skt.bind("tcp://*:3000")
            self.pull_skt.connect(f"tcp://localhost:3000")
            self.poller = zmq.Poller()
            self.poller.register(self.pull_skt, zmq.POLLIN)

        elif self.node_type == Worker:
            self.ctx = zmq.Context()
            self.push_skt = self.ctx.socket(zmq.PUSH)
            self.push_skt.connect("tcp://localhost:2000")

            import atexit
            import signal
            import sys
            import uuid

            # self.worker_id = os.getenv('HOSTNAME') + "-" + str(os.getpid())
            self.worker_id = str(uuid.uuid4()).encode("utf-8")
            self._send_to_controller({"type": "register", "id": self.worker_id})

            @atexit.register
            def termination_handler(sig=None, frame=None):
                if os.getpid() == os.getpgid(0):
                    self._send_to_trainer("END")
                    sys.exit(1)
                sys.exit(0)
            signal.signal(signal.SIGTERM, termination_handler)

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
                        worker_id, messages = self.pull_skt.recv_multipart()

                        data = dill.loads(messages)
                        id_set.add(worker_id)
                        if data == "END":
                            print(f"{worker_id.decode('utf-8')} ended")
                            id_set.discard(worker_id)  # doesn't raise an error if the id doesn't exist
                            if not id_set:
                                break
                            continue
                        print(f"Received {worker_id.decode('utf-8')}")
                        minibatch = [data for _ in id_set]

                        # flatten the minibatch received from multiple workers
                        # functools.reduce has better performance than extending from a empty list
                        minibatch = reduce(operator.iconcat, minibatch, [])
                        self.batch_size = len(minibatch)
                        yield minibatch
                    except Exception as e:
                        print(e)
                        continue
        else:
            raise Exception("Please specify the node type")
    
    @contextmanager
    def recored_compute_time(self, key):
        start = time.time()
        yield
        end = time.time()
        self.compute_time_list.append((key, end - start))

    def _send_to_controller(self, message):
        self.controller_skt.send_pyobj(message, flags=zmq.NOBLOCK)
        if self.controller_skt.recv_pyobj()["status"] != "OK":
            raise Exception("Failed to send message to controller")

    def _recv_from_controller(self):
        return self.controller_skt.recv_pyobj()

    def send_to_trainer(self, message):
        self.push_skt.send_multipart([self.worker_id, dill.dumps(message)])

    def _tweak_batch_size(self, _batch_size):
        self.datapipe.batch.args[2].batch_size = _batch_size

    def _tweak_num_workers(self, _num_workers):
        self.reading_service.num_workers = _num_workers

    def cal_loss_rank(self, loss_tensor):
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.node_type == Trainer:
            return super().__exit__(exc_type, exc_value, traceback)
        elif self.node_type == Worker:
            import sys
            if os.getpid() == os.getpgid(0):
                print("Terminating worker")
                self._send_to_trainer("END")
                sys.exit(1)
            sys.exit(0)
        else:
            raise Exception("Please specify the node type")
