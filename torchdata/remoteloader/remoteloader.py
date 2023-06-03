import queue
import hashlib
import os
import queue
import threading
import time
from typing import Iterable, Optional, Union

import zmq
from agent import GPUAgent, SystemAgent
from utils import get_local_ip

from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.adapter import Adapter
from torchdata.dataloader2.graph._serialization import DataPipe
from torchdata.dataloader2.reading_service import ReadingServiceInterface
from torchdata.datapipes.iter import IterableWrapper
import numpy as np
import dill
dill.settings['recurse'] = True


class RemoteDataloader(DataLoader2):
    r"""
    RemoteDataloader is a remote dataloader that can be used to load data from a remote datapipe.

    Args:
        datapipe: The datapipe to load data from.
        datapipe_adapter_fn: The adapter function to apply to the datapipe.
        reading_service: The reading service to use to load data from the datapipe.
        node_type: The type of the node, can be "TRAINER" or "PREPROCESSOR"
    """

    def __init__(
        self,
        datapipe: Optional[DataPipe],
        datapipe_adapter_fn: Optional[Union[Iterable[Adapter], Adapter]] = None,
        reading_service: Optional[ReadingServiceInterface] = None,
        node_type: str = "TRAINER",
        list_datapipe: Optional[DataPipe] = None,
    ):
        super().__init__(datapipe, datapipe_adapter_fn, reading_service)

        self.status = []
        self.loss = []
        # serialize_datapipe(self.datapipe) as id
        self.id = hashlib.sha1(dill.dumps(self.datapipe)).hexdigest()

        self.ip = get_local_ip()
        self.port = os.environ.get("PORT")
        self.controller_ip = os.environ.get("CONTROLLER_IP")
        self.controller_port = os.environ.get("CONTROLLER_PORT")

        self.node_type = node_type
        self.iter_num = 0

        self.ctx = zmq.Context()
        self.controller_skt = self.ctx.socket(zmq.REQ)
        self.controller_skt.connect(f"tcp://{self.controller_ip}:{self.controller_port}")

        self.recored_time = None

        self.worker_len = 2
        self.message_queue = queue.Queue(self.worker_len)

        if self.node_type == "TRAINER":
            # self.agent = GPUAgent()
            self.payload = {"type": "init", "id": self.id, "datapipe": dill.dumps(self.datapipe), "list_datapipe": dill.dumps(list_datapipe)}
            if self.reading_service is not None:
                self.payload["num_workers"] = self.reading_service.num_workers

            self.controller_skt.send_pyobj(self.payload)
            if self.controller_skt.recv_pyobj()["status"] != "OK":
                raise Exception("Failed to initialize the dataloader")

            self.minibatch_recv = self.ctx.socket(zmq.PULL)
            self.minibatch_recv.bind("tcp://*:5557")
            self.poller = zmq.Poller()
            self.poller.register(self.minibatch_recv, zmq.POLLIN)

        elif self.node_type == "PREPROCESSOR":
            pass

    def iter_sender(self):
        while True:
            events = dict(self.poller.poll())

            if self.minibatch_recv in events and events[self.minibatch_recv] == zmq.POLLIN:
                try:
                    messages = self.minibatch_recv.recv_pyobj()
                    yield messages
                except Exception as e:
                    print(e)
                    continue

    def __iter__(self):
        # self.monitor_thread = threading.Thread(target=self.agent.record, args=(self.status,), daemon=True)
        # self.monitor_thread.start()

        if self.node_type == "PREPROCESSOR":
            return super().__iter__()
        elif self.node_type == "TRAINER":
            while True:
                events = dict(self.poller.poll())
                if self.minibatch_recv in events and events[self.minibatch_recv] == zmq.POLLIN:
                    try:
                        messages = self.minibatch_recv.recv_pyobj()
                        if not self.message_queue.full():
                            self.message_queue.put(messages)
                            continue
                        else:
                            minibatch = []
                            while not self.message_queue.empty():
                                metadata = self.message_queue.get()
                                print("metadata", metadata['id'])
                                minibatch.extend(metadata["data"])
                                print("minibatch", minibatch)
                            yield minibatch
                    except Exception as e:
                        print(e)
                        continue
        else:
            raise Exception("Please specify the node type")

    def __next__(self):
        self.iter_num += 1
        if self.iter_num == 1:
            self.recored_time = time.time()
        self.recored_time = time.time() - self.recored_time

        if self.iter_num % 100 == 0 and self.node_type == "TRAINER":
            self.controller_skt.send_pyobj({"type": "status_update", "id": self.iter_num, "status": self.status, "loss": self.loss})
            self.status.clear()
        print(self.recored_time)

    def record_loss(self, loss):
        self.controller_skt.send_pyobj({"type": "loss_update", "id": self.iter_num, "loss": loss})
        if self.controller_skt.recv_pyobj()["status"] != "OK":
            raise Exception("Failed to record loss")
        
    def record_compute_time(self, recorded_time):
        self.controller_skt.send_pyobj({"type": "time_update", "id": self.iter_num, "time": recorded_time})
        if self.controller_skt.recv_pyobj()["status"] != "OK":
            raise Exception("Failed to record compute time")

    def _tweak_batch_size(self, _batch_size):
        self.datapipe.batch.args[2].batch_size = _batch_size

    def _tweak_num_workers(self, _num_workers):
        self.reading_service.num_workers = _num_workers


    def tweak(self):
        while True:
            message = self.controller_skt.recv_pyobj()
            self.controller_skt.send_pyobj({"status": "OK"})

            if message["type"] == "update_batchsize":
                self._tweak_batch_size(message["batch_size"])
            elif message["type"] == "update_num_workers":
                self._tweak_num_workers(message["num_workers"])
            elif message["type"] == "update_worker_len":
                self.worker_len = message["worker_len"]
            else:
                raise Exception("Unknown message type")

    def shutdown(self) -> None:
        super().shutdown()
        # self.monitor_thread.join()


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
        dp = list_datapipe.sharding_filter().map(_map_fn2).map(lambda x: np.array([x, x + 1])).map(_map_fn).batch(2)
        rdl = RemoteDataloader(dp, list_datapipe=list_datapipe)
        # dl = DataLoader2(list_datapipe)
        for idx, i in enumerate(rdl):
            print(i)
            print(len(i))
            print("-----")
            # time.sleep(1)
