import base64
import threading
import time

import dill
import numpy as np
import zmq
from torch.utils.data.graph import traverse_dps, traverse

from torchdata.dataloader2.communication.queue import (LocalQueue,
                                                       ThreadingQueue)
from torchdata.dataloader2.graph import find_dps, replace_dp
from torchdata.datapipes.iter import IterDataPipe, Filter
from .utils import find_dp, spawn_worker


class Controller:
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:5556")
        self.incoming_queue = ThreadingQueue("incoming")
        self.completed_queue = ThreadingQueue("completed")
        self.incoming_counter = 0
        self.req_len = 2
        self.hash_table = {}

    def start(self):
        start_time = time.time()
        while True:
            # Wait for the next request from the client
            message = self.socket.recv_pyobj()
            print("Sending back a message")
            self.socket.send_pyobj({"status": "OK"})

            self.incoming_queue.put(message)
            self.incoming_counter += 1

            elapsed_time = time.time() - start_time

            # If the batch size has been reached or 10 seconds have passed, process the batch
            if self.incoming_counter % self.req_len == 0 or elapsed_time >= 10:
                threading.Thread(target=self.process_requests).start()
                # self.process_requests()
                start_time = time.time()  # Reset the start time

    def process_requests(self):
        def init_handler(message):
            id = message["id"]
            if id not in self.hash_table:
                self.hash_table[id] = {}
                self.hash_table[id]["gpu_status"] = []
                self.hash_table[id]["loss"] = []
                self.hash_table[id]["compute_time"] = []
                self.hash_table[id]["datapipe"] = message["datapipe"]
                self.hash_table[id]["worker_list"] = []
                if "num_worker" in message:
                    self.hash_table[id]["num_worker"] = message["num_worker"]

        def status_update_handler(message):
            self.hash_table[message["id"]]["gpu_status"].extend(message["gpu_status"])

        def loss_update_handler(message):
            self.hash_table[message["id"]]["loss_info"].extend(message["loss_info"])

        def compute_time_update_handler(message):
            self.hash_table[message["id"]]["compute_time_info"].extend(message["compute_time_info"])

        def forward_dp_handler(message):
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{message['ip']}:{message['port']}")
            socket.send_pyobj(message["datapipe"])

        message_handlers = {
            "init": init_handler,
            "status_update": status_update_handler,
            "loss_update": loss_update_handler,
            "time_update": compute_time_update_handler,
            "forward_dp": forward_dp_handler
        }

        for _ in range(self.req_len):
            message = self.incoming_queue.get()
            message_type = message["type"]
            print(f"Processing {message_type} from {message['id']}")

            handler = message_handlers.get(message_type)
            if handler is not None:
                handler(message)
            else:
                raise Exception(f"Unknown message type: {message_type}")
            # dp = dill.loads(message["datapipe"])
            # list_dp = dill.loads(message["list_datapipe"])
            # # print(self.remove_dp(dp))
            # data_list = [i for i in list_dp]
            # print(data_list)
            # new_dp = self.insert_dp(dp, data_list[:len(data_list)//2], list_dp)
            # for i in new_dp:
            #     print(i)

            self.completed_queue.put(message["id"])

    def _get_completed_task(self, block=True, timeout=0):
        return self.completed_queue.get(block, timeout)

    def placement(self):
        pass

    def encode_datapipe(self, serialized_datapipe):
        return base64.b64encode(serialized_datapipe).decode('utf-8')

    def get_filelist_dp(self, _datapipe):
        def recursive_remove(graph):
            for _, value in graph.items():
                if value[0].__str__() == "ShardingFilterIterDataPipe":
                    return value[1]
                else:
                    return recursive_remove(value[1])
        dps = []
        cache = set()

        def helper(g) -> None:  # pyre-ignore
            for dp_id, (dp, src_graph) in g.items():
                if dp_id in cache:
                    continue
                cache.add(dp_id)
                dps.append(dp)
                helper(src_graph)
        graph = traverse_dps(_datapipe)
        return recursive_remove(graph)

    def insert_dp(self, _datapipe, _list, _list_datapipe):
        F = Filter(_datapipe, lambda x: x in _list)
        print(F(find_dps(traverse_dps(_datapipe), _list_datapipe)))
        new_dp = replace_dp(
            traverse_dps(_datapipe),
            find_dp(traverse_dps(_datapipe), _list_datapipe),
            F(find_dp(traverse_dps(_datapipe), _list_datapipe))
        )

        return new_dp

    def importance_sampling(self, data, num_samples=100, replace=True):
        # Set seed for reproducibility
        np.random.seed(0)

        # Extract keys and corresponding 'loss_rank' values
        keys = []
        loss_info = []
        for key, value in data.items():
            keys.append(key)
            loss_info.append(value['loss_info'])
        loss_info = np.array(loss_info)

        # Compute mean and standard deviation of the data
        mean = np.max(loss_info)
        std_dev = np.std(loss_info)

        # Compute probabilities from the normal distribution
        probabilities = norm.pdf(loss_info, mean, std_dev)

        # Normalize the probabilities so they sum to 1
        probabilities = probabilities / np.sum(probabilities)

        # Use np.random.choice to generate samples
        samples = np.random.choice(keys, size=num_samples, replace=replace, p=probabilities)

        # shuffle samples
        np.random.shuffle(samples)

        return samples


if __name__ == "__main__":
    controller = Controller()
    controller.start()
