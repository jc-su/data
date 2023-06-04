import base64
import threading

import dill
import numpy as np
import scaling
import zmq
from torch.utils.data.graph import traverse_dps, traverse

from torchdata.dataloader2.communication.queue import (LocalQueue,
                                                       ThreadingQueue)
from torchdata.dataloader2.graph import find_dps, replace_dp
from torchdata.datapipes.iter import IterDataPipe, Filter
from utils import find_dp, spawn_worker


class Controller:
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:5556")
        self.incoming_queue = ThreadingQueue("incoming")
        self.completed_queue = ThreadingQueue("completed")
        self.incoming_counter = 0
        self.process_len = 2
        self.hash_table = {}

    def start(self):
        while True:
            # Wait for the next request from the client
            message = self.socket.recv_pyobj()
            self.socket.send_pyobj({"status": "OK"})

            self.incoming_queue.put(message)
            self.incoming_counter += 1

            # Check if it's time to process the batch of 10 requests
            if self.incoming_counter % self.process_len == 0:
                # self.process_requests()
                # # start a new thread
                threading.Thread(target=self.process_requests).start()

    def process_requests(self):
        for _ in range(self.process_len):
            message = self.incoming_queue.get()
            print(message["type"], message["id"])
            if message["type"] == "init":
                self.hash_table[message["id"]] = []
                self.hash_table[message["id"]]["status"] = []
                self.hash_table[message["id"]]["loss"] = []
                self.hash_table[message["id"]]["time"] = []
            elif message["type"] == "status_update":
                self.hash_table[message["id"]].append({"batch_size": message["batch_size"], "num_workers": message["num_workers"]})
            elif message["type"] == "loss_update":
                self.hash_table[message["id"]]["loss"].append({"loss": message["loss"]})
            elif message["type"] == "time_update":
                self.hash_table[message["id"]]["time"].append({"time": message["time"]})
            elif message["type"] == "new":
                spawn_worker(message["id"], "localhost", "5555", "localhost", "5556")
            else:
                raise Exception("Unknown message type")
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

    def sharding(self, data_list, _num_workers):
        shards = np.array_split(data_list, _num_workers)
        return shards

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
        # {ls: (dp, {ls: (dp, {ls: (dp, {})})})}
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

    def _sharding(self, _list, _num_workers):
        return scaling.sharding(_list, _num_workers)


if __name__ == "__main__":
    controller = Controller()
    controller.start()
