import time
from torchdata.dataloader2.graph._serialization import (DataPipe, MapDataPipe,
                                                        clone,
                                                        deserialize_datapipe,
                                                        serialize_datapipe)
from torchdata.dataloader2.communication.queue import ThreadingQueue, LocalQueue
import threading
import zmq
from torch.utils.data.graph import traverse_dps


class Controller:
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:5556")
        self.incoming_queue = LocalQueue("incoming")
        self.completed_queue = LocalQueue("completed")

        # Initialize a counter for incoming requests
        self.incoming_counter = 0

    def start(self):
        while True:
            # Wait for the next request from the client
            message = self.socket.recv_pyobj()

            # Do some 'work'
            self.incoming_queue.put(message)

            # Increase the counter for every incoming request
            self.incoming_counter += 1

            # Check if it's time to process the batch of 10 requests
            if self.incoming_counter % 10 == 0:
                self.process_requests()

            # Send reply back to the client
            self.socket.send(b"Request received")

    def process_requests(self):
        # Add your code here to process every 10 requests
        print("Processing batch of 10 requests...")
        for _ in range(10):
            message = self.incoming_queue.get()
            print(message["status"])
            dp = deserialize_datapipe(message["serialized_datapipe"])
            # dp.batch.args[2].batch_size = 2
            for i in dp:
                print(i)
            # print(dp.batch.args[2].batch_size)
            self.completed_queue.put(message)


    def _add_task(self, fn, args, kwargs):
        self.incoming_queue.put(fn)

    def _get_completed_task(self, block=True, timeout=0):
        return self.completed_queue.get(block, timeout)

    def placement(self):
        pass


if __name__ == "__main__":
    controller = Controller()
    controller.start()
