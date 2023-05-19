from torchdata.dataloader2.graph._serialization import deserialize_datapipe, serialize_datapipe
from dataloader2.communication.queue import ThreadingQueue
import threading

class Controller:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.incoming_queue = ThreadingQueue("incoming")
        self.completed_queue = ThreadingQueue("completed")
        self.workers = []

        # Start worker threads
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)

    def add_task(self, serialized_datapipe):
        self.incoming_queue.put(serialized_datapipe)

    def get_completed_task(self):
        return self.completed_queue.get()

    def _worker_loop(self):
        # TODO: Add support for placement policy
        # TODO: Add support for worker status

