from dataloader2.communication.queue import LocalQueue, ThreadingQueue
import threading

class Controller:
    def __init__(self, num_workers=4):
        self.incoming_queue = ThreadingQueue("incoming")
        self.completed_queue = ThreadingQueue("completed")
        self.workers = [threading.Thread(target=self.worker_function) for _ in range(num_workers)]

    def start(self):
        for worker in self.workers:
            worker.start()