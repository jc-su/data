import zmq
import time
import threading
import hashlib
from queue import Queue

class LoadBalancer:
    def __init__(self):
        self.proxy()

        self.available_workers = set()
        self.worker_heartbeat = {}
        self.ip_hash_map = {}

    def register_worker(self, worker_id):
        self.available_workers.add(worker_id)

    def deregister_worker(self, worker_id):
        self.available_workers.remove(worker_id)

    def monitor_workers(self):
        while True:
            for worker_id in list(self.available_workers):
                if time.time() - self.worker_heartbeat[worker_id] > 10:
                    self.deregister_worker(worker_id)
            time.sleep(1)

    def ip_hash(self, client_ip):
        worker_id = hashlib.sha1(client_ip.encode()).hexdigest() % len(self.available_workers)
        return worker_id

    def route_message(self):
        while True:
            client_id, _, client_ip, request = self.frontend.recv_multipart()
            worker_id = self.ip_hash(client_ip)
            if worker_id in self.available_workers:
                self.backend.send_multipart([worker_id, b"", client_id, request])

    def handle_backend(self):
        while True:
            worker_id, _, client_id, response = self.backend.recv_multipart()
            if response == b"register":
                self.register_worker(worker_id)
                self.worker_heartbeat[worker_id] = time.time()
            elif response == b"deregister":
                self.deregister_worker(worker_id)
            else:
                self.worker_heartbeat[worker_id] = time.time()
                self.frontend.send_multipart([client_id, b"", response])

    def start(self):
        monitor_thread = threading.Thread(target=self.monitor_workers)
        route_thread = threading.Thread(target=self.route_message)
        backend_thread = threading.Thread(target=self.handle_backend)

        monitor_thread.start()
        route_thread.start()
        backend_thread.start()

        monitor_thread.join()
        route_thread.join()
        backend_thread.join()

    def data_plane_proxy(self):
        context = zmq.Context()
        data_plane_frontend = context.socket(zmq.PULL)
        data_plane_frontend.bind("tcp://*:2000")
        data_plane_frontend.set_hwm(1)
        
        
        data_plane_backend = context.socket(zmq.PUSH)
        data_plane_backend.bind("tcp://*:3000")
        data_plane_backend.set_hwm(1)
        
        zmq.proxy(data_plane_frontend, data_plane_backend)


if __name__ == "__main__":
    lb = LoadBalancer()