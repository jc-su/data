import socket
import torch
import zmq
from os import environ
from torchdata.dataloader2.graph._serialization import deserialize_datapipe


def get_gpu_status():
    status = []
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()

        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            gpu_memory_cached = torch.cuda.memory_reserved(i) / 1e9
            status.append({
                "gpu_name": gpu_name,
                "gpu_memory_allocated": gpu_memory_allocated,
                "gpu_memory_cached": gpu_memory_cached
            })
        return status
    else:
        raise Exception("No GPU available")


def get_local_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address
