import socket
import torch
import psutil
from os import environ

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


def get_cpu_status():
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    return {"cpu_count": cpu_count, "cpu_percent": cpu_percent}


def get_memory_status():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / 1e9
    used_memory = memory_info.used / 1e9
    available_memory = memory_info.available / 1e9
    memory_percent = memory_info.percent

    return {"total_memory": total_memory, "used_memory": used_memory, "available_memory": available_memory, "memory_percent": memory_percent}


def get_container_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def composite_trainer_status():
    status = {}
    status["gpu"] = get_gpu_status()
    status["memory"] = get_memory_status()
    status["local_ip_port"] = {"ip": get_container_ip(), "port":environ.get("port")}
    status["controller_ip_port"] = {"ip": environ.get("controller_ip"), "port": environ.get("controller_port")}

    return status


if __name__ == "__main__":
    get_gpu_status()
    get_cpu_status()
    get_memory_status()
