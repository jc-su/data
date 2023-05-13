import time
import psutil
from pynvml import *

class AgentInterface:
    def __init__(self, monitor_interval=1):
        self.monitor_interval = monitor_interval

    def get_status(self):
        pass

    def monitor(self):
        while True:
            status = self.get_status()
            print(status)
            time.sleep(self.monitor_interval)


class SystemAgent(AgentInterface):
    def get_status(self):
        cpu_status = self.get_cpu_status()
        memory_status = self.get_memory_status()
        return {'CPU': cpu_status, 'Memory': memory_status}

    def get_cpu_status(self):
        cpu_utilization = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq().current
        cpu_cores = psutil.cpu_count()
        return {'Utilization': cpu_utilization, 'Frequency': cpu_freq, 'Cores': cpu_cores}

    def get_memory_status(self):
        memory_details = psutil.virtual_memory()
        total_memory = memory_details.total
        available_memory = memory_details.available
        used_memory = memory_details.used
        memory_percent_used = memory_details.percent
        return {'Total': total_memory, 'Available': available_memory, 'Used': used_memory, 'Percent_used': memory_percent_used}


class GPUAgent(AgentInterface):
    def __init__(self, monitor_interval=1):
        super().__init__(monitor_interval)
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()

    def get_status(self):
        statuses = {}
        for i in range(self.device_count):
            statuses[f'Device_{i}'] = self.get_gpu_status(i)
        return statuses

    def get_gpu_status(self, device_id):
        handle = nvmlDeviceGetHandleByIndex(device_id)
        device_name = nvmlDeviceGetName(handle)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        total_memory = meminfo.total
        free_memory = meminfo.free
        used_memory = meminfo.used
        utilization = nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu
        memory_utilization = utilization.memory
        return {'Device_name': device_name, 'Total_memory': total_memory, 'Free_memory': free_memory, 'Used_memory': used_memory, 'GPU_utilization': gpu_utilization, 'Memory_utilization': memory_utilization}

if __name__ == '__main__':
    system_agent = SystemAgent(monitor_interval=1)
    gpu_agent = GPUAgent()

    system_agent.monitor()
    gpu_agent.monitor()
